"""
Analysis-window construction for the (M)SST transforms.
"""
import numpy as np

_WINDOW_BUILDERS = {
    "hann":        lambda N: np.hanning(N),
    "hamming":     lambda N: np.hamming(N),
    "blackman":    lambda N: np.blackman(N),
    "bartlett":    lambda N: np.bartlett(N),
    "rectangular": lambda N: np.ones(N),
}


def _get_window(window, n_size: int) -> np.ndarray:
    """Return an analysis window of length ``n_size``.

    ``window`` may be:
      - a string naming a built-in window, optionally parameterised via a
        tuple, e.g. ("gaussian", 0.4) or ("kaiser", 8.0),
      - a 1-D array already of length n_size (validated and returned),
      - None -> defaults to a Hann window.

    Built-in names: 'hann', 'hamming', 'blackman', 'bartlett', 'rectangular',
    'gaussian' (param = std as a fraction of N), 'kaiser' (param = beta).
    """
    if window is None:
        window = "hann"

    if isinstance(window, np.ndarray):
        if window.shape != (n_size,):
            raise ValueError(
                f"window array length ({window.shape}) must equal n_size ({n_size})"
            )
        return window.astype(np.float64)

    param = None
    if isinstance(window, (tuple, list)):
        window, param = window[0], window[1]

    name = window.lower()

    if name in _WINDOW_BUILDERS:
        return _WINDOW_BUILDERS[name](n_size).astype(np.float64)

    if name == "gaussian":
        std_frac = 0.4 if param is None else float(param)
        n = np.arange(n_size) - (n_size - 1) / 2.0
        sigma = std_frac * (n_size - 1) / 2.0
        return np.exp(-0.5 * (n / sigma) ** 2)

    if name == "kaiser":
        beta = 8.0 if param is None else float(param)
        return np.kaiser(n_size, beta)

    raise ValueError(
        f"Unknown window '{name}'. Choose from "
        f"{sorted(set(_WINDOW_BUILDERS) | {'gaussian', 'kaiser'})}, "
        f"pass a length-{n_size} array, or None."
    )
