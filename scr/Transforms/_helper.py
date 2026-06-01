"""
Helper functions for the (M)SST transforms.

Provides:
  - _check_signal_params : validate framing parameters, return n_chunk
  - _memory_load         : estimate peak memory of the batched transform
  - _resolve_gamma       : turn a fixed float OR an adaptive rule into a
                           per-chunk magnitude threshold

These functions deliberately have no dependency on the transform module so
they can be unit-tested in isolation.
"""
import numpy as np
from typing import Type


def _check_signal_params(signal_len: int,
                         n_size: int,
                         hop: int,
                         nfft: int) -> int:
    """Validate framing parameters and return the number of frames (chunks).

    Raises
    ------
    ValueError
        If nfft < n_size, hop/n_size are non-positive, or the signal is too
        short to yield a single frame.
    """
    if n_size <= 0:
        raise ValueError(f"n_size ({n_size}) must be positive")
    if hop <= 0:
        raise ValueError(f"hop ({hop}) must be positive")
    if nfft < n_size:
        raise ValueError(f"nfft ({nfft}) must be >= n_size ({n_size})")

    n_chunk = 1 + (signal_len - n_size) // hop
    if n_chunk <= 0:
        raise ValueError(
            f"Signal too short ({signal_len} samples) for n_size={n_size} / hop={hop}."
        )
    return n_chunk


def _memory_load(n_chunk: int,
                 nfft: int,
                 dtype: Type[np.number],
                 max_memory_gigabytes: float,
                 n_buffers: int = 5) -> float:
    """Estimate peak memory (GiB) for the batched (M)SST and raise if over cap.

    The dominant allocations are several complex arrays of shape
    (n_chunk, n_freq) with n_freq = nfft (two-sided FFT for complex IQ):
    the STFT, its derivative transform, the instantaneous-frequency map,
    the integer bin map and the output matrix. We budget ``n_buffers`` such
    arrays as a safe upper bound.

    Returns the estimate in GiB so the function doubles as a query, and
    raises MemoryError when it would exceed ``max_memory_gigabytes``.
    """
    n_freq = nfft  # two-sided spectrum for complex input
    itemsize = np.dtype(dtype).itemsize
    total_bytes = n_buffers * n_chunk * n_freq * itemsize
    total_gib = total_bytes / (1024 ** 3)

    if total_gib > max_memory_gigabytes:
        raise MemoryError(
            f"Estimated transform memory {total_gib:.2f} GiB exceeds the limit "
            f"of {max_memory_gigabytes:.2f} GiB. Process the signal in sub-batches "
            f"(see batch_size in sst/msst wrappers), lower nfft, or use a smaller dtype."
        )
    return total_gib


def _resolve_gamma(mag: np.ndarray,
                   gamma,
                   gamma_scale: float = 1.0) -> np.ndarray:
    """Resolve the noise/zero threshold applied to |STFT|.

    ``mag`` : (n_chunk, n_freq) magnitude array.

    ``gamma`` may be:
      - a float        -> fixed scalar threshold (broadcast to all rows),
      - "mad"          -> per-chunk median + k * 1.4826 * MAD (robust; the
                          recommended adaptive default),
      - "median"       -> per-chunk k * median magnitude,
      - "energy"       -> per-chunk k * RMS magnitude,
      - "percentile"   -> per-chunk magnitude percentile (k read as 0-100),
      - "relmax"       -> per-chunk k * max magnitude (k is a small fraction,
                          e.g. 1e-3). This is the recommended default for SST:
                          it rejects window-leakage skirts whose noisy phases
                          would otherwise disperse the reassigned energy.

    ``gamma_scale`` (k) tunes each rule. Returns an (n_chunk, 1) column so it
    broadcasts against ``mag`` row-wise: every time frame adapts to its own
    dynamic range rather than a single rigid global number.
    """
    if not isinstance(gamma, str):
        return np.full((mag.shape[0], 1), float(gamma), dtype=mag.dtype)

    rule = gamma.lower()

    if rule == "relmax":
        k = 0.15 if gamma_scale == 1.0 else gamma_scale
        thr = k * np.max(mag, axis=1, keepdims=True)
    elif rule == "mad":
        med = np.median(mag, axis=1, keepdims=True)
        madev = np.median(np.abs(mag - med), axis=1, keepdims=True)
        thr = med + gamma_scale * 1.4826 * madev
    elif rule == "median":
        thr = gamma_scale * np.median(mag, axis=1, keepdims=True)
    elif rule == "energy":
        thr = gamma_scale * np.sqrt(np.mean(mag ** 2, axis=1, keepdims=True))
    elif rule == "percentile":
        q = float(np.clip(gamma_scale, 0.0, 100.0))
        thr = np.percentile(mag, q, axis=1, keepdims=True)
    else:
        raise ValueError(
            f"Unknown gamma rule '{gamma}'. Use a float or one of "
            f"'mad', 'median', 'energy', 'percentile'."
        )

    # A silent frame has median/MAD == 0; keep a tiny floor so pure numerical
    # noise is never admitted as signal.
    thr = np.maximum(thr, np.finfo(mag.dtype).eps)
    return thr.astype(mag.dtype)
