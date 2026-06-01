"""
Carrier-ridge extraction from a time-frequency magnitude matrix.

For a Doppler-shifted satellite signal the carrier traces an S-curve through
the TF plane. Tracking that ridge gives the carrier's instantaneous frequency
at every time frame, which lets us measure power *on the carrier* instead of
power averaged over a wide band.

The tracker is a Viterbi-style dynamic program:
  - state at time t = which frequency bin the carrier is in
  - emission cost  = -log magnitude (favours strong bins)
  - transition cost = quadratic in bin jump (favours smooth paths)

This handles fading and brief signal dropouts without losing track, as long as
the largest jump per frame stays within ``max_step_bins``.
"""
import numpy as np


def track_ridge(M: np.ndarray,
                max_step_bins: int = 8,
                smoothness: float = 1.0) -> np.ndarray:
    """Find the smooth carrier ridge through a TF matrix.

    Parameters
    ----------
    M : (n_freq, n_frame) complex or real matrix. Only |M| is used.
    max_step_bins : maximum frequency-bin change between consecutive frames.
        Set so that ``max_step_bins * df`` exceeds the steepest Doppler slope
        (Hz/frame) you expect — typically a few hundred Hz for LEO.
    smoothness : transition-cost weight. Higher = smoother ridge, more
        resistant to noise spikes; lower = more responsive to fast drifts.

    Returns
    -------
    ridge : (n_frame,) integer bin indices of the ridge at each frame.
    """
    mag = np.abs(M).astype(np.float64)
    n_freq, n_frame = mag.shape

    # Emission cost: -log magnitude. Add a tiny floor so silent bins are finite.
    floor = mag.max() * 1e-12 + 1e-30
    emit = -np.log(np.maximum(mag, floor))

    # Viterbi forward pass.
    cost = emit[:, 0].copy()
    back = np.empty((n_freq, n_frame), dtype=np.intp)
    back[:, 0] = np.arange(n_freq)

    step = np.arange(-max_step_bins, max_step_bins + 1)
    trans_pen = smoothness * (step.astype(np.float64) ** 2)

    for t in range(1, n_frame):
        # For each candidate target bin k, find the best source bin in
        # [k - max_step_bins, k + max_step_bins]. We do this vectorised by
        # shifting `cost` over all allowed jumps and stacking.
        stacked = np.full((len(step), n_freq), np.inf)
        for i, ds in enumerate(step):
            src_lo = max(0, -ds)
            src_hi = min(n_freq, n_freq - ds)
            dst_lo = src_lo + ds
            dst_hi = src_hi + ds
            stacked[i, dst_lo:dst_hi] = cost[src_lo:src_hi] + trans_pen[i]
        best = stacked.argmin(axis=0)
        cost = stacked[best, np.arange(n_freq)] + emit[:, t]
        back[:, t] = np.arange(n_freq) - step[best]               # source bin

    # Backward pass: trace the cheapest endpoint back to the start.
    ridge = np.empty(n_frame, dtype=np.intp)
    ridge[-1] = cost.argmin()
    for t in range(n_frame - 1, 0, -1):
        ridge[t - 1] = back[ridge[t], t]

    return ridge


def ridge_to_hz(ridge: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Convert ridge bin indices to Hz using the (already-fftshifted) freq axis."""
    return freqs[ridge]


def ridge_power(M: np.ndarray,
                ridge: np.ndarray,
                half_width: int = 1) -> np.ndarray:
    """Sum |M|^2 over a small frequency window centred on the ridge per frame.

    ``half_width`` bins on each side captures main-lobe energy without admitting
    much noise; 1 (i.e. 3 bins total) is a sensible default for a Hann window.
    """
    n_freq, n_frame = M.shape
    P = np.zeros(n_frame, dtype=np.float64)
    powr = np.abs(M) ** 2
    for dk in range(-half_width, half_width + 1):
        idx = np.clip(ridge + dk, 0, n_freq - 1)
        P += powr[idx, np.arange(n_frame)]
    return P
