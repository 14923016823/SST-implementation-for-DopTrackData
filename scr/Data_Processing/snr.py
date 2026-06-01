"""
Signal-strength / SNR extraction from time-frequency matrices.

Two methods are provided:

1. tfsnr_zhao  -- Time-Frequency domain SNR after Zhao, Liu, Li & Jiang
   (J. Appl. Geophys. 107, 2014). Multi-trace cross-spectrum estimator,
   Eqs. (5)-(8). For a single IQ stream we adapt the "adjacent traces" to be
   adjacent time frames (columns), which share a consistent carrier while
   noise decorrelates frame-to-frame.

2. percentile_strength -- a lightweight estimator: pick the strong bins in the
   signal band by percentile, sum their power, normalise by signal-band width
   and divide by the noise-floor power.

All functions take a TF matrix S of shape (n_freq, n_frame) — frequency down
rows, time across columns — matching the transforms' output orientation.
"""
import numpy as np


# ----------------------------------------------------------------------
# Method 1: paper TFSNR (Zhao et al. 2014)
# ----------------------------------------------------------------------
def tfsnr_zhao(S: np.ndarray,
               axis_traces: str = "time",
               eps: float = 1e-30) -> np.ndarray:
    """Time-frequency-domain SNR per Zhao et al. (2014), Eqs. (5)-(8).

    The signal power is estimated from the real part of the cross-spectrum of
    adjacent "traces", exploiting that signal is coherent across neighbours
    while noise is not::

        Px = mean over traces of |X|^2                                 (5)
        Ps = mean over adjacent pairs of Re[ X_i * conj(X_{i+1}) ]      (6)
        Pn = Px - Ps                                                    (7)
        SNR = Ps / Pn                                                   (8)

    Parameters
    ----------
    S : complex TF matrix (n_freq, n_frame).
    axis_traces : "time" treats adjacent time frames (columns) as adjacent
        traces -> returns a per-frequency, per-frame SNR map by sliding the
        pair estimate along time. "frequency" would treat rows as traces (rare;
        provided for completeness in true multi-trace gathers passed transposed).

    Returns
    -------
    snr : real array, same shape as S, the TFSNR(tau, f) map (clipped >= 0).

    Notes
    -----
    This is the genuine cross-spectrum estimator from the paper, applied to a
    single stream via temporal coherence. It is most meaningful for signals
    with a stable instantaneous frequency across neighbouring frames (carriers,
    slowly varying tones) — exactly the Doppler-corrected satellite case.
    """
    if axis_traces == "frequency":
        S = S.T  # treat rows as traces by transposing; result transposed back

    n_freq, n_frame = S.shape
    if n_frame < 2:
        raise ValueError("Need at least 2 frames/traces for the cross-spectrum.")

    # Px(tau,f): instantaneous power, per the paper averaged over the pair that
    # touches each frame. With the boundary handling of Eq. (5) the interior
    # term is |X_i|^2; we use the local two-frame mean so the map aligns in time.
    powr = np.abs(S) ** 2

    # Adjacent-pair cross term Re[X_i conj(X_{i+1})], length n_frame-1.
    cross = np.real(S[:, :-1] * np.conj(S[:, 1:]))      # (n_freq, n_frame-1)

    # Ps and Px aligned on the same frame grid: assign each interior frame the
    # mean of the pairs that include it; replicate at the boundaries.
    Ps = np.empty_like(powr)
    Px = np.empty_like(powr)

    # signal power = average of adjacent cross terms touching the frame
    Ps[:, 1:-1] = 0.5 * (cross[:, :-1] + cross[:, 1:])
    Ps[:, 0]    = cross[:, 0]
    Ps[:, -1]   = cross[:, -1]

    # trace power = average |X|^2 of the frame with its neighbour(s)
    Px[:, 1:-1] = 0.5 * (powr[:, :-2] + powr[:, 2:]) * 0.5 + 0.5 * powr[:, 1:-1]
    Px[:, 1:-1] = (powr[:, :-2] + 2.0 * powr[:, 1:-1] + powr[:, 2:]) / 4.0
    Px[:, 0]    = 0.5 * (powr[:, 0] + powr[:, 1])
    Px[:, -1]   = 0.5 * (powr[:, -1] + powr[:, -2])

    Pn = Px - Ps
    # Cross term can exceed Px or go negative for incoherent (noise) bins;
    # clip Pn to a small positive floor and SNR to >= 0 as the paper's ratio
    # is only meaningful where signal is coherent.
    Pn = np.maximum(Pn, eps)
    snr = np.maximum(Ps, 0.0) / Pn

    if axis_traces == "frequency":
        snr = snr.T
    return snr


def tfsnr_band_power(S: np.ndarray,
                     freqs: np.ndarray,
                     signal_cf: float,
                     signal_bw: float):
    """Collapse a TF matrix into per-frame signal / noise power using a band.

    Returns (P_signal, P_noise, snr_linear) each of length n_frame, where the
    signal band is |f - signal_cf| <= signal_bw and the rest is noise floor
    (mean power per noise bin). Useful for the strength-vs-time curves.
    """
    powr = np.abs(S) ** 2
    sig_mask = np.abs(freqs - signal_cf) <= signal_bw
    noise_mask = ~sig_mask

    P_signal = powr[sig_mask, :].sum(axis=0)
    if noise_mask.any():
        P_noise = powr[noise_mask, :].mean(axis=0)
    else:
        P_noise = np.zeros(S.shape[1])
    snr = P_signal / np.maximum(P_noise, 1e-30)
    return P_signal, P_noise, snr


# ----------------------------------------------------------------------
# Method 2: percentile-based strength
# ----------------------------------------------------------------------
def percentile_strength(S: np.ndarray,
                        freqs: np.ndarray,
                        signal_cf: float = 0.0,
                        signal_bw: float = None,
                        peak_percentile: float = 95.0,
                        noise_percentile: float = 50.0):
    """Lightweight per-frame strength estimator.

    For each time frame:
      1. restrict to the signal band (|f - signal_cf| <= signal_bw), if given;
      2. take bins above ``peak_percentile`` of in-band power as "peaks";
      3. sum the peak power and normalise by the signal-band width (Hz);
      4. estimate the noise floor as the ``noise_percentile`` of out-of-band
         power (median by default) and divide.

    Returns
    -------
    strength : (n_frame,) normalised peak power / noise-floor power (linear).
    P_peak   : (n_frame,) summed in-band peak power, width-normalised.
    P_noise  : (n_frame,) noise-floor power per bin.
    """
    powr = np.abs(S) ** 2
    n_freq, n_frame = powr.shape

    if signal_bw is not None:
        sig_mask = np.abs(freqs - signal_cf) <= signal_bw
    else:
        sig_mask = np.ones(n_freq, dtype=bool)
    noise_mask = ~sig_mask

    df = float(np.median(np.diff(freqs)))
    band_width = max(sig_mask.sum() * df, df)  # Hz spanned by the signal band

    in_band = powr[sig_mask, :]                # (n_sig, n_frame)

    # Per-frame peak threshold within the band.
    thr = np.percentile(in_band, peak_percentile, axis=0, keepdims=True)
    peaks = np.where(in_band >= thr, in_band, 0.0)
    P_peak = peaks.sum(axis=0) / band_width

    # Noise floor from out-of-band bins (or in-band low percentile if no
    # out-of-band bins exist).
    if noise_mask.any():
        P_noise = np.percentile(powr[noise_mask, :], noise_percentile, axis=0)
    else:
        P_noise = np.percentile(in_band, noise_percentile, axis=0)

    strength = P_peak / np.maximum(P_noise, 1e-30)
    return strength, P_peak, P_noise
