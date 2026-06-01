"""
Strength-metric collection with a unified interface.

Every metric takes a TF matrix S (n_freq, n_frame), the frequency axis (Hz),
the sample rate, and a ridge (optional, integer bin indices per frame), and
returns a per-frame strength curve in well-defined units.

Metrics implemented:
  - peak_power_db        : 10*log10 of the carrier-bin power (dB, arbitrary ref)
  - band_power_db        : 10*log10 of integrated in-band power (dB, arbitrary)
  - cn0_dbhz             : carrier-to-noise-density in dB-Hz — the standard
                           satellite-link metric. Carrier power on the ridge
                           divided by noise power spectral density estimated
                           from off-ridge bins.
  - snr_inband_db        : 10*log10 of (in-band power / mean noise-bin power)
  - percentile_strength  : the lightweight method from snr.py, in dB

All metrics consume the same TF matrix, so you can run them on the STFT, on
the SST, and on the MSST and compare directly.
"""
import numpy as np

from .ridge import ridge_power


def _to_db(x, floor=1e-30):
    return 10.0 * np.log10(np.maximum(x, floor))


def peak_power_db(S, freqs, fs, ridge, half_width=1):
    """Power summed over (2*half_width + 1) bins centred on the ridge, in dB."""
    return _to_db(ridge_power(S, ridge, half_width=half_width))


def band_power_db(S, freqs, fs, ridge=None,
                  signal_cf=0.0, signal_bw=5_000.0):
    """Integrated power in a fixed frequency band, in dB.

    If ``ridge`` is supplied the band is centred on the *median* ridge
    frequency instead of ``signal_cf`` — useful for Doppler-shifted carriers.
    """
    if ridge is not None:
        cf = float(np.median(freqs[ridge]))
    else:
        cf = signal_cf
    mask = np.abs(freqs - cf) <= signal_bw
    return _to_db((np.abs(S) ** 2)[mask, :].sum(axis=0))


def snr_inband_db(S, freqs, fs, ridge=None,
                  signal_cf=0.0, signal_bw=5_000.0):
    """In-band power / mean noise-bin power, in dB.

    Same banding logic as band_power_db; the noise floor is the per-frame mean
    of the out-of-band bins.
    """
    if ridge is not None:
        cf = float(np.median(freqs[ridge]))
    else:
        cf = signal_cf
    sig = np.abs(freqs - cf) <= signal_bw
    noise = ~sig
    powr = np.abs(S) ** 2
    P_sig = powr[sig, :].sum(axis=0)
    P_noise = powr[noise, :].mean(axis=0) if noise.any() else np.full(S.shape[1], 1e-30)
    return _to_db(P_sig / np.maximum(P_noise, 1e-30))


def cn0_dbhz(S, freqs, fs, ridge,
             carrier_half_width=1,
             noise_guard_bins=8,
             window_enbw_factor=1.5,
             nperseg=None):
    """Carrier-to-noise density ratio (dB-Hz) — the standard satellite metric.

    C/N0 = 10 * log10( C / N0 ) where
      C  = carrier power (W, arbitrary linear scale), summed near the ridge
      N0 = noise power spectral density (W/Hz), estimated from off-ridge bins

    Parameters
    ----------
    ridge : ridge bin indices per frame (required — defines where the carrier is).
    carrier_half_width : bins on each side of the ridge counted as carrier.
    noise_guard_bins : bins around the ridge excluded from the noise estimate
        to keep window leakage skirts out of the noise floor.
    window_enbw_factor : equivalent-noise-bandwidth factor of the analysis
        window (~1.5 for Hann, ~1.36 for Hamming, ~1.0 for rectangular).
    nperseg : window length used to compute S. Required if not equal to nfft;
        when None, assumed equal to nfft (== S.shape[0] before fftshift, which
        we read as len(freqs)).

    Returns
    -------
    cn0 : (n_frame,) C/N0 estimate in dB-Hz.

    Notes
    -----
    The noise PSD is estimated per frame as mean(|S|^2) over the bins outside
    the [ridge - guard, ridge + guard] region, divided by the bin width times
    the window's equivalent noise bandwidth. The carrier power is the sum of
    |S|^2 over the carrier window; noise leakage into that window is
    subtracted using the same per-bin noise estimate. The result is
    independent of the noise bandwidth chosen, so methods using different
    nfft can be compared on equal footing.
    """
    n_freq, n_frame = S.shape
    df = float(np.median(np.diff(freqs)))         # Hz per bin
    enbw_hz = window_enbw_factor * df             # equivalent noise bandwidth (Hz)

    powr = np.abs(S) ** 2
    cn0 = np.empty(n_frame, dtype=np.float64)

    # Indices for the carrier and guard windows are vectorised per frame.
    bins = np.arange(n_freq)
    for t in range(n_frame):
        k = ridge[t]
        car_lo = max(0, k - carrier_half_width)
        car_hi = min(n_freq, k + carrier_half_width + 1)
        gd_lo  = max(0, k - noise_guard_bins)
        gd_hi  = min(n_freq, k + noise_guard_bins + 1)

        noise_mask = (bins < gd_lo) | (bins >= gd_hi)
        if noise_mask.sum() == 0:
            cn0[t] = np.nan
            continue
        # Per-bin noise power estimate (mean to be robust to a few outliers).
        N_per_bin = np.median(powr[noise_mask, t])     # median is robust to weak interferers
        # Carrier window power, with the noise contribution removed.
        n_car = car_hi - car_lo
        C = powr[car_lo:car_hi, t].sum() - n_car * N_per_bin
        C = max(C, 1e-30)
        # Noise PSD: power per Hz = N_per_bin / enbw.
        N0 = N_per_bin / enbw_hz
        cn0[t] = 10.0 * np.log10(C / max(N0, 1e-30))
    return cn0


def percentile_strength_db(S, freqs, fs, ridge=None,
                           signal_cf=0.0, signal_bw=5_000.0,
                           peak_percentile=95.0, noise_percentile=50.0):
    """Percentile-method strength, in dB. Mirrors snr.percentile_strength.

    If ``ridge`` is supplied, the in-band region is centred on the median ridge.
    """
    if ridge is not None:
        cf = float(np.median(freqs[ridge]))
    else:
        cf = signal_cf
    powr = np.abs(S) ** 2
    sig_mask = np.abs(freqs - cf) <= signal_bw
    noise_mask = ~sig_mask
    df = float(np.median(np.diff(freqs)))
    band_width = max(sig_mask.sum() * df, df)

    in_band = powr[sig_mask, :]
    thr = np.percentile(in_band, peak_percentile, axis=0, keepdims=True)
    peaks = np.where(in_band >= thr, in_band, 0.0)
    P_peak = peaks.sum(axis=0) / band_width

    if noise_mask.any():
        P_noise = np.percentile(powr[noise_mask, :], noise_percentile, axis=0)
    else:
        P_noise = np.percentile(in_band, noise_percentile, axis=0)
    return _to_db(P_peak / np.maximum(P_noise, 1e-30))


# ----------------------------------------------------------------------
# Driver: compute every metric on a TF matrix
# ----------------------------------------------------------------------
def all_metrics(S, freqs, fs, ridge, *,
                signal_bw=5_000.0,
                window_enbw_factor=1.5,
                carrier_half_width=1) -> dict:
    """Compute every metric on one TF matrix.

    Returns
    -------
    dict with keys: peak_db, band_db, snr_db, cn0_dbhz, pc_db.
    """
    return dict(
        peak_db = peak_power_db(S, freqs, fs, ridge,
                                half_width=carrier_half_width),
        band_db = band_power_db(S, freqs, fs, ridge=ridge, signal_bw=signal_bw),
        snr_db  = snr_inband_db(S, freqs, fs, ridge=ridge, signal_bw=signal_bw),
        cn0_dbhz = cn0_dbhz(S, freqs, fs, ridge,
                            carrier_half_width=carrier_half_width,
                            window_enbw_factor=window_enbw_factor),
        pc_db   = percentile_strength_db(S, freqs, fs, ridge=ridge,
                                         signal_bw=signal_bw),
    )
