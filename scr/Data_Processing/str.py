"""
High-level signal-strength extraction for satellite IQ recordings.

Replaces the old per-frame Python loop with the vectorised (M)SST transforms.
Computes strength-vs-time curves from STFT, SST and MSST matrices using both
the Zhao et al. TFSNR method and the lightweight percentile method.
"""
import numpy as np

from scr.Transforms import sst_chunk, msst_chunk
from .snr import tfsnr_band_power, percentile_strength, tfsnr_zhao


def extract_strength(x,
                     fs=25_000.0,
                     window="hann",
                     nperseg=511,
                     nfft=1024,
                     hop=250,
                     signal_cf=0.0,
                     signal_bw=3_000.0,
                     n_iter_msst=2,
                     gamma="relmax",
                     gamma_scale=1.0,
                     dtype=np.complex64,
                     max_memory_gigabytes=2.0,
                     keep_matrices=True):
    """Compute STFT / SST / MSST and derive strength-vs-time curves.

    For long records (e.g. 20M samples) keep ``dtype=np.complex64`` (halves
    output memory and roughly doubles speed). The full TF matrices scale with
    signal length (~0.03 GB per million samples at complex64, per matrix); set
    ``keep_matrices=False`` to drop them from the result after the curves are
    computed if you only need the strength-vs-time output.

    Returns a dict with the time axis, frequency axis, optionally the three TF
    matrices, and per-frame strength curves from both extraction methods.
    """
    x = np.ascontiguousarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1-D.")

    if nperseg % 2 == 0:
        nperseg += 1  # keep odd so a frame is centre-symmetric

    common = dict(n_size=nperseg, hop=hop, nfft=nfft, fs=fs, window=window,
                  gamma=gamma, gamma_scale=gamma_scale, dtype=dtype,
                  max_memory_gigabytes=max_memory_gigabytes, fftshift=True)

    print(f"[extract] {len(x):,} samples | nperseg={nperseg} nfft={nfft} "
          f"hop={hop} | SST + MSST(n_iter={n_iter_msst})")

    sst,  freqs, stft = sst_chunk(x, **common)
    msst, _,     _    = msst_chunk(x, n_iter=n_iter_msst, **common)

    n_frame = stft.shape[1]
    Lh = (nperseg - 1) // 2
    t = (Lh + np.arange(n_frame) * hop) / fs

    # Band-power strength (Zhao TFSNR uses the same band collapse for curves).
    P_stft, N_stft, snr_stft = tfsnr_band_power(stft, freqs, signal_cf, signal_bw)
    P_sst,  N_sst,  snr_sst  = tfsnr_band_power(sst,  freqs, signal_cf, signal_bw)
    P_msst, N_msst, snr_msst = tfsnr_band_power(msst, freqs, signal_cf, signal_bw)

    # Percentile strength on each transform.
    pc_stft, _, _ = percentile_strength(stft, freqs, signal_cf, signal_bw)
    pc_sst,  _, _ = percentile_strength(sst,  freqs, signal_cf, signal_bw)
    pc_msst, _, _ = percentile_strength(msst, freqs, signal_cf, signal_bw)

    # Zhao et al. TFSNR map — computed on the STFT, where the cross-spectrum
    # estimator is valid (SST/MSST concentrate energy into single complex
    # spikes whose adjacent-frame cross terms vanish, so TFSNR is not meaningful
    # on the reassigned matrices). Per-frame in-band TFSNR summarises the map.
    tfsnr_map = tfsnr_zhao(stft, axis_traces="time")
    sig_mask = np.abs(freqs - signal_cf) <= signal_bw
    tfsnr_curve = tfsnr_map[sig_mask, :].mean(axis=0)

    result = dict(
        t=t, freqs=freqs,
        stft=stft, sst=sst, msst=msst,
        P_stft=P_stft, P_sst=P_sst, P_msst=P_msst,
        N_stft=N_stft, N_sst=N_sst, N_msst=N_msst,
        snr_stft=snr_stft, snr_sst=snr_sst, snr_msst=snr_msst,
        pc_stft=pc_stft, pc_sst=pc_sst, pc_msst=pc_msst,
        tfsnr_map=tfsnr_map, tfsnr_curve=tfsnr_curve,
        signal_cf=signal_cf, signal_bw=signal_bw,
    )
    if not keep_matrices:
        result["stft"] = result["sst"] = result["msst"] = None
        result["tfsnr_map"] = None
    return result


# ----------------------------------------------------------------------
# IQ loaders / Doppler correction
# ----------------------------------------------------------------------
def load_iq_fc32(path):
    """Load interleaved complex64 (.fc32) IQ: I,Q,I,Q ... float32 pairs."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def load_doptrack_iq(path, dtype=np.int16):
    """Load a DopTrack L1B .dat file (interleaved int16 I/Q), normalised."""
    raw = np.fromfile(path, dtype=dtype)
    if raw.size % 2:
        raw = raw[:-1]
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    iq /= np.iinfo(dtype).max
    return iq.astype(np.complex64)


def doppler_correction(iq, fs, doppler_hz, t_doppler):
    """Remove a time-varying Doppler offset from baseband IQ."""
    t_samp = np.arange(len(iq)) / fs
    dop = np.interp(t_samp, t_doppler, doppler_hz)
    phase = -2.0 * np.pi * np.cumsum(dop) / fs
    return (iq * np.exp(1j * phase)).astype(np.complex64)
