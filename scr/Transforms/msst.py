"""
Batched Fourier (Multi-)Synchrosqueezing Transform for complex IQ signals.

Design
------
* Works on COMPLEX baseband IQ -> two-sided spectrum (np.fft.fft / fftfreq).
* Fully vectorised: all frames of a (sub-)batch are transformed at once.
* n_iter == 1  -> ordinary synchrosqueezing (SST).
  n_iter  > 1  -> multi-synchrosqueezing (MSST, Yu et al. 2019): the
                  instantaneous-frequency map is reassigned iteratively, then
                  energy is scattered once.
* Adaptive thresholding: gamma may be a float or a per-frame statistic
  ("mad" / "median" / "energy" / "percentile") via _resolve_gamma.
* Memory safety: the public wrappers split the signal into sub-batches of
  frames so a 20M-sample record never materialises one giant matrix.

Returned matrices are oriented (n_freq, n_frame) — frequency down rows, time
across columns — i.e. spectrogram orientation, ready for plotting and SNR.
"""
import numpy as np
from typing import Type, Tuple

from ._helper import _check_signal_params, _memory_load, _resolve_gamma
from ._window import _get_window
from ._chunks import _make_frames


# ----------------------------------------------------------------------
# Core engine — operates on one already-framed batch
# ----------------------------------------------------------------------
def _msst_core(frames: np.ndarray,
               win: np.ndarray,
               dwin: np.ndarray,
               nfft: int,
               fs: float,
               n_iter: int,
               gamma,
               gamma_scale: float,
               dtype) -> Tuple[np.ndarray, np.ndarray]:
    """Run (M)SST on a (n_frame, n_size) batch of complex frames.

    Returns (sst, stft), each (n_frame, n_freq) complex, n_freq == nfft.
    The shared frequency axis is produced once by the caller.
    """
    real_dtype = np.real(np.empty(0, dtype=dtype)).dtype

    # Two-sided STFT and the derivative-window STFT (for reassignment).
    stft   = np.fft.fft(frames * win,  n=nfft, axis=1).astype(dtype)
    stft_d = np.fft.fft(frames * dwin, n=nfft, axis=1).astype(dtype)

    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)            # (n_freq,) two-sided
    n_freq = freqs.size
    df = fs / nfft

    mag = np.abs(stft)
    thr = _resolve_gamma(mag, gamma, gamma_scale)       # (n_frame, 1)
    safe = mag > thr

    # Instantaneous frequency: f - Im(STFT_dwin / STFT) / (2*pi).
    inst_f = np.broadcast_to(freqs, stft.shape).astype(real_dtype).copy()
    correction = np.imag(stft_d[safe] / stft[safe]) / (2.0 * np.pi)
    inst_f[safe] -= correction

    # Map IF to nearest two-sided bin. fftfreq layout: bin = round(f/df) mod nfft
    # handles negative frequencies correctly via modulo wrap.
    bins = np.mod(np.round(inst_f / df).astype(np.intp), n_freq)

    # MSST: iterate the reassignment by following each bin's current target.
    for _ in range(n_iter - 1):
        bins = np.take_along_axis(bins, bins, axis=1)

    # Scatter STFT energy into target bins, per frame.
    sst = np.zeros_like(stft)
    rows = np.broadcast_to(np.arange(stft.shape[0])[:, None], stft.shape)
    np.add.at(sst, (rows[safe], bins[safe]), stft[safe])

    return sst, stft


# ----------------------------------------------------------------------
# Public driver with automatic sub-batching
# ----------------------------------------------------------------------
def msst_chunk(signal: np.ndarray,
               n_size: int,
               hop: int,
               nfft: int,
               fs: float = 1.0,
               window=None,
               n_iter: int = 1,
               gamma="relmax",
               gamma_scale: float = 1.0,
               max_memory_gigabytes: float = 2.0,
               dtype: Type[np.number] = np.complex128,
               fftshift: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched Fourier (M)SST over a complex IQ signal.

    Parameters
    ----------
    signal : 1-D complex (or real) array.
    n_size : frame / window length (samples).
    hop    : step between successive frames (samples).
    nfft   : FFT length (>= n_size). n_freq == nfft (two-sided).
    fs     : sampling frequency (Hz).
    window : window spec accepted by _get_window (default Hann).
    n_iter : 1 = SST, >1 = MSST iterations.
    gamma  : fixed float OR adaptive rule name ("mad"/"median"/"energy"/"percentile").
    gamma_scale : scale factor (k) for the adaptive rule.
    max_memory_gigabytes : cap; frames are processed in sub-batches to respect it.
    dtype  : complex working dtype.
    fftshift : if True, return spectra/axis with DC centred (np.fft.fftshift).

    Returns
    -------
    sst   : (n_freq, n_frame) complex — (multi-)synchrosqueezed transform.
    freqs : (n_freq,) Hz — two-sided frequency axis (centred if fftshift).
    stft  : (n_freq, n_frame) complex — original STFT (same orientation).
    """
    if n_iter < 1:
        raise ValueError(f"n_iter ({n_iter}) must be >= 1")

    signal = np.ascontiguousarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")

    n_chunk = _check_signal_params(len(signal), n_size, hop, nfft)

    win = _get_window(window, n_size)
    dwin = np.gradient(win) * fs
    real_dtype = np.real(np.empty(0, dtype=dtype)).dtype
    win = win.astype(real_dtype)
    dwin = dwin.astype(real_dtype)

    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    n_freq = freqs.size

    # Choose a sub-batch size of frames that fits the memory cap.
    itemsize = np.dtype(dtype).itemsize
    n_buffers = 5
    bytes_per_frame = n_buffers * n_freq * itemsize
    max_bytes = max_memory_gigabytes * (1024 ** 3)
    batch_frames = max(1, int(max_bytes // bytes_per_frame))
    # Validate the per-batch estimate (raises if even one batch is too big,
    # which only happens for absurd nfft/dtype combinations).
    _memory_load(min(batch_frames, n_chunk), nfft, dtype,
                 max_memory_gigabytes, n_buffers=n_buffers)

    all_frames = _make_frames(signal, n_size, hop, n_chunk)

    sst_out  = np.empty((n_chunk, n_freq), dtype=dtype)
    stft_out = np.empty((n_chunk, n_freq), dtype=dtype)

    for start in range(0, n_chunk, batch_frames):
        stop = min(start + batch_frames, n_chunk)
        batch = np.ascontiguousarray(all_frames[start:stop])
        sst_b, stft_b = _msst_core(batch, win, dwin, nfft, fs,
                                   n_iter, gamma, gamma_scale, dtype)
        sst_out[start:stop]  = sst_b
        stft_out[start:stop] = stft_b

    # Orient as (n_freq, n_frame); optionally centre DC.
    sst_out  = sst_out.T
    stft_out = stft_out.T
    if fftshift:
        freqs    = np.fft.fftshift(freqs)
        sst_out  = np.fft.fftshift(sst_out,  axes=0)
        stft_out = np.fft.fftshift(stft_out, axes=0)

    return sst_out, freqs, stft_out


def sst_chunk(signal: np.ndarray,
              n_size: int,
              hop: int,
              nfft: int,
              **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass synchrosqueezing transform (MSST with n_iter=1).

    Thin wrapper over ``msst_chunk``; accepts the same keyword arguments
    except ``n_iter`` which is forced to 1.
    """
    kwargs.pop("n_iter", None)
    return msst_chunk(signal, n_size, hop, nfft, n_iter=1, **kwargs)
