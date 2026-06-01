"""
Batched Fourier (Multi-)Synchrosqueezing Transform for complex IQ signals.
"""
import numpy as np
from typing import Type, Tuple

from ._helper import _check_signal_params, _memory_load, _resolve_gamma
from ._window import _get_window
from ._chunks import _make_frames


def _msst_core(frames, win, dwin, nfft, fs, n_iter, gamma, gamma_scale, dtype):
    """Run (M)SST on a (n_frame, n_size) batch of complex frames."""
    real_dtype = np.real(np.empty(0, dtype=dtype)).dtype

    stft   = np.fft.fft(frames * win,  n=nfft, axis=1).astype(dtype)
    stft_d = np.fft.fft(frames * dwin, n=nfft, axis=1).astype(dtype)

    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    n_freq = freqs.size
    df = fs / nfft

    mag = np.abs(stft)
    thr = _resolve_gamma(mag, gamma, gamma_scale)
    safe = mag > thr

    inst_f = np.broadcast_to(freqs, stft.shape).astype(real_dtype).copy()
    correction = np.imag(stft_d[safe] / stft[safe]) / (2.0 * np.pi)
    inst_f[safe] -= correction

    bins = np.mod(np.round(inst_f / df).astype(np.intp), n_freq)
    for _ in range(n_iter - 1):
        bins = np.take_along_axis(bins, bins, axis=1)

    sst = np.zeros_like(stft)
    rows = np.broadcast_to(np.arange(stft.shape[0])[:, None], stft.shape)
    np.add.at(sst, (rows[safe], bins[safe]), stft[safe])

    return sst, stft


def msst_chunk(signal, n_size, hop, nfft, fs=1.0, window=None, n_iter=1,
               gamma="relmax", gamma_scale=1.0, max_memory_gigabytes=2.0,
               dtype: Type[np.number] = np.complex128, fftshift=True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched Fourier (M)SST over a complex IQ signal.

    Returns (sst, freqs, stft); matrices oriented (n_freq, n_frame).
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

    itemsize = np.dtype(dtype).itemsize
    n_buffers = 5
    bytes_per_frame = n_buffers * n_freq * itemsize
    max_bytes = max_memory_gigabytes * (1024 ** 3)
    batch_frames = max(1, int(max_bytes // bytes_per_frame))
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

    sst_out  = sst_out.T
    stft_out = stft_out.T
    if fftshift:
        freqs    = np.fft.fftshift(freqs)
        sst_out  = np.fft.fftshift(sst_out,  axes=0)
        stft_out = np.fft.fftshift(stft_out, axes=0)

    return sst_out, freqs, stft_out


def sst_chunk(signal, n_size, hop, nfft, **kwargs
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass SST (MSST with n_iter=1). Same kwargs as msst_chunk."""
    kwargs.pop("n_iter", None)
    return msst_chunk(signal, n_size, hop, nfft, n_iter=1, **kwargs)