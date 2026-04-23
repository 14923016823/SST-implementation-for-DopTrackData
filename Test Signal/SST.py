import numpy as np
from scipy.signal import get_window
from scipy.fft import fft, fftfreq, fftshift


def SST(x, fs=1.0, window='hann', nperseg=None, nfft=None, hop=None):
    """
    Synchrosqueezing Transform (SST) for large real or complex IQ signals.

    The signal is processed frame-by-frame with a configurable hop size,
    making it feasible for signals with millions of samples.

    Intended usage for satellite IQ data:
        - Set hop=nfft for non-overlapping frames (one FFT frame per column)
        - Each group of fs/nfft frames covers 1 second of data

    Parameters
    ----------
    x       : 1-D array, real or complex input signal.
    fs      : float, sampling frequency in Hz.
    window  : str or tuple, window type for scipy.signal.get_window.
    nperseg : int, window length (forced odd). Default: nfft - 1 (odd, full window).
    nfft    : int, FFT length. Default: 2048.
    hop     : int, number of samples to advance between frames.
              Default: nfft (non-overlapping frames).
              Use hop=1 for maximum time resolution (very slow for large x).
              Use hop=nfft for fast processing of large files.

    Returns
    -------
    f    : 1-D array, frequency axis (Hz), sorted low-to-high.
           Full two-sided [-fs/2, +fs/2] for complex input.
    t    : 1-D array, time axis (s), one entry per frame (frame centre time).
    TFR  : 2-D complex array, shape (len(f), n_frames), STFT.
    RTFR : 2-D complex array, shape (len(f), n_frames), synchrosqueezed TFR.

    Both TFR and RTFR are fftshift-ed so row 0 = most negative frequency.
    Display with imshow(..., origin='lower', extent=[t[0],t[-1],f[0],f[-1]]).
    """
    x = np.asarray(x)
    is_complex = np.iscomplexobj(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1-D.")
    xlen = x.shape[0]

    # --- parameters ----------------------------------------------------------
    if nfft is None:
        nfft = 2048
    if nperseg is None:
        nperseg = nfft - 1          # largest odd number <= nfft
    if nperseg % 2 == 0:
        nperseg += 1
    if hop is None:
        hop = nfft                  # non-overlapping by default

    h    = get_window(window, nperseg).astype(float)
    Dh   = np.gradient(h)
    Lh   = (nperseg - 1) // 2
    h_norm = np.linalg.norm(h)

    n_out = nfft if is_complex else nfft // 2

    # Frequency axis (un-shifted), in Hz
    xi = fftfreq(nfft, 1.0 / fs)   # shape (nfft,)
    df = fs / nfft                  # bin width in Hz

    # --- frame centres -------------------------------------------------------
    # Frame i is centred at sample index:  Lh + i * hop
    # We start at Lh so the first window fits entirely in the signal,
    # and stop before the last window would exceed the signal.
    frame_starts  = np.arange(Lh, xlen - Lh, hop)   # first sample in window
    n_frames      = len(frame_starts)
    frame_centres = frame_starts  # centre index = start + Lh, but we track start

    print(f"SST: {xlen} samples, {n_frames} frames, "
          f"hop={hop}, nperseg={nperseg}, nfft={nfft}")

    # --- output arrays -------------------------------------------------------
    TFR  = np.zeros((n_out, n_frames), dtype=np.complex128)
    RTFR = np.zeros((n_out, n_frames), dtype=np.complex128)

    k_vec     = np.arange(nfft)
    threshold = 1e-8 * np.mean(np.abs(x) ** 2)

    for i, centre in enumerate(frame_starts):
        # --- extract windowed segment ----------------------------------------
        start = centre - Lh
        end   = centre + Lh + 1     # exactly nperseg samples (no boundary issues
                                     # because we ensured Lh <= centre <= xlen-Lh-1)
        seg    = x[start:end]
        h_seg  = h
        dh_seg = Dh

        # --- zero-pad and FFT ------------------------------------------------
        buf    = np.zeros(nfft, dtype=np.complex128)
        buf_dh = np.zeros(nfft, dtype=np.complex128)

        buf   [:nperseg] = seg * h_seg  / h_norm
        buf_dh[:nperseg] = seg * dh_seg / h_norm

        Xk    = fft(buf)
        Xk_dh = fft(buf_dh)

        # Phase correction for window position (see derivation in docstring)
        # Corrects for the segment starting at 'start', not 0
        phase = np.exp(-2j * np.pi * k_vec * start / nfft)
        Xk    = Xk    * phase
        Xk_dh = Xk_dh * phase

        # Store STFT (positive half or full spectrum)
        TFR[:, i] = Xk[:n_out]

        # --- synchrosqueezing ------------------------------------------------
        col      = Xk[:n_out]
        sig_mask = np.abs(col) > threshold
        if not np.any(sig_mask):
            continue

        j         = np.where(sig_mask)[0]
        tfr_sl    = col[j]
        tfr_dh_sl = Xk_dh[j]

        # Instantaneous frequency in Hz
        omega_hat = xi[j] - (fs / (2.0 * np.pi)) * np.imag(
            tfr_dh_sl / (tfr_sl + 1e-12)
        )

        # Convert to bin index
        if is_complex:
            k_vals = np.round(omega_hat / df).astype(int) % nfft
            valid  = np.ones(len(k_vals), dtype=bool)
        else:
            k_vals = np.round(omega_hat / df).astype(int)
            valid  = (k_vals >= 0) & (k_vals < n_out)

        np.add.at(RTFR[:, i], k_vals[valid], tfr_sl[valid])

    # --- time axis (frame centre times in seconds) ---------------------------
    t = frame_starts / fs      # centre sample / fs = time in seconds

    # --- fftshift ------------------------------------------------------------
    if is_complex:
        f    = fftshift(xi)
        TFR  = fftshift(TFR,  axes=0)
        RTFR = fftshift(RTFR, axes=0)
    else:
        f   = xi[:n_out]
        # TFR and RTFR already built over n_out bins, no shift needed

    return f, t, TFR, RTFR


# ---------------------------------------------------------------------------
def plot_sst(f, t, data, db_range=50, ax=None, title='', cmap='inferno'):
    """
    Display a TFR (STFT or SST output) on a dB scale.

    Parameters
    ----------
    f        : frequency axis (Hz) from SST().
    t        : time axis (s) from SST().
    data     : complex or real 2-D array (len(f) x len(t)).
    db_range : float, dB window to display (default 50).
    ax       : matplotlib Axes, or None to create a new figure.
    title    : plot title string.
    cmap     : colormap name.

    Returns
    -------
    fig, ax, im
    """
    import matplotlib.pyplot as plt

    mag  = np.abs(data)
    eps  = mag[mag > 0].min() * 1e-3 if np.any(mag > 0) else 1e-12
    db   = 20.0 * np.log10(mag + eps)
    vmax = db.max()
    vmin = vmax - db_range

    extent = [t[0], t[-1], f[0], f[-1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    im = ax.imshow(db, aspect='auto', origin='lower', extent=extent,
                   vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, label='dB')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    return fig, ax, im