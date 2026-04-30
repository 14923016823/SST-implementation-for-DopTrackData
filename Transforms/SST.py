import numpy as np
from scipy.signal import get_window
from scipy.fft import fft, fftfreq, fftshift


def SST(x, fs=1.0, window='hann', nperseg=None, nfft=None, hop=None,
        threshold=None):
    """
    Synchrosqueezing Transform (SST) for real or complex IQ signals.

    For complex IQ (satellite data tuned to DC) the full two-sided spectrum
    [-fs/2, +fs/2] is returned, fftshift-ed so DC is in the centre row.

    Parameters
    ----------
    x         : 1-D array, real or complex input signal.
    fs        : float, sampling frequency in Hz.
    window    : str or tuple, window type for scipy.signal.get_window.
    nperseg   : int, window length (forced odd).
                Default: nfft // 8  (short window is CRITICAL for SST quality).
                SST needs a short window so the IF estimate is accurate within
                each frame. Unlike STFT, a longer window makes SST WORSE because
                the derivative estimate averages over too many different IF values.
    nfft      : int, FFT length (zero-padding for freq resolution). Default: 512.
    hop       : int, samples between successive frames.
                Default: nperseg // 4  (~75% overlap).
                SST requires overlap -- non-overlapping frames give no improvement
                over STFT.
    threshold : float, minimum |Xk| to attempt reassignment.
                Default: 1e-3 * mean(|x|^2)^0.5  (aggressive suppression of noise
                bins whose omega_hat values are random and scatter energy everywhere).
                Increase if SST looks noisy; decrease if signal is too faint.

    Returns
    -------
    f    : 1-D array, frequency axis (Hz), sorted low-to-high.
           Full two-sided [-fs/2, +fs/2] for complex input.
    t    : 1-D array, time axis (s), one entry per frame centre.
    TFR  : 2-D complex array (len(f) x n_frames), STFT.
    RTFR : 2-D complex array (len(f) x n_frames), synchrosqueezed TFR.

    Both TFR and RTFR are fftshift-ed: row 0 = most negative frequency.
    Display with:
        db = 20*np.log10(np.abs(RTFR) + 1e-12)
        plt.imshow(db, aspect='auto', origin='lower',
                   extent=[t[0],t[-1],f[0],f[-1]],
                   vmin=db.max()-50, vmax=db.max())
    """
    x = np.asarray(x)
    is_complex = np.iscomplexobj(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1-D.")
    xlen = x.shape[0]

    # --- parameters ----------------------------------------------------------
    if nfft is None:
        nfft = 512
    if nperseg is None:
        nperseg = max(7, nfft // 8)     # short window -- critical for SST
    if nperseg % 2 == 0:
        nperseg += 1
    if hop is None:
        hop = max(1, nperseg // 4)      # 75% overlap

    h      = get_window(window, nperseg).astype(float)
    Dh     = np.gradient(h)             # window derivative, sample units
    Lh     = (nperseg - 1) // 2
    h_norm = np.linalg.norm(h)

    n_out = nfft if is_complex else nfft // 2
    xi    = fftfreq(nfft, 1.0 / fs)    # bin frequencies in Hz, un-shifted
    df    = fs / nfft

    # Frame centres: clipped so window never reads outside x
    frame_centres = np.arange(Lh, xlen - Lh, hop)
    n_frames      = len(frame_centres)

    # Threshold: suppress noise bins whose omega_hat is meaningless
    if threshold is None:
        threshold = 1e-3 * np.sqrt(np.mean(np.abs(x) ** 2))

    print(f"SST: {xlen:,} samples | {n_frames:,} frames | "
          f"nperseg={nperseg} | nfft={nfft} | hop={hop} | "
          f"overlap={100*(1-hop/nperseg):.0f}%")

    TFR  = np.zeros((n_out, n_frames), dtype=np.complex128)
    RTFR = np.zeros((n_out, n_frames), dtype=np.complex128)

    k_vec = np.arange(nfft)

    for i, centre in enumerate(frame_centres):
        start = centre - Lh
        seg   = x[start : start + nperseg]

        buf    = np.zeros(nfft, dtype=np.complex128)
        buf_dh = np.zeros(nfft, dtype=np.complex128)
        buf   [:nperseg] = seg * h  / h_norm
        buf_dh[:nperseg] = seg * Dh / h_norm

        # Phase correction for window position (segment starts at 'start', not 0)
        phase = np.exp(-2j * np.pi * k_vec * start / nfft)
        Xk    = fft(buf)    * phase
        Xk_dh = fft(buf_dh) * phase

        TFR[:, i] = Xk[:n_out]

        # --- synchrosqueezing ------------------------------------------------
        # Instantaneous frequency operator:
        #   omega_hat[k] = xi[k]  -  fs/(2*pi) * Im( Xk_dh[k] / Xk[k] )
        #
        # Only applied to bins significantly above the noise floor.
        # Noise bins have random phase -> random omega_hat -> scatter RTFR.

        col      = Xk[:n_out]
        sig_mask = np.abs(col) > threshold
        if not np.any(sig_mask):
            continue

        j         = np.where(sig_mask)[0]
        tfr_sl    = col[j]
        tfr_dh_sl = Xk_dh[j]

        omega_hat = xi[j] - (fs / (2.0 * np.pi)) * np.imag(
            tfr_dh_sl / (tfr_sl + 1e-12)
        )

        if is_complex:
            k_vals = np.round(omega_hat / df).astype(int) % nfft
            valid  = np.ones(len(k_vals), dtype=bool)
        else:
            k_vals = np.round(omega_hat / df).astype(int)
            valid  = (k_vals >= 0) & (k_vals < n_out)

        np.add.at(RTFR[:, i], k_vals[valid], tfr_sl[valid])

    # --- fftshift so row 0 = most negative frequency -------------------------
    t = frame_centres / fs

    if is_complex:
        f    = fftshift(xi)
        TFR  = fftshift(TFR,  axes=0)
        RTFR = fftshift(RTFR, axes=0)
    else:
        f = xi[:n_out]

    return f, t, TFR, RTFR


# ---------------------------------------------------------------------------
def plot_sst(f, t, data, db_range=50, ax=None, title='', cmap='inferno'):
    """
    Display a TFR (STFT or SST) on a dB scale with correct axes.

    Parameters
    ----------
    f        : frequency axis (Hz) from SST().
    t        : time axis (s) from SST().
    data     : complex 2-D array from SST().
    db_range : dB window. Increase to show fainter features.
    ax       : matplotlib Axes, or None to create a new figure.
    title    : plot title.
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


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fs = 1000.0
    T  = 1.0
    t  = np.arange(int(fs * T)) / fs

    f0, f1 = -200.0, 200.0
    phase  = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t ** 2)
    x      = np.exp(1j * phase) + 0.05 * (
        np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    )

    # nfft=512 for frequency resolution, nperseg=63 (short!), hop=16
    f, t_out, TFR, RTFR = SST(x, fs=fs, nfft=512, nperseg=63, hop=16)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_sst(f, t_out, TFR,  db_range=40, ax=axes[0], title='STFT')
    plot_sst(f, t_out, RTFR, db_range=40, ax=axes[1], title='SST')
    plt.tight_layout()
    plt.savefig('sst_demo.png', dpi=150)
    plt.show()