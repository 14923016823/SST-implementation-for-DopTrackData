import numpy as np
from scipy.signal import get_window, stft
from scipy.fft import fft, fftfreq, fftshift


def SST(x, fs=1.0, window='hann', nperseg=None, nfft=None, hop=None,
        threshold=None, n_iters=1, alpha=1.0):
    """
    Multi-Synchrosqueezing Transform (MSST).

    Performs `n_iters` successive synchrosqueezing passes on the STFT.
    Each pass reassigns energy using the instantaneous frequency operator,
    progressively sharpening the TF ridges.

    Parameters
    ----------
    x         : 1-D array, real or complex input signal.
    fs        : float, sampling frequency in Hz.
    window    : str or tuple, window type for scipy.signal.get_window.
    nperseg   : int, window length (forced odd).
    nfft      : int, FFT length. Default: 512.
    hop       : int, samples between frames. Default: nperseg // 4.
    threshold : float, minimum |Xk| for reassignment.
    n_iters   : int, number of SST passes (default 1 = standard SST).
                2-3 passes typically give MSST-quality sharpening.
                Beyond 4 passes rarely helps and may amplify noise.
    alpha   : float in (0, 1], step size for frequency reassignment.
              alpha=1.0 (default) is standard SST - full reassignment to
              omega_hat. alpha<1 is partial reassignment - energy moves
              only a fraction of the way toward omega_hat each iteration,
              which can give more stable convergence when components are
              close together and IF estimates are noisy.

    Returns
    -------
    f       : 1-D array, frequency axis (Hz).
    t       : 1-D array, time axis (s).
    TFR     : 2-D complex array, STFT (n_freqs x n_frames).
    RTFR    : 2-D complex array, final synchrosqueezed TFR after n_iters passes.
    RTFR_hist : list of n_iters arrays, the RTFR after each individual pass
                (useful for comparing convergence).
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
        nperseg = max(7, nfft // 8)
    if nperseg % 2 == 0:
        nperseg += 1
    if hop is None:
        hop = max(1, nperseg // 4)
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be within (0, 1], got {alpha}")

    h      = get_window(window, nperseg).astype(float)
    Dh     = np.gradient(h) * fs
    Lh     = (nperseg - 1) // 2
    h_norm = np.linalg.norm(h, ord=2)

    n_out = nfft if is_complex else nfft // 2
    xi    = fftfreq(nfft, 1.0 / fs)
    df    = fs / nfft

    frame_centres = np.arange(Lh, xlen - Lh, hop)
    n_frames      = len(frame_centres)

    if threshold is None:
        x_power = np.abs(x) ** 2
        kernel = np.ones(nperseg) / nperseg
        local_power = np.convolve(x_power, kernel, mode='same')
        threshold = 1e-6 * np.sqrt(local_power)[frame_centres]
    elif np.isscalar(threshold):
        threshold = np.full(n_frames, threshold)
    elif isinstance(threshold, (list, np.ndarray)):
        threshold = np.asarray(threshold)
        if threshold.shape != (n_frames,):
            raise ValueError(f"threshold array must have shape ({n_frames},), got {threshold.shape}")
    
    else:
        raise ValueError(f"threshold must be None, a scalar, or an array-like, got {type(threshold)}")

    print(f"MSST ({n_iters} iter{'s' if n_iters > 1 else ''}): "
          f"{xlen:,} samples | {n_frames:,} frames | "
          f"nperseg={nperseg} | nfft={nfft} | hop={hop} | "
          f"overlap={100*(1-hop/nperseg):.0f}%")

    # --- compute STFT and derivative STFT once -------------------------------

    TFR   = np.zeros((n_out, n_frames), dtype=np.complex128)
    TFR_dh = np.zeros((n_out, n_frames), dtype=np.complex128)

    k_vec = np.arange(nfft)

    for i, centre in enumerate(frame_centres):
        start = centre - Lh
        seg   = x[start : start + nperseg]

        buf    = np.zeros(nfft, dtype=np.complex128)
        buf_dh = np.zeros(nfft, dtype=np.complex128)
        buf   [:nperseg] = seg * h  / h_norm
        buf_dh[:nperseg] = seg * Dh / h_norm

        phase     = np.exp(-2j * np.pi * k_vec * start / nfft)
        Xk        = fft(buf)    * phase
        Xk_dh     = fft(buf_dh) * phase

        TFR   [:, i] = Xk   [:n_out]
        TFR_dh[:, i] = Xk_dh[:n_out]

    # --- multi-pass synchrosqueezing -----------------------------------------
    # Each pass takes the current RTFR as the spectrum to reassign.
    # On pass 1 this is just the STFT; on subsequent passes it is the
    # output of the previous squeeze, which is already sharper.
    # The IF operator (omega_hat) is always derived from the *original*
    # STFT ratio Xk_dh/Xk — this is what distinguishes MSST from
    # simply re-running SST on a new signal.

    RTFR_hist = []
    current   = TFR.copy()      # spectrum that gets reassigned each pass

    for iteration in range(n_iters):
        RTFR = np.zeros((n_out, n_frames), dtype=np.complex128)

        for i in range(n_frames):
            col    = current[:, i]
            col_dh = TFR_dh[:, i]      # derivative always from original STFT

            sig_mask = np.abs(col) > threshold[i]
            if not np.any(sig_mask):
                continue

            j         = np.where(sig_mask)[0]
            tfr_sl    = col[j]
            tfr_dh_sl = col_dh[j]

            # IF estimate — uses original Xk_dh but current Xk
            omega_hat = xi[j] - (1.0 / (2.0 * np.pi)) * np.imag(
                tfr_dh_sl / (tfr_sl + 1e-12)
            )
            if alpha < 1.0:
                # Partial reassignment: move only alpha of the way from
                # current bin frequency to omega_hat.
                # current_freq is the actual frequency of bin j.
                current_freq = xi[j]
                omega_hat    = current_freq + alpha * (omega_hat - current_freq)

            if is_complex:
                k_vals = np.round(omega_hat / df).astype(int) % nfft
                valid  = np.ones(len(k_vals), dtype=bool)
            else:
                k_vals = np.round(omega_hat / df).astype(int)
                valid  = (k_vals >= 0) & (k_vals < n_out)

            np.add.at(RTFR[:, i], k_vals[valid], tfr_sl[valid])

        RTFR_hist.append(RTFR.copy())
        current = RTFR          # next pass squeezes the already-squeezed result
        print(f"  pass {iteration+1}/{n_iters} done")

    # --- fftshift ------------------------------------------------------------
    t = (frame_centres - Lh) / fs
    t_last = (frame_centres[-1] + Lh) / fs
    t = np.hstack((t, t_last))

    if is_complex:
        f    = fftshift(xi)
        TFR  = fftshift(TFR,  axes=0)
        RTFR = fftshift(RTFR, axes=0)
        RTFR_hist = [fftshift(R, axes=0) for R in RTFR_hist]
    else:
        f = xi[:n_out]

    return f, t, TFR, RTFR, RTFR_hist


# ---------------------------------------------------------------------------
def plot_sst(f, t, data, db_range=50, ax=None, title='', cmap='turbo'):
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

    extent = [t[0], t[-1], f[-1], f[0]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    im = ax.imshow(db, aspect='auto', origin='upper', extent=extent,
                   vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, label='dB')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(title)
    ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    return fig, ax, im


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fs = 100
    t = np.arange(0, 4, 1/fs) 

    print(f"The time vector goes until: {t[-1]:.3f}")
    
    # --- Multicomponent test signal ---------
    A_1 = 1 - 0.6 * np.exp(-0.3 * t)
    A_2 = np.exp(-0.4*t)
    
    phi_1 =  2*np.pi*(38*t+3*np.sin(1.5*t))
    phi_2 = 2*np.pi*(15*t+2*np.sin(4.5*t))
    
    s_1 = A_1 * np.sin(phi_1)
    s_2 = A_2 * np.sin(phi_2)
    
    x = s_1 + s_2
    

    # --- Plot the time sequence ------------
    
    #plt.plot(t, x)    
    #plt.show()

    f, t_out, TFR, RTFR, RTFR_hist = SST(x, fs=fs, nperseg=33, nfft=1024, hop=1, n_iters=20, alpha=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(14, 12))
    plot_sst(t_out, f, TFR.T,  db_range=60, ax=axes[0, 0], title='STFT')
    for i in range(5):
        row = (i+1) // 3
        col = (i+1) % 3
        
        plot_sst(t_out, f, RTFR_hist[i].T, db_range=40, ax=axes[row, col], title=f'SST iteration: {i+1}')
    plt.tight_layout()
    plt.savefig('Test_Results/sst_demo.png', dpi=150)
    
    fig, ax = plt.subplots(figsize=(14,12))
    plot_sst(t_out, f, TFR.T, db_range=20, ax=ax, title="STFT")
    plt.savefig("Test_Results/Multicomponent_test_stft.png", dpi=150)
    
    fig, ax = plt.subplots(figsize=(14,12))
    plot_sst(t_out, f, RTFR_hist[0].T, db_range=20, ax=ax, title="Syncrhosqueezed STFT")
    plt.savefig("Test_Results/Multicomponent_test_sst.png", dpi=150)
    
    fig, ax = plt.subplots(figsize=(14,12))
    plot_sst(t_out, f, RTFR.T, db_range=30, ax=ax, title="Synchrosqueezed STFT 20-th iteration")
    plt.savefig("Test_Results/Multicomponent_test_sst20.png", dpi=150)
    #plt.show()
    
    # --- Plot power of the actual signals
    '''fig, axes = plt.subplots(3, figsize=(14,12))
    power_s_1 = 10*np.log(np.abs(s_1)**2+1e-12)
    power_s_2 = 10*np.log(np.abs(s_2)**2+1e-12)
    power_x = 10*np.log(np.abs(x)**2+1e-12)
    plot_list = [power_s_1, power_s_2, power_x]
    title_list = ["First signal component power", " Second signal component power", "Sum of first and second signal components power"]
    
    for i, ax in enumerate(axes):
        ax.plot(t, plot_list[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (dB)")
        #ax.set_title(title_list[i])
    plt.savefig('Test_Results/test_signal_power.png', dpi=150)'''