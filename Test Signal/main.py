import numpy as np
import matplotlib.pyplot as plt
from SST import SST, plot_sst
 
data_folder = "Data/"
file        = "FUNcube-1_39444_202601010247.fc32"
 
if __name__ == "__main__":
 
    # --- load ----------------------------------------------------------------
    x  = np.fromfile(data_folder + file, dtype=np.complex64)
    fs = 25_000          # Hz
    nfft    = 2048
    nperseg = 2047       # odd, slightly smaller than nfft for zero-padding
 
    # hop = nfft  -->  non-overlapping frames, one FFT column per nfft samples
    # With fs=25000 and nfft=2048:
    #   frames per second  = 25000 / 2048 ≈ 12.2  frames/s
    #   total frames       = 16_500_000 / 2048  ≈ 8_057  columns
    #   output matrix size = 2048 rows × 8_057 cols  (manageable)
    hop = nfft
 
    print(f"Signal: {len(x):,} samples  |  {len(x)/fs:.1f} s  |  "
          f"~{len(x)//hop} frames")
 
    f, t, x_stft, x_sst = SST(x, fs=fs, window='hann', nperseg=nperseg, nfft=nfft, hop=hop)
 
    # --- dB conversion -------------------------------------------------------
    sst_db  = 20 * np.log10(np.abs(x_sst)  + 1e-12)
    stft_db = 20 * np.log10(np.abs(x_stft) + 1e-12)
 
    # --- plot ----------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
 
    plot_sst(f, t, x_stft, db_range=50, ax=axs[0], title='STFT')
    plot_sst(f, t, x_sst,  db_range=50, ax=axs[1], title='SST (synchrosqueezed)')
 
    # Difference panel: diverging colormap centred at 0
    diff = sst_db - stft_db
    ext  = [t[0], t[-1], f[0], f[-1]]
    lim  = np.percentile(np.abs(diff), 98)   # robust symmetric limits
    im   = axs[2].imshow(diff, aspect='auto', origin='lower', extent=ext,
                         cmap='RdBu_r', vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axs[2], label='ΔdB')
    axs[2].set_title('SST − STFT (dB)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Frequency (Hz)')
    axs[2].axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
 
    plt.tight_layout()
    plt.savefig('sst_output.png', dpi=150)
    plt.show()
    print("Saved sst_output.png")
 
