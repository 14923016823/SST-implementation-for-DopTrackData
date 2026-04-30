import numpy as np
import matplotlib.pyplot as plt
from Transforms import SST

data_folder = "Data/"
file        = "FUNcube-1_39444_202601010247.fc32"

if __name__ == "__main__":

    x  = np.fromfile(data_folder + file, dtype=np.complex64)
    fs = 25_000      # Hz

    nfft    = 2048
    nperseg = 255
    hop     = 512

    f, t, x_stft, x_sst = SST(x, fs=fs, window='hann',
                               nfft=nfft, nperseg=nperseg, hop=hop)

    stft_db = 20 * np.log10(np.abs(x_stft) + 1e-12)
    sst_db  = 20 * np.log10(np.abs(x_sst)  + 1e-12)

    # --- Diagnostics: print the actual dB statistics -------------------------
    print(f"STFT dB: min={stft_db.min():.1f}  p1={np.percentile(stft_db,1):.1f}  "
          f"p50={np.percentile(stft_db,50):.1f}  p99={np.percentile(stft_db,99):.1f}  "
          f"max={stft_db.max():.1f}")
    print(f"SST  dB: min={sst_db.min():.1f}  p1={np.percentile(sst_db,1):.1f}  "
          f"p50={np.percentile(sst_db,50):.1f}  p99={np.percentile(sst_db,99):.1f}  "
          f"max={sst_db.max():.1f}")

    # --- Percentile-based colour scaling -------------------------------------
    # Use the 99.9th percentile as vmax and the 1st percentile as vmin.
    # This stretches the colormap across the actual data range, not noise floor.
    # Both panels use the SAME vmin/vmax so they are directly comparable.
    vmax = max(np.percentile(stft_db, 99.9), np.percentile(sst_db, 99.9))
    vmin = min(np.percentile(stft_db,  1.0), np.percentile(sst_db,  1.0))
    print(f"Colour range: vmin={vmin:.1f} dB  vmax={vmax:.1f} dB")

    # Waterfall: transpose so rows=time, cols=freq; origin='upper' -> t=0 at top
    stft_w = stft_db.T
    sst_w  = sst_db.T
    extent_w = [f[0], f[-1], t[-1], t[0]]   # [left, right, bottom, top]

    fig, axs = plt.subplots(1, 3, figsize=(20, 9))
    imshow_kw = dict(aspect='auto', origin='upper', extent=extent_w,
                     vmin=vmin, vmax=vmax, cmap='inferno')

    im0 = axs[0].imshow(stft_w, **imshow_kw)
    axs[0].set_title('STFT')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Time (s)')
    axs[0].axvline(0, color='white', lw=0.5, ls='--', alpha=0.5)
    fig.colorbar(im0, ax=axs[0], label='dB')

    im1 = axs[1].imshow(sst_w, **imshow_kw)
    axs[1].set_title('SST (synchrosqueezed)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Time (s)')
    axs[1].axvline(0, color='white', lw=0.5, ls='--', alpha=0.5)
    fig.colorbar(im1, ax=axs[1], label='dB')

    # Difference panel: anchored symmetrically around 0
    # Red = SST gained energy (sharp ridge), Blue = SST lost energy (smeared halo)
    diff = sst_db - stft_db
    # Use a tight percentile range so the ridge feature dominates, not outliers
    lim  = np.percentile(np.abs(diff), 99)
    im2  = axs[2].imshow(diff.T, aspect='auto', origin='upper', extent=extent_w,
                         cmap='RdBu_r', vmin=-lim, vmax=lim)
    axs[2].set_title('SST − STFT (dB)\nRed = energy gained, Blue = energy moved away')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Time (s)')
    axs[2].axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    fig.colorbar(im2, ax=axs[2], label='ΔdB')

    plt.suptitle(file, fontsize=9)
    plt.tight_layout()
    plt.savefig('Results/sst_output.png', dpi=150, bbox_inches='tight')
    plt.show()