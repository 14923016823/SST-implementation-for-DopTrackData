"""
Plotting for the strength-extraction pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def _db(p):
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def plot_tf_panels(res, savepath=None, title_tag="", db_floor=-60.0):
    """Three stacked spectrograms: STFT, SST, MSST (power in dB)."""
    t, f = res["t"], res["freqs"]
    mats = [("STFT", res["stft"]), ("SST", res["sst"]), ("MSST", res["msst"])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, sharey=True)
    tag = f" — {title_tag}" if title_tag else ""
    fig.suptitle(f"Time-Frequency representations{tag}", fontsize=13, y=0.995)

    extent = [t[0], t[-1], f[0] / 1e3, f[-1] / 1e3]
    for ax, (name, M) in zip(axes, mats):
        P = _db(np.abs(M) ** 2)
        vmax = P.max()
        im = ax.imshow(P, origin="lower", aspect="auto", extent=extent,
                       vmin=vmax + db_floor, vmax=vmax, cmap="magma")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(name, loc="left", fontsize=11)
        fig.colorbar(im, ax=ax, label="Power (dB)", pad=0.01)
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140)
        print(f"[plot] TF panels -> {savepath}")
    plt.close(fig)
    return fig


def plot_comparison(res, savepath=None, title_tag="", db_floor=-60.0):
    """Two-panel comparison: STFT spectrogram with the tracked ridge overlaid,
    and C/N0 / peak power / band power / percentile-strength curves for STFT,
    SST and MSST on a shared time axis.
    """
    t, f = res["t"], res["freqs"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})
    tag = f" — {title_tag}" if title_tag else ""
    fig.suptitle(f"Strength-method comparison{tag}", fontsize=13, y=0.995)

    # Panel 1: STFT spectrogram with ridge overlay.
    ax = axes[0]
    P = _db(np.abs(res["stft"]) ** 2)
    vmax = P.max()
    extent = [t[0], t[-1], f[0] / 1e3, f[-1] / 1e3]
    im = ax.imshow(P, origin="lower", aspect="auto", extent=extent,
                   vmin=vmax + db_floor, vmax=vmax, cmap="magma")
    ax.plot(t, res["ridge_hz"] / 1e3, "c-", lw=1.0, alpha=0.85,
            label="tracked ridge")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("STFT with carrier ridge", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="Power (dB)", pad=0.01)

    # Panels 2-5: each metric, all three transforms.
    metric_titles = [
        ("cn0_dbhz", "C/N\u2080 (dB-Hz) — standard satellite-link metric"),
        ("peak_db",  "Carrier-bin power (dB, arbitrary)"),
        ("snr_db",   "In-band / out-of-band SNR (dB)"),
        ("pc_db",    "Percentile-method strength (dB)"),
    ]
    colours = {"stft": "steelblue", "sst": "tomato", "msst": "seagreen"}

    for ax, (key, title) in zip(axes[1:], metric_titles):
        for tf_name in ("stft", "sst", "msst"):
            ax.plot(t, res["metrics"][tf_name][key],
                    lw=1.0, color=colours[tf_name],
                    label=tf_name.upper(), alpha=0.85)
        ax.set_ylabel(key)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=3)
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140)
        print(f"[plot] comparison -> {savepath}")
    plt.close(fig)
    return fig


def plot_strength(res, smooth_s=1.0, savepath=None, title_tag=""):
    """Strength-vs-time: band power per transform + both extraction methods."""
    t = res["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0

    def smooth(a):
        if smooth_s <= 0 or dt <= 0:
            return a
        return gaussian_filter1d(a, sigma=max(smooth_s / dt, 1e-6))

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    tag = f" — {title_tag}" if title_tag else ""
    fig.suptitle(f"Signal strength vs time{tag}", fontsize=13, y=0.995)

    # Panel 1: band power (dB) for the three transforms.
    ax = axes[0]
    ax.plot(t, _db(smooth(res["P_stft"])), lw=1.2, color="steelblue", label="STFT")
    ax.plot(t, _db(smooth(res["P_sst"])),  lw=1.2, color="tomato", label="SST", alpha=0.85)
    ax.plot(t, _db(smooth(res["P_msst"])), lw=1.2, color="seagreen", label="MSST", alpha=0.85)
    ax.plot(t, _db(smooth(res["N_stft"])), lw=0.8, color="grey", ls="--", label="noise floor")
    ax.set_ylabel("Signal-band power (dB)")
    ax.set_title("In-band power (band-collapse method)", loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2: percentile-method strength (dB).
    ax = axes[1]
    ax.plot(t, _db(smooth(res["pc_stft"])), lw=1.2, color="steelblue", label="STFT")
    ax.plot(t, _db(smooth(res["pc_sst"])),  lw=1.2, color="tomato", label="SST", alpha=0.85)
    ax.plot(t, _db(smooth(res["pc_msst"])), lw=1.2, color="seagreen", label="MSST", alpha=0.85)
    ax.set_ylabel("Strength (dB)")
    ax.set_title("Percentile peak power / noise-floor", loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 3: noise-floor rejection — in-band power relative to noise floor.
    ax = axes[2]
    r_stft = _db(smooth(res["P_stft"])) - _db(smooth(res["N_stft"]))
    r_sst  = _db(smooth(res["P_sst"]))  - _db(smooth(res["N_sst"]))
    r_msst = _db(smooth(res["P_msst"])) - _db(smooth(res["N_msst"]))
    ax.plot(t, r_stft, lw=1.0, color="steelblue", label="STFT")
    ax.plot(t, r_sst,  lw=1.0, color="tomato",    label="SST")
    ax.plot(t, r_msst, lw=1.0, color="seagreen",  label="MSST")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("In-band / noise (dB)"); ax.set_xlabel("Time (s)")
    ax.set_title("Signal-band power above noise floor (higher = better SNR)",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140)
        print(f"[plot] strength -> {savepath}")
    plt.close(fig)
    return fig
