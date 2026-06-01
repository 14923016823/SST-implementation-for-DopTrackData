"""
Multi-method strength comparison and batch processing over many recordings.
"""
from pathlib import Path
import json
import numpy as np

from scr.Transforms import sst_chunk, msst_chunk
from .ridge import track_ridge, ridge_to_hz
from .metrics import all_metrics
from .str import load_iq_fc32, load_doptrack_iq


def compare_methods(iq, fs, *,
                    nperseg=1023, nfft=2048, hop=250,
                    window="hann", signal_bw=5_000.0,
                    n_iter_msst=2, gamma="relmax", gamma_scale=0.15,
                    max_step_bins=8, ridge_smoothness=1.0,
                    window_enbw_factor=1.5,
                    dtype=np.complex64, max_memory_gigabytes=2.0) -> dict:
    """Compute STFT/SST/MSST, extract the carrier ridge from each, and run
    every metric on every transform. Returns a result dict that the plotting
    routines and batch driver consume.
    """
    common = dict(n_size=nperseg, hop=hop, nfft=nfft, fs=fs, window=window,
                  gamma=gamma, gamma_scale=gamma_scale, dtype=dtype,
                  max_memory_gigabytes=max_memory_gigabytes, fftshift=True)

    sst, freqs, stft = sst_chunk(iq, **common)
    msst, _, _       = msst_chunk(iq, n_iter=n_iter_msst, **common)

    # Time axis (centre of each frame).
    Lh = (nperseg - 1) // 2
    t = (Lh + np.arange(stft.shape[1]) * hop) / fs

    # Track the ridge on the STFT (most reliable — non-reassigned, smooth
    # magnitudes). The SST and MSST then measure power at the *same* ridge,
    # so all methods evaluate the carrier at the same TF locations.
    ridge = track_ridge(stft, max_step_bins=max_step_bins,
                        smoothness=ridge_smoothness)
    ridge_hz = ridge_to_hz(ridge, freqs)

    metrics = {
        "stft": all_metrics(stft, freqs, fs, ridge,
                            signal_bw=signal_bw,
                            window_enbw_factor=window_enbw_factor),
        "sst":  all_metrics(sst,  freqs, fs, ridge,
                            signal_bw=signal_bw,
                            window_enbw_factor=window_enbw_factor),
        "msst": all_metrics(msst, freqs, fs, ridge,
                            signal_bw=signal_bw,
                            window_enbw_factor=window_enbw_factor),
    }
    return dict(
        t=t, freqs=freqs,
        stft=stft, sst=sst, msst=msst,
        ridge=ridge, ridge_hz=ridge_hz,
        metrics=metrics,
        nperseg=nperseg, nfft=nfft, hop=hop, fs=fs, signal_bw=signal_bw,
    )


# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------
def batch_strength(file_paths, out_dir, fs, *,
                   loader=None, keep_matrices=False,
                   **kwargs):
    """Run compare_methods on every file in ``file_paths``.

    Writes per-file .npz outputs and a single summary JSON to ``out_dir``. Use
    ``loader=load_iq_fc32`` or ``loader=load_doptrack_iq``; if ``None`` the
    loader is chosen by suffix (``.fc32`` -> fc32, anything else -> DopTrack).

    Returns the summary dict (per-file median C/N0 etc.) for use in plots.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for p in file_paths:
        p = Path(p)
        if loader is None:
            ld = load_iq_fc32 if p.suffix == ".fc32" else load_doptrack_iq
        else:
            ld = loader
        print(f"[batch] {p.name} ...")
        iq = ld(p)
        res = compare_methods(iq, fs=fs, **kwargs)

        # Per-file summary statistics (median across the pass, for ranking).
        row = {"file": p.name, "n_samples": int(len(iq)),
               "duration_s": float(len(iq) / fs)}
        for tf_name, m in res["metrics"].items():
            for metric_name, curve in m.items():
                finite = curve[np.isfinite(curve)]
                row[f"{tf_name}_{metric_name}_median"] = (
                    float(np.median(finite)) if finite.size else float("nan"))
        summary.append(row)

        # Per-file arrays (curves only by default; full matrices are heavy).
        save = dict(t=res["t"], ridge_hz=res["ridge_hz"])
        for tf_name, m in res["metrics"].items():
            for metric_name, curve in m.items():
                save[f"{tf_name}_{metric_name}"] = curve
        if keep_matrices:
            save["stft"] = res["stft"]; save["sst"] = res["sst"]; save["msst"] = res["msst"]
            save["freqs"] = res["freqs"]
        np.savez(out_dir / f"strength_{p.stem}.npz", **save)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[batch] {len(summary)} files -> {out_dir}/summary.json")
    return summary
