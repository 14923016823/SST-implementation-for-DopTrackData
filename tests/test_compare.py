"""
Run the multi-method strength comparison (STFT vs SST vs MSST × five metrics)
on synthetic or real IQ data.

    python tests/test_compare.py                  # synthetic Doppler-arc pass
    python tests/test_compare.py file.fc32        # real complex64 IQ
"""
import sys
from pathlib import Path
import numpy as np

# allow running from project root or from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.Data_Processing import (compare_methods, plot_comparison,
                                  load_iq_fc32, load_doptrack_iq)
from tests.test_run import synthetic_pass


def main():
    out = Path("Test_Results"); out.mkdir(exist_ok=True)
    fs = 25_000.0

    if len(sys.argv) > 1:
        p = Path(sys.argv[1]); tag = p.stem
        print(f"Loading {p} ...")
        iq = load_iq_fc32(p) if p.suffix == ".fc32" else load_doptrack_iq(p)
    else:
        print("No file given — synthetic Doppler-arc pass.")
        iq = synthetic_pass(fs=fs); tag = "synthetic_arc"

    res = compare_methods(
        iq, fs=fs,
        nperseg=1023, nfft=2048, hop=250,    # longer window: cleaner IF estimator
        window="hann", signal_bw=5_000.0,
        n_iter_msst=2, gamma="relmax", gamma_scale=0.15,
        max_step_bins=12, ridge_smoothness=1.0,
        window_enbw_factor=1.5,              # Hann
        dtype=np.complex64, max_memory_gigabytes=2.0,
    )

    plot_comparison(res, savepath=out / f"compare_{tag}.png", title_tag=tag)
    print(f"  median C/N0 STFT/SST/MSST (dB-Hz): "
          f"{np.nanmedian(res['metrics']['stft']['cn0_dbhz']):.1f}  "
          f"{np.nanmedian(res['metrics']['sst']['cn0_dbhz']):.1f}  "
          f"{np.nanmedian(res['metrics']['msst']['cn0_dbhz']):.1f}")

    np.savez(out / f"compare_{tag}.npz",
             t=res["t"], ridge_hz=res["ridge_hz"],
             **{f"{tf}_{m}": res["metrics"][tf][m]
                for tf in ("stft", "sst", "msst")
                for m in ("cn0_dbhz", "peak_db", "snr_db", "pc_db", "band_db")})
    print(f"  arrays -> {out / f'compare_{tag}.npz'}")


if __name__ == "__main__":
    main()
