"""
Test / demo run for the (M)SST strength-extraction pipeline.

Usage
-----
    python test_run.py                      # synthetic CubeSat pass
    python test_run.py path/to/file.fc32    # real complex64 IQ
    python test_run.py path/to/file.dat     # DopTrack int16 IQ

Produces, in Test_Results/:
    tf_panels_<tag>.png    STFT / SST / MSST spectrograms
    strength_<tag>.png     strength-vs-time curves (both methods)
    strength_<tag>.npz     saved arrays
"""
import sys
from pathlib import Path
import numpy as np

# allow running from project root or from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.Data_Processing import (extract_strength, plot_tf_panels, plot_strength,
                                  load_iq_fc32, load_doptrack_iq)


def synthetic_pass(fs=25_000.0, duration=60.0, seed=0, doppler_arc=True):
    """A CubeSat-like pass on RAW IQ (no Doppler correction applied).

    With ``doppler_arc=True`` the carrier sweeps across the band (a classic
    S-curve, high -> low frequency) as it would in an uncorrected recording,
    so the strength curve is measured on a moving carrier.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    elev = np.sin(np.pi * t / duration)
    A = np.maximum(elev, 0.05) ** 0.5

    # Doppler frequency arc: tanh-shaped sweep from +f_max to -f_max.
    if doppler_arc:
        f_max = 4_000.0
        dop_f = -f_max * np.tanh(6.0 * (t / duration - 0.5))
    else:
        dop_f = np.zeros_like(t)
    carrier_ph = 2 * np.pi * np.cumsum(dop_f) / fs
    carrier = A * np.exp(1j * carrier_ph)

    baud = 1_200.0
    b0, b1 = int(duration / 3 * fs), int(2 * duration / 3 * fs)
    spb = int(fs / baud)
    n_bits = len(t) // spb + 2
    bits = rng.integers(0, 2, n_bits)
    fsk_dev = np.repeat(bits, spb)[:len(t)]
    fsk_dev = fsk_dev * 1_200 - 600
    fsk_ph = 2 * np.pi * np.cumsum(dop_f + fsk_dev) / fs   # FSK rides the arc
    fsk = np.zeros(len(t), dtype=complex)
    fsk[b0:b1] = 0.5 * A[b0:b1] * np.exp(1j * fsk_ph[b0:b1])

    noise = 0.25 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    return (carrier + fsk + noise).astype(np.complex64)


def main():
    out = Path("Test_Results"); out.mkdir(exist_ok=True)
    fs = 25_000.0
    signal_cf, signal_bw = 0.0, 5_000.0

    if len(sys.argv) > 1:
        p = Path(sys.argv[1]); tag = p.stem
        print(f"Loading {p} ...")
        iq = load_iq_fc32(p) if p.suffix == ".fc32" else load_doptrack_iq(p)
        # NO Doppler correction: strength is measured on the RAW IQ. The carrier
        # may sweep across the band during a pass, so the band is widened to span
        # the full Doppler excursion rather than parking the carrier at DC.
        # (load + doppler_correction remain available in scr.Data_Processing if a
        #  TLE-derived correction is wanted later.)
        signal_cf = 0.0
        signal_bw = 5_000.0
    else:
        print("No file given — generating synthetic pass.")
        iq = synthetic_pass(fs=fs); tag = "synthetic_pass"

    res = extract_strength(iq, fs=fs, window="hann",
                           nperseg=511, nfft=1024, hop=250,
                           signal_cf=signal_cf, signal_bw=signal_bw,
                           n_iter_msst=2, gamma="relmax", gamma_scale=1.0,
                           max_memory_gigabytes=2.0)

    plot_tf_panels(res, savepath=out / f"tf_panels_{tag}.png", title_tag=tag)
    plot_strength(res, smooth_s=1.0, savepath=out / f"strength_{tag}.png", title_tag=tag)

    np.savez(out / f"strength_{tag}.npz",
             t=res["t"], freqs=res["freqs"],
             P_stft=res["P_stft"], P_sst=res["P_sst"], P_msst=res["P_msst"],
             pc_stft=res["pc_stft"], pc_sst=res["pc_sst"], pc_msst=res["pc_msst"])
    print(f"Saved arrays -> {out / f'strength_{tag}.npz'}")


if __name__ == "__main__":
    main()
