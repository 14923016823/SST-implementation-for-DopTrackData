# SST / MSST strength extraction for satellite IQ

Vectorised, memory-bounded Synchrosqueezing (SST) and Multi-Synchrosqueezing
(MSST) transforms for long complex-baseband IQ recordings, with two
signal-strength estimators built on top.

## Layout

```
scr/
  Transforms/
    _helper.py   param checks, memory-load estimate, adaptive gamma threshold
    _window.py   analysis windows (hann/hamming/blackman/bartlett/rect/gaussian/kaiser)
    _chunks.py   zero-copy overlapping-frame view (as_strided)
    msst.py      _msst_core engine + sst_chunk / msst_chunk public wrappers
  Data_Processing/
    snr.py       tfsnr_zhao (paper method) + percentile_strength + band-power
    str.py       extract_strength driver + IQ loaders + Doppler correction
    str_plot.py  TF spectrogram panels + strength-vs-time plots
test_run.py      synthetic demo / real-file runner -> Test_Results/
```

## Quick start

```python
from scr.Data_Processing import extract_strength, plot_tf_panels, plot_strength

res = extract_strength(iq, fs=25_000.0, nperseg=511, nfft=1024, hop=250,
                       signal_cf=0.0, signal_bw=3_000.0,
                       n_iter_msst=2, gamma="relmax", dtype=np.complex64)
plot_tf_panels(res, savepath="tf.png")
plot_strength(res, savepath="strength.png")
```

Or run `python test_run.py` (synthetic) / `python test_run.py file.fc32` (real).

## Key design decisions (read these)

- **Complex IQ, two-sided spectrum.** The transforms use `np.fft.fft` /
  `fftfreq` (not `rfft`) because satellite IQ is complex baseband. Output
  matrices are oriented `(n_freq, n_frame)` and DC-centred via `fftshift`.

- **Vectorisation + memory bound.** All frames of a sub-batch are FFT'd at
  once. `max_memory_gigabytes` caps the *working* set by streaming sub-batches;
  results are bit-identical to a single batch. Note the *output* matrices still
  scale with signal length (~0.03 GB per 1M samples per matrix at complex64),
  so use `dtype=np.complex64` for long records and `keep_matrices=False` if you
  only need the strength curves. A 20M-sample run is ~12 s per transform and
  ~0.6 GB per output matrix at complex64.

- **Adaptive gamma.** The reassignment threshold can be a fixed float or a
  per-frame statistic: `"relmax"` (default, fraction of the frame's peak
  magnitude — the right choice for SST, gates the window-leakage skirts whose
  noisy phases otherwise disperse the reassigned energy), `"mad"`, `"median"`,
  `"energy"`, `"percentile"`. Tune with `gamma_scale`. For `"relmax"` the
  default fraction (~0.15) cleanly separates carrier from noise floor; lower it
  for weak signals, raise it for very strong ones.

- **Two strength estimators.**
  - `tfsnr_zhao` — faithful implementation of Zhao, Liu, Li & Jiang (2014),
    Eqs. (5)-(8): the adjacent-trace cross-spectrum SNR. For a single IQ stream
    the "adjacent traces" are adjacent **time frames** (carrier coherent
    frame-to-frame, noise decorrelated). **Computed on the STFT**, not on
    SST/MSST: reassignment concentrates energy into single complex spikes whose
    adjacent-frame cross terms vanish, so the estimator is only meaningful on a
    non-reassigned representation.
  - `percentile_strength` — pick in-band bins above a percentile, sum their
    power, normalise by signal-band width (Hz), divide by the noise-floor
    power. Lightweight and works directly on SST/MSST.

## Validation

On a 2 kHz tone in complex noise, energy concentration around the carrier bin:
STFT 0.54 -> SST 1.00 -> MSST 1.00, peak at the correct frequency. TFSNR on the
STFT reads ~1000+ at the carrier row vs ~0.4 off-carrier. Memory sub-batching
is bit-identical to single-batch.
