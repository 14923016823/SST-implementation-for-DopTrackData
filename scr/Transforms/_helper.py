"""
Helper functions for the (M)SST transforms.
"""
import numpy as np
from typing import Type


def _check_signal_params(signal_len: int,
                         n_size: int,
                         hop: int,
                         nfft: int) -> int:
    """Validate framing parameters and return the number of frames (chunks)."""
    if n_size <= 0:
        raise ValueError(f"n_size ({n_size}) must be positive")
    if hop <= 0:
        raise ValueError(f"hop ({hop}) must be positive")
    if nfft < n_size:
        raise ValueError(f"nfft ({nfft}) must be >= n_size ({n_size})")

    n_chunk = 1 + (signal_len - n_size) // hop
    if n_chunk <= 0:
        raise ValueError(
            f"Signal too short ({signal_len} samples) for n_size={n_size} / hop={hop}."
        )
    return n_chunk


def _memory_load(n_chunk: int,
                 nfft: int,
                 dtype: Type[np.number],
                 max_memory_gigabytes: float,
                 n_buffers: int = 5) -> float:
    """Estimate peak memory (GiB) for the batched (M)SST and raise if over cap."""
    n_freq = nfft  # two-sided spectrum for complex input
    itemsize = np.dtype(dtype).itemsize
    total_bytes = n_buffers * n_chunk * n_freq * itemsize
    total_gib = total_bytes / (1024 ** 3)

    if total_gib > max_memory_gigabytes:
        raise MemoryError(
            f"Estimated transform memory {total_gib:.2f} GiB exceeds the limit "
            f"of {max_memory_gigabytes:.2f} GiB. Process the signal in sub-batches, "
            f"lower nfft, or use a smaller dtype."
        )
    return total_gib


def _resolve_gamma(mag: np.ndarray,
                   gamma,
                   gamma_scale: float = 1.0) -> np.ndarray:
    """Resolve the noise/zero threshold applied to |STFT|.

    gamma may be a float (fixed scalar) or a per-frame rule:
    "relmax" (default for SST), "mad", "median", "energy", "percentile".
    Returns an (n_chunk, 1) column that broadcasts row-wise.
    """
    if not isinstance(gamma, str):
        return np.full((mag.shape[0], 1), float(gamma), dtype=mag.dtype)

    rule = gamma.lower()

    if rule == "relmax":
        k = 0.15 if gamma_scale == 1.0 else gamma_scale
        thr = k * np.max(mag, axis=1, keepdims=True)
    elif rule == "mad":
        med = np.median(mag, axis=1, keepdims=True)
        madev = np.median(np.abs(mag - med), axis=1, keepdims=True)
        thr = med + gamma_scale * 1.4826 * madev
    elif rule == "median":
        thr = gamma_scale * np.median(mag, axis=1, keepdims=True)
    elif rule == "energy":
        thr = gamma_scale * np.sqrt(np.mean(mag ** 2, axis=1, keepdims=True))
    elif rule == "percentile":
        q = float(np.clip(gamma_scale, 0.0, 100.0))
        thr = np.percentile(mag, q, axis=1, keepdims=True)
    else:
        raise ValueError(
            f"Unknown gamma rule '{gamma}'. Use a float or one of "
            f"'relmax', 'mad', 'median', 'energy', 'percentile'."
        )

    thr = np.maximum(thr, np.finfo(mag.dtype).eps)
    return thr.astype(mag.dtype)