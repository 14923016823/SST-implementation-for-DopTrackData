"""
Overlapping-frame generator shared by the transforms.
"""
import numpy as np


def _make_frames(signal: np.ndarray,
                 n_size: int,
                 hop: int,
                 n_chunk: int) -> np.ndarray:
    """Build a (n_chunk, n_size) view of overlapping frames without copying.

    Uses as_strided so all frames are exposed as one 2-D array and the FFT can
    run once over axis=1 instead of in a Python loop. The signal must be
    C-contiguous (callers ensure this with np.ascontiguousarray).
    """
    s = signal.strides[0]
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_chunk, n_size),
        strides=(hop * s, s),
        writeable=False,
    )
