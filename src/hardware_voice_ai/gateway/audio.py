"""PCM shaping for the gateway. Samples stay in memory."""

import numpy as np


def as_columns(samples: np.ndarray) -> np.ndarray:
    """Copy float32 PCM into shape ``(samples, channels)``."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("audio must have shape (samples, channels)")
    copied = np.array(data, dtype=np.float32, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError("audio samples must be finite")
    return copied


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels. Models in this gateway take one channel."""
    columns = as_columns(samples)
    return np.mean(columns, axis=1, dtype=np.float64).astype(np.float32)


def match_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    """Return ``(samples, channels)``. Mono is duplicated. Extra channels mix down."""
    if channels < 1:
        raise ValueError("channels must be at least 1")
    columns = as_columns(samples)
    if columns.shape[1] == channels:
        return columns
    mono = np.mean(columns, axis=1, dtype=np.float64).astype(np.float32)
    return np.repeat(mono.reshape(-1, 1), channels, axis=1)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. A production resampler can replace this call.

    ``samples`` is ``(samples,)`` or ``(samples, channels)``. The result
    keeps the same rank. Equal rates return a copy.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    vector = samples.ndim == 1
    columns = as_columns(samples)
    if src_rate == dst_rate:
        out = columns
    else:
        count = int(columns.shape[0])
        out_count = max(1, int(round(count * dst_rate / src_rate)))
        if count == 1 or out_count == 1:
            out = np.repeat(columns[:1], out_count, axis=0)
        else:
            position = np.linspace(0, count - 1, out_count)
            left = np.floor(position).astype(np.int64)
            right = np.minimum(left + 1, count - 1)
            frac = (position - left).astype(np.float32).reshape(-1, 1)
            out = columns[left] * (1.0 - frac) + columns[right] * frac
            out = out.astype(np.float32, copy=False)
    if vector:
        return np.ascontiguousarray(out.reshape(-1), dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def iter_chunks(samples: np.ndarray, frame: int):
    if frame < 1:
        raise ValueError("frame must be positive")
    for start in range(0, int(samples.shape[0]), frame):
        yield samples[start : start + frame]
