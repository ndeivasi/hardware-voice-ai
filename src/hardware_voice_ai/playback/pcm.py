"""In-memory PCM helpers for speaker playback.

Samples stay in process memory. Nothing here writes a file.
"""

import numpy as np


def as_pcm(samples: np.ndarray, channels: int) -> np.ndarray:
    """Copy float32 PCM into shape ``(samples, channels)``.

    A one-dimensional array is mono. The copy detaches the caller so a
    backend can reuse its buffer. Non-finite values are rejected.
    """
    if channels < 1:
        raise ValueError("channels must be at least 1")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        if channels != 1:
            raise ValueError(
                f"mono PCM cannot fill {channels} speaker channels"
            )
        data = data[:, None]
    if data.ndim != 2 or int(data.shape[1]) != channels:
        raise ValueError(
            f"PCM shape {data.shape} does not match {channels} channels"
        )
    copied = np.array(data, dtype=np.float32, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError("audio samples must be finite")
    return copied


def apply_volume(samples: np.ndarray, volume: float) -> np.ndarray:
    """Scale PCM by ``volume`` without hard-clipping.

    ``volume`` is in ``[0, 1]``. Audio already inside ``[-1, 1]`` stays
    inside that range, and unity gain leaves those samples unchanged.
    A chunk whose peak would exceed full scale is scaled down so the
    peak is 1. That is attenuation of the whole chunk, not a clipped
    waveform.
    """
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        return np.array(data, dtype=np.float32, copy=True)
    peak = float(np.max(np.abs(data)))
    if volume == 1.0 and peak <= 1.0:
        return np.array(data, dtype=np.float32, copy=True)
    gained = data * np.float32(volume)
    gained_peak = float(np.max(np.abs(gained))) if gained.size else 0.0
    if gained_peak > 1.0:
        gained = gained * np.float32(1.0 / gained_peak)
    return np.ascontiguousarray(gained, dtype=np.float32)
