"""Local cue audio.

Offline and error cues are synthesized in this process. They do not use a
cloud path in either Audio Model setting. No cue audio is stored on disk.
"""

from enum import StrEnum

import numpy as np

CUE_RATE_HZ = 16_000


class LocalCue(StrEnum):
    OFFLINE = "offline"
    ERROR = "error"


def render_cue(cue: LocalCue, *, sample_rate_hz: int = CUE_RATE_HZ) -> np.ndarray:
    """Return float32 PCM shaped ``(samples, 1)`` for a local cue."""
    if cue is LocalCue.OFFLINE:
        return _pattern(sample_rate_hz, (220.0, 0.18))
    if cue is LocalCue.ERROR:
        return _pattern(sample_rate_hz, (880.0, 0.08), (0.0, 0.04), (880.0, 0.08))
    raise ValueError(f"unknown cue {cue!r}")


def _pattern(rate: int, *parts: tuple[float, float]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for frequency, duration in parts:
        count = int(duration * rate)
        if count <= 0:
            continue
        if frequency <= 0:
            chunks.append(np.zeros(count, dtype=np.float32))
            continue
        time = np.arange(count, dtype=np.float32) / rate
        chunks.append((0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32))
    if not chunks:
        wave = np.zeros(1, dtype=np.float32)
    else:
        wave = np.concatenate(chunks).astype(np.float32)
    _fade(wave, rate)
    return wave.reshape(-1, 1)


def _fade(wave: np.ndarray, rate: int) -> None:
    count = min(int(0.005 * rate), wave.shape[0] // 2)
    if count <= 0:
        return
    ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
    wave[:count] *= ramp
    wave[-count:] *= ramp[::-1]
