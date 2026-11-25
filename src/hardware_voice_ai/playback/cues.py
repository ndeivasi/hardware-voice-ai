"""Local wake, offline, and error cues.

Cues are synthesized in this process. They do not read a file and they do
not use a cloud audio path. The same tones play for Audio Model ``local``
and ``remote``.
"""

from enum import StrEnum

import numpy as np

from hardware_voice_ai.audio.frame import AudioFormat

_AMPLITUDE = np.float32(0.25)


class Cue(StrEnum):
    """Local earcons. These are not spoken replies from the Audio Model."""

    WAKE = "wake"
    OFFLINE = "offline"
    ERROR = "error"


def render_cue(cue: Cue, fmt: AudioFormat) -> np.ndarray:
    """Render one cue at ``fmt`` as float32 PCM shaped ``(samples, channels)``.

    The amplitude stays at or below 0.25 so a later volume in ``[0, 1]``
    cannot push the cue into clipping. Each cue has a different pattern so
    a listener can tell them apart:

    - ``wake``: a short rising pair, 880 Hz then 1320 Hz
    - ``offline``: two low 220 Hz tones
    - ``error``: three short 660 Hz beeps
    """
    if not isinstance(cue, Cue):
        raise ValueError("cue must be wake, offline, or error")
    pieces = [_tone(freq, _samples(ms, fmt), fmt) for freq, ms in _pattern(cue)]
    mono = pieces[0]
    for piece in pieces[1:]:
        mono = np.concatenate((mono, piece))
    shaped = mono[:, None]
    if fmt.channels == 1:
        return shaped
    return np.repeat(shaped, fmt.channels, axis=1)


def _pattern(cue: Cue) -> tuple[tuple[float, int], ...]:
    if cue is Cue.WAKE:
        return ((880.0, 80), (0.0, 20), (1320.0, 80))
    if cue is Cue.OFFLINE:
        return ((220.0, 90), (0.0, 40), (220.0, 90))
    return ((660.0, 50), (0.0, 30), (660.0, 50), (0.0, 30), (660.0, 50))


def _samples(ms: int, fmt: AudioFormat) -> int:
    return max(1, fmt.sample_rate_hz * ms // 1000)


def _tone(freq: float, count: int, fmt: AudioFormat) -> np.ndarray:
    if freq <= 0.0:
        return np.zeros(count, dtype=np.float32)
    position = np.arange(count, dtype=np.float32)
    wave = np.sin(np.float32(2.0 * np.pi * freq) * position / np.float32(fmt.sample_rate_hz))
    faded = wave * _AMPLITUDE * _fade(count)
    return np.asarray(faded, dtype=np.float32)


def _fade(count: int) -> np.ndarray:
    ramp = min(64, count // 2)
    if ramp < 1:
        return np.ones(count, dtype=np.float32)
    envelope = np.ones(count, dtype=np.float32)
    slope = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    envelope[:ramp] = slope
    envelope[-ramp:] = slope[::-1]
    return envelope
