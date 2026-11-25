"""Local cues.

Cues are synthesized in this process. They do not read a file and they do
not use a cloud audio path. The same tones play for Audio Model ``local``
and ``remote``. ``thinking`` is an earcon. It is not a session state.
"""

from enum import StrEnum

import numpy as np

from hardware_voice_ai.audio.frame import AudioFormat

_DEFAULT_AMPLITUDE = 0.25


class Cue(StrEnum):
    """Local earcons. These are not spoken replies from the Audio Model."""

    WAKE = "wake"
    OFFLINE = "offline"
    ERROR = "error"
    THINKING = "thinking"
    LISTENING = "listening"
    FOLLOW_UP_END = "follow_up_end"


# (frequency Hz, duration ms, amplitude before volume). 0 Hz is silence.
_PATTERNS: dict[Cue, tuple[tuple[float, int, float], ...]] = {
    Cue.WAKE: (
        (880.0, 80, _DEFAULT_AMPLITUDE),
        (0.0, 20, _DEFAULT_AMPLITUDE),
        (1320.0, 80, _DEFAULT_AMPLITUDE),
    ),
    Cue.OFFLINE: (
        (220.0, 90, _DEFAULT_AMPLITUDE),
        (0.0, 40, _DEFAULT_AMPLITUDE),
        (220.0, 90, _DEFAULT_AMPLITUDE),
    ),
    Cue.ERROR: (
        (660.0, 50, _DEFAULT_AMPLITUDE),
        (0.0, 30, _DEFAULT_AMPLITUDE),
        (660.0, 50, _DEFAULT_AMPLITUDE),
        (0.0, 30, _DEFAULT_AMPLITUDE),
        (660.0, 50, _DEFAULT_AMPLITUDE),
    ),
    Cue.THINKING: ((440.0, 120, _DEFAULT_AMPLITUDE),),
    Cue.LISTENING: ((1100.0, 120, _DEFAULT_AMPLITUDE),),
    Cue.FOLLOW_UP_END: ((330.0, 80, 0.15),),
}


def render_cue(cue: Cue, fmt: AudioFormat) -> np.ndarray:
    """Render one cue at ``fmt`` as float32 PCM shaped ``(samples, channels)``.

    Amplitude is the tone's own level, at or below 0.25, so a later volume
    in ``[0, 1]`` cannot push the cue into clipping. Each cue has a different
    pattern:

    - ``wake``: a short rising pair, 880 Hz then 1320 Hz, amplitude 0.25
    - ``offline``: two low 220 Hz tones, amplitude 0.25
    - ``error``: three short 660 Hz beeps, amplitude 0.25
    - ``thinking``: one 440 Hz tone, 120 ms, amplitude 0.25
    - ``listening``: one 1100 Hz tone, 120 ms, amplitude 0.25
    - ``follow_up_end``: one 330 Hz tone, 80 ms, amplitude 0.15
    """
    if not isinstance(cue, Cue):
        raise ValueError(f"cue must be one of: {', '.join(item.value for item in Cue)}")
    pieces = [
        _tone(freq, _samples(ms, fmt), fmt, amplitude)
        for freq, ms, amplitude in _PATTERNS[cue]
    ]
    mono = pieces[0]
    for piece in pieces[1:]:
        mono = np.concatenate((mono, piece))
    shaped = mono[:, None]
    if fmt.channels == 1:
        return shaped
    return np.repeat(shaped, fmt.channels, axis=1)


def _samples(ms: int, fmt: AudioFormat) -> int:
    return max(1, fmt.sample_rate_hz * ms // 1000)


def _tone(freq: float, count: int, fmt: AudioFormat, amplitude: float) -> np.ndarray:
    if freq <= 0.0:
        return np.zeros(count, dtype=np.float32)
    position = np.arange(count, dtype=np.float32)
    wave = np.sin(np.float32(2.0 * np.pi * freq) * position / np.float32(fmt.sample_rate_hz))
    faded = wave * np.float32(amplitude) * _fade(count)
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
