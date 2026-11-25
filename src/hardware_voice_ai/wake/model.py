"""Wake-model contract. Implementations stay on device."""

from dataclasses import dataclass
from typing import Protocol

from hardware_voice_ai.audio.frame import AudioFrame


@dataclass(frozen=True, slots=True)
class WakeScore:
    """One frame's wake confidence.

    ``confidence`` is in ``[0, 1]``. ``channel`` is the microphone column that
    produced it. Models may score every channel and report the strongest one.
    """

    confidence: float
    channel: int


class WakeModel(Protocol):
    """A swappable on-device wake model.

    ``score`` consumes one frame and returns a confidence. ``reset`` drops
    model memory (for mute, or after an accepted wake) without writing audio.
    """

    model_id: str
    wake_word: str

    def score(self, frame: AudioFrame) -> WakeScore: ...

    def reset(self) -> None: ...
