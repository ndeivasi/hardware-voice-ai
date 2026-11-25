"""Shared fixtures for wake-word tests. No microphone and no network."""

from dataclasses import dataclass, field

import numpy as np

from hardware_voice_ai.audio.frame import AudioFormat, AudioFrame
from hardware_voice_ai.capture.port import CaptureRequest, CaptureStopReason
from hardware_voice_ai.wake.events import WakeEvent
from hardware_voice_ai.wake.model import WakeScore


@dataclass
class ScriptedModel:
    """Confidence script. ``reset`` does not rewind the script."""

    scores: list[float]
    wake_word: str = "hey computer"
    model_id: str = "scripted"
    resets: int = 0
    _index: int = 0

    def score(self, frame: AudioFrame) -> WakeScore:
        del frame
        if self._index >= len(self.scores):
            raise AssertionError("scripted model ran out of scores")
        value = self.scores[self._index]
        self._index += 1
        return WakeScore(confidence=value, channel=0)

    def reset(self) -> None:
        self.resets += 1


@dataclass
class RecordingCapture:
    requests: list[CaptureRequest] = field(default_factory=list)
    frames: list[AudioFrame] = field(default_factory=list)
    stop_reasons: list[CaptureStopReason] = field(default_factory=list)

    def start(self, request: CaptureRequest) -> None:
        self.requests.append(request)

    def on_frame(self, frame: AudioFrame) -> None:
        self.frames.append(frame)

    def stop(self, reason: CaptureStopReason) -> None:
        self.stop_reasons.append(reason)


@dataclass
class MemoryLog:
    records: list[tuple[WakeEvent, bool, str]] = field(default_factory=list)

    def record(self, event: WakeEvent, *, accepted: bool, audio_model) -> None:
        self.records.append((event, accepted, audio_model.value))


def silence(n_samples: int, channels: int = 1) -> np.ndarray:
    if channels == 1:
        return np.zeros(n_samples, dtype=np.float32)
    return np.zeros((n_samples, channels), dtype=np.float32)


def feed_scores(engine, scores: list[float], *, channels: int = 1):
    fmt = AudioFormat(sample_rate_hz=16000, channels=channels, frame_samples=320)
    events = []
    for index, _score in enumerate(scores):
        frame = AudioFrame(
            samples=np.zeros((320, channels), dtype=np.float32),
            format=fmt,
            sample_index=index * 320,
        )
        event = engine.process(frame)
        if event is not None:
            events.append(event)
    return events
