"""Opt-in local detection log. Raw audio is never written."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.events import WakeEvent

# Metadata a tester needs. There is no sample array, path to a recording,
# or transcript of always-on audio.
LOG_FIELDS = (
    "event",
    "accepted",
    "timestamp_ms",
    "wake_word",
    "model_id",
    "confidence",
    "decision_latency_ms",
    "channels",
    "winning_channel",
    "sample_rate_hz",
    "audio_model",
    "decision_frame_index",
    "word_end_frame_index",
    "word_end_sample_index",
)


class DetectionLog:
    """Append-only JSON lines on the local filesystem.

    The log is off unless a path is configured. Each record describes a wake
    decision. It does not contain PCM.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = path
        self._clock = time.time if clock is None else clock

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(self, event: WakeEvent, *, accepted: bool, audio_model: AudioModel) -> None:
        if self.path is None:
            return
        payload = {
            "event": "wake_detection",
            "accepted": accepted,
            "timestamp_ms": int(self._clock() * 1000),
            "wake_word": event.wake_word,
            "model_id": event.model_id,
            "confidence": event.confidence,
            "decision_latency_ms": event.decision_latency_ms,
            "channels": event.channels,
            "winning_channel": event.winning_channel,
            "sample_rate_hz": event.sample_rate_hz,
            "audio_model": audio_model.value,
            "decision_frame_index": event.decision_frame_index,
            "word_end_frame_index": event.word_end_frame_index,
            "word_end_sample_index": event.word_end_sample_index,
        }
        if tuple(payload) != LOG_FIELDS:
            raise RuntimeError("detection log fields drifted from the allowlist")
        line = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
