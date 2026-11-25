"""A wake decision. Metadata only: this object never carries samples."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WakeEvent:
    """An accepted candidate from the wake engine.

    ``decision_latency_ms`` is the time from the estimated end of the wake
    word to this decision. ``word_end_sample_index`` is the exclusive stream
    index of that end, so capture can keep audio from just before it.
    """

    wake_word: str
    model_id: str
    confidence: float
    decision_latency_ms: int
    winning_channel: int
    channels: int
    sample_rate_hz: int
    frame_samples: int
    decision_frame_index: int
    word_end_frame_index: int
    word_end_sample_index: int
    decision_sample_index: int
