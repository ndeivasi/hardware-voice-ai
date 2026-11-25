"""Threshold, endpoint, and confirm-window logic for any wake model."""

from dataclasses import dataclass

from hardware_voice_ai.audio.frame import AudioFrame
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.wake.events import WakeEvent
from hardware_voice_ai.wake.model import WakeModel


@dataclass
class _Candidate:
    start_frame: int
    peak: float
    word_end_frame: int
    word_end_sample: int
    frames_above: int
    ended: bool
    channel: int


class WakeEngine:
    """Turn per-frame scores into a wake decision.

    A candidate starts when confidence crosses ``accept_threshold``. The word
    end is the last frame that stays near the peak. The decision is emitted
    ``confirm_ms`` later (the 200–300 ms budget) and only if the score stayed
    above the threshold long enough and the candidate was not still going.
    Short spikes and energy that never ends are rejected.
    """

    def __init__(self, model: WakeModel, config: DeviceConfig) -> None:
        self.model = model
        self._threshold = config.accept_threshold
        self._frame_ms = config.frame_ms
        self._min_positive_frames = config.min_positive_ms // config.frame_ms
        self._confirm_frames = config.confirm_ms // config.frame_ms
        self._cooldown_frames = config.cooldown_ms // config.frame_ms
        self._max_word_frames = config.max_word_ms // config.frame_ms
        self._end_ratio = config.end_ratio
        self._frame_index = -1
        self._candidate: _Candidate | None = None
        self._cooldown = 0
        self._hold_off = False

    def reset(self) -> None:
        """Drop an in-progress candidate and the model's memory."""
        self._candidate = None
        self._cooldown = 0
        self._hold_off = False
        self.model.reset()

    def process(self, frame: AudioFrame) -> WakeEvent | None:
        self._frame_index += 1
        score = self.model.score(frame)
        confidence = float(score.confidence)
        if self._cooldown > 0:
            self._cooldown -= 1
            return None
        if self._hold_off:
            if confidence < self._threshold:
                self._hold_off = False
            else:
                return None

        candidate = self._candidate
        if candidate is None:
            if confidence >= self._threshold:
                self._candidate = _Candidate(
                    start_frame=self._frame_index,
                    peak=confidence,
                    word_end_frame=self._frame_index,
                    word_end_sample=frame.end_sample_index,
                    frames_above=1,
                    ended=False,
                    channel=score.channel,
                )
            return None

        if not candidate.ended:
            if (self._frame_index - candidate.start_frame) >= self._max_word_frames:
                self._candidate = None
                self._hold_off = True
                self.model.reset()
                return None
            if confidence >= candidate.peak * self._end_ratio:
                self._extend(candidate, confidence, score.channel, frame)
                return None
            candidate.ended = True

        if confidence >= candidate.peak * self._end_ratio:
            candidate.ended = False
            self._extend(candidate, confidence, score.channel, frame)
            return None

        elapsed_frames = self._frame_index - candidate.word_end_frame
        if elapsed_frames < self._confirm_frames:
            return None

        finished = candidate
        self._candidate = None
        if (
            finished.frames_above < self._min_positive_frames
            or finished.peak < self._threshold
        ):
            return None

        self._cooldown = self._cooldown_frames
        self.model.reset()
        latency_ms = (frame.end_sample_index - finished.word_end_sample) * 1000 // frame.format.sample_rate_hz
        return WakeEvent(
            wake_word=self.model.wake_word,
            model_id=self.model.model_id,
            confidence=finished.peak,
            decision_latency_ms=latency_ms,
            winning_channel=finished.channel,
            channels=frame.format.channels,
            sample_rate_hz=frame.format.sample_rate_hz,
            frame_samples=frame.format.frame_samples,
            decision_frame_index=self._frame_index,
            word_end_frame_index=finished.word_end_frame,
            word_end_sample_index=finished.word_end_sample,
            decision_sample_index=frame.end_sample_index,
        )

    def _extend(
        self,
        candidate: _Candidate,
        confidence: float,
        channel: int,
        frame: AudioFrame,
    ) -> None:
        if confidence > candidate.peak:
            candidate.peak = confidence
            candidate.channel = channel
        candidate.word_end_frame = self._frame_index
        candidate.word_end_sample = frame.end_sample_index
        if confidence >= self._threshold:
            candidate.frames_above += 1
