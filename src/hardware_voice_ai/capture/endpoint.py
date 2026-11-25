"""Energy endpoint for a post-wake utterance.

This is a basic loudness check, not a neural voice-activity model. It treats
the loudest channel as the one that matters, so a quiet second microphone
cannot hide speech on another.
"""

from dataclasses import dataclass

import numpy as np

ENDPOINT_SILENCE = "silence"
ENDPOINT_LEADING_SILENCE = "leading_silence"
ENDPOINT_MAX_DURATION = "max_duration"


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """When capture decides the user has finished speaking.

    ``speech_tail_ms`` is shorter than the wake path's 240 ms confirm window
    on purpose. The wake word itself sits earlier in the pre-roll. Looking
    only at the last 200 ms means that word does not, by itself, arm the
    short trailing-silence timer. A command that has already started in that
    tail still arms it, and the whole pre-roll stays in the utterance either way.

    Durations are design defaults for a powered device, not measurements from
    a chosen board.
    """

    end_silence_ms: int = 700
    max_leading_silence_ms: int = 1_600
    max_utterance_ms: int = 15_000
    min_speech_rms: float = 0.02
    speech_to_noise: float = 3.0
    noise_percentile: float = 20.0
    speech_tail_ms: int = 200

    def __post_init__(self) -> None:
        if self.end_silence_ms <= 0:
            raise ValueError("end_silence_ms must be positive")
        if self.max_leading_silence_ms < self.end_silence_ms:
            raise ValueError(
                "max_leading_silence_ms must cover at least one end-silence window"
            )
        if self.max_utterance_ms <= 0:
            raise ValueError("max_utterance_ms must be positive")
        if self.min_speech_rms < 0:
            raise ValueError("min_speech_rms must be >= 0")
        if self.speech_to_noise < 1:
            raise ValueError("speech_to_noise must be >= 1")
        if not 0 <= self.noise_percentile <= 100:
            raise ValueError("noise_percentile must be between 0 and 100")
        if self.speech_tail_ms <= 0:
            raise ValueError("speech_tail_ms must be positive")


def channel_rms(samples: np.ndarray) -> float:
    """Maximum per-channel RMS of ``samples`` with shape ``(samples, channels)``."""
    if samples.size == 0:
        return 0.0
    shaped = np.asarray(samples, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(shaped), axis=0))
    return float(np.max(rms))


def noise_floor(samples: np.ndarray, frame_samples: int, percentile: float) -> float:
    """Quiet-part loudness of a pre-roll, using the loudest channel in each chunk."""
    if samples.shape[0] == 0:
        return 0.0
    step = max(int(frame_samples), 1)
    values = [
        channel_rms(samples[start : start + step])
        for start in range(0, samples.shape[0], step)
    ]
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def speech_threshold(floor: float, config: EndpointConfig) -> float:
    return max(config.min_speech_rms, floor * config.speech_to_noise)


class EndpointTracker:
    """Watch post-wake frames and report why the utterance should end, if it should."""

    def __init__(
        self,
        config: EndpointConfig,
        sample_rate_hz: int,
        threshold: float,
        speech_already_started: bool,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.config = config
        self.sample_rate_hz = sample_rate_hz
        self.threshold = threshold
        self.speech_started = speech_already_started
        self._silence_samples = 0
        self._post_samples = 0

    def observe(self, samples: np.ndarray) -> str | None:
        count = int(samples.shape[0])
        self._post_samples += count
        if channel_rms(samples) >= self.threshold:
            self.speech_started = True
            self._silence_samples = 0
        else:
            self._silence_samples += count
        if self._milliseconds(self._post_samples) >= self.config.max_utterance_ms:
            return ENDPOINT_MAX_DURATION
        if self.speech_started:
            if self._milliseconds(self._silence_samples) >= self.config.end_silence_ms:
                return ENDPOINT_SILENCE
            return None
        if self._milliseconds(self._silence_samples) >= self.config.max_leading_silence_ms:
            return ENDPOINT_LEADING_SILENCE
        return None

    def _milliseconds(self, sample_count: int) -> int:
        return sample_count * 1000 // self.sample_rate_hz


def tail_is_speech(
    samples: np.ndarray,
    sample_rate_hz: int,
    threshold: float,
    tail_ms: int,
) -> bool:
    """True when the end of the pre-roll is already above the speech threshold."""
    if samples.shape[0] == 0:
        return False
    tail_rows = sample_rate_hz * tail_ms // 1000
    if tail_rows <= 0:
        return False
    tail = samples[-tail_rows:]
    return channel_rms(tail) >= threshold
