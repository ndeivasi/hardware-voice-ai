"""Reference on-device keyword model.

The product wake word is not chosen yet. This model is the default stand-in:
it spots a fixed tonal sequence labeled ``hey computer``. It does not recognize
the spoken English phrase. Swap ``WakeModel`` when the word and a trained
model are selected. See ``docs/wake-word.md``.
"""

from dataclasses import dataclass

import numpy as np

from hardware_voice_ai.audio.frame import AudioFrame
from hardware_voice_ai.wake.defaults import DEFAULT_WAKE_WORD, REFERENCE_MODEL_ID
from hardware_voice_ai.wake.model import WakeScore

DEFAULT_CENTERS_HZ: tuple[int, ...] = (300, 500, 700, 900, 1200, 1600, 2000, 2600)
# A 320 ms pattern at 20 ms per step. The sequence is the acoustic stand-in
# for the placeholder wake word, not a transcription of speech.
DEFAULT_PATTERN: tuple[int, ...] = (0, 3, 6, 2, 5, 7, 1, 4, 2, 6, 3, 0, 5, 7, 4, 1)
# A frame counts as a step only when one analysis band dominates. That rejects
# flat noise and most speech, which are not single-bin tones.
TONAL_RATIO = 0.85
# Ignore numerical residue so a muted or silent channel cannot advance.
SILENCE_ENERGY = 1e-6
STEP_CONFIDENCE = 0.95
COMPLETE_CONFIDENCE = 0.99


@dataclass(frozen=True, slots=True)
class WakeTemplate:
    """A swappable acoustic pattern for ``ReferenceKeywordModel``."""

    wake_word: str
    model_id: str
    sample_rate_hz: int
    frame_samples: int
    centers_hz: tuple[int, ...]
    pattern: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.wake_word.strip():
            raise ValueError("wake_word must not be empty")
        if not self.pattern:
            raise ValueError("pattern must contain at least one step")
        if any(step < 0 or step >= len(self.centers_hz) for step in self.pattern):
            raise ValueError("pattern steps must index centers_hz")
        if any(
            self.pattern[index] == self.pattern[index + 1]
            for index in range(len(self.pattern) - 1)
        ):
            raise ValueError("pattern steps must change band from one step to the next")


def default_template() -> WakeTemplate:
    return WakeTemplate(
        wake_word=DEFAULT_WAKE_WORD,
        model_id=REFERENCE_MODEL_ID,
        sample_rate_hz=16000,
        frame_samples=320,
        centers_hz=DEFAULT_CENTERS_HZ,
        pattern=DEFAULT_PATTERN,
    )


def make_template(
    pattern: tuple[int, ...],
    wake_word: str,
    *,
    model_id: str = REFERENCE_MODEL_ID,
    sample_rate_hz: int = 16000,
    frame_samples: int = 320,
    centers_hz: tuple[int, ...] = DEFAULT_CENTERS_HZ,
) -> WakeTemplate:
    return WakeTemplate(
        wake_word=wake_word,
        model_id=model_id,
        sample_rate_hz=sample_rate_hz,
        frame_samples=frame_samples,
        centers_hz=centers_hz,
        pattern=pattern,
    )


class ReferenceKeywordModel:
    """Sequence spotter over per-channel band energy.

    Each channel keeps its own place in the pattern, and the frame score is
    the strongest channel. v1 can run one microphone; the loop does not assume
    that. Level is normalized out so a quieter far-field tone still matches
    the same pattern. This is not beamforming and not acoustic echo cancellation.
    """

    def __init__(self, template: WakeTemplate, *, channels: int) -> None:
        if channels < 1:
            raise ValueError("channels must be at least 1")
        self.template = template
        self.model_id = template.model_id
        self.wake_word = template.wake_word
        self._channels = channels
        self._positions = [0] * channels

    def reset(self) -> None:
        self._positions = [0] * self._channels

    def score(self, frame: AudioFrame) -> WakeScore:
        if frame.format.sample_rate_hz != self.template.sample_rate_hz:
            raise ValueError("frame sample rate does not match the wake template")
        if frame.format.frame_samples != self.template.frame_samples:
            raise ValueError("frame size does not match the wake template")
        if frame.format.channels != self._channels:
            raise ValueError(
                f"expected {self._channels} channels, got {frame.format.channels}"
            )
        best = 0.0
        best_channel = 0
        for channel in range(self._channels):
            confidence = self._advance(channel, frame.samples[:, channel])
            if confidence > best:
                best = confidence
                best_channel = channel
        return WakeScore(confidence=best, channel=best_channel)

    def _advance(self, channel: int, mono: np.ndarray) -> float:
        band = dominant_band(
            mono,
            sample_rate_hz=self.template.sample_rate_hz,
            centers_hz=self.template.centers_hz,
        )
        pattern = self.template.pattern
        position = self._positions[channel]
        if band is None:
            self._positions[channel] = 0
            return 0.0
        if position < len(pattern) and band == pattern[position]:
            position += 1
            if position == len(pattern):
                self._positions[channel] = 0
                return COMPLETE_CONFIDENCE
            self._positions[channel] = position
            return STEP_CONFIDENCE
        if position > 0 and band == pattern[position - 1]:
            # The same tone still occupies this hop (a slower or shifted step).
            return STEP_CONFIDENCE
        if band == pattern[0]:
            self._positions[channel] = 1
            return STEP_CONFIDENCE
        self._positions[channel] = 0
        return 0.0


def dominant_band(
    mono: np.ndarray,
    *,
    sample_rate_hz: int,
    centers_hz: tuple[int, ...],
) -> int | None:
    """Index of the band that dominates this hop, or ``None`` when it is not tonal."""
    if float(np.dot(mono, mono)) < SILENCE_ENERGY:
        return None
    features = band_features(mono, sample_rate_hz=sample_rate_hz, centers_hz=centers_hz)
    peak = float(features.max())
    if peak < TONAL_RATIO:
        return None
    return int(features.argmax())


def band_features(
    mono: np.ndarray,
    *,
    sample_rate_hz: int,
    centers_hz: tuple[int, ...],
) -> np.ndarray:
    """Unit-length energy at each template center frequency."""
    spectrum = np.abs(np.fft.rfft(mono))
    n = mono.shape[0]
    energies = np.empty(len(centers_hz), dtype=np.float64)
    for index, hz in enumerate(centers_hz):
        bin_index = int(round(hz * n / sample_rate_hz))
        energies[index] = spectrum[bin_index]
    norm = float(np.linalg.norm(energies))
    if norm < 1e-12:
        return np.zeros(len(centers_hz), dtype=np.float64)
    return energies / norm


def render_pattern(
    template: WakeTemplate,
    *,
    channels: int = 1,
    active_channel: int = 0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Synthesize the template as PCM the reference model can score.

    The result is one-dimensional when ``channels`` is 1, otherwise
    ``(samples, channels)`` with silence on the inactive microphones.
    """
    if channels < 1:
        raise ValueError("channels must be at least 1")
    if active_channel < 0 or active_channel >= channels:
        raise ValueError("active_channel is outside the frame")
    steps = [
        _tone(
            template.centers_hz[band],
            template.frame_samples,
            template.sample_rate_hz,
            amplitude,
        )
        for band in template.pattern
    ]
    mono = np.concatenate(steps)
    if channels == 1:
        return mono
    mixed = np.zeros((mono.shape[0], channels), dtype=np.float32)
    mixed[:, active_channel] = mono
    return mixed


def _tone(frequency_hz: int, n: int, sample_rate_hz: int, amplitude: float) -> np.ndarray:
    # Integer FFT bin so the hop lands in one analysis band.
    k = int(round(frequency_hz * n / sample_rate_hz))
    time = np.arange(n, dtype=np.float64)
    wave = amplitude * np.sin(2.0 * np.pi * k * time / n)
    return wave.astype(np.float32)
