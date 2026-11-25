"""PCM frames shared by the microphone, the wake model, and the pre-roll."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Stream layout. Channel count is part of the format so callers are not
    fixed to a single microphone.
    """

    sample_rate_hz: int
    channels: int
    frame_samples: int

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels < 1:
            raise ValueError("channels must be at least 1")
        if self.frame_samples < 1:
            raise ValueError("frame_samples must be positive")

    @property
    def frame_ms(self) -> int:
        return self.frame_samples * 1000 // self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One hop of float32 PCM in ``[-1, 1]``, shape ``(frame_samples, channels)``.

    ``sample_index`` is the stream index of the first sample. The object copies
    its samples so a backend can reuse its read buffer.
    """

    samples: np.ndarray
    format: AudioFormat
    sample_index: int

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        expected = (self.format.frame_samples, self.format.channels)
        if samples.shape != expected:
            raise ValueError(
                f"frame shape {samples.shape} does not match format {expected}"
            )
        if not np.isfinite(samples).all():
            raise ValueError("frame samples must be finite")
        if self.sample_index < 0:
            raise ValueError("sample_index must be >= 0")
        object.__setattr__(self, "samples", samples)

    @property
    def end_sample_index(self) -> int:
        """Exclusive stream index of the sample after this frame."""
        return self.sample_index + self.format.frame_samples


@dataclass(frozen=True, slots=True)
class AudioSnapshot:
    """A copy of recent PCM. This is the pre-roll handed to capture.

    The array is in memory only. Nothing in the wake path writes it to disk.
    """

    samples: np.ndarray
    sample_rate_hz: int
    start_sample_index: int
    end_sample_index: int

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        if samples.ndim != 2:
            raise ValueError("snapshot samples must have shape (samples, channels)")
        object.__setattr__(self, "samples", samples)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    def offset_of_sample(self, sample_index: int) -> int | None:
        """Index into ``samples`` for a stream sample, if the snapshot still holds it."""
        if sample_index < self.start_sample_index or sample_index >= self.end_sample_index:
            return None
        return sample_index - self.start_sample_index
