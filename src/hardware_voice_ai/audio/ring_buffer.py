"""In-memory pre-roll shared by wake detection and later utterance capture."""

import threading

import numpy as np

from hardware_voice_ai.audio.frame import AudioFrame, AudioSnapshot


class PreRollBuffer:
    """A ring of the latest microphone audio.

    Capacity is a few hundred milliseconds to about a second. The buffer is
    process-local: ``snapshot`` returns a copy and nothing here is written to
    disk. A gap in ``sample_index`` (hardware mute dropped frames, or the
    buffer was cleared) discards audio from before the gap so a later capture
    cannot include it.
    """

    def __init__(self, capacity_samples: int, channels: int, sample_rate_hz: int) -> None:
        if capacity_samples < 1:
            raise ValueError("capacity_samples must be positive")
        if channels < 1:
            raise ValueError("channels must be at least 1")
        self.capacity = capacity_samples
        self.channels = channels
        self.sample_rate_hz = sample_rate_hz
        self._data = np.zeros((capacity_samples, channels), dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._end_index = 0
        self._lock = threading.Lock()

    def write(self, frame: AudioFrame) -> None:
        samples = frame.samples
        if samples.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {samples.shape[1]}"
            )
        with self._lock:
            if self._filled > 0 and frame.sample_index != self._end_index:
                self._filled = 0
                self._write = 0
            self._write_unlocked(samples, frame.sample_index)

    def snapshot(self) -> AudioSnapshot:
        with self._lock:
            if self._filled == 0:
                samples = np.zeros((0, self.channels), dtype=np.float32)
            elif self._filled < self.capacity:
                samples = self._data[: self._filled].copy()
            else:
                samples = np.concatenate(
                    (self._data[self._write :], self._data[: self._write]),
                    axis=0,
                )
            end = self._end_index
            start = end - int(samples.shape[0])
        return AudioSnapshot(
            samples=samples,
            sample_rate_hz=self.sample_rate_hz,
            start_sample_index=start,
            end_sample_index=end,
        )

    def clear(self) -> None:
        """Drop retained audio. The stream clock (``_end_index``) is kept."""
        with self._lock:
            self._data.fill(0)
            self._write = 0
            self._filled = 0

    def _write_unlocked(self, samples: np.ndarray, sample_index: int) -> None:
        n = int(samples.shape[0])
        if n >= self.capacity:
            self._data[:] = samples[-self.capacity :]
            self._write = 0
            self._filled = self.capacity
            self._end_index = sample_index + n
            return
        end = self._write + n
        if end <= self.capacity:
            self._data[self._write : end] = samples
        else:
            first = self.capacity - self._write
            self._data[self._write :] = samples[:first]
            self._data[: n - first] = samples[first:]
        self._write = (self._write + n) % self.capacity
        self._filled = min(self.capacity, self._filled + n)
        self._end_index = sample_index + n
