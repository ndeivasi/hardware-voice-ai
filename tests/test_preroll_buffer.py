"""Pre-roll ring buffer."""

import numpy as np

from hardware_voice_ai.audio.frame import AudioFormat, AudioFrame
from hardware_voice_ai.audio.ring_buffer import PreRollBuffer


def _frame(index: int, value: float, channels: int = 1, n: int = 4) -> AudioFrame:
    samples = np.full((n, channels), value, dtype=np.float32)
    if channels > 1:
        samples[:, 1:] = value + 1
    fmt = AudioFormat(sample_rate_hz=16000, channels=channels, frame_samples=n)
    return AudioFrame(samples=samples, format=fmt, sample_index=index)


def test_snapshot_is_chronological_and_independent() -> None:
    buffer = PreRollBuffer(capacity_samples=8, channels=1, sample_rate_hz=16000)
    buffer.write(_frame(0, 1.0))
    buffer.write(_frame(4, 2.0))
    snap = buffer.snapshot()
    assert snap.samples[:, 0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert snap.start_sample_index == 0
    assert snap.end_sample_index == 8
    snap.samples[0, 0] = 99
    assert buffer.snapshot().samples[0, 0] == 1


def test_capacity_drops_the_oldest_samples() -> None:
    buffer = PreRollBuffer(capacity_samples=6, channels=1, sample_rate_hz=16000)
    buffer.write(_frame(0, 1.0))
    buffer.write(_frame(4, 2.0))
    buffer.write(_frame(8, 3.0))
    snap = buffer.snapshot()
    # 12 samples written, capacity 6, so the last 6: two of the 2s and four 3s.
    assert snap.samples[:, 0].tolist() == [2, 2, 3, 3, 3, 3]
    assert snap.start_sample_index == 6
    assert snap.end_sample_index == 12
    assert snap.offset_of_sample(6) == 0
    assert snap.offset_of_sample(5) is None
    assert snap.offset_of_sample(11) == 5


def test_gap_discards_audio_from_before_the_gap() -> None:
    buffer = PreRollBuffer(capacity_samples=20, channels=1, sample_rate_hz=16000)
    buffer.write(_frame(0, 1.0))
    buffer.write(_frame(10_000, 4.0))
    snap = buffer.snapshot()
    assert snap.samples[:, 0].tolist() == [4, 4, 4, 4]
    assert snap.start_sample_index == 10_000
    assert snap.end_sample_index == 10_004


def test_clear_drops_retained_audio() -> None:
    buffer = PreRollBuffer(capacity_samples=20, channels=1, sample_rate_hz=16000)
    buffer.write(_frame(0, 1.0))
    buffer.clear()
    snap = buffer.snapshot()
    assert snap.samples.shape == (0, 1)
    assert snap.start_sample_index == snap.end_sample_index
    buffer.write(_frame(500, 3.0))
    snap = buffer.snapshot()
    assert snap.start_sample_index == 500
    assert snap.samples[:, 0].tolist() == [3, 3, 3, 3]


def test_multiple_channels_keep_their_columns() -> None:
    buffer = PreRollBuffer(capacity_samples=8, channels=2, sample_rate_hz=16000)
    buffer.write(_frame(0, 0.25, channels=2))
    snap = buffer.snapshot()
    assert snap.channels == 2
    assert snap.samples.shape == (4, 2)
    assert snap.samples[0, 0] == np.float32(0.25)
    assert snap.samples[0, 1] == np.float32(1.25)
