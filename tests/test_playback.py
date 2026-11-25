"""Speaker playback without a physical speaker or a network."""

import socket
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hardware_voice_ai.audio.frame import AudioFormat, AudioSnapshot
from hardware_voice_ai.audio.source import PcmAudioSource
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import AudioBackendError, ConfigurationError
from hardware_voice_ai.playback.constants import STOP_LATENCY_BUDGET_MS
from hardware_voice_ai.capture.utterance import UtteranceCapture
from hardware_voice_ai.capture.handoff import QueueUtteranceSink
from hardware_voice_ai.playback.cues import Cue, render_cue
from hardware_voice_ai.playback.player import (
    PlaybackStatus,
    PlaybackStopReason,
    SpeakerPlayback,
)
from hardware_voice_ai.playback.source import MemoryReplyAudio, StubReplyAudio
from hardware_voice_ai.playback.speaker import (
    MemorySpeaker,
    NullSpeaker,
    SoundDeviceSpeaker,
    build_speaker,
)
from hardware_voice_ai.runtime import WakeRuntime
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.session.state import DeviceState, IllegalTransitionError
from hardware_voice_ai.wake.events import WakeEvent
from hardware_voice_ai.wake.reference import default_template, render_pattern
from tests.support import MemoryLog, RecordingCapture, ScriptedModel, silence


def _fmt(channels: int = 1) -> AudioFormat:
    return AudioFormat(sample_rate_hz=16_000, channels=channels, frame_samples=320)


def _wait_until(predicate, message: str) -> None:
    deadline = time.monotonic() + 2.0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.005)


def _event() -> WakeEvent:
    return WakeEvent(
        wake_word="hey computer",
        model_id="reference-keyword-v1",
        confidence=0.99,
        decision_latency_ms=240,
        winning_channel=0,
        channels=1,
        sample_rate_hz=16000,
        frame_samples=320,
        decision_frame_index=10,
        word_end_frame_index=4,
        word_end_sample_index=1600,
        decision_sample_index=4800,
    )


def _snapshot() -> AudioSnapshot:
    samples = np.zeros((320, 1), dtype=np.float32)
    return AudioSnapshot(
        samples=samples,
        sample_rate_hz=16000,
        start_sample_index=0,
        end_sample_index=320,
    )


def _open_turn(mode: AudioModel = AudioModel.REMOTE):
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(
        RecordingCapture(),
        audio_model=mode,
        playback=playback,
    )
    assert session.on_wake_accepted(_event(), _snapshot()) is True
    playback.wait()
    session.close_capture()
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    return session, playback, speaker


def test_reply_starts_before_the_rest_arrives() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    stub = StubReplyAudio(16_000, 1)
    first = np.full(3200, 0.3, dtype=np.float32)
    rest = np.full(3200, 0.6, dtype=np.float32)
    playback.play_reply(stub)
    stub.push(first)
    _wait_until(
        lambda: len(speaker.accepted()) == len(first) and speaker.played_samples > 0,
        "the first chunk was not playing",
    )
    assert np.allclose(speaker.accepted()[:, 0], first)
    assert speaker.played_samples < len(first) + len(rest)
    stub.push(rest)
    stub.finish()
    result = playback.wait()
    assert result.status is PlaybackStatus.COMPLETED
    assert result.reason is PlaybackStopReason.COMPLETED
    assert np.allclose(speaker.heard()[:, 0], np.concatenate((first, rest)))
    playback.close()


def test_completion_plays_the_whole_reply() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    audio = np.linspace(-0.4, 0.4, 1600, dtype=np.float32)
    source = MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1)
    playback.play_reply(source)
    result = playback.wait()
    assert result.status is PlaybackStatus.COMPLETED
    assert result.samples_dropped == 0
    assert result.samples_played == len(audio)
    assert np.allclose(speaker.heard()[:, 0], audio)
    playback.close()


def test_stop_drops_only_the_queued_tail_within_the_budget() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    audio = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    stub = StubReplyAudio(16_000, 1)
    stub.push(audio)
    playback.play_reply(stub)
    _wait_until(
        lambda: len(speaker.accepted()) == len(audio) and speaker.queued_samples > 0,
        "reply audio did not fill the speaker queue",
    )
    queued_ms = speaker.queued_samples * 1000 / speaker.format.sample_rate_hz
    assert queued_ms <= STOP_LATENCY_BUDGET_MS
    accepted = speaker.accepted()
    queued = speaker.queued_samples
    started = time.perf_counter()
    result = playback.stop()
    elapsed = time.perf_counter() - started
    assert elapsed < STOP_LATENCY_BUDGET_MS / 1000
    assert result.status is PlaybackStatus.STOPPED
    assert result.reason is PlaybackStopReason.STOP
    assert result.status is not PlaybackStatus.COMPLETED
    assert speaker.queued_samples == 0
    assert result.samples_dropped == queued
    assert result.samples_dropped <= speaker.format.sample_rate_hz * speaker.queue_ms // 1000
    heard = speaker.heard()
    assert len(heard) == len(accepted) - queued
    assert np.allclose(heard[:, 0], audio[: len(heard)])
    assert len(heard) < len(audio)
    playback.close()


def test_volume_scales_without_hard_clipping() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker, volume=0.5)
    audio = np.full(1600, 0.8, dtype=np.float32)
    playback.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1))
    playback.wait()
    heard = speaker.heard()[:, 0]
    assert np.allclose(heard, 0.4)
    assert float(np.max(np.abs(heard))) <= 1.0

    unity = MemorySpeaker(_fmt())
    full = SpeakerPlayback(unity, volume=1.0)
    identity = np.full(640, 1.0, dtype=np.float32)
    full.play_reply(MemoryReplyAudio(identity, sample_rate_hz=16_000, channels=1))
    full.wait()
    assert np.allclose(unity.heard()[:, 0], identity)

    hot = MemorySpeaker(_fmt())
    limited = SpeakerPlayback(hot, volume=1.0)
    rate = 16_000
    sine = (
        np.sin(2 * np.pi * 440 * np.arange(rate, dtype=np.float32) / rate).astype(np.float32)
        * np.float32(1.5)
    )
    limited.play_reply(MemoryReplyAudio(sine, sample_rate_hz=rate, channels=1))
    limited.wait()
    played = hot.heard()[:, 0]
    assert np.max(np.abs(played)) <= 1.0 + 1e-5
    assert np.allclose(played, sine / np.float32(1.5), atol=1e-5)
    clipped = np.clip(sine, -1.0, 1.0)
    assert not np.allclose(played, clipped, atol=1e-4)
    playback.close()
    full.close()
    limited.close()


def test_volume_changes_apply_to_the_next_chunk() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker, volume=1.0)
    stub = StubReplyAudio(16_000, 1)
    first = np.full(3200, 0.5, dtype=np.float32)
    second = np.full(3200, 0.5, dtype=np.float32)
    playback.play_reply(stub)
    stub.push(first)
    _wait_until(lambda: len(speaker.accepted()) == len(first), "first chunk was not written")
    playback.set_volume(0.5)
    stub.push(second)
    stub.finish()
    playback.wait()
    heard = speaker.heard()[:, 0]
    assert np.allclose(heard[: len(first)], 0.5)
    assert np.allclose(heard[len(first) :], 0.25)
    playback.close()


def test_volume_bounds_are_rejected() -> None:
    playback = SpeakerPlayback(MemorySpeaker(_fmt()))
    with pytest.raises(ConfigurationError):
        playback.set_volume(1.5)
    with pytest.raises(ConfigurationError):
        playback.set_volume(-0.1)
    with pytest.raises(ConfigurationError):
        SpeakerPlayback(MemorySpeaker(_fmt()), volume=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        DeviceConfig(volume=2)
    with pytest.raises(ConfigurationError):
        DeviceConfig(playback_queue_ms=0)
    with pytest.raises(ConfigurationError):
        DeviceConfig(playback_queue_ms=STOP_LATENCY_BUDGET_MS + 1)
    with pytest.raises(ConfigurationError):
        MemorySpeaker(_fmt(), queue_ms=500)
    config = DeviceConfig.from_env(
        {"HVA_VOLUME": "0.4", "HVA_PLAYBACK_QUEUE_MS": "40", "HVA_OUTPUT_DEVICE": "hw:0"}
    )
    assert config.volume == 0.4
    assert config.playback_queue_ms == 40
    assert config.output_device == "hw:0"
    playback.close()


@pytest.mark.parametrize("mode", [AudioModel.LOCAL, AudioModel.REMOTE])
def test_cues_are_local_and_distinct_in_either_mode(mode: AudioModel, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fmt = _fmt()
    rendered = [render_cue(cue, fmt) for cue in Cue]
    assert len({item.tobytes() for item in rendered}) == len(Cue)
    for cue, expected in zip(Cue, rendered, strict=True):
        assert float(np.max(np.abs(expected))) <= 0.25
        speaker = MemorySpeaker(fmt)
        playback = SpeakerPlayback(speaker, volume=1.0)
        session = SessionOrchestrator(RecordingCapture(), audio_model=mode, playback=playback)
        result = session.play_cue(cue)
        finished = playback.wait()
        assert result.status in {PlaybackStatus.PLAYING, PlaybackStatus.COMPLETED}
        assert finished.status is PlaybackStatus.COMPLETED
        assert np.allclose(speaker.heard(), expected)
        assert session.state is DeviceState.IDLE
        playback.close()
    assert list(tmp_path.iterdir()) == []


def test_cues_do_not_open_a_socket(monkeypatch) -> None:
    def blocked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("playback opened a socket")

    monkeypatch.setattr(socket, "socket", blocked)
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    for cue in Cue:
        playback.play_cue(cue)
        playback.wait()
    playback.play_reply(
        MemoryReplyAudio(np.full(320, 0.2, dtype=np.float32), sample_rate_hz=16_000, channels=1)
    )
    playback.wait()
    playback.close()


def test_playback_package_does_not_reference_a_cloud_client() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "hardware_voice_ai" / "playback"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for banned in (
        "websocket",
        "urllib",
        "http.client",
        "requests",
        "speech_to_text",
        "text_to_speech",
        "import socket",
    ):
        assert banned not in text


def test_null_speaker_does_not_retain_samples() -> None:
    speaker = NullSpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    audio = np.full(2000, 0.2, dtype=np.float32)
    playback.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1))
    result = playback.wait()
    assert result.status is PlaybackStatus.COMPLETED
    assert result.samples_played == len(audio)
    assert speaker.retains_samples() is False
    assert not any(isinstance(value, np.ndarray) for value in vars(speaker).values())
    playback.close()


def test_reply_format_must_match_the_speaker() -> None:
    playback = SpeakerPlayback(MemorySpeaker(_fmt()))
    with pytest.raises(ConfigurationError, match="format"):
        playback.play_reply(StubReplyAudio(8_000, 1))
    playback.close()


def test_stereo_reply_keeps_both_channels() -> None:
    fmt = _fmt(channels=2)
    speaker = MemorySpeaker(fmt)
    playback = SpeakerPlayback(speaker)
    left = np.full(640, 0.2, dtype=np.float32)
    right = np.full(640, -0.4, dtype=np.float32)
    audio = np.column_stack((left, right))
    playback.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=2))
    playback.wait()
    assert np.allclose(speaker.heard(), audio)
    playback.close()


def test_orchestrator_completion_returns_to_idle() -> None:
    session, playback, speaker = _open_turn(AudioModel.REMOTE)
    before = len(speaker.heard())
    audio = np.full(960, 0.25, dtype=np.float32)
    stub = StubReplyAudio(16_000, 1)
    session.play_reply(stub)
    assert session.state is DeviceState.SPEAKING
    stub.push(audio)
    stub.finish()
    playback.wait()
    if playback.is_playing:
        playback.wait()
    assert session.state is DeviceState.IDLE
    assert session.follow_up_is_active is True
    assert session.last_reply_result is not None
    assert session.last_reply_result.status is PlaybackStatus.COMPLETED
    assert session.playback_status().status is PlaybackStatus.COMPLETED
    listening = render_cue(Cue.LISTENING, _fmt())
    heard = speaker.heard()[before:]
    assert np.allclose(heard[: len(audio), 0], audio)
    assert np.allclose(heard[len(audio) :], listening)
    playback.close()


def test_reply_plays_locally_for_both_audio_model_modes() -> None:
    audio = np.linspace(0.1, 0.3, 640, dtype=np.float32)
    heard = {}
    for mode in (AudioModel.LOCAL, AudioModel.REMOTE):
        session, playback, speaker = _open_turn(mode)
        before = len(speaker.heard())
        session.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1))
        playback.wait()
        if playback.is_playing:
            playback.wait()
        heard_now = speaker.heard()[before:]
        listening = render_cue(Cue.LISTENING, _fmt())
        assert np.allclose(heard_now[: len(audio), 0], audio)
        assert np.allclose(heard_now[len(audio) :], listening)
        heard[mode] = heard_now[: len(audio), 0].copy()
        assert session.state is DeviceState.IDLE
        playback.close()
    assert np.allclose(heard[AudioModel.LOCAL], audio)
    assert np.allclose(heard[AudioModel.REMOTE], audio)


def test_stop_request_is_stopped_not_completed() -> None:
    session, playback, _speaker = _open_turn()
    stub = StubReplyAudio(16_000, 1)
    stub.push(np.linspace(-0.2, 0.2, 16_000, dtype=np.float32))
    session.play_reply(stub)
    _wait_until(lambda: playback.speaker.played_samples > 0, "reply did not start")
    result = session.stop_playback()
    assert session.state is DeviceState.IDLE
    assert result.status is PlaybackStatus.STOPPED
    assert result.reason is PlaybackStopReason.STOP
    assert session.last_reply_result is not None
    assert session.last_reply_result.status is PlaybackStatus.STOPPED
    playback.close()


def test_play_reply_rejects_the_wrong_state() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(RecordingCapture(), playback=playback)
    source = StubReplyAudio(16_000, 1)
    with pytest.raises(IllegalTransitionError):
        session.play_reply(source)
    session.on_wake_accepted(_event(), _snapshot())
    assert session.state is DeviceState.CAPTURING
    with pytest.raises(IllegalTransitionError, match="after capture has closed"):
        session.play_reply(source)
    assert session.state is DeviceState.CAPTURING
    playback.close()


def test_cue_during_capture_does_not_leave_capturing() -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(RecordingCapture(), playback=playback)
    session.on_wake_accepted(_event(), _snapshot())
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    session.play_cue(Cue.OFFLINE)
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    assert session.on_wake_accepted(_event(), _snapshot()) is False
    playback.close()


def test_mute_stops_playback_and_refuses_new_audio() -> None:
    session, playback, speaker = _open_turn()
    stub = StubReplyAudio(16_000, 1)
    stub.push(np.linspace(-0.3, 0.3, 16_000, dtype=np.float32))
    session.play_reply(stub)
    _wait_until(lambda: speaker.played_samples > 0, "reply did not start")
    session.apply_mute(True)
    assert session.state is DeviceState.IDLE
    assert session.is_muted is True
    assert session.last_reply_result is not None
    assert session.last_reply_result.status is PlaybackStatus.STOPPED
    assert session.last_reply_result.reason is PlaybackStopReason.MUTE
    assert speaker.queued_samples == 0
    held = speaker.heard().copy()
    assert session.play_cue(Cue.ERROR).reason is PlaybackStopReason.MUTE
    assert session.play_reply(StubReplyAudio(16_000, 1)).reason is PlaybackStopReason.MUTE
    assert session.state is DeviceState.IDLE
    assert np.array_equal(speaker.heard(), held)
    session.apply_mute(False)
    assert session.is_muted is False
    assert np.array_equal(speaker.heard(), held)
    session.on_wake_accepted(_event(), _snapshot())
    playback.wait()
    assert len(speaker.heard()) > len(held)
    playback.close()


def test_barge_in_stops_the_reply_and_accepts_the_new_wake() -> None:
    session, playback, speaker = _open_turn()
    capture = session.capture
    stub = StubReplyAudio(16_000, 1)
    audio = np.linspace(-0.4, 0.4, 16_000, dtype=np.float32)
    before = len(speaker.accepted())
    stub.push(audio)
    session.play_reply(stub)
    _wait_until(
        lambda: len(speaker.accepted()) >= before + len(audio),
        "reply was not queued",
    )
    assert session.on_wake_accepted(_event(), _snapshot()) is True
    assert session.state is DeviceState.CAPTURING
    assert session.last_reply_result is not None
    assert session.last_reply_result.status is PlaybackStatus.STOPPED
    assert session.last_reply_result.reason is PlaybackStopReason.BARGE_IN
    assert session.last_reply_result.samples_played < len(audio)
    assert len(capture.requests) == 2
    playback.wait()
    playback.close()


def test_runtime_barge_in_on_a_second_wake_word() -> None:
    word = render_pattern(default_template())
    gap = silence(int(1.2 * 16_000))
    samples = np.concatenate((silence(1600), word, gap, word, silence(8000)))
    config = DeviceConfig()
    speaker = MemorySpeaker(config.audio_format)
    playback = SpeakerPlayback(speaker)
    capture = RecordingCapture()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        capture=capture,
        playback=playback,
    )
    while runtime.orchestrator.state is not DeviceState.CAPTURING:
        assert runtime.poll()
    runtime.orchestrator.close_capture()
    stub = StubReplyAudio(config.sample_rate_hz, config.channels)
    stub.push(np.linspace(-0.2, 0.2, 16_000, dtype=np.float32))
    runtime.orchestrator.play_reply(stub)
    assert runtime.orchestrator.state is DeviceState.SPEAKING
    _wait_until(lambda: speaker.played_samples > 0, "runtime reply did not start")
    while runtime.poll():
        pass
    assert runtime.orchestrator.last_reply_result is not None
    assert runtime.orchestrator.last_reply_result.reason is PlaybackStopReason.BARGE_IN
    assert runtime.orchestrator.last_reply_result.status is PlaybackStatus.STOPPED
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert len(capture.requests) == 2
    runtime.close()


def test_pcm_backend_is_memory_and_sounddevice_import_is_optional(monkeypatch) -> None:
    fmt = _fmt()
    speaker = build_speaker(fmt, "pcm", queue_ms=40)
    assert isinstance(speaker, MemorySpeaker)
    assert speaker.queue_ms == 40
    with pytest.raises(ConfigurationError, match="Unknown audio backend"):
        build_speaker(fmt, "vendor")
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(AudioBackendError, match="sounddevice"):
        SoundDeviceSpeaker(fmt)


def test_sounddevice_speaker_uses_a_short_callback_queue(monkeypatch) -> None:
    class FakeStream:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    class FakeSoundDevice:
        def __init__(self) -> None:
            self.stream: FakeStream | None = None

        def OutputStream(self, **kwargs):
            self.stream = FakeStream(**kwargs)
            return self.stream

    fake = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    fmt = _fmt(channels=2)
    speaker = SoundDeviceSpeaker(fmt, device=4, queue_ms=80)
    assert fake.stream is not None
    assert fake.stream.started is True
    assert fake.stream.kwargs["device"] == 4
    assert fake.stream.kwargs["channels"] == 2
    assert fake.stream.kwargs["dtype"] == "float32"
    assert fake.stream.kwargs["latency"] == "low"
    assert fake.stream.kwargs["blocksize"] == 320
    callback = fake.stream.kwargs["callback"]
    block = np.full((320, 2), 0.25, dtype=np.float32)
    speaker.write(block)
    out = np.zeros((320, 2), dtype=np.float32)
    callback(out, 320, None, None)
    assert np.allclose(out, 0.25)
    assert speaker.played_samples == 320
    speaker.write(block)
    dropped = speaker.drop_queued()
    assert dropped == 320
    assert speaker.queued_samples == 0
    assert speaker.retains_samples() is False
    speaker.close()
    assert fake.stream.stopped is True
    assert fake.stream.closed is True


class _FakeOutput:
    """PortAudio stand-in. It does not run a callback thread."""

    def __init__(self) -> None:
        self.stream: _FakeOutputStream | None = None

    def OutputStream(self, **kwargs):
        self.stream = _FakeOutputStream(**kwargs)
        return self.stream


class _FakeOutputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _live_speaker(monkeypatch, queue_ms: int = 200) -> tuple[SoundDeviceSpeaker, object]:
    fake = _FakeOutput()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    speaker = SoundDeviceSpeaker(_fmt(), queue_ms=queue_ms)
    assert fake.stream is not None
    return speaker, fake.stream.kwargs["callback"]


def test_live_flush_waits_until_the_callback_drains(monkeypatch) -> None:
    speaker, callback = _live_speaker(monkeypatch)
    block = np.full((320, 1), 0.5, dtype=np.float32)
    speaker.write(block)
    assert speaker.queued_samples == 320
    finished = threading.Event()

    def drain() -> None:
        speaker.flush()
        finished.set()

    thread = threading.Thread(target=drain)
    thread.start()
    time.sleep(0.05)
    assert finished.is_set() is False
    assert speaker.queued_samples == 320
    out = np.zeros((320, 1), dtype=np.float32)
    callback(out, 320, None, None)
    thread.join(timeout=1.0)
    assert finished.is_set() is True
    assert speaker.queued_samples == 0
    assert speaker.played_samples == 320
    assert np.allclose(out, 0.5)
    speaker.close()


def test_stop_during_live_flush_drops_the_tail(monkeypatch) -> None:
    speaker, _callback = _live_speaker(monkeypatch)
    playback = SpeakerPlayback(speaker)
    audio = np.full(320, 0.4, dtype=np.float32)
    playback.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1))
    _wait_until(
        lambda: speaker.queued_samples == 320 and playback.is_playing,
        "live playback completed before the queue was quiet",
    )
    result = playback.stop()
    assert result.status is PlaybackStatus.STOPPED
    assert result.reason is PlaybackStopReason.STOP
    assert result.status is not PlaybackStatus.COMPLETED
    assert speaker.queued_samples == 0
    assert speaker.played_samples == 0
    assert result.samples_dropped == 320
    playback.close()


def test_mute_during_live_flush_drops_the_tail(monkeypatch) -> None:
    speaker, _callback = _live_speaker(monkeypatch)
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(RecordingCapture(), playback=playback)
    session.on_wake_accepted(_event(), _snapshot())
    _wait_until(
        lambda: speaker.queued_samples > 0 and playback.is_cue_playing,
        "wake cue was not held in the live queue",
    )
    session.apply_mute(True)
    assert session.is_muted is True
    assert playback.status().status is PlaybackStatus.STOPPED
    assert playback.status().reason is PlaybackStopReason.MUTE
    assert speaker.queued_samples == 0
    assert speaker.played_samples == 0
    playback.close()


def test_live_flush_completes_after_the_callback_drains(monkeypatch) -> None:
    speaker, callback = _live_speaker(monkeypatch)
    playback = SpeakerPlayback(speaker)
    audio = np.full(320, 0.4, dtype=np.float32)
    playback.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1))
    _wait_until(
        lambda: speaker.queued_samples == 320 and playback.is_playing,
        "reply was marked complete while still queued",
    )
    out = np.zeros((320, 1), dtype=np.float32)
    callback(out, 320, None, None)
    result = playback.wait()
    assert result.status is PlaybackStatus.COMPLETED
    assert speaker.played_samples == 320
    assert speaker.queued_samples == 0
    assert np.allclose(out[:, 0], audio)
    playback.close()


def _wake_scores() -> list[float]:
    return [0.9] * 6 + [0.0] * 12


def _cue_runtime(monkeypatch, samples: np.ndarray, scores: list[float], capture):
    speaker, callback = _live_speaker(monkeypatch, queue_ms=80)
    playback = SpeakerPlayback(speaker)
    config = replace(
        DeviceConfig(),
        accept_threshold=0.6,
        min_positive_ms=120,
        confirm_ms=240,
        cooldown_ms=0,
        max_word_ms=1000,
        end_ratio=0.85,
    )
    log = MemoryLog()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        model=ScriptedModel(scores=scores),
        capture=capture,
        detection_log=log,
        playback=playback,
    )
    return runtime, playback, callback, log


def _release_cue(playback: SpeakerPlayback, callback) -> None:
    out = np.zeros((320, 1), dtype=np.float32)
    for _ in range(40):
        if not playback.is_playing and playback.speaker.queued_samples == 0:
            break
        callback(out, 320, None, None)
        time.sleep(0.001)
    playback.wait()


def test_wake_cue_is_not_in_the_captured_utterance(monkeypatch) -> None:
    wake = _wake_scores()
    during = 8
    after = 4
    trailing = 40
    scores = wake + [0.0] * (during + after + trailing)
    tone = np.full(320, 0.8, dtype=np.float32)
    quiet = np.zeros(320, dtype=np.float32)
    samples = np.concatenate(
        [quiet] * len(wake) + [tone] * during + [tone] * after + [quiet] * trailing
    )
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink)
    runtime, playback, callback, _log = _cue_runtime(monkeypatch, samples, scores, capture)
    while runtime.orchestrator.state is not DeviceState.CAPTURING:
        assert runtime.poll()
    assert playback.is_cue_playing is True
    for _ in range(during):
        assert runtime.poll()
    assert len(capture._frames) == during
    for frame in capture._frames:
        assert float(np.max(np.abs(frame))) == 0.0
    _release_cue(playback, callback)
    assert playback.is_cue_playing is False
    while runtime.poll():
        pass
    utterance = sink.take()
    assert utterance is not None
    loud = np.count_nonzero(np.abs(utterance.samples[:, 0]) > 0.5)
    assert loud == after * 320
    runtime.close()


def test_wake_during_a_local_cue_is_ignored(monkeypatch) -> None:
    one = _wake_scores()
    scores = one + one
    samples = np.zeros(320 * len(scores), dtype=np.float32)
    capture = RecordingCapture()
    runtime, playback, callback, log = _cue_runtime(monkeypatch, samples, scores, capture)
    runtime.orchestrator.play_cue(Cue.OFFLINE)
    assert playback.is_cue_playing is True
    for _ in range(len(one)):
        assert runtime.poll()
    assert capture.requests == []
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert [accepted for _event, accepted, _model in log.records] == [False]
    _release_cue(playback, callback)
    while runtime.poll():
        pass
    assert len(capture.requests) == 1
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert [accepted for _event, accepted, _model in log.records] == [False, True]
    runtime.close()


def test_help_lists_volume(capsys) -> None:
    from hardware_voice_ai.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--volume" in capsys.readouterr().out
