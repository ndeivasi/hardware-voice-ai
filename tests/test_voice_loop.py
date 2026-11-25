"""One Audio Model loop: utterance audio in, spoken reply toward the speaker."""

import sys
import threading
import time

import numpy as np

from hardware_voice_ai.audio.frame import AudioSnapshot
from hardware_voice_ai.audio.source import PcmAudioSource
from hardware_voice_ai.audio_model.events import Failure, ManualClock, TurnOutcome
from hardware_voice_ai.audio_model.factory import build_voice_loop
from hardware_voice_ai.audio_model.gateway import GatewayAudio, GatewayEnd, GatewayError
from hardware_voice_ai.audio_model.local import LocalAudioModel
from hardware_voice_ai.audio_model.remote import RemoteAudioModel
from hardware_voice_ai.audio_model.utterance import CapturedUtterance, utterance_from_request
from hardware_voice_ai.capture.port import CaptureRequest
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ModelUnavailable
from hardware_voice_ai.mute.control import InMemoryMuteControl
from hardware_voice_ai.playback.player import PlaybackStatus, PlaybackStopReason, SpeakerPlayback
from hardware_voice_ai.playback.speaker import MemorySpeaker as PlaybackMemorySpeaker
from hardware_voice_ai.playback.speaker import SoundDeviceSpeaker
from hardware_voice_ai.runtime import WakeRuntime
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.session.state import DeviceState
from hardware_voice_ai.speaker.port import MemorySpeaker
from hardware_voice_ai.wake.events import WakeEvent
from hardware_voice_ai.wake.reference import default_template, render_pattern
from tests.support import RecordingCapture, silence


TOKEN = "test-token"


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
    return AudioSnapshot(
        samples=np.zeros((320, 1), dtype=np.float32),
        sample_rate_hz=16000,
        start_sample_index=0,
        end_sample_index=320,
    )


def _utterance(
    model: AudioModel = AudioModel.LOCAL,
    samples: np.ndarray | None = None,
    *,
    rate: int = 16000,
) -> CapturedUtterance:
    if samples is None:
        samples = np.full((640, 1), 0.25, dtype=np.float32)
    return CapturedUtterance(
        wake=_event(),
        audio_model=model,
        samples=samples,
        sample_rate_hz=rate,
        pre_roll_sample_count=320,
        wake_word_end_offset=160,
        endpoint="handed_off",
    )


def _wait_idle(session) -> None:
    """``submit_utterance`` can return while speaker flush is still running."""
    deadline = time.monotonic() + 2.0
    while session.state is not DeviceState.IDLE:
        if time.monotonic() > deadline:
            raise AssertionError("session did not return to idle after playback flush")
        time.sleep(0.001)


def _ready(voice, model: AudioModel):
    capture = RecordingCapture()
    session = SessionOrchestrator(capture, audio_model=model, voice=voice)
    assert session.on_wake_accepted(_event(), _snapshot()) is True
    session.close_capture()
    assert session.state is DeviceState.CAPTURING
    return session


def _local_voice(engine=None, *, clock=None, speaker=None):
    speaker = speaker or MemorySpeaker()
    if engine is None:
        from hardware_voice_ai.audio_model.local import ReferenceAudioEngine

        engine = ReferenceAudioEngine()
    model = LocalAudioModel(engine)
    from hardware_voice_ai.audio_model.loop import VoiceLoop

    return VoiceLoop(
        model,
        speaker,
        first_audio_timeout_s=8.0,
        turn_timeout_s=30.0,
        clock=clock or ManualClock(),
    )


class ScriptedSession:
    def __init__(self, events, clock: ManualClock | None = None) -> None:
        self.events = list(events)
        self.clock = clock
        self.sent: list[np.ndarray] = []
        self.cancels = 0
        self.closed = False
        self.started = None
        self.finished = False

    def start_turn(self, *, sample_rate_hz: int, channels: int) -> None:
        self.started = (sample_rate_hz, channels)

    def send_audio(self, samples: np.ndarray) -> None:
        self.sent.append(np.array(samples, dtype=np.float32, copy=True))

    def finish_audio(self) -> None:
        self.finished = True

    def send_cancel(self) -> None:
        self.cancels += 1

    def receive(self, timeout_s: float):
        del timeout_s
        if self.events:
            return self.events.pop(0)
        if self.clock is not None:
            self.clock.advance(100)
        return None

    def close(self) -> None:
        self.closed = True


def _remote_voice(session: ScriptedSession, *, clock=None, speaker=None, timeouts=(8.0, 30.0)):
    speaker = speaker or MemorySpeaker()
    from hardware_voice_ai.audio_model.loop import VoiceLoop

    clock = clock or ManualClock()
    model = RemoteAudioModel(lambda: session, token=TOKEN)
    return VoiceLoop(
        model,
        speaker,
        first_audio_timeout_s=timeouts[0],
        turn_timeout_s=timeouts[1],
        clock=clock,
    )


def test_local_model_speaks_a_reply_and_walks_the_states() -> None:
    voice = _local_voice()
    session = _ready(voice, AudioModel.LOCAL)
    first = session.submit_utterance(_utterance())
    assert first.outcome is TurnOutcome.SPOKEN
    assert first.cue is None
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
    assert session.trace == [
        DeviceState.IDLE,
        DeviceState.WOKEN,
        DeviceState.CAPTURING,
        DeviceState.SPEAKING,
        DeviceState.IDLE,
    ]
    reply = voice.speaker.reply_audio()
    assert reply.shape == (6400, 1)
    assert voice.speaker.reply_rate == 16000
    assert voice.speaker.endings == 1
    assert voice.speaker.stops == 0
    assert voice.speaker.cues == []
    assert float(np.max(np.abs(reply))) > 0.0


def test_same_utterance_is_deterministic_and_level_changes_the_reply() -> None:
    def spoken(level: float) -> np.ndarray:
        voice = _local_voice()
        session = _ready(voice, AudioModel.LOCAL)
        result = session.submit_utterance(
            _utterance(samples=np.full((640, 1), level, dtype=np.float32))
        )
        assert result.outcome is TurnOutcome.SPOKEN
        return voice.speaker.reply_audio()

    quiet = spoken(0.25)
    assert np.array_equal(quiet, spoken(0.25))
    assert not np.array_equal(quiet, spoken(0.8))


def test_local_silence_is_low_confidence_and_the_next_wake_works() -> None:
    voice = _local_voice()
    session = _ready(voice, AudioModel.LOCAL)
    result = session.submit_utterance(
        _utterance(samples=np.zeros((640, 1), dtype=np.float32))
    )
    assert result.outcome is TurnOutcome.LOW_CONFIDENCE
    assert result.cue == "error"
    assert result.reply_sample_count == 0
    assert voice.speaker.reply_audio().shape[0] == 0
    assert voice.speaker.cues[0][0] == "error"
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
    assert session.on_wake_accepted(_event(), _snapshot()) is True


def test_local_turn_does_not_import_the_tls_transport() -> None:
    sys.modules.pop("hardware_voice_ai.audio_model.tls", None)
    voice = _local_voice()
    session = _ready(voice, AudioModel.LOCAL)
    result = session.submit_utterance(_utterance())
    assert result.outcome is TurnOutcome.SPOKEN
    assert "hardware_voice_ai.audio_model.tls" not in sys.modules


def test_unavailable_local_model_stays_on_device_and_listening_recovers() -> None:
    voice = _local_voice(engine=None)
    voice.model = LocalAudioModel(None)
    session = _ready(voice, AudioModel.LOCAL)
    result = session.submit_utterance(_utterance())
    assert result.outcome is TurnOutcome.UNAVAILABLE
    assert result.message == "local audio model is unavailable"
    assert result.cue == "error"
    assert voice.speaker.cues[0][0] == "error"
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
    assert session.on_wake_accepted(_event(), _snapshot()) is True


def test_local_timeout_plays_an_error_cue() -> None:
    clock = ManualClock()

    class Slow:
        def run(self, samples, sample_rate_hz, cancel, deadline):
            del samples, sample_rate_hz, cancel
            deadline.clock.advance(deadline.turn_s)
            if deadline.expired(False):
                yield Failure(TurnOutcome.TIMEOUT, "audio model timed out")

    voice = _local_voice(engine=Slow(), clock=clock)
    session = _ready(voice, AudioModel.LOCAL)
    result = session.submit_utterance(_utterance())
    assert result.outcome is TurnOutcome.TIMEOUT
    assert result.cue == "error"
    _wait_idle(session)
    assert session.state is DeviceState.IDLE


def test_raising_model_does_not_stick_the_session_or_leak_the_exception() -> None:
    class Boom:
        def run(self, samples, sample_rate_hz, cancel, deadline):
            del samples, sample_rate_hz, cancel, deadline
            raise RuntimeError("boom test-token")
            yield  # pragma: no cover

    voice = _local_voice(engine=Boom())
    session = _ready(voice, AudioModel.LOCAL)
    result = session.submit_utterance(_utterance())
    assert result.outcome is TurnOutcome.ERROR
    assert result.message == "audio model failed"
    assert "test-token" not in result.message
    assert "boom" not in result.message
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
    assert session.on_wake_accepted(_event(), _snapshot()) is True


def test_remote_reply_is_audio() -> None:
    reply = np.linspace(-0.4, 0.4, 320, dtype=np.float32).reshape(-1, 1)
    session_io = ScriptedSession(
        [
            GatewayAudio(reply, 16000),
            GatewayEnd(),
        ]
    )
    voice = _remote_voice(session_io)
    session = _ready(voice, AudioModel.REMOTE)
    utterance = _utterance(AudioModel.REMOTE, samples=np.full((640, 2), 0.3, dtype=np.float32))
    result = session.submit_utterance(utterance)
    assert result.outcome is TurnOutcome.SPOKEN
    assert result.cue is None
    assert session_io.started == (16000, 2)
    sent = np.concatenate(session_io.sent, axis=0)
    assert sent.shape == utterance.samples.shape
    assert np.allclose(sent, utterance.samples)
    played = voice.speaker.reply_audio()
    assert np.allclose(played, reply)
    assert voice.speaker.cues == []
    _wait_idle(session)
    assert session.trace[-3:] == [
        DeviceState.CAPTURING,
        DeviceState.SPEAKING,
        DeviceState.IDLE,
    ]
    assert session_io.closed is True
    assert session_io.finished is True
    assert session_io.cancels == 0


def test_speaker_receives_the_first_chunk_before_the_model_continues() -> None:
    first = np.full((320, 1), 0.2, dtype=np.float32)
    second = np.full((320, 1), -0.2, dtype=np.float32)
    seen: list[int] = []

    class Paced(ScriptedSession):
        def __init__(self) -> None:
            super().__init__([])
            self.phase = 0
            self.speaker = None

        def receive(self, timeout_s: float):
            del timeout_s
            self.phase += 1
            if self.phase == 1:
                return GatewayAudio(first, 16000)
            if self.phase == 2:
                seen.append(int(self.speaker.reply_audio().shape[0]))
                return GatewayAudio(second, 16000)
            return GatewayEnd()

    paced = Paced()
    voice = _remote_voice(paced)
    paced.speaker = voice.speaker
    session = _ready(voice, AudioModel.REMOTE)
    result = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert result.outcome is TurnOutcome.SPOKEN
    assert seen == [320]
    assert voice.speaker.reply_audio().shape == (640, 1)


def test_remote_without_a_gateway_is_offline_and_wake_still_works() -> None:
    sys.modules.pop("hardware_voice_ai.audio_model.tls", None)
    voice = build_voice_loop(DeviceConfig(), speaker=MemorySpeaker(), clock=ManualClock())
    session = _ready(voice, AudioModel.REMOTE)
    result = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert result.outcome is TurnOutcome.UNAVAILABLE
    assert result.cue == "offline"
    assert result.message == "remote audio model is unavailable"
    assert voice.speaker.cues[0][0] == "offline"
    assert "hardware_voice_ai.audio_model.tls" not in sys.modules
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
    assert session.on_wake_accepted(_event(), _snapshot()) is True


def test_remote_timeout_plays_the_offline_cue() -> None:
    clock = ManualClock()
    scripted = ScriptedSession([], clock)
    voice = _remote_voice(scripted, clock=clock)
    session = _ready(voice, AudioModel.REMOTE)
    result = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert result.outcome is TurnOutcome.TIMEOUT
    assert result.cue == "offline"
    assert scripted.cancels == 1
    assert scripted.closed is True
    _wait_idle(session)
    assert session.state is DeviceState.IDLE


def test_remote_error_and_low_confidence_are_not_played_as_speech() -> None:
    scripted = ScriptedSession(
        [GatewayError("low_confidence", f"nope {TOKEN}")]
    )
    voice = _remote_voice(scripted)
    session = _ready(voice, AudioModel.REMOTE)
    result = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert result.outcome is TurnOutcome.LOW_CONFIDENCE
    assert result.cue == "error"
    assert TOKEN not in result.message
    assert voice.speaker.reply_audio().shape[0] == 0


def test_cancel_stops_the_reply_without_a_cue() -> None:
    holder: dict = {}

    class CancelSession(ScriptedSession):
        def send_audio(self, samples: np.ndarray) -> None:
            super().send_audio(samples)
            if len(self.sent) == 1:
                holder["session"].cancel()

    scripted = CancelSession([])
    voice = _remote_voice(scripted)
    session = _ready(voice, AudioModel.REMOTE)
    holder["session"] = session
    result = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert result.outcome is TurnOutcome.CANCELLED
    assert result.cue is None
    assert voice.speaker.cues == []
    assert voice.speaker.stops >= 1
    assert scripted.cancels == 1
    assert len(scripted.sent) == 1
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
    assert session.on_wake_accepted(_event(), _snapshot()) is True


def test_mute_during_the_model_keeps_the_microphone_loop_running() -> None:
    """The runtime, not the test, runs the model off the microphone thread."""

    class Blocking(ScriptedSession):
        def __init__(self) -> None:
            super().__init__([])
            self.entered = threading.Event()

        def receive(self, timeout_s: float):
            self.entered.set()
            time.sleep(0.01 if timeout_s == 0 else min(timeout_s, 0.02))
            return None

    scripted = Blocking()
    voice = _remote_voice(scripted)
    mute = InMemoryMuteControl()
    config = DeviceConfig()
    word = render_pattern(default_template())
    samples = np.concatenate((silence(1600), word, silence(48_000)))
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        voice=voice,
        mute=mute,
    )
    started = False
    while runtime.poll():
        time.sleep(0)
        if scripted.entered.is_set():
            started = True
            break
    assert started
    polls_while_model_runs = 0
    for _ in range(20):
        assert runtime.poll() is True
        polls_while_model_runs += 1
    assert polls_while_model_runs == 20
    assert scripted.closed is False
    assert runtime.orchestrator.state is DeviceState.SPEAKING
    mute.set_muted(True)
    runtime.wait_for_turns()
    assert runtime.results[-1].outcome is TurnOutcome.CANCELLED
    assert runtime.results[-1].cue is None
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.orchestrator.is_muted is True
    assert runtime.speaker.stops >= 1
    runtime.close()


def test_submit_refuses_a_bad_utterance_without_calling_the_model() -> None:
    called = []

    class Guard:
        def run(self, samples, sample_rate_hz, cancel, deadline):
            del samples, sample_rate_hz, cancel, deadline
            called.append("ran")
            yield Failure(TurnOutcome.ERROR, "should not run")

    voice = _local_voice(engine=Guard())
    session = _ready(voice, AudioModel.LOCAL)

    mismatch = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert mismatch.outcome is TurnOutcome.ERROR
    assert mismatch.message == "utterance does not match the Audio Model setting"
    assert called == []
    _wait_idle(session)
    assert session.state is DeviceState.IDLE

    session.on_wake_accepted(_event(), _snapshot())
    session.close_capture()
    empty = session.submit_utterance(
        _utterance(samples=np.zeros((0, 1), dtype=np.float32))
    )
    assert empty.outcome is TurnOutcome.ERROR
    assert empty.message == "utterance audio is empty"
    assert called == []

    session.on_wake_accepted(_event(), _snapshot())
    refused = session.submit_utterance(_utterance())
    assert refused.outcome is TurnOutcome.REFUSED
    assert session.state is DeviceState.CAPTURING
    assert called == []
    session.cancel()
    assert session.state is DeviceState.IDLE


def test_close_capture_still_does_not_start_the_model() -> None:
    called = []

    class Guard:
        def run(self, samples, sample_rate_hz, cancel, deadline):
            del samples, sample_rate_hz, cancel, deadline
            called.append("ran")
            if False:
                yield Failure(TurnOutcome.ERROR, "unused")

    voice = _local_voice(engine=Guard())
    capture = RecordingCapture()
    session = SessionOrchestrator(capture, audio_model=AudioModel.LOCAL, voice=voice)
    assert session.on_wake_accepted(_event(), _snapshot()) is True
    session.close_capture()
    assert session.state is DeviceState.CAPTURING
    assert called == []


def test_opener_failure_is_unavailable_without_the_token() -> None:
    def opener():
        raise ModelUnavailable(f"down {TOKEN}")

    voice = _remote_voice(ScriptedSession([]))
    voice.model = RemoteAudioModel(opener, token=TOKEN)
    session = _ready(voice, AudioModel.REMOTE)
    result = session.submit_utterance(_utterance(AudioModel.REMOTE))
    assert result.outcome is TurnOutcome.UNAVAILABLE
    assert result.message == "remote audio model is unavailable"
    assert TOKEN not in result.message
    _wait_idle(session)
    assert session.state is DeviceState.IDLE


def test_always_on_audio_does_not_open_a_gateway_session() -> None:
    sys.modules.pop("hardware_voice_ai.audio_model.tls", None)
    opened: list[str] = []

    def opener():
        opened.append("opened")
        raise ModelUnavailable("remote audio model is unavailable")

    config = DeviceConfig()
    voice = build_voice_loop(config, session_opener=opener, clock=ManualClock())
    word = render_pattern(default_template())
    samples = np.concatenate((silence(1600), word, silence(8000)))
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        voice=voice,
    )
    runtime.run()
    assert opened == []
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert "hardware_voice_ai.audio_model.tls" not in sys.modules
    runtime.orchestrator.close_capture()
    result = runtime.orchestrator.submit_utterance(_utterance(AudioModel.REMOTE))
    assert opened == ["opened"]
    assert result.outcome is TurnOutcome.UNAVAILABLE
    _wait_idle(runtime.orchestrator)
    assert runtime.orchestrator.state is DeviceState.IDLE
    runtime.close()


def test_default_runtime_does_not_import_tls_while_listening() -> None:
    sys.modules.pop("hardware_voice_ai.audio_model.tls", None)
    config = DeviceConfig()
    samples = np.concatenate(
        (silence(1600), render_pattern(default_template()), silence(4000))
    )
    runtime = WakeRuntime(config, PcmAudioSource(samples, config.audio_format))
    runtime.run()
    assert isinstance(runtime.speaker, MemorySpeaker)
    assert runtime.speaker.reply_audio().shape[0] == 0
    assert "hardware_voice_ai.audio_model.tls" not in sys.modules
    runtime.close()


def test_wake_during_speaking_barges_in_and_starts_a_new_capture() -> None:
    """A second wake while the reply is playing cancels that turn.

    The microphone loop pauses before the second wake word, waits until the
    runtime's model thread has entered ``speaking``, then reads the rest.
    """

    class Hold(ScriptedSession):
        def __init__(self) -> None:
            super().__init__([])
            self.reply_started = threading.Event()

        def receive(self, timeout_s: float):
            del timeout_s
            if not self.events and not self.reply_started.is_set():
                chunk = np.full((320, 1), 0.2, dtype=np.float32)
                self.events.append(GatewayAudio(chunk, 16000))
            if self.reply_started.is_set():
                time.sleep(0.01)
                return None
            event = self.events.pop(0)
            self.reply_started.set()
            return event

    class LimitedSource:
        def __init__(self, samples: np.ndarray, fmt, limit: int) -> None:
            self._inner = PcmAudioSource(samples, fmt)
            self.format = fmt
            self.limit = limit
            self.reads = 0

        def read(self):
            if self.reads >= self.limit:
                return None
            frame = self._inner.read()
            if frame is not None:
                self.reads += 1
            return frame

        def close(self) -> None:
            self._inner.close()

    scripted = Hold()
    voice = _remote_voice(scripted)
    config = DeviceConfig()
    word = render_pattern(default_template())
    samples = np.concatenate(
        (silence(1600), word, silence(32_000), word, silence(32_000))
    )
    # Past the 1600 ms leading-silence endpoint, and still before the second word.
    pause_frames = (1600 + len(word) + 30_000) // config.frame_samples
    source = LimitedSource(samples, config.audio_format, pause_frames)
    runtime = WakeRuntime(config, source, voice=voice)
    while runtime.poll():
        pass
    assert scripted.reply_started.wait(2)
    for _ in range(50):
        if runtime.orchestrator.state is DeviceState.SPEAKING:
            break
        time.sleep(0.01)
    assert runtime.orchestrator.state is DeviceState.SPEAKING
    source.limit = 10**9
    while runtime.poll():
        pass
    trace = list(runtime.orchestrator.trace)
    saw_speaking = False
    barged = False
    for state in trace:
        if state is DeviceState.SPEAKING:
            saw_speaking = True
        elif saw_speaking and state is DeviceState.WOKEN:
            barged = True
            break
    runtime.close()
    assert barged
    assert voice.speaker.stops >= 1
    assert any(result.outcome is TurnOutcome.CANCELLED for result in runtime.results)


class _FakeOutput:
    """PortAudio stand-in. The test drives the callback; nothing is played live."""

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


def _drain(callback, frames: int = 320) -> None:
    out = np.zeros((frames, 1), dtype=np.float32)
    callback(out, frames, None, None)


def _wait_until(predicate, message: str) -> None:
    deadline = time.monotonic() + 2.0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.001)


def _flush_runtime(monkeypatch, samples: np.ndarray):
    """Runtime whose speaker blocks in ``flush`` until the test drains it."""
    fake = _FakeOutput()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    config = DeviceConfig(audio_model="local")
    speaker = SoundDeviceSpeaker(config.audio_format, queue_ms=200)
    playback = SpeakerPlayback(speaker, volume=config.volume)
    assert fake.stream is not None
    callback = fake.stream.kwargs["callback"]
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        playback=playback,
    )
    return runtime, speaker, playback, callback


def _reach_flush_tail(runtime, speaker, playback, callback) -> None:
    """Play the reply until ``flush`` is waiting on a queued tail.

    A full queue can also mean ``write`` is blocked. One drained frame that
    is not replaced is the flushing tail: nothing further is coming.
    """
    pumped = 0
    while pumped < 100:
        session = runtime.orchestrator
        if (
            not session.model_is_running
            and session.state is DeviceState.SPEAKING
            and playback.is_playing
            and not playback.is_cue_playing
            and speaker.queued_samples > 320
            and speaker.played_samples > 0
        ):
            before = speaker.queued_samples
            _drain(callback)
            time.sleep(0.02)
            if (
                0 < speaker.queued_samples < before
                and playback.is_playing
                and not playback.is_cue_playing
            ):
                return
            pumped += 1
            continue
        if speaker.queued_samples >= 320:
            _drain(callback)
            pumped += 1
        elif not runtime.poll():
            time.sleep(0.001)
    raise AssertionError("playback did not stop in flush with a queued tail")


def test_memory_speaker_flush_idles_through_the_finished_callback() -> None:
    """In-memory playback, including the default speaker, idles from the reply callback.

    ``SpeakerPlayback`` around ``MemorySpeaker`` and the runtime's default
    ``NullSpeaker`` both finish ``flush`` on the playback thread.
    ``_on_reply_finished`` is what records that completion and leaves ``speaking``.
    """
    config = DeviceConfig(audio_model="local")
    word = render_pattern(default_template())
    samples = np.concatenate((silence(1600), word, silence(32_000)))
    for playback in (
        SpeakerPlayback(PlaybackMemorySpeaker(config.audio_format), volume=config.volume),
        None,
    ):
        runtime = WakeRuntime(
            config,
            PcmAudioSource(samples, config.audio_format),
            playback=playback,
        )
        try:
            runtime.run()
            runtime.wait_for_turns()
            _wait_until(
                lambda runtime=runtime: (
                    runtime.orchestrator.state is DeviceState.IDLE
                    and runtime.orchestrator.last_reply_result is not None
                    and runtime.orchestrator.last_reply_result.status
                    is PlaybackStatus.COMPLETED
                    and runtime.orchestrator.last_reply_result.reason
                    is PlaybackStopReason.COMPLETED
                ),
                "in-memory flush did not finish through the reply callback",
            )
        finally:
            runtime.close()


def test_wake_during_flush_tail_stops_playback_and_session_stays_speaking(monkeypatch) -> None:
    """Speaking lasts through speaker flush, and a wake in that tail cuts it.

    The model thread only finishes the reply stream. ``flush`` is still
    playing that tail. The session stays in ``speaking`` until the tail
    drains or a wake stops it and opens a new capture.
    """
    config = DeviceConfig(audio_model="local")
    word = render_pattern(default_template())
    quiet = np.concatenate((silence(1600), word, silence(32_000)))
    runtime, speaker, playback, callback = _flush_runtime(monkeypatch, quiet)
    try:
        _reach_flush_tail(runtime, speaker, playback, callback)
        assert runtime.orchestrator.state is DeviceState.SPEAKING
        assert runtime.orchestrator.model_is_running is False
        assert playback.is_playing is True
        while speaker.queued_samples > 0:
            _drain(callback)
        _wait_until(
            lambda: (
                runtime.orchestrator.state is DeviceState.IDLE
                and not playback.is_playing
                and runtime.orchestrator.last_reply_result is not None
                and runtime.orchestrator.last_reply_result.status is PlaybackStatus.COMPLETED
            ),
            "flush did not finish through the reply callback",
        )
        assert runtime.orchestrator.state is DeviceState.IDLE
        assert runtime.orchestrator.last_reply_result.reason is PlaybackStopReason.COMPLETED
    finally:
        runtime.close()

    gap = silence(32_000)
    both = np.concatenate((silence(1600), word, gap, word, gap))
    pause_frames = (1600 + len(word) + 30_000) // config.frame_samples

    class LimitedSource:
        def __init__(self) -> None:
            self._inner = PcmAudioSource(both, config.audio_format)
            self.format = config.audio_format
            self.limit = pause_frames
            self.reads = 0

        def read(self):
            if self.reads >= self.limit:
                return None
            frame = self._inner.read()
            if frame is not None:
                self.reads += 1
            return frame

        def close(self) -> None:
            self._inner.close()

    source = LimitedSource()
    fake = _FakeOutput()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    hold = SoundDeviceSpeaker(config.audio_format, queue_ms=200)
    held_playback = SpeakerPlayback(hold, volume=config.volume)
    assert fake.stream is not None
    held_callback = fake.stream.kwargs["callback"]
    runtime = WakeRuntime(config, source, playback=held_playback)
    try:
        _reach_flush_tail(runtime, hold, held_playback, held_callback)
        assert runtime.orchestrator.state is DeviceState.SPEAKING
        assert runtime.orchestrator.model_is_running is False
        assert held_playback.is_playing is True
        assert hold.queued_samples > 0
        source.limit = 10**9
        while runtime.poll():
            if runtime.orchestrator.state is DeviceState.CAPTURING:
                break
        assert runtime.orchestrator.state is DeviceState.CAPTURING
        assert runtime.orchestrator.capture_is_active is True
        assert runtime.orchestrator.last_reply_result is not None
        assert runtime.orchestrator.last_reply_result.status is not None
        assert runtime.orchestrator.last_reply_result.reason is PlaybackStopReason.BARGE_IN
        assert runtime.capture.holds_audio is True
    finally:
        runtime.close()


def test_wake_as_the_previous_turn_finishes_stays_in_capture() -> None:
    """A wake accepted while the old turn finishes is not forced back to idle."""
    config = DeviceConfig(audio_model="local")
    word = render_pattern(default_template())
    samples = np.concatenate(
        (silence(1600), word, silence(32_000), word, silence(8_000))
    )
    pause_frames = (1600 + len(word) + 30_000) // config.frame_samples

    class LimitedSource:
        def __init__(self) -> None:
            self._inner = PcmAudioSource(samples, config.audio_format)
            self.format = config.audio_format
            self.limit = pause_frames
            self.reads = 0

        def read(self):
            if self.reads >= self.limit:
                return None
            frame = self._inner.read()
            if frame is not None:
                self.reads += 1
            return frame

        def close(self) -> None:
            self._inner.close()

    source = LimitedSource()
    runtime = WakeRuntime(config, source)
    finishing = threading.Event()
    wake_accepted = threading.Event()
    original = runtime.orchestrator._finish_turn

    def finish_after_the_new_wake(turn_id: int) -> None:
        finishing.set()
        assert wake_accepted.wait(2)
        original(turn_id)

    runtime.orchestrator._finish_turn = finish_after_the_new_wake
    while runtime.poll():
        pass
    assert finishing.wait(2)
    assert runtime.orchestrator.state is DeviceState.SPEAKING
    source.limit = 10**9
    while runtime.poll():
        if (
            runtime.orchestrator.state is DeviceState.CAPTURING
            and runtime.orchestrator.capture_is_active
        ):
            break
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.capture_is_active is True
    wake_accepted.set()
    runtime.wait_for_turns()
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.capture_is_active is True
    assert runtime.capture.holds_audio is True
    runtime.close()


def test_end_of_utterance_speaks_and_a_second_wake_is_accepted() -> None:
    """A finished utterance leaves capturing through speaking and back to idle."""
    config = DeviceConfig(audio_model="local")
    word = render_pattern(default_template())
    gap = silence(32_000)
    samples = np.concatenate((silence(1600), word, gap, word, gap))
    runtime = WakeRuntime(config, PcmAudioSource(samples, config.audio_format))
    runtime.run()
    runtime.wait_for_turns()
    # Idle follows speaker flush, which can finish just after the model thread.
    _wait_until(
        lambda: (
            runtime.orchestrator.state is DeviceState.IDLE
            and runtime.orchestrator.last_reply_result is not None
            and runtime.orchestrator.last_reply_result.status is PlaybackStatus.COMPLETED
        ),
        "spoken reply did not return to idle after flush",
    )
    runtime.close()
    trace = runtime.orchestrator.trace
    entered_speaking = sum(
        1
        for previous, current in zip(trace, trace[1:])
        if current is DeviceState.SPEAKING and previous is not DeviceState.SPEAKING
    )
    assert entered_speaking == 2
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.orchestrator.trace[-2:] == [DeviceState.SPEAKING, DeviceState.IDLE]
    assert runtime.speaker.reply_audio().shape[0] > 0
    assert runtime.capture.holds_audio is False
    assert runtime.utterances is not None
    assert runtime.utterances.take() is None


def test_utterance_adapter_keeps_the_pre_roll_and_the_continuation() -> None:
    request = CaptureRequest(
        wake=_event(),
        pre_roll=_snapshot(),
        audio_model=AudioModel.REMOTE,
    )
    continuation = np.full((160, 1), 0.5, dtype=np.float32)
    utterance = utterance_from_request(request, continuation)
    assert utterance.audio_model is AudioModel.REMOTE
    assert utterance.pre_roll_sample_count == 320
    assert utterance.samples.shape == (480, 1)
    assert np.all(utterance.samples[:320] == 0)
    assert np.all(utterance.samples[320:] == 0.5)
    assert utterance.endpoint == "handed_off"


def test_overlong_utterance_is_not_sent() -> None:
    sent = []

    class Spy(ScriptedSession):
        def send_audio(self, samples: np.ndarray) -> None:
            sent.append(samples)

    voice = _remote_voice(Spy([]))
    session = _ready(voice, AudioModel.REMOTE)
    samples = np.zeros((31, 1), dtype=np.float32)
    result = session.submit_utterance(_utterance(AudioModel.REMOTE, samples, rate=1))
    assert result.outcome is TurnOutcome.ERROR
    assert "longer" in result.message
    assert sent == []
    _wait_idle(session)
    assert session.state is DeviceState.IDLE
