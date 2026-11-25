"""Buffer a post-wake utterance and hand it to the session orchestrator.

``UtteranceCapture`` is the ``CapturePort`` the wake runtime calls:

- ``start`` receives the wake event, the pre-roll snapshot, and the Audio
  Model for the turn.
- ``on_frame`` receives microphone frames after the decision frame.
- On end-of-utterance this object calls ``SessionOrchestrator.close_capture``.
  That calls back into ``stop`` with reason ``end_of_utterance``.
- ``stop(cancel)`` and ``stop(mute)`` drop the buffer and do not hand audio on.

``on_frame`` calls ``close_capture`` on the session orchestrator. It does not
call back into the wake runtime, the wake model, or the microphone source.
The wake runtime remains the only reader of the device microphone.
"""

import threading

import numpy as np

from hardware_voice_ai.audio.frame import AudioFrame
from hardware_voice_ai.capture.endpoint import (
    EndpointConfig,
    EndpointTracker,
    noise_floor,
    speech_threshold,
    tail_is_speech,
)
from hardware_voice_ai.capture.handoff import (
    CaptureLifecycle,
    CapturedUtterance,
    UtteranceSink,
)
from hardware_voice_ai.capture.log import CaptureLog
from hardware_voice_ai.capture.port import (
    CaptureRequest,
    CaptureStopReason,
)

ENDPOINT_REQUESTED = "requested"


class CaptureError(RuntimeError):
    """The capture handoff cannot proceed with this audio or session."""


class UtteranceCapture:
    """``CapturePort`` that keeps the pre-roll and the following frames.

    ``WakeRuntime`` installs this class by default and binds it to the real
    session orchestrator. Pass an instance as ``capture=`` to choose the
    endpoint or the sink; the runtime still calls ``bind_session``. Direct
    use outside the runtime must bind a session before end-of-utterance.

    The finished utterance is delivered to ``sink`` only for end-of-utterance.
    Cancel and hardware mute discard it. Nothing is written to disk unless an
    opt-in ``CaptureLog`` was given, and that log stores metadata only.
    """

    def __init__(
        self,
        sink: UtteranceSink,
        *,
        endpoint: EndpointConfig | None = None,
        session: CaptureLifecycle | None = None,
        log: CaptureLog | None = None,
    ) -> None:
        self._sink = sink
        self._endpoint = endpoint if endpoint is not None else EndpointConfig()
        self._session = session
        self._log = log if log is not None else CaptureLog()
        self._lock = threading.RLock()
        self._active = False
        self._pre_roll = np.zeros((0, 1), dtype=np.float32)
        self._frames: list[np.ndarray] = []
        self._request: CaptureRequest | None = None
        self._tracker: EndpointTracker | None = None
        self._next_index = 0
        self._frame_samples = 0
        self._rate = 0
        self._channels = 0
        self._endpoint_cause: str | None = None
        self._wake_word_end_offset = 0

    def bind_session(self, session: CaptureLifecycle) -> None:
        """Connect end-of-utterance to ``SessionOrchestrator.close_capture``."""
        with self._lock:
            self._session = session

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def holds_audio(self) -> bool:
        """True only while a turn is open and its samples are still in memory."""
        with self._lock:
            return self._active

    def start(self, request: CaptureRequest) -> None:
        with self._lock:
            if self._active:
                raise CaptureError("capture is already running")
            self._begin(request)
            summary = (
                "event=start"
                f" audio_model={request.audio_model.value}"
                f" sample_rate_hz={self._rate}"
                f" channels={self._channels}"
                f" pre_roll_samples={self._pre_roll.shape[0]}"
            )
        self._log.write(summary)

    def on_frame(self, frame: AudioFrame) -> None:
        session: CaptureLifecycle | None = None
        missing_session = False
        with self._lock:
            if not self._active:
                return
            self._accept_frame(frame)
            assert self._tracker is not None
            cause = self._tracker.observe(frame.samples)
            if cause is None:
                return
            self._endpoint_cause = cause
            session = self._session
            missing_session = session is None
        if missing_session:
            raise CaptureError(
                "bind_session before the utterance can end; "
                "end-of-utterance calls SessionOrchestrator.close_capture"
            )
        assert session is not None
        session.close_capture()

    def stop(self, reason: CaptureStopReason) -> None:
        utterance: CapturedUtterance | None = None
        with self._lock:
            if not self._active:
                return
            cause = self._endpoint_cause or ENDPOINT_REQUESTED
            if reason is CaptureStopReason.END_OF_UTTERANCE:
                utterance = self._build(cause)
            post_samples = sum(frame.shape[0] for frame in self._frames)
            self._release()
            summary = (
                "event=stop"
                f" reason={reason.value}"
                f" endpoint={cause}"
                f" post_samples={post_samples}"
                f" handed_off={utterance is not None}"
            )
        self._log.write(summary)
        if utterance is not None:
            self._sink.accept_utterance(utterance)

    def _begin(self, request: CaptureRequest) -> None:
        pre_roll = request.pre_roll
        wake = request.wake
        samples = np.array(pre_roll.samples, dtype=np.float32, copy=True)
        if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 1:
            raise CaptureError("pre-roll must contain at least one sample and one channel")
        if not np.isfinite(samples).all():
            raise CaptureError("pre-roll samples must be finite")
        expected_rows = pre_roll.end_sample_index - pre_roll.start_sample_index
        if samples.shape[0] != expected_rows:
            raise CaptureError("pre-roll length does not match its sample indexes")
        if pre_roll.sample_rate_hz != wake.sample_rate_hz:
            raise CaptureError("pre-roll sample rate does not match the wake event")
        if samples.shape[1] != wake.channels:
            raise CaptureError("pre-roll channel count does not match the wake event")
        if wake.channels < 1 or wake.sample_rate_hz <= 0 or wake.frame_samples < 1:
            raise CaptureError("wake event does not describe a usable microphone format")
        if pre_roll.end_sample_index != wake.decision_sample_index:
            raise CaptureError("pre-roll must end at the wake decision sample")
        if not (
            pre_roll.start_sample_index
            < wake.word_end_sample_index
            <= pre_roll.end_sample_index
        ):
            raise CaptureError("pre-roll must include the end of the wake word")
        floor = noise_floor(samples, wake.frame_samples, self._endpoint.noise_percentile)
        threshold = speech_threshold(floor, self._endpoint)
        started = tail_is_speech(
            samples,
            wake.sample_rate_hz,
            threshold,
            self._endpoint.speech_tail_ms,
        )
        self._request = request
        self._pre_roll = samples
        self._frames = []
        self._rate = wake.sample_rate_hz
        self._channels = int(samples.shape[1])
        self._frame_samples = wake.frame_samples
        self._next_index = pre_roll.end_sample_index
        self._wake_word_end_offset = wake.word_end_sample_index - pre_roll.start_sample_index
        self._endpoint_cause = None
        self._tracker = EndpointTracker(
            self._endpoint,
            wake.sample_rate_hz,
            threshold,
            started,
        )
        self._active = True

    def _accept_frame(self, frame: AudioFrame) -> None:
        if frame.sample_index != self._next_index:
            raise CaptureError(
                f"frame sample_index {frame.sample_index} does not continue "
                f"the utterance at {self._next_index}"
            )
        if (
            frame.format.sample_rate_hz != self._rate
            or frame.format.channels != self._channels
            or frame.format.frame_samples != self._frame_samples
        ):
            raise CaptureError("frame format does not match the wake handoff")
        self._frames.append(frame.samples)
        self._next_index = frame.end_sample_index

    def _build(self, endpoint: str) -> CapturedUtterance:
        assert self._request is not None
        parts = [self._pre_roll, *self._frames]
        audio = np.concatenate(parts, axis=0)
        return CapturedUtterance(
            wake=self._request.wake,
            audio_model=self._request.audio_model,
            samples=audio,
            sample_rate_hz=self._rate,
            pre_roll_sample_count=int(self._pre_roll.shape[0]),
            wake_word_end_offset=self._wake_word_end_offset,
            endpoint=endpoint,
        )

    def _release(self) -> None:
        channels = self._channels if self._channels > 0 else 1
        self._active = False
        self._pre_roll = np.zeros((0, channels), dtype=np.float32)
        self._frames = []
        self._request = None
        self._tracker = None
        self._endpoint_cause = None
