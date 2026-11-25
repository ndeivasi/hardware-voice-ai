"""Serve one remote turn on an already-open byte stream.

One connection is one turn. The caller closes the stream. This module does
not open a socket. The device client is the other end: it sends the
utterance after wake and reads spoken audio back.
"""

import hmac
import threading

import numpy as np

from hardware_voice_ai.errors import ConfigurationError, RemoteBackendError
from hardware_voice_ai.gateway.audio import iter_chunks, match_channels, resample
from hardware_voice_ai.gateway.backend import SpeechBackend, SpokenChunk
from hardware_voice_ai.gateway.catalog import parse_remote_backend
from hardware_voice_ai.gateway.framing import (
    AUDIO,
    TEXT,
    FrameReader,
    FramingError,
    decode_audio,
    decode_text,
    encode_audio,
    encode_text,
)
from hardware_voice_ai.gateway.settings import GatewaySettings

MAX_UTTERANCE_S = 30.0
_MIN_RATE = 8_000
_MAX_RATE = 48_000


class TurnGateway:
    """One loaded backend. A second connection is refused until the turn ends."""

    def __init__(self, settings: GatewaySettings, backend: SpeechBackend) -> None:
        self.settings = settings
        self.backend = backend
        self._lock = threading.Lock()

    def handle(self, stream) -> None:
        if not self._lock.acquire(blocking=False):
            _try_error(stream, "unavailable", "remote audio model is unavailable")
            return
        try:
            serve_turn(stream, self.backend, self.settings)
        finally:
            self._lock.release()


def serve_turn(
    stream,
    backend: SpeechBackend,
    settings: GatewaySettings,
    *,
    max_utterance_s: float = MAX_UTTERANCE_S,
) -> None:
    """Read one device turn and write the spoken reply, or one error."""
    reader = FrameReader(stream)
    try:
        _exchange(reader, stream, backend, settings, max_utterance_s)
    except (BrokenPipeError, ConnectionError):
        return
    except (RemoteBackendError, ConfigurationError):
        _try_error(stream, "unavailable", "remote audio model is unavailable")
    except FramingError:
        _try_error(stream, "failed", "audio model failed")
    except Exception:
        _try_error(stream, "failed", "audio model failed")


def _exchange(reader, stream, backend, settings: GatewaySettings, max_utterance_s: float) -> None:
    hello = _expect_text(reader, "hello")
    if hello.get("audio_model") != "remote":
        raise FramingError("hello is not the remote audio model")
    _check_token(settings.token, hello.get("authorization", None))
    _check_backend(settings.backend_id, hello.get("backend", None))

    start = _expect_text(reader, "audio_start")
    rate = _rate(start.get("sample_rate_hz"))
    channels = _channels(start.get("channels"))
    pieces: list[np.ndarray] = []
    total = 0
    limit = int(rate * max_utterance_s)
    while True:
        frame = reader.read_frame()
        if frame is None:
            raise FramingError("turn ended before audio_end")
        kind, payload = frame
        if kind == TEXT:
            message = decode_text(payload)
            name = message["type"]
            if name == "cancel":
                return
            if name == "audio_end":
                break
            raise FramingError("unexpected control during audio")
        if kind != AUDIO:
            raise FramingError("unknown frame type")
        samples = decode_audio(payload, channels)
        total += int(samples.shape[0])
        if total > limit:
            raise FramingError("utterance is longer than the turn cap")
        pieces.append(np.array(samples, dtype=np.float32, copy=True))

    if not pieces:
        _send_error(stream, "low_confidence", "audio was hard to hear")
        return
    utterance = np.concatenate(pieces, axis=0)
    if not np.isfinite(utterance).all():
        raise FramingError("utterance audio is not finite")

    cancel = threading.Event()
    reply_rate: int | None = None
    reply_channels: int | None = None
    wrote_audio = False
    for chunk in backend.speak(utterance, rate, channels, cancel):
        spoken = _fit(chunk, rate, channels)
        if reply_rate is None:
            reply_rate = rate
            reply_channels = channels
            _write(stream, encode_text({
                "type": "reply_start",
                "sample_rate_hz": reply_rate,
                "channels": reply_channels,
            }))
        frame_samples = max(1, rate * 20 // 1000)
        for piece in iter_chunks(spoken, frame_samples):
            _write(stream, encode_audio(piece))
            wrote_audio = True
    if not wrote_audio:
        _send_error(stream, "failed", "audio model failed")
        return
    _write(stream, encode_text({"type": "end"}))


def _fit(chunk: SpokenChunk, rate: int, channels: int) -> np.ndarray:
    shaped = match_channels(chunk.samples, channels)
    return resample(shaped, chunk.sample_rate_hz, rate)


def _check_token(expected: str, presented: object) -> None:
    if presented is None:
        presented = ""
    if not isinstance(presented, str):
        _reject()
    if not hmac.compare_digest(expected, presented):
        _reject()


def _reject() -> None:
    raise RemoteBackendError("gateway rejected the session")


def _check_backend(running: str, requested: object) -> None:
    if requested is None:
        return
    if not isinstance(requested, str):
        raise ConfigurationError("remote backend must be a string.")
    if parse_remote_backend(requested) != running:
        raise ConfigurationError("gateway is not serving that audio model.")


def _expect_text(reader: FrameReader, name: str) -> dict:
    frame = reader.read_frame()
    if frame is None:
        raise FramingError("session ended early")
    kind, payload = frame
    if kind != TEXT:
        raise FramingError("expected a control frame")
    message = decode_text(payload)
    if message.get("type") != name:
        raise FramingError(f"expected {name}")
    return message


def _rate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not _MIN_RATE <= value <= _MAX_RATE:
        raise FramingError("audio sample rate is invalid")
    return value


def _channels(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8:
        raise FramingError("audio channel count is invalid")
    return value


def _write(stream, data: bytes) -> None:
    stream.write(data)


def _send_error(stream, code: str, message: str) -> None:
    _write(stream, encode_text({"type": "error", "code": code, "message": message}))


def _try_error(stream, code: str, message: str) -> None:
    try:
        _send_error(stream, code, message)
    except (BrokenPipeError, ConnectionError, OSError):
        return
