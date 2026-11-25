"""One streaming session with the cloud gateway.

The session carries this turn's utterance audio out and spoken-reply audio
back. Control messages are errors, end of turn, and cancel. Cancel is not
audio. The transport is injected, so this class does not open a socket itself.
"""

from typing import Protocol

import numpy as np

from hardware_voice_ai.audio_model.framing import (
    AUDIO,
    TEXT,
    FramingError,
    decode_audio,
    decode_header,
    decode_text,
    encode_audio,
    encode_text,
)
from hardware_voice_ai.audio_model.gateway import (
    GatewayAudio,
    GatewayEnd,
    GatewayError,
    GatewayMessage,
    redact,
)


class DuplexStream(Protocol):
    def read(self, n: int, timeout_s: float) -> bytes: ...

    def write(self, data: bytes) -> None: ...

    def close(self) -> None: ...


class BufferStream:
    """In-memory duplex used by tests. An empty read raises ``TimeoutError``."""

    def __init__(self, incoming: bytes = b"") -> None:
        self.incoming = bytearray(incoming)
        self.outgoing = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionError("session is closed")
        self.outgoing.extend(data)

    def read(self, n: int, timeout_s: float) -> bytes:
        del timeout_s
        if self.closed and not self.incoming:
            return b""
        if not self.incoming:
            raise TimeoutError
        take = min(n, len(self.incoming))
        out = bytes(self.incoming[:take])
        del self.incoming[:take]
        return out

    def close(self) -> None:
        self.closed = True


class GatewaySession(Protocol):
    def start_turn(self, *, sample_rate_hz: int, channels: int) -> None: ...

    def send_audio(self, samples: np.ndarray) -> None: ...

    def finish_audio(self) -> None: ...

    def send_cancel(self) -> None: ...

    def receive(self, timeout_s: float) -> GatewayMessage | None: ...

    def close(self) -> None: ...


class FramedGatewaySession:
    """Gateway session over a length-prefixed duplex stream."""

    def __init__(self, stream: DuplexStream, *, token: str = "") -> None:
        self._stream = stream
        self._token = token
        self._send_channels: int | None = None
        self._reply_rate: int | None = None
        self._reply_channels: int | None = None
        self._closed = False

    def __repr__(self) -> str:
        return "FramedGatewaySession()"

    def start_turn(self, *, sample_rate_hz: int, channels: int) -> None:
        if sample_rate_hz <= 0 or channels < 1:
            raise FramingError("turn audio format is invalid")
        hello = {"type": "hello", "audio_model": "remote"}
        if self._token:
            hello["authorization"] = self._token
        self._write_text(hello)
        self._write_text(
            {
                "type": "audio_start",
                "sample_rate_hz": int(sample_rate_hz),
                "channels": int(channels),
            }
        )
        self._send_channels = int(channels)

    def send_audio(self, samples: np.ndarray) -> None:
        if self._send_channels is None:
            raise FramingError("audio sent before the turn started")
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self._send_channels or array.shape[0] < 1:
            raise FramingError("audio chunk does not match the turn format")
        self._write(encode_audio(array))

    def finish_audio(self) -> None:
        self._write_text({"type": "audio_end"})

    def send_cancel(self) -> None:
        self._write_text({"type": "cancel"})

    def receive(self, timeout_s: float) -> GatewayMessage | None:
        while True:
            frame = self._read_frame(timeout_s)
            if frame is None:
                return None
            kind, payload = frame
            if kind == AUDIO:
                if self._reply_rate is None or self._reply_channels is None:
                    raise FramingError("reply audio arrived before reply_start")
                samples = decode_audio(payload, self._reply_channels)
                return GatewayAudio(samples=samples, sample_rate_hz=self._reply_rate)
            message = decode_text(payload)
            kind_name = message["type"]
            if kind_name == "reply_start":
                self._reply_rate = _positive_int(message.get("sample_rate_hz"))
                self._reply_channels = _positive_int(message.get("channels"))
                continue
            if kind_name == "error":
                code = message.get("code", "failed")
                detail = message.get("message", "")
                if not isinstance(code, str) or not isinstance(detail, str):
                    raise FramingError("error control is malformed")
                return GatewayError(code=code, message=redact(detail, self._token))
            if kind_name == "end":
                return GatewayEnd()
            raise FramingError("unknown gateway control")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()

    def _write_text(self, payload: dict) -> None:
        self._write(encode_text(payload))

    def _write(self, data: bytes) -> None:
        if self._closed:
            raise ConnectionError("session is closed")
        self._stream.write(data)

    def _read_frame(self, timeout_s: float) -> tuple[int, bytes] | None:
        try:
            header = _read_exact(self._stream, 5, timeout_s)
        except TimeoutError:
            return None
        length, kind = decode_header(header)
        if length == 0:
            return kind, b""
        try:
            payload = _read_exact(self._stream, length, timeout_s)
        except TimeoutError as exc:
            raise FramingError("frame ended before it was complete") from exc
        if kind == TEXT and payload == b"":
            raise FramingError("control frame is empty")
        return kind, payload


def _read_exact(stream: DuplexStream, count: int, timeout_s: float) -> bytes:
    buffer = bytearray()
    while len(buffer) < count:
        try:
            part = stream.read(count - len(buffer), timeout_s)
        except TimeoutError:
            if buffer:
                raise FramingError("frame ended before it was complete") from None
            raise
        if part == b"":
            if buffer:
                raise FramingError("frame ended before it was complete")
            raise ConnectionError("gateway closed the session")
        buffer.extend(part)
    return bytes(buffer)


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FramingError("reply format is invalid")
    return value
