"""Length-prefixed frames for one remote Audio Model turn.

This is the server side of the device session. The device client sends the
same layout: a big-endian uint32 length, a uint8 type, then the payload.
Type 1 is UTF-8 JSON. Type 2 is int16 little-endian PCM. This module does
not open a socket.

Device to gateway:

- ``{"type":"hello","audio_model":"remote"}`` and, when a token is set,
  ``"authorization"``. ``"backend"`` is optional.
- ``{"type":"audio_start","sample_rate_hz":16000,"channels":1}``
- audio frames
- ``{"type":"audio_end"}``
- ``{"type":"cancel"}``

Gateway to device:

- ``{"type":"reply_start","sample_rate_hz":16000,"channels":1}``
- audio frames, the spoken reply
- ``{"type":"error","code":"...","message":"..."}``
- ``{"type":"end"}``

There is no transcript frame. The spoken reply is audio.
"""

import json
import struct

import numpy as np

TEXT = 1
AUDIO = 2
_HEADER = struct.Struct(">IB")
MAX_PAYLOAD = 1_048_576


class FramingError(ValueError):
    """A frame does not follow the remote Audio Model session."""


def encode_text(payload: dict) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return encode_frame(TEXT, body)


def encode_audio(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.ascontiguousarray(np.rint(clipped * 32767.0).astype("<i2"))
    return encode_frame(AUDIO, pcm.tobytes())


def encode_frame(kind: int, payload: bytes) -> bytes:
    if kind not in (TEXT, AUDIO):
        raise FramingError("unknown frame type")
    if len(payload) > MAX_PAYLOAD:
        raise FramingError("frame is too large")
    return _HEADER.pack(len(payload), kind) + payload


def decode_header(header: bytes) -> tuple[int, int]:
    if len(header) != _HEADER.size:
        raise FramingError("short frame header")
    length, kind = _HEADER.unpack(header)
    if kind not in (TEXT, AUDIO) or length < 0 or length > MAX_PAYLOAD:
        raise FramingError("invalid frame header")
    return int(length), int(kind)


def decode_text(payload: bytes) -> dict:
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FramingError("control frame is not JSON") from exc
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise FramingError("control frame has no type")
    return message


def decode_audio(payload: bytes, channels: int) -> np.ndarray:
    if channels < 1:
        raise FramingError("audio channel count is invalid")
    if len(payload) % (2 * channels) != 0:
        raise FramingError("audio frame does not match the channel count")
    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    if pcm.size == 0:
        raise FramingError("audio frame is empty")
    samples = np.array((pcm / 32767.0).reshape(-1, channels), dtype=np.float32, copy=True)
    samples.setflags(write=False)
    return samples


class FrameReader:
    """Pull complete frames from a stream that may return short reads."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._buffer = bytearray()

    def read_frame(self) -> tuple[int, bytes] | None:
        header = self._read_exact(_HEADER.size)
        if header is None:
            return None
        length, kind = decode_header(header)
        if length == 0:
            return kind, b""
        payload = self._read_exact(length)
        if payload is None:
            raise FramingError("frame ended before it was complete")
        return kind, payload

    def _read_exact(self, count: int) -> bytes | None:
        while len(self._buffer) < count:
            part = self._stream.read(count - len(self._buffer))
            if not part:
                if not self._buffer:
                    return None
                raise FramingError("frame ended before it was complete")
            self._buffer.extend(part)
        out = bytes(self._buffer[:count])
        del self._buffer[:count]
        return out
