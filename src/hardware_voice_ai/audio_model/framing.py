"""Length-prefixed frames for one remote Audio Model turn.

Text frames are JSON control. Audio frames are int16 PCM. This module does
not open a socket.
"""

import json
import struct

import numpy as np

TEXT = 1
AUDIO = 2
_HEADER = struct.Struct(">IB")
MAX_PAYLOAD = 1_048_576


class FramingError(ValueError):
    """A frame or control message does not follow the gateway protocol."""


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
        raise FramingError("reply channel count is invalid")
    if len(payload) % (2 * channels) != 0:
        raise FramingError("audio frame does not match the channel count")
    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    if pcm.size == 0:
        raise FramingError("audio frame is empty")
    samples = np.array((pcm / 32767.0).reshape(-1, channels), dtype=np.float32, copy=True)
    samples.setflags(write=False)
    return samples


def iter_chunks(samples: np.ndarray, frame: int):
    if frame < 1:
        raise ValueError("frame must be positive")
    for start in range(0, int(samples.shape[0]), frame):
        yield samples[start : start + frame]
