"""Gateway framing, credentials, and the TLS transport boundary."""

import socket

import numpy as np
import pytest

from hardware_voice_ai.audio_model.framing import AUDIO, TEXT, decode_audio, decode_header, decode_text
from hardware_voice_ai.audio_model.gateway import GatewayConfig, parse_gateway_url, redact
from hardware_voice_ai.audio_model.session import BufferStream, FramedGatewaySession
from hardware_voice_ai.audio_model.tls import SocketStream, connect_framed_session
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ConfigurationError, ModelUnavailable

TOKEN = "test-token"


def _frames(data: bytes) -> list[tuple[int, bytes]]:
    frames = []
    offset = 0
    while offset < len(data):
        length, kind = decode_header(data[offset : offset + 5])
        offset += 5
        payload = data[offset : offset + length]
        assert len(payload) == length
        offset += length
        frames.append((kind, payload))
    return frames


def test_gateway_url_is_tls_and_the_credential_stays_out_of_it() -> None:
    assert parse_gateway_url("") == ("", 0)
    assert parse_gateway_url("tls://gateway.example:443") == ("gateway.example", 443)
    with pytest.raises(ConfigurationError, match="cleartext"):
        parse_gateway_url("http://gateway.example:443")
    with pytest.raises(ConfigurationError, match="HVA_GATEWAY_TOKEN"):
        parse_gateway_url(f"tls://{TOKEN}@gateway.example:443")
    with pytest.raises(ConfigurationError):
        DeviceConfig(gateway=GatewayConfig(url="ws://gateway.example:443"))


def test_config_repr_hides_the_gateway_token() -> None:
    config = DeviceConfig.from_env(
        {
            "HVA_GATEWAY_URL": "tls://gateway.example:8443",
            "HVA_GATEWAY_TOKEN": TOKEN,
        }
    )
    assert config.gateway.url == "tls://gateway.example:8443"
    assert config.gateway.token == TOKEN
    rendered = repr(config)
    assert TOKEN not in rendered
    assert "token=set" in repr(config.gateway)
    with pytest.raises(ConfigurationError, match="single line"):
        DeviceConfig.from_env({"HVA_GATEWAY_TOKEN": "one\ntwo"})


def test_framed_session_sends_audio_and_cancel_as_control() -> None:
    stream = BufferStream()
    session = FramedGatewaySession(stream, token=TOKEN)
    samples = np.array([[0.5], [-0.25], [0.0], [1.0]], dtype=np.float32)
    session.start_turn(sample_rate_hz=16000, channels=1)
    session.send_audio(samples)
    session.finish_audio()
    session.send_cancel()
    assert TOKEN not in repr(session)
    frames = _frames(bytes(stream.outgoing))
    texts = [decode_text(payload) for kind, payload in frames if kind == TEXT]
    audio = [payload for kind, payload in frames if kind == AUDIO]
    assert [message["type"] for message in texts] == [
        "hello",
        "audio_start",
        "audio_end",
        "cancel",
    ]
    assert texts[0]["audio_model"] == "remote"
    assert texts[0]["authorization"] == TOKEN
    assert set(texts[0]) == {"type", "audio_model", "authorization"}
    assert texts[1] == {"type": "audio_start", "sample_rate_hz": 16000, "channels": 1}
    assert texts[-1] == {"type": "cancel"}
    assert len(audio) == 1
    assert np.allclose(decode_audio(audio[0], 1), samples, atol=1 / 32767)
    blob = bytes(stream.outgoing).lower()
    for vendor in (b"openai", b"whisper", b"elevenlabs", b"deepgram", b"anthropic"):
        assert vendor not in blob


def test_reply_audio_and_errors_redact_the_token() -> None:
    reply = np.full((4, 1), 0.25, dtype=np.float32)
    from hardware_voice_ai.audio_model.framing import encode_audio, encode_text

    incoming = b"".join(
        [
            encode_text({"type": "reply_start", "sample_rate_hz": 16000, "channels": 1}),
            encode_audio(reply),
            encode_text({"type": "error", "code": "failed", "message": f"nope {TOKEN}"}),
        ]
    )
    session = FramedGatewaySession(BufferStream(incoming), token=TOKEN)
    audio = session.receive(1)
    error = session.receive(1)
    assert np.allclose(audio.samples, reply, atol=1 / 32767)
    assert error.code == "failed"
    assert error.message == "nope [redacted]"
    assert TOKEN not in error.message
    assert redact(f"x {TOKEN} y", TOKEN) == "x [redacted] y"


def test_socket_stream_roundtrip_stays_on_a_local_pair() -> None:
    left, right = socket.socketpair()
    try:
        stream = SocketStream(left, 1.0)
        stream.write(b"abc")
        assert right.recv(3) == b"abc"
        right.sendall(b"wxyz")
        assert stream.read(4, 1.0) == b"wxyz"
    finally:
        left.close()
        right.close()


def test_tls_connect_failure_does_not_include_the_token() -> None:
    def boom(host: str, port: int, timeout_s: float) -> socket.socket:
        assert host == "gateway.example"
        assert port == 443
        assert timeout_s == 5.0
        raise OSError("network is down")

    with pytest.raises(ModelUnavailable, match="unavailable") as caught:
        connect_framed_session(
            "tls://gateway.example:443",
            TOKEN,
            connector=boom,
        )
    assert TOKEN not in str(caught.value)


def test_device_audio_model_code_does_not_name_a_vendor() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "hardware_voice_ai" / "audio_model"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py")).lower()
    for name in ("openai", "whisper", "elevenlabs", "deepgram", "anthropic", "assemblyai"):
        assert name not in text
    names = {path.name for path in root.rglob("*.py")}
    assert names.isdisjoint(
        {"stt.py", "tts.py", "llm.py", "speech_to_text.py", "text_to_speech.py"}
    )


def test_a_truncated_frame_fails_instead_of_being_read_again() -> None:
    from hardware_voice_ai.audio_model.framing import FramingError

    session = FramedGatewaySession(BufferStream(b"\x00\x00"), token="")
    with pytest.raises(FramingError):
        session.receive(1)


def test_empty_hello_omits_authorization_when_there_is_no_token() -> None:
    stream = BufferStream()
    session = FramedGatewaySession(stream, token="")
    session.start_turn(sample_rate_hz=16000, channels=1)
    message = decode_text(_frames(bytes(stream.outgoing))[0][1])
    assert message == {"type": "hello", "audio_model": "remote"}
    assert "authorization" not in message
