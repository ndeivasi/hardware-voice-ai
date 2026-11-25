"""Remote speech-to-speech adapters. No GPU, download, or live service."""

import contextlib
import shutil
import socket
import ssl
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from hardware_voice_ai.errors import ConfigurationError, RemoteBackendError
from hardware_voice_ai.gateway import (
    DEFAULT_REMOTE_BACKEND,
    LOCAL_STAND_IN_ID,
    REMOTE_BACKENDS,
    parse_remote_backend,
)
from hardware_voice_ai.gateway.backend import SpokenChunk
from hardware_voice_ai.gateway.framing import (
    AUDIO,
    TEXT,
    FrameReader,
    decode_audio,
    decode_text,
    encode_audio,
    encode_text,
)
from hardware_voice_ai.gateway.gpu import ensure_gpu
from hardware_voice_ai.gateway.loader import load_backend
from hardware_voice_ai.gateway.minicpm import MiniCPMSession, load_minicpm
from hardware_voice_ai.gateway.moshi import MoshiBackend, MoshiStepper, load_moshi
from hardware_voice_ai.gateway.qwen import QwenBackend, QwenSession, load_qwen, prepare_audio_inputs
from hardware_voice_ai.gateway.settings import GatewaySettings, settings_from_env
from hardware_voice_ai.gateway.tls import SocketStream, load_server_context
from hardware_voice_ai.gateway.turn import TurnGateway, serve_turn
from hardware_voice_ai.session.audio_model import AudioModel


class MemoryStream:
    def __init__(self, incoming: bytes = b"", read_size: int | None = None) -> None:
        self.incoming = bytearray(incoming)
        self.outgoing = bytearray()
        self.read_size = read_size
        self.closed = False

    def read(self, n: int) -> bytes:
        if not self.incoming:
            return b""
        take = min(n, len(self.incoming))
        if self.read_size is not None:
            take = min(take, self.read_size)
        out = bytes(self.incoming[:take])
        del self.incoming[:take]
        return out

    def write(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.outgoing.extend(data)

    def close(self) -> None:
        self.closed = True


class Scripted:
    def __init__(self, chunks: list[SpokenChunk]) -> None:
        self.chunks = chunks
        self.seen: list[tuple[np.ndarray, int, int]] = []

    def speak(self, utterance, sample_rate_hz, channels, cancel):
        self.seen.append((np.array(utterance, copy=True), sample_rate_hz, channels))
        if cancel.is_set():
            return
        yield from self.chunks


def _pcm(rows: int, channels: int = 1, value: float = 0.25) -> np.ndarray:
    return np.full((rows, channels), value, dtype=np.float32)


def _client(
    pcm: np.ndarray,
    *,
    token: str = "",
    backend: str | None = None,
    rate: int = 16_000,
    cancel: bool = False,
) -> bytes:
    hello: dict[str, object] = {"type": "hello", "audio_model": "remote"}
    if token:
        hello["authorization"] = token
    if backend is not None:
        hello["backend"] = backend
    parts = [
        encode_text(hello),
        encode_text({
            "type": "audio_start",
            "sample_rate_hz": rate,
            "channels": int(pcm.shape[1]),
        }),
        encode_audio(pcm),
    ]
    parts.append(encode_text({"type": "cancel" if cancel else "audio_end"}))
    return b"".join(parts)


def _frames(blob: bytes) -> list[tuple[int, bytes]]:
    reader = FrameReader(MemoryStream(blob))
    found = []
    while True:
        frame = reader.read_frame()
        if frame is None:
            return found
        found.append(frame)


def _texts(blob: bytes) -> list[dict]:
    return [decode_text(payload) for kind, payload in _frames(blob) if kind == TEXT]


def _audio(blob: bytes, channels: int) -> np.ndarray:
    pieces = [
        decode_audio(payload, channels)
        for kind, payload in _frames(blob)
        if kind == AUDIO
    ]
    if not pieces:
        return np.zeros((0, channels), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def test_remote_catalog_defaults_to_moshi_and_keeps_the_local_stand_in() -> None:
    assert DEFAULT_REMOTE_BACKEND == "moshi"
    assert REMOTE_BACKENDS == ("moshi", "minicpm-o-4.5", "qwen3-omni")
    assert LOCAL_STAND_IN_ID == "reference-audio-v1"
    assert parse_remote_backend("Moshi") == "moshi"
    assert parse_remote_backend("minicpm") == "minicpm-o-4.5"
    assert parse_remote_backend("qwen3-omni-instruct") == "qwen3-omni"
    assert GatewaySettings().backend_id == "moshi"
    assert {item.value for item in AudioModel} == {"local", "remote"}
    with pytest.raises(ConfigurationError, match="local Audio Model stand-in"):
        parse_remote_backend("reference-audio-v1")
    with pytest.raises(ConfigurationError, match="moshi, minicpm-o-4.5, or qwen3-omni"):
        parse_remote_backend("stt")


def test_settings_hide_the_token_and_reject_a_url_host() -> None:
    settings = settings_from_env(
        {
            "HVA_REMOTE_BACKEND": "qwen3-omni-instruct",
            "HVA_GATEWAY_TOKEN": "super-secret",
            "HVA_GATEWAY_HOST": "gpu.example",
            "HVA_GATEWAY_PORT": "9443",
        }
    )
    assert settings.backend_id == "qwen3-omni"
    assert settings.token == "super-secret"
    assert settings.host == "gpu.example"
    assert settings.port == 9443
    rendered = repr(settings)
    assert "super-secret" not in rendered
    assert "token=set" in rendered
    with pytest.raises(ConfigurationError, match="host"):
        GatewaySettings(host="tls://gpu.example")
    with pytest.raises(ConfigurationError, match="single line"):
        GatewaySettings(token="one\ntwo")


def test_text_and_audio_frames_match_the_device_session() -> None:
    hello = {"type": "hello", "audio_model": "remote", "authorization": "device-token"}
    frame = encode_text(hello)
    assert frame[4] == TEXT
    assert int.from_bytes(frame[:4], "big") == len(frame) - 5
    assert b'"type":"hello"' in frame
    assert decode_text(frame[5:]) == hello

    samples = np.array([[0.5], [-1.0]], dtype=np.float32)
    audio = encode_audio(samples)
    assert audio[4] == AUDIO
    decoded = decode_audio(audio[5:], 1)
    assert decoded.shape == (2, 1)
    assert np.allclose(decoded[:, 0], samples[:, 0], atol=1 / 32767)


def test_turn_uses_the_gateway_default_when_hello_omits_backend() -> None:
    reply = SpokenChunk(_pcm(1600), 16_000)
    backend = Scripted([reply])
    stream = MemoryStream(_client(_pcm(640)), read_size=3)
    serve_turn(stream, backend, GatewaySettings(backend_id="moshi"))
    assert len(backend.seen) == 1
    utterance, rate, channels = backend.seen[0]
    assert utterance.shape == (640, 1)
    assert rate == 16_000
    assert channels == 1
    messages = _texts(bytes(stream.outgoing))
    assert [item["type"] for item in messages] == ["reply_start", "end"]
    assert messages[0]["sample_rate_hz"] == 16_000
    assert messages[0]["channels"] == 1
    heard = _audio(bytes(stream.outgoing), 1)
    assert heard.shape == (1600, 1)
    assert np.allclose(heard, 0.25, atol=1 / 32767)
    assert all(item["type"] != "transcript" for item in messages)


def test_hello_backend_must_match_the_loaded_model() -> None:
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])
    stream = MemoryStream(_client(_pcm(320), backend="qwen3-omni"))
    serve_turn(stream, backend, GatewaySettings(backend_id="moshi"))
    assert backend.seen == []
    message = _texts(bytes(stream.outgoing))[0]
    assert message["type"] == "error"
    assert message["code"] == "unavailable"


def test_exact_device_hello_with_token_round_trips() -> None:
    token = "device-token"
    reply = SpokenChunk(np.full((2400, 1), 0.5, dtype=np.float32), 24_000)
    backend = Scripted([reply])
    pcm = _pcm(320, channels=2, value=0.4)
    stream = MemoryStream(_client(pcm, token=token))
    serve_turn(stream, backend, GatewaySettings(token=token))
    assert backend.seen[0][0].shape == (320, 2)
    start = _texts(bytes(stream.outgoing))[0]
    assert start["channels"] == 2
    assert start["sample_rate_hz"] == 16_000
    heard = _audio(bytes(stream.outgoing), 2)
    assert heard.shape[0] == 1600
    assert np.allclose(heard[:, 0], heard[:, 1], atol=1 / 32767)
    assert token.encode() not in bytes(stream.outgoing)


def test_bad_token_is_not_echoed() -> None:
    token = "super-secret-token"
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])
    stream = MemoryStream(_client(_pcm(320), token="nope"))
    serve_turn(stream, backend, GatewaySettings(token=token))
    assert backend.seen == []
    raw = bytes(stream.outgoing)
    assert token.encode() not in raw
    assert b"nope" not in raw
    assert _texts(raw)[0]["code"] == "unavailable"


def test_cancel_before_audio_end_does_not_run_the_model() -> None:
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])
    stream = MemoryStream(_client(_pcm(320), cancel=True))
    serve_turn(stream, backend, GatewaySettings())
    assert backend.seen == []
    assert stream.outgoing == b""


def test_empty_utterance_is_low_confidence_audio_not_a_transcript() -> None:
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])
    hello = encode_text({"type": "hello", "audio_model": "remote"})
    start = encode_text({"type": "audio_start", "sample_rate_hz": 16000, "channels": 1})
    end = encode_text({"type": "audio_end"})
    stream = MemoryStream(hello + start + end)
    serve_turn(stream, backend, GatewaySettings())
    assert backend.seen == []
    message = _texts(bytes(stream.outgoing))[0]
    assert message == {
        "type": "error",
        "code": "low_confidence",
        "message": "audio was hard to hear",
    }


def test_overlong_utterance_is_refused() -> None:
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])
    stream = MemoryStream(_client(_pcm(640)))
    serve_turn(stream, backend, GatewaySettings(), max_utterance_s=0.02)
    assert backend.seen == []
    assert _texts(bytes(stream.outgoing))[0]["code"] == "failed"


def test_model_failure_does_not_leak_internals() -> None:
    class Boom:
        def speak(self, utterance, sample_rate_hz, channels, cancel):
            del utterance, sample_rate_hz, channels, cancel
            raise RuntimeError("super-secret-token")

    stream = MemoryStream(_client(_pcm(320)))
    serve_turn(stream, Boom(), GatewaySettings())
    raw = bytes(stream.outgoing)
    assert b"super-secret-token" not in raw
    assert _texts(raw)[0]["message"] == "audio model failed"


def test_a_second_connection_is_refused_while_a_turn_holds_the_model() -> None:
    gateway = TurnGateway(GatewaySettings(), Scripted([SpokenChunk(_pcm(320), 16_000)]))
    assert gateway._lock.acquire(blocking=False)
    stream = MemoryStream(_client(_pcm(320)))
    try:
        gateway.handle(stream)
    finally:
        gateway._lock.release()
    assert _texts(bytes(stream.outgoing))[0]["code"] == "unavailable"


def test_one_stream_stops_after_a_single_turn() -> None:
    first = _client(_pcm(320))
    second = _client(_pcm(320, value=0.1))
    stream = MemoryStream(first + second)
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])
    serve_turn(stream, backend, GatewaySettings())
    assert len(backend.seen) == 1
    assert stream.incoming[:5] == second[:5]


def test_moshi_stepper_calls_encode_step_and_decode() -> None:
    class Mimi:
        sample_rate = 24_000
        frame_rate = 12.5

        def __init__(self) -> None:
            self.encoded = []
            self.batches = []

        def streaming(self, batch: int):
            self.batches.append(batch)
            return contextlib.nullcontext()

        def encode(self, frame: np.ndarray) -> np.ndarray:
            assert frame.shape == (1, 1, 1920)
            self.encoded.append(frame.copy())
            return frame

        def decode(self, tokens: object) -> np.ndarray:
            assert tokens == "audio-tokens"
            return np.full((1, 1, 1920), 0.2, dtype=np.float32)

    class Tokens:
        def __getitem__(self, item: object) -> str:
            assert item == (slice(None), slice(1, None))
            return "audio-tokens"

    class LM:
        def __init__(self) -> None:
            self.steps = 0

        def streaming(self, batch: int):
            assert batch == 1
            return contextlib.nullcontext()

        def step(self, codes: np.ndarray):
            assert codes.shape == (1, 1, 1920)
            self.steps += 1
            if self.steps == 1:
                return None
            return Tokens()

    mimi = Mimi()
    lm = LM()
    stepper = MoshiStepper(mimi, lm)
    heard = list(stepper.run(np.ones(3840, dtype=np.float32), threading.Event()))
    assert lm.steps == 2
    assert mimi.batches == [1]
    assert len(heard) == 1
    assert heard[0].shape == (1920,)
    backend = MoshiBackend(stepper)
    chunks = list(
        backend.speak(np.ones((1600, 1), dtype=np.float32), 16_000, 1, threading.Event())
    )
    assert chunks[0].sample_rate_hz == 24_000
    assert chunks[0].samples.ndim == 2


def test_moshi_stepper_stops_when_cancelled() -> None:
    class Mimi:
        sample_rate = 24_000
        frame_rate = 12.5

        def streaming(self, batch: int):
            del batch
            return contextlib.nullcontext()

        def encode(self, frame: np.ndarray) -> np.ndarray:
            return frame

        def decode(self, tokens: object) -> np.ndarray:
            del tokens
            return np.ones((1, 1, 1920), dtype=np.float32)

    class Tokens:
        def __getitem__(self, item: object) -> str:
            del item
            return "audio-tokens"

    cancel = threading.Event()

    class LM:
        def streaming(self, batch: int):
            del batch
            return contextlib.nullcontext()

        def step(self, codes: np.ndarray):
            del codes
            cancel.set()
            return Tokens()

    heard = list(MoshiStepper(Mimi(), LM()).run(np.ones(3840, dtype=np.float32), cancel))
    assert len(heard) == 1


def test_minicpm_session_prefills_audio_and_drops_text() -> None:
    class Model:
        def __init__(self) -> None:
            self.calls = []

        def reset_session(self, **kwargs) -> None:
            self.calls.append(("reset", kwargs["reset_token2wav_cache"]))

        def init_token2wav_cache(self, **kwargs) -> None:
            self.calls.append(kwargs["prompt_speech_16k"].shape)

        def streaming_prefill(self, **kwargs) -> None:
            self.calls.append(("prefill", kwargs["omni_mode"], kwargs["is_last_chunk"], kwargs["msgs"]))

        def streaming_generate(self, **kwargs):
            self.calls.append(kwargs["generate_audio"])
            assert kwargs["use_tts_template"] is True
            yield (np.ones(2400, dtype=np.float32), "spoken inside the model")

    model = Model()
    ref = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
    session = MiniCPMSession(model, ref)
    heard = list(session.run(np.ones(100, dtype=np.float32), threading.Event()))
    assert heard[0].shape == (2400,)
    assert model.calls[0] == ("reset", True)
    assert model.calls[1] == (160,)
    system = model.calls[2]
    assert system[1] is False
    assert system[3][0]["role"] == "system"
    user = model.calls[3]
    assert user[2] is True
    assert user[3][0]["content"][0].shape == (16_000,)
    assert model.calls[4] is True
    assert b"spoken inside the model" not in heard[0].tobytes()


def test_qwen_session_generates_audio_from_memory() -> None:
    class Processor:
        def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
            self.conversation = conversation
            assert add_generation_prompt is True
            assert tokenize is False
            return "turn"

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            box = {"input_ids": np.array([[1]], dtype=np.int64)}

            class Inputs(dict):
                def to(self, *args, **kwargs):
                    del args, kwargs
                    return self

            return Inputs(box)

    class Model:
        device = "cpu"
        dtype = np.float32

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return ({"sequences": None}, np.full(2400, 0.5, dtype=np.float32))

    processor = Processor()
    model = Model()
    waveform = np.ones(1600, dtype=np.float32)
    session = QwenSession(model, processor, speaker="Ethan")
    heard = session.run(waveform, threading.Event())
    assert heard is not None
    assert heard.shape == (2400,)
    audio = processor.conversation[0]["content"][0]["audio"]
    assert isinstance(audio, np.ndarray)
    assert processor.kwargs["audio"][0] is audio
    assert processor.kwargs["images"] is None
    assert model.kwargs["speaker"] == "Ethan"
    assert model.kwargs["use_audio_in_video"] is False
    chunks = list(QwenBackend(session).speak(waveform.reshape(-1, 1), 16_000, 1, threading.Event()))
    assert chunks[0].sample_rate_hz == 24_000
    with pytest.raises(RemoteBackendError, match="in-memory"):
        prepare_audio_inputs([{
            "role": "user",
            "content": [{"type": "audio", "audio": "https://example.invalid/utterance.wav"}],
        }])


def test_loaders_fail_closed_without_vendor_libraries(monkeypatch) -> None:
    real = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        roots = ("moshi", "transformers", "torch", "librosa")
        if name in roots or name.startswith(tuple(root + "." for root in roots)):
            raise ImportError("absent")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked)
    moshi = GatewaySettings(backend_id="moshi", device="cpu")
    minicpm = GatewaySettings(
        backend_id="minicpm-o-4.5",
        device="cpu",
        minicpm_ref_audio="clip.wav",
    )
    qwen = GatewaySettings(backend_id="qwen3-omni", device="cpu")
    with pytest.raises(RemoteBackendError, match="Moshi is not installed"):
        load_moshi(moshi)
    with pytest.raises(RemoteBackendError, match="MiniCPM-o is not installed"):
        load_minicpm(minicpm)
    with pytest.raises(RemoteBackendError, match="Qwen3-Omni is not installed"):
        load_qwen(qwen)
    with pytest.raises(RemoteBackendError, match="not installed"):
        load_backend(moshi)


def test_minicpm_ref_clip_is_required_and_not_invented() -> None:
    with pytest.raises(RemoteBackendError, match="HVA_MINICPM_REF_AUDIO"):
        load_minicpm(GatewaySettings(backend_id="minicpm-o-4.5", device="cpu"))
    with pytest.raises(RemoteBackendError, match="does not include one"):
        MiniCPMSession(object(), np.zeros(0, dtype=np.float32))


def test_cuda_is_required_unless_the_operator_overrides(monkeypatch) -> None:
    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Torch:
        cuda = Cuda()

    monkeypatch.setitem(sys.modules, "torch", Torch())
    with pytest.raises(RemoteBackendError, match="Pi-class"):
        ensure_gpu("cuda")
    ensure_gpu("cpu")


def test_listener_does_not_invent_credentials(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not generate"):
        load_server_context("", "")
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("not a certificate", encoding="utf-8")
    key.write_text("not a key", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="could not be loaded"):
        load_server_context(str(cert), str(key))


def test_cli_selects_a_backend_without_loading_weights(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HVA_GATEWAY_TOKEN", "super-secret")
    seen = {}

    def serve(settings, backend) -> None:
        seen["settings"] = settings
        seen["backend"] = backend

    def load(settings):
        assert settings.backend_id == "minicpm-o-4.5"
        return "loaded-stand-in"

    from hardware_voice_ai.gateway.__main__ import main

    code = main(
        ["--backend", "minicpm-o-4.5", "--host", "127.0.0.1", "--port", "8443"],
        serve=serve,
        load=load,
    )
    assert code == 0
    assert seen["backend"] == "loaded-stand-in"
    assert "super-secret" not in capsys.readouterr().out
    monkeypatch.setenv("HVA_REMOTE_BACKEND", "reference-audio-v1")
    assert main([], serve=serve, load=load) == 2


def test_one_tls_connection_carries_one_turn(tmp_path: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required to mint a throwaway listener certificate")
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-days", "1", "-nodes", "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    context = load_server_context(str(cert), str(key))
    assert context.minimum_version is ssl.TLSVersion.TLSv1_2
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.bind(("127.0.0.1", 0))
    raw.listen(1)
    raw.settimeout(5)
    port = raw.getsockname()[1]
    settings = GatewaySettings(token="turn-token", host="127.0.0.1", port=port)
    backend = Scripted([SpokenChunk(_pcm(320), 16_000)])

    def accept_one() -> None:
        try:
            conn, _addr = raw.accept()
        except OSError:
            return
        try:
            tls_conn = context.wrap_socket(conn, server_side=True)
        except ssl.SSLError:
            conn.close()
            return
        try:
            TurnGateway(settings, backend).handle(SocketStream(tls_conn))
        finally:
            tls_conn.close()

    thread = threading.Thread(target=accept_one, daemon=True)
    thread.start()
    client_ctx = ssl.create_default_context()
    client_ctx.load_verify_locations(str(cert))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            with client_ctx.wrap_socket(sock, server_hostname="localhost") as tls:
                tls.sendall(_client(_pcm(320), token="turn-token"))
                received = bytearray()
                while True:
                    part = tls.recv(4096)
                    if not part:
                        break
                    received.extend(part)
    finally:
        raw.close()
        thread.join(timeout=5)
    assert len(backend.seen) == 1
    messages = _texts(bytes(received))
    assert [item["type"] for item in messages] == ["reply_start", "end"]
    assert b"turn-token" not in bytes(received)
