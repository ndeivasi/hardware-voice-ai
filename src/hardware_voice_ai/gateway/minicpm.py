"""MiniCPM-o 4.5 speech-to-speech adapter.

The published half-duplex path is ``streaming_prefill`` on the utterance,
then ``streaming_generate`` with ``generate_audio=True``. That method is
inside the one model. This gateway does not add a text-to-speech stage
after it. Weights are ``openbmb/MiniCPM-o-4_5`` (Apache-2.0, 9B).

The model card's conversational path needs a reference clip for the voice.
The operator sets ``HVA_MINICPM_REF_AUDIO`` on the GPU host. This repo does
not ship that clip.
"""

import threading

import numpy as np

from hardware_voice_ai.errors import RemoteBackendError
from hardware_voice_ai.gateway.audio import resample, to_mono
from hardware_voice_ai.gateway.backend import SpokenChunk
from hardware_voice_ai.gateway.gpu import ensure_gpu
from hardware_voice_ai.gateway.settings import GatewaySettings

MINICPM_IN_HZ = 16_000
MINICPM_OUT_HZ = 24_000
_CHUNK = 16_000


class MiniCPMSession:
    """One half-duplex turn against an object with the model-card methods."""

    def __init__(self, model, ref_audio: np.ndarray) -> None:
        self.model = model
        self.ref_audio = np.asarray(ref_audio, dtype=np.float32).reshape(-1)
        if self.ref_audio.size == 0:
            raise RemoteBackendError(
                "MiniCPM-o needs a reference clip. Set HVA_MINICPM_REF_AUDIO "
                "on the GPU host to a local file. This repo does not include one."
            )

    def run(self, pcm_16k: np.ndarray, cancel: threading.Event):
        if cancel.is_set():
            return
        model = self.model
        model.reset_session(reset_token2wav_cache=True)
        model.init_token2wav_cache(prompt_speech_16k=self.ref_audio)
        session_id = "turn"
        model.streaming_prefill(
            session_id=session_id,
            msgs=[{
                "role": "system",
                "content": [
                    "Clone the voice in the provided audio prompt.",
                    self.ref_audio,
                    "Please assist the user with a short spoken reply.",
                ],
            }],
            omni_mode=False,
            is_last_chunk=True,
        )
        audio = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        count = int(np.ceil(audio.size / _CHUNK))
        for index in range(count):
            if cancel.is_set():
                return
            piece = audio[index * _CHUNK : (index + 1) * _CHUNK]
            is_last = index == count - 1
            if is_last and piece.size < _CHUNK:
                piece = np.pad(piece, (0, _CHUNK - piece.size))
            model.streaming_prefill(
                session_id=session_id,
                msgs=[{"role": "user", "content": [piece]}],
                omni_mode=False,
                is_last_chunk=is_last,
            )
        generated = model.streaming_generate(
            session_id=session_id,
            generate_audio=True,
            use_tts_template=True,
            enable_thinking=False,
            do_sample=True,
            max_new_tokens=512,
            length_penalty=1.1,
        )
        for item in generated:
            if cancel.is_set():
                return
            wav = item[0] if isinstance(item, tuple) else item
            pcm = _pcm(wav)
            if pcm.size:
                yield pcm


class MiniCPMBackend:
    def __init__(self, session: MiniCPMSession) -> None:
        self._session = session

    def speak(self, utterance, sample_rate_hz, channels, cancel):
        del channels
        mono = resample(to_mono(utterance), sample_rate_hz, MINICPM_IN_HZ)
        for chunk in self._session.run(mono, cancel):
            yield SpokenChunk(chunk, MINICPM_OUT_HZ)


def load_minicpm(settings: GatewaySettings) -> MiniCPMBackend:
    """Load MiniCPM-o on the gateway machine. The device never calls this."""
    if not settings.minicpm_ref_audio:
        raise RemoteBackendError(
            "MiniCPM-o needs a reference clip. Set HVA_MINICPM_REF_AUDIO "
            "on the GPU host to a local file. This repo does not include one."
        )
    ensure_gpu(settings.device)
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RemoteBackendError(
            "MiniCPM-o is not installed on this host. Install the minicpm extra "
            "on the GPU machine. The device does not download the weights."
        ) from exc
    import torch

    model = AutoModel.from_pretrained(
        settings.minicpm_model,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=True,
    )
    model.eval()
    if settings.device != "cpu":
        model = model.to(settings.device)
    model.init_tts()
    ref = _load_ref(settings.minicpm_ref_audio)
    return MiniCPMBackend(MiniCPMSession(model, ref))


def _load_ref(path: str) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RemoteBackendError(
            "MiniCPM-o needs librosa on the GPU host to read HVA_MINICPM_REF_AUDIO."
        ) from exc
    audio, _rate = librosa.load(path, sr=MINICPM_IN_HZ, mono=True)
    return np.asarray(audio, dtype=np.float32)


def _pcm(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    return np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
