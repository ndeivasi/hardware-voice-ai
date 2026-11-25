"""Qwen3-Omni Instruct speech-to-speech adapter.

The published Transformers call is ``Qwen3OmniMoeForConditionalGeneration``
with ``Qwen3OmniMoeProcessor`` and ``model.generate``. The model id is
``Qwen/Qwen3-Omni-30B-A3B-Instruct`` (Apache-2.0). Spoken audio comes back
from that one generate call. This gateway does not write the utterance to
disk and does not pass a URL to the processor.

``qwen_omni_utils.process_mm_info`` loads paths and URLs. The utterance is
already PCM, so the processor receives that waveform directly. An operator
can replace ``prepare`` on ``QwenSession`` if a different wiring is required.
"""

import threading

import numpy as np

from hardware_voice_ai.errors import RemoteBackendError
from hardware_voice_ai.gateway.audio import resample, to_mono
from hardware_voice_ai.gateway.backend import SpokenChunk
from hardware_voice_ai.gateway.gpu import ensure_gpu
from hardware_voice_ai.gateway.settings import GatewaySettings

QWEN_IN_HZ = 16_000
QWEN_OUT_HZ = 24_000


class QwenSession:
    """One audio-only turn against the published generate call."""

    def __init__(self, model, processor, *, speaker: str = "Ethan", prepare=None) -> None:
        self.model = model
        self.processor = processor
        self.speaker = speaker
        self.prepare = prepare if prepare is not None else prepare_audio_inputs

    def run(self, pcm_16k: np.ndarray, cancel: threading.Event) -> np.ndarray | None:
        if cancel.is_set():
            return None
        waveform = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            return None
        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "audio": waveform}],
        }]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios = self.prepare(conversation)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        _text_ids, audio = self.model.generate(
            **inputs,
            speaker=self.speaker,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=False,
        )
        del _text_ids
        if audio is None:
            return None
        return _pcm(audio)


class QwenBackend:
    def __init__(self, session: QwenSession) -> None:
        self._session = session

    def speak(self, utterance, sample_rate_hz, channels, cancel):
        del channels
        mono = resample(to_mono(utterance), sample_rate_hz, QWEN_IN_HZ)
        audio = self._session.run(mono, cancel)
        if audio is None or audio.size == 0 or cancel.is_set():
            return
        yield SpokenChunk(audio, QWEN_OUT_HZ)


def prepare_audio_inputs(conversation: list) -> list[np.ndarray]:
    """Pull in-memory waveforms out of the chat content.

    A path or URL is refused so the utterance is not read back from disk
    and is not fetched.
    """
    found: list[np.ndarray] = []
    for message in conversation:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "audio":
                continue
            audio = part.get("audio")
            if not isinstance(audio, np.ndarray):
                raise RemoteBackendError(
                    "Qwen3-Omni received audio that is not an in-memory waveform."
                )
            found.append(np.asarray(audio, dtype=np.float32))
    if not found:
        raise RemoteBackendError("Qwen3-Omni received no audio.")
    return found


def load_qwen(settings: GatewaySettings) -> QwenBackend:
    """Load Qwen3-Omni Instruct on the gateway machine."""
    ensure_gpu(settings.device)
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    except ImportError as exc:
        raise RemoteBackendError(
            "Qwen3-Omni is not installed on this host. Install the qwen extra "
            "on the GPU machine. The device does not download the weights."
        ) from exc
    # "auto" matches the model card when the operator asked for a GPU.
    # A specific cuda:N index is passed through. flash_attention_2 is not
    # required; the installed Transformers default stays swappable.
    device_map = "auto" if settings.device == "cuda" else settings.device
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        settings.qwen_model,
        dtype="auto",
        device_map=device_map,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(settings.qwen_model)
    return QwenBackend(QwenSession(model, processor, speaker=settings.qwen_speaker))


def _pcm(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    return np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
