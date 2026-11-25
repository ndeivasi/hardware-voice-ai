"""Moshi speech-to-speech adapter.

The published streaming loop is Mimi encode, ``LMGen.step``, then Mimi
decode. Weights default to ``kyutai/moshiko-pytorch-bf16`` (7B, CC-BY 4.0).
Text tokens stay inside the model. This module imports Moshi only when
``load_moshi`` runs.
"""

import contextlib
import threading
from collections.abc import Iterator

import numpy as np

from hardware_voice_ai.errors import RemoteBackendError
from hardware_voice_ai.gateway.audio import resample, to_mono
from hardware_voice_ai.gateway.backend import SpokenChunk
from hardware_voice_ai.gateway.gpu import ensure_gpu
from hardware_voice_ai.gateway.settings import GatewaySettings

MOSHI_RATE_HZ = 24_000
MOSHI_FRAME_RATE = 12.5


class MoshiStepper:
    """Feed 24 kHz mono frames through an injected Mimi and LMGen.

    ``mimi.encode`` / ``mimi.decode`` and ``lm_gen.step`` are the Kyutai
    calls. Tests pass fakes with those methods. ``to_frame`` turns one
    frame of float32 PCM into the tensor the loaded Mimi expects.
    """

    def __init__(self, mimi, lm_gen, *, to_frame=None, inference=None) -> None:
        self.mimi = mimi
        self.lm_gen = lm_gen
        self.sample_rate_hz = int(getattr(mimi, "sample_rate", MOSHI_RATE_HZ))
        frame_rate = float(getattr(mimi, "frame_rate", MOSHI_FRAME_RATE))
        self.frame_samples = max(1, int(round(self.sample_rate_hz / frame_rate)))
        self.to_frame = to_frame if to_frame is not None else _numpy_frame
        self.inference = inference if inference is not None else contextlib.nullcontext

    def run(self, pcm_mono: np.ndarray, cancel: threading.Event) -> Iterator[np.ndarray]:
        if cancel.is_set() or pcm_mono.size == 0:
            return
        frame = self.frame_samples
        padded = np.asarray(pcm_mono, dtype=np.float32).reshape(-1)
        remainder = len(padded) % frame
        if remainder:
            padded = np.pad(padded, (0, frame - remainder))
        contexts = []
        if hasattr(self.mimi, "streaming"):
            contexts.append(self.mimi.streaming(1))
        if hasattr(self.lm_gen, "streaming"):
            contexts.append(self.lm_gen.streaming(1))
        with contextlib.ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            with self.inference():
                for start in range(0, len(padded), frame):
                    if cancel.is_set():
                        return
                    codes = self.mimi.encode(self.to_frame(padded[start : start + frame]))
                    tokens = self.lm_gen.step(codes)
                    if tokens is None:
                        continue
                    audio_tokens = tokens[:, 1:]
                    yield _pcm_from_model(self.mimi.decode(audio_tokens))


class MoshiBackend:
    """``SpeechBackend`` for one loaded Moshi stepper."""

    def __init__(self, stepper: MoshiStepper) -> None:
        self._stepper = stepper

    def speak(self, utterance, sample_rate_hz, channels, cancel):
        del channels
        mono = resample(to_mono(utterance), sample_rate_hz, self._stepper.sample_rate_hz)
        for chunk in self._stepper.run(mono, cancel):
            if cancel.is_set() or chunk.size == 0:
                return
            yield SpokenChunk(chunk, self._stepper.sample_rate_hz)


def load_moshi(settings: GatewaySettings) -> MoshiBackend:
    """Load Moshi on the gateway machine. The device never calls this."""
    ensure_gpu(settings.device)
    try:
        from moshi.models import LMGen, loaders
    except ImportError as exc:
        raise RemoteBackendError(
            "Moshi is not installed on this host. Install the moshi extra "
            "on the GPU machine. The device does not download the weights."
        ) from exc
    import torch

    info = loaders.CheckpointInfo.from_hf_repo(settings.moshi_repo)
    mimi = info.get_mimi(device=settings.device)
    lm = info.get_moshi(device=settings.device, dtype=torch.bfloat16)
    gen_kwargs = {"temp": 0.8, "temp_text": 0.7}
    gen_kwargs.update(dict(getattr(info, "lm_gen_config", {}) or {}))
    lm_gen = LMGen(lm, **gen_kwargs)
    stepper = MoshiStepper(
        mimi,
        lm_gen,
        to_frame=lambda pcm: _torch_frame(pcm, settings.device),
        inference=torch.no_grad,
    )
    return MoshiBackend(stepper)


def _numpy_frame(pcm: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(pcm, dtype=np.float32).reshape(1, 1, -1)


def _torch_frame(pcm: np.ndarray, device: str):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))
    return tensor.view(1, 1, -1).to(device)


def _pcm_from_model(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    return np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1))
