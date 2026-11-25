"""Optional openWakeWord adapter.

The package is not a dependency of the default path. Construct this with a
runner that implements ``predict(int16_mono) -> dict[str, float]`` — the
openWakeWord ``Model`` does. Tests pass a fake runner so the score mapping
does not need the library or a microphone.
"""

import numpy as np

from hardware_voice_ai.audio.frame import AudioFrame
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.wake.defaults import OPENWAKEWORD_MODEL_ID
from hardware_voice_ai.wake.model import WakeScore

# openWakeWord scores 80 ms of 16 kHz mono audio per predict() call.
OWW_CHUNK_SAMPLES = 1280
OWW_SAMPLE_RATE_HZ = 16_000


class OpenWakeWordModel:
    """Map a pretrained openWakeWord posterior onto ``WakeScore``.

    Neural posteriors are often a single hot frame. Pair this model with a
    ``min_positive_ms`` of one frame (20 ms at the default hop). The reference
    model's longer minimum is for its multi-step pattern, not for this adapter.

    Channels are not mixed by averaging. Each chunk uses the microphone with
    the highest energy so a quiet channel cannot cancel a louder one.
    """

    def __init__(self, wake_word: str, runner: object, *, channels: int, sample_rate_hz: int) -> None:
        if sample_rate_hz != OWW_SAMPLE_RATE_HZ:
            raise ConfigurationError("openWakeWord requires 16000 Hz audio.")
        if channels < 1:
            raise ValueError("channels must be at least 1")
        if not wake_word.strip():
            raise ValueError("wake_word must not be empty")
        self.model_id = OPENWAKEWORD_MODEL_ID
        self.wake_word = wake_word
        self._runner = runner
        self._channels = channels
        self._pending = np.zeros((0, channels), dtype=np.float32)

    def reset(self) -> None:
        self._pending = np.zeros((0, self._channels), dtype=np.float32)
        reset = getattr(self._runner, "reset", None)
        if callable(reset):
            reset()

    def score(self, frame: AudioFrame) -> WakeScore:
        if frame.format.channels != self._channels:
            raise ValueError(
                f"expected {self._channels} channels, got {frame.format.channels}"
            )
        if frame.format.sample_rate_hz != OWW_SAMPLE_RATE_HZ:
            raise ValueError("openWakeWord requires 16000 Hz audio.")
        self._pending = np.concatenate((self._pending, frame.samples), axis=0)
        best: WakeScore | None = None
        while self._pending.shape[0] >= OWW_CHUNK_SAMPLES:
            chunk = self._pending[:OWW_CHUNK_SAMPLES]
            self._pending = self._pending[OWW_CHUNK_SAMPLES:]
            best = self._score_chunk(chunk)
        if best is None:
            return WakeScore(confidence=0.0, channel=0)
        return best

    def _score_chunk(self, chunk: np.ndarray) -> WakeScore:
        energy = np.mean(chunk * chunk, axis=0)
        channel = int(np.argmax(energy))
        pcm = np.clip(chunk[:, channel] * 32768.0, -32768, 32767).astype(np.int16)
        predict = getattr(self._runner, "predict")
        prediction = predict(pcm)
        try:
            confidence = float(prediction[self.wake_word])
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(
                f"openWakeWord runner did not return a score for {self.wake_word!r}."
            ) from exc
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        return WakeScore(confidence=confidence, channel=channel)
