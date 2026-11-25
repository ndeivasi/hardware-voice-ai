"""Utterance audio handed from capture into the one Audio Model.

Field names match the capture handoff: ``wake``, ``audio_model``, ``samples``,
``sample_rate_hz``, ``pre_roll_sample_count``, ``wake_word_end_offset``, and
``endpoint``. A capture implementation can pass its own object here; this
module does not read the microphone and does not decide that speech has ended.
"""

from dataclasses import dataclass

import numpy as np

from hardware_voice_ai.audio.frame import AudioSnapshot
from hardware_voice_ai.audio_model.defaults import MAX_UTTERANCE_S
from hardware_voice_ai.capture.port import CaptureRequest
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.events import WakeEvent


@dataclass(frozen=True, slots=True)
class CapturedUtterance:
    """One post-wake utterance, kept in memory only.

    ``samples`` is float32 PCM shaped ``(samples, channels)`` in about
    ``[-1, 1]``. The leading ``pre_roll_sample_count`` rows are the wake
    snapshot. ``wake_word_end_offset`` is the row where that wake word ends.
    """

    wake: WakeEvent
    audio_model: AudioModel
    samples: np.ndarray
    sample_rate_hz: int
    pre_roll_sample_count: int
    wake_word_end_offset: int
    endpoint: str

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        if samples.ndim != 2:
            raise ValueError("utterance samples must have shape (samples, channels)")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])


def utterance_from_request(
    request: CaptureRequest,
    continuation: np.ndarray | None = None,
    *,
    endpoint: str = "handed_off",
) -> CapturedUtterance:
    """Narrow adapter from a capture request to utterance audio.

    ``continuation`` is the audio the caller kept after the pre-roll. This
    does not look for silence, does not open the microphone, and does not
    keep any always-on audio beyond that pre-roll and continuation.
    """
    pre_roll = _as_matrix(request.pre_roll)
    if continuation is None:
        samples = pre_roll
    else:
        extra = np.array(continuation, dtype=np.float32, copy=True)
        if extra.ndim == 1:
            extra = extra.reshape(-1, pre_roll.shape[1])
        if extra.ndim != 2 or extra.shape[1] != pre_roll.shape[1]:
            raise ValueError("continuation must match the pre-roll channel count")
        samples = np.concatenate((pre_roll, extra), axis=0)
    wake = request.wake
    offset = wake.word_end_sample_index - request.pre_roll.start_sample_index
    return CapturedUtterance(
        wake=wake,
        audio_model=request.audio_model,
        samples=samples,
        sample_rate_hz=request.pre_roll.sample_rate_hz,
        pre_roll_sample_count=int(pre_roll.shape[0]),
        wake_word_end_offset=int(offset),
        endpoint=endpoint,
    )


def utterance_problem(utterance: object) -> str | None:
    """Return a stable reason to refuse this utterance, or ``None`` if it can run."""
    samples = getattr(utterance, "samples", None)
    if samples is None:
        return "utterance audio is missing"
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        return "utterance audio is empty"
    if not np.isfinite(array).all():
        return "utterance audio is not finite"
    try:
        rate = int(getattr(utterance, "sample_rate_hz"))
    except (TypeError, ValueError):
        return "utterance sample rate is invalid"
    if rate <= 0:
        return "utterance sample rate is invalid"
    if array.shape[0] > rate * MAX_UTTERANCE_S:
        return "utterance is longer than the model will accept"
    model = getattr(utterance, "audio_model", None)
    if model not in (AudioModel.LOCAL, AudioModel.REMOTE):
        return "utterance audio model must be local or remote"
    return None


def _as_matrix(snapshot: AudioSnapshot) -> np.ndarray:
    samples = np.array(snapshot.samples, dtype=np.float32, copy=True)
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("pre-roll must contain at least one sample and one channel")
    return samples
