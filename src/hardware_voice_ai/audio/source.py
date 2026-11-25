"""Microphone backends.

``PcmAudioSource`` feeds tests and other in-process audio. ``SoundDeviceAudioSource``
is the live PortAudio backend for a Linux board. Both yield the same
``AudioFrame`` shape, with one column per microphone.
"""

from typing import Protocol

import numpy as np

from hardware_voice_ai.audio.frame import AudioFormat, AudioFrame
from hardware_voice_ai.errors import AudioBackendError, ConfigurationError


class AudioSource(Protocol):
    """A single reader of the live microphone (or a stand-in).

    Live backends block inside ``read`` until a frame is available. Quiet
    rooms still produce frames. ``None`` means a finite source has ended.
    """

    format: AudioFormat

    def read(self) -> AudioFrame | None: ...

    def close(self) -> None: ...


class PcmAudioSource:
    """Finite float32 PCM. Mono input may be one-dimensional."""

    def __init__(self, samples: np.ndarray, fmt: AudioFormat) -> None:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[1] != fmt.channels:
            raise ValueError(
                f"PCM shape {data.shape} does not match {fmt.channels} channels"
            )
        self._samples = data
        self._pos = 0
        self.format = fmt

    def read(self) -> AudioFrame | None:
        end = self._pos + self.format.frame_samples
        if end > len(self._samples):
            return None
        chunk = self._samples[self._pos : end]
        index = self._pos
        self._pos = end
        return AudioFrame(samples=chunk, format=self.format, sample_index=index)

    def close(self) -> None:
        return None


class SoundDeviceAudioSource:
    """Live capture through ``sounddevice`` (PortAudio).

    The import is deferred so tests and the reference model run without the
    audio extra and without a microphone. Samples are converted to float32
    and kept in memory.
    """

    def __init__(self, fmt: AudioFormat, device: str | int | None = None) -> None:
        sd = _import_sounddevice()
        self.format = fmt
        self.overflows = 0
        self._sample_index = 0
        self._stream = sd.InputStream(
            samplerate=fmt.sample_rate_hz,
            channels=fmt.channels,
            dtype="int16",
            blocksize=fmt.frame_samples,
            device=device,
        )
        try:
            self._stream.start()
        except Exception as exc:
            self._stream.close()
            raise AudioBackendError(f"Could not open the microphone: {exc}") from exc

    def read(self) -> AudioFrame:
        data, overflowed = self._stream.read(self.format.frame_samples)
        if overflowed:
            self.overflows += 1
        samples = np.asarray(data, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        samples = samples / np.float32(32768.0)
        frame = AudioFrame(
            samples=samples,
            format=self.format,
            sample_index=self._sample_index,
        )
        self._sample_index += self.format.frame_samples
        return frame

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def build_audio_source(
    fmt: AudioFormat,
    backend: str,
    *,
    samples: np.ndarray | None = None,
    device: str | int | None = None,
) -> AudioSource:
    if backend == "pcm":
        if samples is None:
            raise ConfigurationError("The pcm audio backend requires samples.")
        return PcmAudioSource(samples, fmt)
    if backend == "sounddevice":
        return SoundDeviceAudioSource(fmt, device=device)
    raise ConfigurationError(
        f"Unknown audio backend {backend!r}. Use 'sounddevice' or 'pcm'."
    )


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioBackendError(
            "The sounddevice audio backend is not installed. "
            "Install the 'audio' extra to open a microphone."
        ) from exc
    return sd
