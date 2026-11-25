# hardware-voice-ai

On-device wake-word detection, microphone capture, and speaker playback for a hardware voice assistant. The approved design is [docs/architecture.md](docs/architecture.md). Wake-word assumptions are in [docs/wake-word.md](docs/wake-word.md). Microphone capture after an accepted wake is in [docs/capture.md](docs/capture.md). Speaker playback is in [docs/playback.md](docs/playback.md).

Always-on microphone audio stays in this process. An accepted wake moves the session from `idle` to `woken` to `capturing` and hands a short pre-roll to microphone capture. The device has one Audio Model setting, `local` or `remote`, default `remote`. Capture keeps that utterance for the Audio Model. This package does not run either model. Spoken-reply audio and local wake, offline, and error cues play on the device speaker in both modes.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The tests synthesize audio and do not need a microphone, a speaker, a model download, or a network connection.

## On a device

```bash
pip install -e '.[audio]'
python -m hardware_voice_ai
```

The live backend is PortAudio through `sounddevice`, for both the microphone and the speaker. Hardware is not chosen yet; the default is one channel at 16 kHz, and frames are shaped `(samples, channels)`. `HVA_VOLUME` sets speaker gain from 0 to 1 (default 1). `HVA_DETECTION_LOG` opts into a local JSON-lines log of detection metadata. Raw audio is not written.

`python -m hardware_voice_ai --help` lists the wake-model, wake-word, audio-model, volume, and backend flags. Other knobs are `HVA_*` environment variables read by `DeviceConfig.from_env`.
