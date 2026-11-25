# hardware-voice-ai

On-device wake-word detection for a hardware voice assistant. The approved design is [docs/architecture.md](docs/architecture.md). Wake-word assumptions, the microphone backend, and the capture handoff are in [docs/wake-word.md](docs/wake-word.md).

Always-on microphone audio stays in this process. An accepted wake moves the session from `idle` to `woken` to `capturing` and hands a short pre-roll to the capture port. The device has one Audio Model setting, `local` or `remote`, default `remote`. This package records that choice on the handoff. It does not run either audio model.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The tests synthesize audio and do not need a microphone, a model download, or a network connection.

## On a device

```bash
pip install -e '.[audio]'
python -m hardware_voice_ai
```

The live backend is PortAudio through `sounddevice`. Hardware is not chosen yet; the default is one channel at 16 kHz, and frames are shaped `(samples, channels)` so a later board can expose more microphones. `HVA_DETECTION_LOG` opts into a local JSON-lines log of detection metadata. Raw audio is not written.

`python -m hardware_voice_ai --help` lists the wake-model, wake-word, audio-model, and backend flags. Other knobs are `HVA_*` environment variables read by `DeviceConfig.from_env`.
