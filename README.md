# hardware-voice-ai

On-device wake-word detection, microphone capture, speaker playback, and one Audio Model voice loop. The approved design is [docs/architecture.md](docs/architecture.md). Wake-word assumptions are in [docs/wake-word.md](docs/wake-word.md). Microphone capture after an accepted wake is in [docs/capture.md](docs/capture.md). Speaker playback is in [docs/playback.md](docs/playback.md). The voice loop is in [docs/audio-model.md](docs/audio-model.md).

Always-on microphone audio stays in this process. An accepted wake moves the session from `idle` to `woken` to `capturing` and hands a short pre-roll to microphone capture. The device has one Audio Model setting, `local` or `remote`, default `remote`. After capture ends the utterance, that one model takes the audio and produces the spoken reply. Playback, volume, and local wake, offline, and error cues stay on the device speaker in both modes. Speech-to-text and text-to-speech are not separate stages.

The remote adapters are Moshi (default), MiniCPM-o 4.5, and Qwen3-Omni Instruct. They run in a separate gateway process on a GPU host. The local stand-in stays `reference-audio-v1`. See [docs/remote-backends.md](docs/remote-backends.md).

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The tests synthesize audio and do not need a microphone, a speaker, a model download, or a live network. One gateway test listens on localhost with a certificate it creates for that run and deletes with the temp directory.

## On a device

```bash
pip install -e '.[audio]'
python -m hardware_voice_ai
```

The live backend is PortAudio through `sounddevice`, for both the microphone and the speaker. Hardware is not chosen yet; the default is one channel at 16 kHz, and frames are shaped `(samples, channels)`. `HVA_VOLUME` sets speaker gain from 0 to 1 (default 1). `HVA_DETECTION_LOG` opts into a local JSON-lines log of detection metadata. Raw audio is not written.

`python -m hardware_voice_ai --help` lists the wake-model, wake-word, audio-model, volume, and backend flags. Other knobs are `HVA_*` environment variables read by `DeviceConfig.from_env`.

## Remote gateway

On a GPU host, not on the Pi:

```bash
pip install -e '.[moshi]'
python -m hardware_voice_ai.gateway --backend moshi
```

`--backend` is `moshi` (default), `minicpm-o-4.5`, or `qwen3-omni`. `HVA_GATEWAY_TOKEN` is required. This process does not dial out from the device; `DeviceConfig` has no gateway URL. A later device client should use `tls://<gpu-host>:8443` and the same token. Certificates are the operator's. This repo does not create them. Details are in [docs/remote-backends.md](docs/remote-backends.md).
