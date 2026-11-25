# Microphone capture

This is the on-device capture path for an accepted wake. It implements the `CapturePort` left by wake-word detection: buffer the utterance, include the pre-roll, stop on end-of-utterance or cancel or hardware mute, and hand the audio to the session orchestrator for the Audio Model.

The Audio Model is one setting, `local` or `remote`, default `remote`. `local` means on-device. `remote` means a cloud service. That one model receives this utterance and produces the spoken reply. Capture does not run the model. There is no separate speech-to-text stage and no text-to-speech stage.

Wake-word detection is the always-on path already on `main`. This document is microphone capture only. The names match that path's handoff (`CapturePort`, `CaptureRequest`, `CaptureStopReason`, `WakeEvent`, `AudioSnapshot`, `AudioFrame`, `SessionOrchestrator.close_capture`) so `WakeRuntime` can take an `UtteranceCapture` as `capture=`.

## Assumptions

The board, microphone, speaker, wake word, and Audio Model vendor are not chosen yet.

- **Device.** A powered Linux-class single-board computer (Raspberry Pi-class or similar) with at least one microphone and a speaker. No GPIO map is assumed. Hardware mute arrives as `stop(mute)` from the session orchestrator, which is driven by the wake path's `MuteControl`.
- **Who opens the microphone.** The wake runtime is the only reader. It opens one swappable source and shares that stream with the pre-roll ring. Capture does not open a second device. v1 of that source can be a single microphone; the frames here are shaped `(samples, channels)` and capture keeps every channel.
- **PCM.** float32 samples in `[-1, 1]`. The wake path's default stream is 16 kHz, 20 ms frames (320 samples), one channel. Capture accepts whatever rate, frame size, and channel count the wake event and the pre-roll agree on.
- **Pre-roll.** The `CaptureRequest.pre_roll` snapshot, not a second ring buffer. The wake path's default length is 800 ms, inside a few hundred milliseconds to about a second, and that path keeps the length between 200 ms and 1 s. The snapshot runs through the decision frame and still holds audio from before the wake word ended. Capture keeps the whole snapshot. `wake_word_end_offset` marks the exclusive end of the wake word inside it. The decision frame is not delivered again; the first `on_frame` starts at `pre_roll.end_sample_index`, which equals `wake.decision_sample_index`.
- **Endpoint.** Energy on the loudest channel, not a neural voice-activity model and not beamforming. Defaults: 700 ms of trailing silence after speech has started, 1600 ms of leading silence if the last 200 ms of the pre-roll is still quiet, and 15 s maximum after the wake. The 200 ms tail is shorter than the wake path's 240 ms confirm window, so the wake word alone does not arm the short timer. The threshold is at least 0.02 RMS and at least three times the 20th-percentile loudness of the pre-roll, which is how steady room noise is separated from a louder command. These are starting points for a quiet powered device, not measurements.
- **Audio Model.** Copied from `CaptureRequest.audio_model` onto the utterance. Capture does not open a network connection for `remote` and does not run a local model for `local`.

## Handoff

```python
from hardware_voice_ai.runtime import WakeRuntime

runtime = WakeRuntime(config, source)
# runtime.capture is an UtteranceCapture bound to runtime.orchestrator.
# A finished utterance is queued on runtime.utterances.
```

`WakeRuntime` binds that capture to the real `SessionOrchestrator` before audio starts. Callers do not call `bind_session`. A custom `UtteranceCapture` passed as `capture=` is bound the same way; `utterances` is `None` in that case because the caller already holds the sink.

`WakeRuntime` calls:

| Call | When | What capture does |
| --- | --- | --- |
| `start(CaptureRequest)` | Wake accepted, session moved `idle → woken → capturing` | Keep `wake` (no samples of its own), `pre_roll`, and `audio_model`. |
| `on_frame(AudioFrame)` | Each later microphone frame while the tap is open | Append it. On end-of-utterance, call `session.close_capture()`. |
| `stop(end_of_utterance)` | `SessionOrchestrator.close_capture` | Hand a `CapturedUtterance` to the sink. Leave the session in `capturing`. |
| `stop(cancel)` | `SessionOrchestrator.cancel` | Drop the buffer. Do not call the sink. Session returns to `idle`. |
| `stop(mute)` | `SessionOrchestrator.apply_mute(True)` | Drop the buffer. Do not call the sink. Session returns to `idle`. |

`on_frame` does not call the wake runtime, the wake model, or the microphone source. The only callback is `close_capture` on the orchestrator that was bound.

`close_capture` on the wake path stops the tap and does not enter `transcribing`, and it does not run an Audio Model. The sink is how this package gives that orchestrator the audio for the later model step. `QueueUtteranceSink.accept_utterance` only enqueues. It runs on the microphone thread, under the orchestrator's lock, so it returns without starting a model. The orchestrator drains the queue when it routes the turn:

- `audio_model == local`: the utterance stays on the device for the local Audio Model.
- `audio_model == remote`: a later issue may send this utterance, and only this utterance, to the cloud gateway. Always-on audio is not in the buffer.

`CapturedUtterance` fields:

- `wake`: the `WakeEvent` from the request. It carries no samples.
- `audio_model`: `local` or `remote`.
- `samples`: float32, shape `(samples, channels)`, pre-roll first, then `on_frame` audio.
- `sample_rate_hz`, `channels`.
- `pre_roll_sample_count`: how many leading rows came from the snapshot.
- `wake_word_end_offset`: row where the wake word ends. Rows before it are audio from before that end, including the lead-in the pre-roll held.
- `endpoint`: `silence`, `leading_silence`, `max_duration`, or `requested` when `close_capture` ran before the energy check finished.

Cancel and mute do not produce an utterance. After `stop`, capture drops its copy. `holds_audio` is true only while a turn is open.

## Privacy

Raw audio is not written by default. `CaptureLog` is off unless it is constructed with a local file path. A path containing `://` is rejected. Each line is one text event (`start` or `stop`) with the Audio Model label, channel count, sample rate, pre-roll length, stop reason, and endpoint cause. The line does not contain samples.

## What this package does not do

- The wake-word detector, or opening a second microphone. The wake runtime is the only reader of the device.
- The local or remote Audio Model, the cloud gateway, and speaker playback.
- A speech-to-text, text-destination, or text-to-speech setting or pipeline stage.
- Choosing the board, the microphones, or the production wake word.
