# Wake-word detection

This package is the on-device wake path from the [architecture baseline](architecture.md). Always-on audio stays on the device. An accepted wake notifies the session orchestrator. There is no speech-to-text stage and no text-to-speech stage.

The session states are `idle → woken → capturing → transcribing → thinking → speaking → idle`. This work takes `idle → woken → capturing`. Cancel and hardware mute return to `idle` from any later state this package can enter. A wake during capture is ignored. A wake during speaker playback stops that playback and starts a new turn; see [playback.md](playback.md).

## Assumptions

The product wake word, language, board, and microphones are not chosen yet. The defaults below are explicit stand-ins so the on-device path can run and be tested.

### Wake word and language

- Default label: `hey computer`.
- The label is English and unbranded. It is metadata, not a claim that the default model understands spoken English.
- Changing the label without changing the model is rejected. A different word needs a template or another model id, so a detection is not reported as a word the model does not spot.

### Wake model

- Default model id: `reference-keyword-v1`.
- It is a portable, on-device sequence spotter. Each 20 ms frame is scored by which of eight tonal bands dominates. The default pattern is 16 steps (320 ms) over these centers, in hertz: `300, 500, 700, 900, 1200, 1600, 2000, 2600`. The band indexes are `0, 3, 6, 2, 5, 7, 1, 4, 2, 6, 3, 0, 5, 7, 4, 1`.
- Level is normalized per frame, so a quieter copy of the same pattern still matches. That is not far-field speech performance. There is no beamforming and no acoustic echo cancellation.
- The spotter keeps one sequence position per channel and reports the strongest channel. It does not assume a single microphone. The default configuration uses one channel.
- A shift of a few milliseconds still lands in one band. A half-frame mix of two tones does not complete the pattern. The reference model is frame-oriented on purpose; it is not a shift-invariant speech model.
- `render_pattern` synthesizes the pattern in memory for tests and for playing the stand-in tone into a microphone. It does not write a file.
- Swap point: implement `WakeModel` (`score` and `reset`) and select it with `HVA_WAKE_MODEL`, or pass `extra_models` to `build_wake_model`. The reference builder refuses a wake word that does not match its template.
- Spoken-word swap: model id `openwakeword`, with the `wake` extra installed. `HVA_WAKE_WORD` is the pretrained model name (for example a model chosen later). openWakeWord posteriors are brief, so `HVA_MIN_POSITIVE_MS` must be one frame (`20`). The adapter is tested with a fake runner and does not download weights in the default tests. Audio stays in-process. The adapter scores the highest-energy channel of each 80 ms chunk rather than averaging channels together.

### Audio backend

- Internal PCM is float32 in `[-1, 1]`, shape `(frame_samples, channels)`.
- Default format: 16 kHz, 20 ms frames (320 samples), one channel. The reference model is built for that rate and frame size.
- `sounddevice` (PortAudio) is the live microphone backend for a Linux-class board. It is an optional extra so tests run without a microphone. `SoundDeviceAudioSource` converts int16 device audio to float32 and keeps it in memory.
- `PcmAudioSource` is the in-process backend used by tests and by injected audio.
- The runtime is the only reader of the source. Wake detection and the capture tap share that read. They do not open a second microphone stream.

### Hardware and mute

- Target: a powered Linux-class single-board computer with at least one microphone. No specific board, GPIO map, or microphone array is assumed.
- v1 can run with one microphone. Interfaces take a channel count.
- `MuteControl` is the authoritative physical mute. `InMemoryMuteControl` is the stand-in until a board driver exists.
- Engaging mute returns the session to `idle`, stops capture, clears the pre-roll, resets the wake model, and drops microphone frames until mute is released. Those dropped frames are not scored and are not kept for a later capture.

### Audio Model setting

- The only device setting for where the turn's model runs is Audio Model: `local` or `remote`.
- Default: `remote`.
- The wake path copies the setting onto the capture request. `local` and `remote` share the same wake and handoff behavior. Neither model is started here, and no cloud session is opened.

### Latency and false accepts

Design targets from the architecture, not measurements on a board:

- The engine estimates the end of the wake word as the last frame that stays near the confidence peak.
- It then waits `confirm_ms` before accepting. The default is 240 ms, inside the 200–300 ms budget. `confirm_ms` cannot be set outside that budget.
- The reported `decision_latency_ms` is the time from that estimated end to the decision, using the frame's sample clock.
- A candidate must stay above `accept_threshold` (default 0.8) for at least `min_positive_ms` (default 280 ms, which the 320 ms reference pattern satisfies). A one-frame spike is rejected. A neural posterior can use a one-frame minimum when that model is selected.
- Energy that never falls is abandoned at `max_word_ms` (default 1000 ms) and cannot accept until it drops.
- After an accept, a cooldown (default 800 ms) suppresses another decision. The reference model also requires the whole tonal sequence on one channel, so noise, a single tone, a reversed pattern, and a partial pattern do not wake.
- The pre-roll default is 800 ms, within "a few hundred milliseconds to about a second."

### Privacy and logs

- Always-on audio is written only to the in-memory pre-roll and to the wake model.
- The package does not import a network client. Audio can leave the device only after a wake, and only from later work, and only when Audio Model is `remote`.
- Detection logs are off unless `HVA_DETECTION_LOG` is a local file path. Each line is JSON metadata: word, model, confidence, latency, channel, audio-model setting, and whether the session accepted the wake. The log does not contain samples. A URL path is rejected.

## Handoff, and what issue 2 owns

On an accepted wake the orchestrator moves `idle → woken`, calls `CapturePort.start` with a `CaptureRequest`, then moves to `capturing`.

`CaptureRequest` carries:

- the `WakeEvent` (no samples): word, model, confidence, latency, winning channel, and the sample index where the wake word ended
- `pre_roll`: a copy of the ring buffer through the decision frame, which still holds audio from before the wake word ended
- `audio_model` for the turn

The decision frame is the last frame in the snapshot. Later microphone frames are delivered to `CapturePort.on_frame` until the tap stops. Issue 2 should treat the snapshot as the start of the utterance and `on_frame` as the continuation.

`PendingCapturePort` records that handoff and drops later samples. Tests that need the tap to stay open pass it in explicitly. `WakeRuntime` does not use it by default. The default capture is `UtteranceCapture`, and the runtime binds that object to the real `SessionOrchestrator` before audio starts. Callers do not call `bind_session`.

`UtteranceCapture` keeps the pre-roll plus the frames after the decision. Silence ends the utterance through `SessionOrchestrator.close_capture`. That stops the tap with `end_of_utterance` and leaves the session in `capturing`. The utterance, including the pre-roll, is queued on `runtime.utterances` for the Audio Model. See `docs/capture.md`.

`close_capture` does not run a model and does not enter `transcribing`.

Cancel calls `stop(cancel)` and returns to `idle`. Hardware mute calls `stop(mute)`, clears the pre-roll, and returns to `idle`.

Issue 2 should not upload always-on audio. Only the post-wake utterance, including the pre-roll that was handed over, is in scope for a later turn, and only when Audio Model is `remote`.

## What this package does not do

- The local or remote Audio Model and the cloud gateway (issues 3–5). Speaker playback is [playback.md](playback.md). Post-wake utterance capture is `UtteranceCapture`; see [capture.md](capture.md).
- A separate speech-to-text or text-to-speech setting or pipeline stage.
- Choosing the production wake word, training a far-field speech model, or selecting the board.
