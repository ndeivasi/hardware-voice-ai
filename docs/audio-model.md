# Audio Model voice loop

This is one voice loop for issues 3, 4, and 5. The device has one Audio Model setting, `local` or `remote`, default `remote`. That model receives a captured utterance and produces the spoken reply. Speech-to-text and text-to-speech are inside that model. They are not device settings and they are not separate pipeline stages.

The approved baseline is [architecture.md](architecture.md). This note records how the loop is wired. It does not change that baseline.

## What runs

The device states are `idle → woken → capturing → speaking → idle`.

After an accepted wake the session is `capturing`. `UtteranceCapture` decides that the user has stopped speaking and calls `SessionOrchestrator.close_capture`. That stops the tap, enqueues the utterance, and leaves the session in `capturing`. It does not start the model. `accept_utterance` only enqueues, because it runs under the orchestrator lock.

When `poll` sees that queued utterance, it moves the session to `speaking` before it reads another frame, then starts the one Audio Model on a separate thread and keeps reading the microphone. A wake in that gap, or any wake during `speaking`, cancels the turn, stops playback, and starts a new capture. The spoken reply is played with `play_reply`. A local failure cue is played with `play_cue`. `_finish_turn` clears that turn once, under the session lock, and only while it is still the one speaking. If playback is still flushing, the session stays in `speaking` until flush completes; `_on_reply_finished` is what returns it to `idle`. A wake in that tail stops playback and starts a new capture. If a new capture is already open, the finishing turn leaves it alone. Cancel and hardware mute return to `idle` without playing a cue. Hardware mute does not resume playback when it is released.

`submit_utterance` is also public for a caller that already holds a `CapturedUtterance`:

```python
result = runtime.orchestrator.submit_utterance(utterance)
```

## Handoff from capture

`CapturedUtterance` is the object capture queues. `submit_utterance` accepts that object, or anything with the same fields:

- `wake`
- `audio_model` (`local` or `remote`)
- `samples` float32, shape `(samples, channels)`
- `sample_rate_hz`
- `pre_roll_sample_count`
- `wake_word_end_offset`
- `endpoint`

`CapturedUtterance` is that object. `utterance_from_request(request, continuation)` builds one from the `CaptureRequest` the stub already stored, plus any post-wake samples the caller kept. It does not read the microphone.

The session setting wins. A mismatched label, empty audio, non-finite audio, or more than 30 seconds is rejected and is not sent anywhere. The session returns to `idle` so a later wake still works. Submitting while the capture tap is still open is refused and leaves the session in `capturing`.

## Speaker port

Speaker playback is implemented in [playback.md](playback.md). The voice loop writes the spoken reply through `SpeakerPort`, and the runtime plays those chunks with `play_reply` and local cues with `play_cue`:

- `start_reply(sample_rate_hz, channels)`
- `write_reply(samples)` as chunks arrive
- `end_reply()` when the spoken reply is complete
- `stop()` for immediate silence on cancel, mute, or a failed turn
- `play_cue(cue, samples, sample_rate_hz)` for local offline and error audio

`MemorySpeaker` in this package records that audio in memory and does not open a device. Playback's `MemorySpeaker` and `NullSpeaker` are the stand-ins for the device speaker. `stop` rejects later reply writes. Speaker methods must not call back into the session.

Wake, offline, and error cues are synthesized in-process. They never use the gateway. Cancel and mute do not play a cue; they stop playback and return to idle. Releasing mute does not resume audio.

## `local`

The turn stays on the device. No gateway session is opened and the TLS module is not imported.

`reference-audio-v1` is the built-in stand-in (`HVA_LOCAL_MODEL`, default). It maps the utterance level to a short deterministic tone so the loop can run without a download. Near-silence (RMS below 0.01 on the loudest channel) is low confidence: an error cue, no reply. It is not a conversational model. Which local model ships on a board is still an open question. Pass another engine to `LocalAudioModel`, or set an unknown `HVA_LOCAL_MODEL`. An unknown id is unavailable at turn time and does not stop the wake loop.

## `remote`

Default. After `submit_utterance`, the device opens one authenticated TLS session and streams that utterance. There is no always-on socket. The wake loop does not open one. If `HVA_GATEWAY_URL` is empty, the model is unavailable, a local offline cue plays, and listening continues.

The device does not name a vendor. The gateway holds provider adapters. The device speaks one framed RPC:

```
uint32 big-endian payload length | uint8 type | payload
type 1: UTF-8 JSON control
type 2: int16 little-endian PCM, interleaved channels, clipped from float [-1, 1]
```

Device to gateway:

- `{"type":"hello","audio_model":"remote","authorization":"<token>"}` — `authorization` is omitted when no token is set
- `{"type":"audio_start","sample_rate_hz":16000,"channels":1}`
- audio frames
- `{"type":"audio_end"}`
- `{"type":"cancel"}` when the turn is cancelled or times out. Cancel is control, not audio.

Gateway to device:

- `{"type":"reply_start","sample_rate_hz":16000,"channels":1}`
- audio frames — the spoken reply
- `{"type":"error","code":"...","message":"..."}`
- `{"type":"end"}`

Control messages are errors, end of turn, and cancel. The spoken reply is audio. Confirmation, including a possible mishearing, is spoken in that reply. It is not a separate text message.

`HVA_GATEWAY_URL` must be `tls://host:port`. A cleartext scheme is rejected. The credential is `HVA_GATEWAY_TOKEN` and is rejected if it is placed in the URL. The token is not included in `repr` or in model error text. It is sent only on that session. The token authenticates the device to the gateway. It is not a provider API key, and no provider key belongs in this repo.

`socket` and `ssl` are imported only from `audio_model/tls.py`, and only when a turn actually connects.

## Failures

| Case | Result | Cue |
| --- | --- | --- |
| `remote` and no gateway, or the connection fails | unavailable | offline |
| `remote` turn exceeds the timeout | timeout | offline |
| `local` model id is not on the device | unavailable | error |
| `local` turn exceeds the timeout | timeout | error |
| near-silence, or the gateway says `low_confidence` | low confidence | error |
| protocol or other model failure | error | error |
| cancel or hardware mute | cancelled | none |

Defaults are 8 seconds to the first reply audio and 30 seconds for the whole turn (`HVA_FIRST_AUDIO_TIMEOUT_S`, `HVA_TURN_TIMEOUT_S`). The architecture's "under about 2 seconds" figure is a design goal, not this failure cutoff. A hung model gives up. The microphone loop keeps reading frames while the model runs.

Hardware mute during a turn cancels the model, stops the speaker, and ignores a new wake until mute is released.

## What this does not do

- Microphone capture, silence detection, or a second open of the microphone (issue 2, already implemented).
- A second speaker implementation. Playback is [playback.md](playback.md): `play_reply`, `play_cue`, volume, and stop. `MemorySpeaker` in this loop records the reply for tests. `MemorySpeaker` and `NullSpeaker` in playback are the speaker stand-ins.
- A speech-to-text, text-destination, or text-to-speech stage or setting.
- A cloud vendor client, a model download, or a stored provider credential.
- Persisting utterance or reply audio. The in-memory record is not written to disk.
