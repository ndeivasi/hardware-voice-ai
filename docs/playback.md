# Speaker playback

This is the on-device speaker path. It plays two kinds of audio:

- Spoken-reply audio from the Audio Model. Issues 3–5 will feed that stream in later. Until then, `StubReplyAudio` is the stand-in. It is not a model and it does not open a network connection.
- Local wake, offline, and error cues. These are synthesized in this process in both Audio Model modes. They do not use a cloud audio path.

Playback is local when Audio Model is `local` and when it is `remote`. `remote` is the default. The spoken reply in that mode is streamed audio played here; this package does not fetch it.

The session orchestrator is the caller. After capture closes it can move `capturing → speaking` and play a reply. A finished reply returns the session to `idle`. Cancel, a stop request, hardware mute, and a new accepted wake during speaking also return to `idle`. A wake during capture is still ignored.

## Assumptions

The board, speaker, and Audio Model are not chosen yet.

- **Device.** A powered Linux-class single-board computer (Raspberry Pi-class or similar) with a speaker. No GPIO map, ALSA device name, or amplifier is assumed. The live backend is PortAudio through `sounddevice`, the same optional extra as the microphone. `MemorySpeaker` and `NullSpeaker` stand in when there is no hardware.
- **PCM.** float32 samples in `[-1, 1]`, shape `(samples, channels)`. The default stream is 16 kHz, 20 ms frames (320 samples), one channel, matching the microphone. Channel count comes from the device config. Reply audio must already be in that format. This path does not resample.
- **Queue and stop latency.** The design goal is under about 200 ms from a stop cue to the speaker going quiet. The software queue defaults to 80 ms and cannot be configured above 200 ms. `stop` discards that queue. On the live backend, `flush` waits until the callback has played the queue, or until stop or close wakes it. Playback stays in progress during that wait, so mute and stop can still drop the tail. The callback may already be holding one frame (20 ms at the default); that frame can finish. These are design bounds, not measurements on a board.
- **Volume.** Gain in `[0, 1]`. `1` is unity. `0` is silence and is not hardware mute. Unity gain does not change samples that are already inside `[-1, 1]`. A chunk that would exceed full scale is scaled down so its peak is 1. The waveform is not hard-clipped.
- **Cues.** `wake` (rising pair, 880 Hz then 1320 Hz), `offline` (two 220 Hz tones), and `error` (three 660 Hz beeps). Amplitude is 0.25 before volume. The same render is used for `local` and `remote`. While a cue is playing, the runtime does not score that audio and does not forward it into the captured utterance: those frames are silence, so the cue cannot leave the device with a `remote` turn. A wake decision during a cue is ignored. Echo cancellation while a spoken reply is playing is a later follow-up.
- **Streaming.** `ReplyAudio.read` may return a chunk, an empty array when nothing is ready, or `None` when the reply is finished. Playback writes the first chunk without waiting for `finish`.
- **Mute.** Hardware mute is authoritative. It stops playback, returns the session to idle, and refuses new reply and cue audio until it is released. Releasing mute does not resume the cut sound.
- **Privacy.** Raw audio is not written to disk. `NullSpeaker` and `SoundDeviceSpeaker` do not keep a sample history. `MemorySpeaker` keeps samples in RAM so tests can inspect them. That object is the test double, not the live backend.

## Interface

```python
from hardware_voice_ai.playback import Cue, SpeakerPlayback, StubReplyAudio
from hardware_voice_ai.playback.speaker import MemorySpeaker, build_speaker

speaker = build_speaker(config.audio_format, "sounddevice", queue_ms=config.playback_queue_ms)
playback = SpeakerPlayback(speaker, volume=config.volume)

# The session orchestrator owns these calls.
session.play_reply(reply_stream)   # capturing → speaking; local in both modes
session.play_cue(Cue.OFFLINE)      # local earcon, no cloud path
session.set_volume(0.4)            # [0, 1], applies to the next block
session.stop_playback()            # cut the queue; speaking → idle
session.playback_status()          # playing, completed, or stopped
session.last_reply_result          # the reply's own result after a later cue
```

`play_reply` accepts any `ReplyAudio`. `StubReplyAudio.push` / `finish` is the in-memory stand-in for the Audio Model stream. `MemoryReplyAudio` is a buffer that is already complete, used for cues and tests.

`PlaybackResult.status` is `playing`, `completed`, or `stopped`. `completed` means the reply or cue finished and the queue was played out. `stopped` means it was cut. `reason` says why: `completed`, `stop`, `barge_in`, `mute`, or `replaced`.

A new accepted wake while the session is `speaking` stops playback with reason `barge_in`, then starts a new capture and plays the local wake cue. The wake cue is also played on an accepted wake from idle.

Swap the hardware by implementing `Speaker`: `write`, `drop_queued`, `flush`, `close`, and the queued / played / dropped counters. `build_speaker` selects `sounddevice` or `pcm` (`MemorySpeaker`). The orchestrator takes a `SpeakerPlayback`, so the session does not know which backend is underneath.

## What this package does not do

- The local or remote Audio Model, the cloud gateway, or a vendor adapter.
- A speech-to-text, text-destination, or text-to-speech stage. Reply audio is one stream.
- Microphone capture. That is `docs/capture.md`. A bad capture frame still cancels the turn and the loop keeps listening.
- Choosing the board, the speaker, or the production cue sounds.
- Acoustic echo cancellation while a spoken reply is playing. Local cues are kept out of the capture tap instead. Echo cancellation for barge-in during the reply can sit behind the same speaker interface later.
