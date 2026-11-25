# Remote speech-to-speech adapters

Audio Model stays one setting: `local` or `remote`, default `remote`. This note is the remote side of that setting. The device streams one turn's audio to the gateway. The gateway runs one speech-to-speech model and streams the spoken reply back. Speech-to-text and text-to-speech are inside that model. They are not stages and they are not settings.

The local stand-in stays `reference-audio-v1`. It is not selectable here. These three models need a GPU. They will not run on a Pi-class board. Wake, capture, and speaker playback stay on the device.

The approved baseline is [architecture.md](architecture.md).

## Which model

The user-facing choice is the gateway's backend. It is not a third Audio Model label.

| What you pass | Model | Weights |
| --- | --- | --- |
| `moshi` (default) | [Moshi](https://github.com/kyutai-labs/moshi) | 7B, CC-BY 4.0, repo `kyutai/moshiko-pytorch-bf16` |
| `minicpm-o-4.5` | [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | 9B, Apache-2.0 |
| `qwen3-omni` | [Qwen3-Omni Instruct](https://github.com/QwenLM/Qwen3-Omni) | Apache-2.0, `Qwen/Qwen3-Omni-30B-A3B-Instruct`, heavier than the other two |

`minicpm`, `minicpm-o`, and `qwen3-omni-instruct` are accepted aliases. `reference-audio-v1` is rejected.

One gateway process loads one model at startup. A device hello may include `"backend"`. If it is present, it must name the model this process loaded. If it is absent, the process default is used. The device client that does not send `backend` still works: it gets whatever this process was started with.

## Point the device at a GPU host

Run the gateway on the machine that has the GPU. Run the device process on the board. They are not the same machine.

On the GPU host, create a TLS certificate and an unguessable token yourself. This repository does not generate either one, and it does not contain a vendor API key. `HVA_GATEWAY_TOKEN` authenticates the device to the gateway. It is not sent to Moshi, MiniCPM-o, or Qwen.

```bash
# GPU host
export HVA_GATEWAY_CERT=/path/to/cert.pem
export HVA_GATEWAY_KEY=/path/to/key.pem
export HVA_GATEWAY_TOKEN='operator-chosen-secret'
export HVA_REMOTE_BACKEND=moshi
pip install -e '.[moshi]'
python -m hardware_voice_ai.gateway --host 0.0.0.0 --port 8443
```

`--backend` overrides `HVA_REMOTE_BACKEND`. The other extras are `.[minicpm]` and `.[qwen]`.

On the device, after an accepted wake, the existing remote transport opens one authenticated TLS session and closes it at the end of the turn. There is no always-on audio socket.

```bash
# Device
export HVA_AUDIO_MODEL=remote
export HVA_GATEWAY_URL=tls://gpu-host.example:8443
export HVA_GATEWAY_TOKEN='operator-chosen-secret'
```

Use a name or address the device can route to. `0.0.0.0` is only the gateway bind address. The URL scheme is `tls://`. A cleartext scheme is not this transport. Do not put the token in the URL.

`HVA_GATEWAY_URL` and the framed session are the device side of this same transport. This process is the server. The wake loop does not import it.

## What the session carries

One connection is one turn. Frames are a big-endian uint32 length, a uint8 type, then the payload. Type 1 is UTF-8 JSON. Type 2 is int16 little-endian PCM, interleaved, clipped from float `[-1, 1]`.

The device sends `hello`, `audio_start`, audio, then `audio_end`. `cancel` stops the turn. The gateway sends `reply_start`, audio, then `end`. Failures are `error` objects. Those control messages are not the spoken reply.

Nothing on the session is a transcript. Confirmation and failure, when the model speaks them, are audio.

The gateway resamples the reply to the sample rate and channel count the device announced in `audio_start`. Speaker playback on the device does not resample. The resampler is linear and is meant to be replaced if a deployment wants a better one.

## Per-model wiring

Each adapter implements `SpeechBackend.speak`: utterance PCM in, spoken PCM out. The constructors below are the current vendor calls. Swap the object if the upstream API changes. The device session does not change with it.

### Moshi

`loaders.CheckpointInfo.from_hf_repo` loads Mimi and the 7B LM from `HVA_MOSHI_REPO` (default `kyutai/moshiko-pytorch-bf16`). Each 24 kHz frame goes through `mimi.encode`, `LMGen.step`, and `mimi.decode`. Text tokens from the step are not forwarded. Input is mixed to mono and resampled to 24 kHz. `HVA_MOSHI_REPO` can point at another Kyutai checkpoint the same loader understands.

### MiniCPM-o 4.5

`AutoModel.from_pretrained` loads `HVA_MINICPM_MODEL` (default `openbmb/MiniCPM-o-4_5`) with audio and the model's own speech decoder enabled, and vision left off, matching the model card's audio-only note. A turn calls `streaming_prefill` and then `streaming_generate(generate_audio=True)`. The text half of each generated pair is discarded.

The published conversational path needs a reference clip for the voice. Set `HVA_MINICPM_REF_AUDIO` to a local file on the GPU host. This repository does not include one. The clip is not uploaded by the device.

### Qwen3-Omni Instruct

`Qwen3OmniMoeForConditionalGeneration.from_pretrained` loads `HVA_QWEN_MODEL` (default `Qwen/Qwen3-Omni-30B-A3B-Instruct`). The turn builds an audio-only chat, runs `apply_chat_template` and the processor, then `generate` with `speaker` (default `Ethan`, override `HVA_QWEN_SPEAKER`). The model card also shows `flash_attention_2`. This process does not require it.

The utterance stays in memory. It is not written to a wav and it is not passed as a URL. `qwen_omni_utils.process_mm_info` is the upstream helper for paths and URLs. `QwenSession.prepare` can be replaced with that helper if a deployment wants it. The default does not fetch anything.

## Assumptions

- The board is still unchosen. The device is a Linux-class single-board computer with a microphone and a speaker. The GPU host is whatever machine can hold the selected weights. No CUDA device index, driver version, or board GPIO is assumed. `HVA_REMOTE_DEVICE` defaults to `cuda`. `cpu` is an explicit override for a workstation experiment. It is not a Pi path. The bind address is a hostname or an IPv4 address.
- The device capture format is what `audio_start` says, usually 16 kHz mono. Moshi's codec is 24 kHz. MiniCPM-o takes 16 kHz and returns 24 kHz. Qwen3-Omni Instruct returns 24 kHz. The gateway converts so the device hears the rate it asked for.
- Models that take one channel hear the mean of the device channels. A mono reply is duplicated when the device asked for more than one channel.
- A turn is half duplex on this session: the device finishes the utterance, then the reply streams. Moshi can run frame by frame inside the gateway. Barge-in is still the device cancelling the turn. A cancel that arrives before `audio_end` does not run the model.
- Utterances longer than 30 seconds are rejected. That matches the device cap.
- One process serves one loaded model. A second connection during a turn is refused. Loading all three models at once is not attempted.
- Weights download only on the GPU host, through the vendor library, and only when the operator starts that backend. Tests do not download them. A Hugging Face token, if a mirror needs one, is the operator's `HF_TOKEN`. None is stored here.
- Vendor code moves. The stable surface is the framed session and `SpeechBackend`. The loaders are the swappable part.

## What this does not do

- A speech-to-text, language-model, or text-to-speech setting.
- Run Moshi, MiniCPM-o, or Qwen3-Omni on the device.
- Replace `reference-audio-v1`.
- Invent a certificate, a gateway token, or a provider key.
- Keep the utterance or the reply on disk.
