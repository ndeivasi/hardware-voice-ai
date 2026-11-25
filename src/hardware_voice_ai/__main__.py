"""Run always-on wake detection against the configured microphone."""

import argparse
import os
import sys
from dataclasses import replace

from hardware_voice_ai.audio.source import build_audio_source
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import AudioBackendError, ConfigurationError
from hardware_voice_ai.runtime import WakeRuntime
from hardware_voice_ai.session.audio_model import parse_audio_model
from hardware_voice_ai.wake.defaults import REFERENCE_MODEL_ID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hardware-voice-ai",
        description=(
            "Listen on-device for the wake word. Always-on audio stays on this "
            "machine. An accepted wake is handed to the session orchestrator."
        )
    )
    parser.add_argument(
        "--audio-backend",
        choices=("sounddevice", "pcm"),
        help="Microphone backend. pcm is for injected audio; the device uses sounddevice.",
    )
    parser.add_argument("--wake-model", help="Wake model id to load.")
    parser.add_argument("--wake-word", help="Wake-word label for the selected model.")
    parser.add_argument(
        "--audio-model",
        choices=("local", "remote"),
        help="Audio Model setting. Default is remote.",
    )
    args = parser.parse_args(argv)
    try:
        config = DeviceConfig.from_env(os.environ)
        changes: dict[str, object] = {}
        if args.audio_backend:
            changes["audio_backend"] = args.audio_backend
        if args.wake_model:
            changes["wake_model_id"] = args.wake_model
        if args.wake_word:
            changes["wake_word"] = args.wake_word
        if args.audio_model:
            changes["audio_model"] = parse_audio_model(args.audio_model)
        if changes:
            config = replace(config, **changes)
        source = build_audio_source(
            config.audio_format,
            config.audio_backend,
            device=_input_device(config.input_device),
        )
        runtime = WakeRuntime(config, source)
    except (ConfigurationError, AudioBackendError) as exc:
        print(f"hardware-voice-ai: {exc}", file=sys.stderr)
        return 2
    print(
        f"wake model={config.wake_model_id} label={config.wake_word!r} "
        f"audio_model={config.audio_model.value} channels={config.channels} "
        f"detection_log={'off' if config.detection_log_path is None else config.detection_log_path}",
        flush=True,
    )
    if config.wake_model_id == REFERENCE_MODEL_ID:
        print(
            "reference-keyword-v1 spots its bundled tonal pattern. "
            "It does not recognize the label as spoken English. "
            "See docs/wake-word.md to swap models.",
            flush=True,
        )
    try:
        runtime.run()
    except KeyboardInterrupt:
        return 0
    finally:
        runtime.close()
    return 0


def _input_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
