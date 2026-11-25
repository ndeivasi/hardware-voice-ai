"""Run the remote Audio Model gateway on a GPU host.

    python -m hardware_voice_ai.gateway --backend moshi
"""

import argparse
import os
import sys

from hardware_voice_ai.errors import ConfigurationError, RemoteBackendError
from hardware_voice_ai.gateway.catalog import REMOTE_BACKENDS
from hardware_voice_ai.gateway.loader import load_backend
from hardware_voice_ai.gateway.settings import settings_from_env


def main(argv: list[str] | None = None, *, serve=None, load=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hardware-voice-ai.gateway",
        description=(
            "Serve one remote speech-to-speech adapter on a GPU host. "
            "The device Audio Model setting stays local or remote."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=REMOTE_BACKENDS,
        help="Adapter to load. Default is moshi, or HVA_REMOTE_BACKEND.",
    )
    parser.add_argument("--host", help="Bind address. Default is 0.0.0.0.")
    parser.add_argument("--port", type=int, help="TCP port. Default is 8443.")
    args = parser.parse_args(argv)
    try:
        settings = settings_from_env(
            os.environ,
            backend=args.backend,
            host=args.host,
            port=args.port,
        )
        runner = serve
        if runner is None:
            from hardware_voice_ai.gateway.tls import load_server_context, serve as runner

            # Reject a missing certificate before the vendor library downloads weights.
            load_server_context(settings.certfile, settings.keyfile)
        backend = (load or load_backend)(settings)
    except (ConfigurationError, RemoteBackendError) as exc:
        print(f"hardware-voice-ai gateway: {exc}", file=sys.stderr)
        return 2
    print(
        f"gateway backend={settings.backend_id} "
        f"listen={settings.host}:{settings.port} "
        f"token={'set' if settings.token else 'unset'}",
        flush=True,
    )
    try:
        runner(settings, backend)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
