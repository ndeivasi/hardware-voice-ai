"""Gateway process settings. Credentials come from the environment."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.gateway.catalog import DEFAULT_REMOTE_BACKEND, parse_remote_backend

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8443
DEFAULT_DEVICE = "cuda"
DEFAULT_MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
DEFAULT_MINICPM_MODEL = "openbmb/MiniCPM-o-4_5"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_QWEN_SPEAKER = "Ethan"


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """How this GPU process serves one remote backend.

    ``token`` is the operator-chosen secret a device must send as
    ``authorization``. It is not a model-vendor key. An empty token is
    rejected. ``repr`` does not include the secret.
    """

    backend_id: str = DEFAULT_REMOTE_BACKEND
    token: str = ""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    certfile: str = ""
    keyfile: str = ""
    device: str = DEFAULT_DEVICE
    moshi_repo: str = DEFAULT_MOSHI_REPO
    minicpm_model: str = DEFAULT_MINICPM_MODEL
    minicpm_ref_audio: str = ""
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_speaker: str = DEFAULT_QWEN_SPEAKER

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_id", parse_remote_backend(self.backend_id))
        if not isinstance(self.token, str) or "\n" in self.token or "\r" in self.token:
            raise ConfigurationError("gateway token must be a single line.")
        token = self.token.strip()
        if not token:
            raise ConfigurationError(
                "Set HVA_GATEWAY_TOKEN. The gateway does not start without "
                "an operator-chosen secret."
            )
        object.__setattr__(self, "token", token)
        host = self.host.strip()
        if not host or any(mark in host for mark in ("/", ":", "?", "#", "@")):
            raise ConfigurationError("gateway host must be a hostname or address.")
        object.__setattr__(self, "host", host)
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ConfigurationError("gateway port must be from 1 to 65535.")
        if self.device not in {"cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ConfigurationError("gateway device must be cuda or an explicit cpu override.")
        for name in ("moshi_repo", "minicpm_model", "qwen_model", "qwen_speaker"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must not be empty.")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.minicpm_ref_audio, str):
            raise ConfigurationError("minicpm_ref_audio must be a path.")
        object.__setattr__(self, "minicpm_ref_audio", self.minicpm_ref_audio.strip())
        object.__setattr__(self, "certfile", str(self.certfile).strip())
        object.__setattr__(self, "keyfile", str(self.keyfile).strip())

    def __repr__(self) -> str:
        presence = "set" if self.token else "unset"
        return (
            f"GatewaySettings(backend_id={self.backend_id!r}, token={presence}, "
            f"host={self.host!r}, port={self.port})"
        )


def settings_from_env(
    env: Mapping[str, str] | None = None,
    *,
    backend: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> GatewaySettings:
    """Read operator settings. An empty backend name means Moshi."""
    values = os.environ if env is None else env
    chosen = backend if backend else values.get("HVA_REMOTE_BACKEND", "").strip()
    if not chosen:
        chosen = DEFAULT_REMOTE_BACKEND
    listen_host = host if host else values.get("HVA_GATEWAY_HOST", "").strip() or DEFAULT_HOST
    if port is None:
        listen_port = _port(values.get("HVA_GATEWAY_PORT", ""))
    else:
        listen_port = port
    token = values.get("HVA_GATEWAY_TOKEN", "")
    return GatewaySettings(
        backend_id=chosen,
        token=token,
        host=listen_host,
        port=listen_port,
        certfile=values.get("HVA_GATEWAY_CERT", ""),
        keyfile=values.get("HVA_GATEWAY_KEY", ""),
        device=values.get("HVA_REMOTE_DEVICE", "").strip() or DEFAULT_DEVICE,
        moshi_repo=values.get("HVA_MOSHI_REPO", "").strip() or DEFAULT_MOSHI_REPO,
        minicpm_model=values.get("HVA_MINICPM_MODEL", "").strip() or DEFAULT_MINICPM_MODEL,
        minicpm_ref_audio=values.get("HVA_MINICPM_REF_AUDIO", ""),
        qwen_model=values.get("HVA_QWEN_MODEL", "").strip() or DEFAULT_QWEN_MODEL,
        qwen_speaker=values.get("HVA_QWEN_SPEAKER", "").strip() or DEFAULT_QWEN_SPEAKER,
    )


def _port(raw: str) -> int:
    if raw.strip() == "":
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError("HVA_GATEWAY_PORT must be an integer.") from exc
