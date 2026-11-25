"""Gateway address and the messages on one remote turn.

The device speaks this protocol to a gateway. It does not name a model
vendor. The gateway is what holds provider adapters. Credentials stay in
the process environment and are redacted from ``repr``.
"""

from dataclasses import dataclass

import numpy as np

from hardware_voice_ai.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """How to reach the gateway for Audio Model ``remote``.

    ``url`` is empty until someone sets ``HVA_GATEWAY_URL``. An empty URL
    means the remote model is unavailable; the wake loop still runs.
    ``token`` authenticates the device to the gateway. It is not a provider key.
    """

    url: str = ""
    token: str = ""

    def __repr__(self) -> str:
        presence = "set" if self.token else "unset"
        return f"GatewayConfig(url={self.url!r}, token={presence})"


@dataclass(frozen=True, slots=True)
class GatewayAudio:
    samples: np.ndarray
    sample_rate_hz: int


@dataclass(frozen=True, slots=True)
class GatewayError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class GatewayEnd:
    pass


GatewayMessage = GatewayAudio | GatewayError | GatewayEnd


def parse_gateway_url(url: str) -> tuple[str, int]:
    """Accept ``tls://host:port`` or an empty string.

    An empty string is "not configured". Anything else must be TLS so the
    utterance is not sent in cleartext. The credential is not accepted in
    the URL, because URLs end up in logs.
    """
    if url == "":
        return "", 0
    if "://" not in url:
        raise ConfigurationError("Audio Model gateway URL must be tls://host:port.")
    scheme, rest = url.split("://", 1)
    if scheme != "tls":
        raise ConfigurationError(
            "Audio Model gateway URL must use tls:// so utterance audio is not sent in cleartext."
        )
    if "@" in rest:
        raise ConfigurationError(
            "Put the gateway credential in HVA_GATEWAY_TOKEN, not in the URL."
        )
    if any(mark in rest for mark in "/?#"):
        raise ConfigurationError("Audio Model gateway URL must be tls://host:port.")
    if rest.count(":") != 1:
        raise ConfigurationError("Audio Model gateway URL must be tls://host:port.")
    host, port_text = rest.split(":")
    if not host or any(character.isspace() for character in host):
        raise ConfigurationError("Audio Model gateway URL must be tls://host:port.")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ConfigurationError("Audio Model gateway port must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ConfigurationError("Audio Model gateway port is out of range.")
    return host, port


def redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text
