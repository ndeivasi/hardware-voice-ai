"""TLS transport for one remote turn.

This is the only module that imports a socket. The wake loop does not import
it. A session is opened only after a captured utterance is submitted and
only when a gateway URL is configured.
"""

import socket
import ssl
from collections.abc import Callable

from hardware_voice_ai.audio_model.gateway import parse_gateway_url
from hardware_voice_ai.audio_model.session import FramedGatewaySession
from hardware_voice_ai.errors import ModelUnavailable

Connector = Callable[[str, int, float], socket.socket]


class SocketStream:
    """Byte stream over one connected socket. Short reads are returned as-is."""

    def __init__(self, sock: socket.socket, timeout_s: float) -> None:
        self._sock = sock
        self._timeout_s = timeout_s

    def read(self, n: int, timeout_s: float) -> bytes:
        self._sock.settimeout(timeout_s if timeout_s > 0 else self._timeout_s)
        try:
            return self._sock.recv(n)
        except socket.timeout as exc:
            raise TimeoutError from exc

    def write(self, data: bytes) -> None:
        self._sock.settimeout(self._timeout_s)
        self._sock.sendall(data)

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


def connect_framed_session(
    url: str,
    token: str,
    *,
    timeout_s: float = 5.0,
    connector: Connector | None = None,
) -> FramedGatewaySession:
    """Open one authenticated TLS session. The token is not placed in the URL."""
    host, port = parse_gateway_url(url)
    if not host or port <= 0:
        raise ModelUnavailable("remote audio model is unavailable")
    connect = connector if connector is not None else _default_connector
    try:
        sock = connect(host, port, timeout_s)
    except (OSError, TimeoutError, ssl.SSLError) as exc:
        raise ModelUnavailable("remote audio model is unavailable") from exc
    return FramedGatewaySession(SocketStream(sock, timeout_s), token=token)


def _default_connector(host: str, port: int, timeout_s: float) -> socket.socket:
    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    raw = socket.create_connection((host, port), timeout=timeout_s)
    try:
        return context.wrap_socket(raw, server_hostname=host)
    except Exception:
        raw.close()
        raise
