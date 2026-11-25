"""TLS listener for the GPU gateway.

This is the only gateway module that imports a socket. Each accepted
connection is one turn, then the connection closes. There is no always-on
audio socket. The certificate and key are the operator's. This module does
not generate them.
"""

import socket
import ssl

from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.gateway.settings import GatewaySettings
from hardware_voice_ai.gateway.turn import READ_TIMEOUT_S, TurnGateway


class SocketStream:
    """Byte stream over one accepted socket.

    ``read`` raises ``TimeoutError`` when the socket wait expires. A closed
    peer returns ``b""``.
    """

    def __init__(self, sock: socket.socket, timeout_s: float = READ_TIMEOUT_S) -> None:
        self._sock = sock
        self.set_timeout(timeout_s)

    def set_timeout(self, timeout_s: float) -> None:
        self._timeout_s = timeout_s
        self._sock.settimeout(timeout_s)

    def read(self, n: int) -> bytes:
        self._sock.settimeout(self._timeout_s)
        try:
            return self._sock.recv(n)
        except socket.timeout as exc:
            raise TimeoutError from exc

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


def load_server_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    """Load the operator's certificate. Missing paths are a configuration error."""
    if not certfile or not keyfile:
        raise ConfigurationError(
            "Set HVA_GATEWAY_CERT and HVA_GATEWAY_KEY. "
            "The gateway does not generate credentials."
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_cert_chain(certfile, keyfile)
    except (OSError, ssl.SSLError) as exc:
        raise ConfigurationError(
            "The gateway certificate could not be loaded."
        ) from exc
    return context


def serve(settings: GatewaySettings, backend) -> None:
    """Listen until the process is interrupted.

    One accepted connection is handled to completion before the next accept.
    A second client waits in the listen backlog; it is not yet a turn, so it
    does not receive an ``unavailable`` frame. The accepted socket is given a
    read timeout before the TLS handshake so a stall cannot sit forever.
    """
    context = load_server_context(settings.certfile, settings.keyfile)
    gateway = TurnGateway(settings, backend)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((settings.host, settings.port))
        server.listen(1)
        while True:
            conn, _addr = server.accept()
            conn.settimeout(READ_TIMEOUT_S)
            try:
                tls_conn = context.wrap_socket(conn, server_side=True)
            except (ssl.SSLError, socket.timeout, TimeoutError, OSError):
                conn.close()
                continue
            try:
                gateway.handle(SocketStream(tls_conn))
            finally:
                try:
                    tls_conn.close()
                except OSError:
                    pass
