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
from hardware_voice_ai.gateway.turn import TurnGateway


class SocketStream:
    """Byte stream over one accepted socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def read(self, n: int) -> bytes:
        return self._sock.recv(n)

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
    """Listen until the process is interrupted. One connection, one turn."""
    context = load_server_context(settings.certfile, settings.keyfile)
    gateway = TurnGateway(settings, backend)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((settings.host, settings.port))
        server.listen(1)
        while True:
            conn, _addr = server.accept()
            try:
                tls_conn = context.wrap_socket(conn, server_side=True)
            except ssl.SSLError:
                conn.close()
                continue
            try:
                gateway.handle(SocketStream(tls_conn))
            finally:
                try:
                    tls_conn.close()
                except OSError:
                    pass
