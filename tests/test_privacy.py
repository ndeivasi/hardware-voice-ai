"""Always-on audio stays on the device. One remote turn may open TLS."""

import ast
from pathlib import Path


FORBIDDEN_ROOTS = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "requests",
    "websocket",
    "websockets",
    "aiohttp",
    "httpx",
}

# The remote Audio Model is the only code allowed to open a socket, and only
# for one post-wake turn. It may use the standard library TLS stack.
TLS_FILE = "tls.py"
TLS_MODULES = {"socket", "ssl"}


def test_only_the_remote_transport_imports_a_network_stack() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "hardware_voice_ai"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                root_name = module.split(".", 1)[0]
                if root_name not in FORBIDDEN_ROOTS:
                    continue
                if path.name == TLS_FILE and root_name in TLS_MODULES:
                    continue
                offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_there_is_no_speech_to_text_or_text_to_speech_package() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "hardware_voice_ai"
    names = [path.name for path in root.rglob("*.py")]
    assert not any("stt" in name or "tts" in name for name in names)
    banned = {"speech_to_text.py", "text_to_speech.py", "llm.py"}
    assert banned.isdisjoint(names)
