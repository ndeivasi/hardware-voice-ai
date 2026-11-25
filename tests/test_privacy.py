"""The wake path stays on the device."""

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


def test_package_does_not_import_a_network_stack() -> None:
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
                if root_name in FORBIDDEN_ROOTS:
                    offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_there_is_no_speech_to_text_or_text_to_speech_package() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "hardware_voice_ai"
    names = [path.name for path in root.rglob("*.py")]
    assert not any("stt" in name or "tts" in name for name in names)
