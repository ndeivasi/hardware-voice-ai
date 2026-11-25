"""GPU requirement shared by the remote adapters.

These models do not run on a Pi-class board. ``cpu`` is an explicit
operator override for a workstation, not the device path.
"""

from hardware_voice_ai.errors import RemoteBackendError


def ensure_gpu(device: str) -> None:
    if device == "cpu":
        return
    try:
        import torch
    except ImportError as exc:
        raise RemoteBackendError(
            "Remote speech-to-speech backends need a GPU and PyTorch. "
            "Install the backend extra on the GPU host."
        ) from exc
    cuda = getattr(torch, "cuda", None)
    available = bool(cuda is not None and cuda.is_available())
    if device.startswith("cuda") and not available:
        raise RemoteBackendError(
            "Remote speech-to-speech backends need a GPU. "
            "Point the gateway at a GPU host. A Pi-class board cannot run them."
        )
