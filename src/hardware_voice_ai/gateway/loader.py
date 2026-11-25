"""Build the one remote backend this gateway process will serve."""

from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.gateway.settings import GatewaySettings


def load_backend(settings: GatewaySettings):
    """Import and construct the selected adapter.

    The import happens here so the wake path, and a gateway configured for
    a different adapter, do not load the others.
    """
    if settings.backend_id == "moshi":
        from hardware_voice_ai.gateway.moshi import load_moshi

        return load_moshi(settings)
    if settings.backend_id == "minicpm-o-4.5":
        from hardware_voice_ai.gateway.minicpm import load_minicpm

        return load_minicpm(settings)
    if settings.backend_id == "qwen3-omni":
        from hardware_voice_ai.gateway.qwen import load_qwen

        return load_qwen(settings)
    raise ConfigurationError(
        "Remote backend must be moshi, minicpm-o-4.5, or qwen3-omni."
    )
