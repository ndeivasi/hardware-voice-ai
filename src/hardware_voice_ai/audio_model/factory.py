"""Build the one Audio Model the device setting selects."""

from hardware_voice_ai.audio_model.defaults import REFERENCE_LOCAL_MODEL_ID
from hardware_voice_ai.audio_model.events import Clock
from hardware_voice_ai.audio_model.gateway import GatewayConfig
from hardware_voice_ai.audio_model.local import LocalAudioModel, LocalEngine, ReferenceAudioEngine
from hardware_voice_ai.audio_model.loop import VoiceLoop
from hardware_voice_ai.audio_model.remote import RemoteAudioModel, SessionOpener
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ModelUnavailable
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.speaker.port import MemorySpeaker, SpeakerPort


def build_local_engine(model_id: str) -> LocalEngine | None:
    """Return the on-device engine, or ``None`` when that id is not on the device.

    Unknown ids are unavailable at turn time. They do not download a model
    and they do not stop the wake loop from starting.
    """
    if model_id == REFERENCE_LOCAL_MODEL_ID:
        return ReferenceAudioEngine()
    return None


def build_voice_loop(
    config: DeviceConfig,
    *,
    speaker: SpeakerPort | None = None,
    session_opener: SessionOpener | None = None,
    clock: Clock | None = None,
) -> VoiceLoop:
    """Build the single loop for ``config.audio_model``.

    ``session_opener`` replaces the gateway connection in tests. The default
    opener imports the TLS transport only when a turn runs and a URL is set.
    """
    sink = speaker if speaker is not None else MemorySpeaker()
    if config.audio_model is AudioModel.LOCAL:
        model = LocalAudioModel(build_local_engine(config.local_model_id))
    else:
        opener = session_opener if session_opener is not None else _gateway_opener(config.gateway)
        model = RemoteAudioModel(opener, token=config.gateway.token)
    return VoiceLoop(
        model,
        sink,
        first_audio_timeout_s=config.first_audio_timeout_s,
        turn_timeout_s=config.turn_timeout_s,
        clock=clock,
    )


def _gateway_opener(gateway: GatewayConfig) -> SessionOpener:
    def open_session():
        if not gateway.url:
            raise ModelUnavailable("remote audio model is unavailable")
        from hardware_voice_ai.audio_model.tls import connect_framed_session

        return connect_framed_session(gateway.url, gateway.token)

    return open_session
