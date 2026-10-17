import io
import wave

import numpy as np

from scripts.web import options, request_config, to_wav
from src.core.config.settings import ENGINE_VOICES, Config


def test_to_wav_is_16_bit_mono_and_clips():
    wav = to_wav(np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32), 24000)
    with wave.open(io.BytesIO(wav)) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) == (1, 2, 24000, 4)
        samples = np.frombuffer(f.readframes(4), dtype="<i2")
    assert samples.tolist() == [0, 16383, 32767, -32767]


def test_request_config_is_always_voxcpm2_and_clamps():
    base = Config(tts_engine="kokoro", voice="hm_omega")
    cfg = request_config(base, {"engine": "kokoro", "voice": "hm_psi", "speed": 9, "steps": 1})
    assert (cfg.tts_engine, cfg.voice) == ("voxcpm2", ENGINE_VOICES["voxcpm2"][0])
    assert (cfg.speed, cfg.voxcpm2_steps) == (2.0, 2)
    assert (base.tts_engine, base.speed) == ("kokoro", 1.0)  # the server's config is left alone


def test_options_offer_only_voxcpm2_voices():
    opts = options(Config(tts_engine="kokoro", voice="hm_omega"))
    assert opts["voices"] == ENGINE_VOICES["voxcpm2"]
    assert opts["defaults"]["voice"] == ENGINE_VOICES["voxcpm2"][0]
