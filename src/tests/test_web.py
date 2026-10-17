import base64

import numpy as np

from scripts import web
from scripts.web import options, request_config, to_pcm16
from src.core.config.settings import ENGINE_VOICES, Config


def test_to_pcm16_is_little_endian_16_bit_and_clips():
    pcm = np.frombuffer(base64.b64decode(to_pcm16(np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32))), dtype="<i2")
    assert pcm.tolist() == [0, 16383, 32767, -32767]


def test_speak_streams_each_chunk_then_stats(monkeypatch):
    chunks = [np.zeros(2400, dtype=np.float32), np.zeros(4800, dtype=np.float32)]
    monkeypatch.setattr(web.tts, "synthesize", lambda text, cfg: iter(chunks))
    messages = list(web.speak("नमस्ते", Config()))
    assert [len(base64.b64decode(m["audio"])) for m in messages[:2]] == [4800, 9600]
    assert messages[2]["stats"]["audio"] == 0.3


def test_closing_speak_early_stops_synthesis_and_frees_the_lock(monkeypatch):
    closed = []

    def synthesize(text, cfg):
        try:
            while True:
                yield np.zeros(10, dtype=np.float32)
        finally:
            closed.append(True)

    monkeypatch.setattr(web.tts, "synthesize", synthesize)
    messages = web.speak("नमस्ते", Config())
    next(messages)
    messages.close()
    assert closed and not web._tts_lock.locked()


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
