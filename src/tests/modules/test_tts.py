import threading

import numpy as np
import pytest

from src.core.config.settings import Config
from src.modules.tts import services


def test_empty_text_yields_nothing(monkeypatch):
    for name in ("_synthesize_voxcpm2", "_synthesize_kokoro_mlx", "_synthesize_kokoro_torch"):
        monkeypatch.setattr(services, name, lambda text, voice, cfg: 1 / 0)
    assert list(services.synthesize("   ", Config())) == []


def test_resolve_voice_keeps_valid_pairs(monkeypatch):
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxcpm2": ["indian_male"], "kokoro": ["hm_omega", "hm_psi"]})
    assert services.resolve_voice(Config(tts_engine="voxcpm2", voice="indian_male")) == ("voxcpm2", "indian_male")
    assert services.resolve_voice(Config(tts_engine="kokoro", voice="hm_psi")) == ("kokoro", "hm_psi")


def test_resolve_voice_fixes_mismatched_voice(monkeypatch):
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxcpm2": ["indian_male"], "kokoro": ["hm_omega", "hm_psi"]})
    assert services.resolve_voice(Config(tts_engine="voxcpm2", voice="hm_psi")) == ("voxcpm2", "indian_male")
    assert services.resolve_voice(Config(tts_engine="kokoro", voice="hf_alpha")) == ("kokoro", "hm_omega")


def test_voxcpm2_falls_back_to_kokoro_where_it_cannot_run(monkeypatch):
    monkeypatch.setattr(services, "ENGINE_VOICES", {"kokoro": ["hm_omega", "hm_psi"]})
    assert services.resolve_voice(Config(tts_engine="voxcpm2", voice="indian_male")) == ("kokoro", "hm_omega")


def test_every_voxcpm2_voice_has_a_reference_clip_and_transcript():
    for voice in ["indian_male"]:
        assert (services.VOICES_DIR / f"{voice}.wav").is_file()
        assert (services.VOICES_DIR / f"{voice}.txt").read_text().strip()


def test_dispatches_to_configured_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxcpm2": ["indian_male"], "kokoro": ["hm_omega", "hm_psi"]})
    monkeypatch.setattr(services, "APPLE_SILICON", True)
    monkeypatch.setattr(
        services, "_synthesize_voxcpm2", lambda text, voice, cfg: calls.append(("voxcpm2", voice)) or []
    )
    monkeypatch.setattr(
        services, "_synthesize_kokoro_mlx", lambda text, voice, cfg: calls.append(("kokoro", voice)) or []
    )
    list(services.synthesize("नमस्ते", Config(tts_engine="voxcpm2", voice="indian_male")))
    list(services.synthesize("नमस्ते", Config(tts_engine="kokoro", voice="hm_psi")))
    assert calls == [("voxcpm2", "indian_male"), ("kokoro", "hm_psi")]


def test_mlx_work_runs_on_one_thread_whichever_thread_asks(monkeypatch):
    threads = []

    def fake_engine(text, voice, cfg):
        threads.append(threading.get_ident())
        yield np.zeros(2, dtype=np.float32)

    monkeypatch.setattr(services, "APPLE_SILICON", True)
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxcpm2": ["indian_male"], "kokoro": ["hm_omega"]})
    monkeypatch.setattr(services, "_synthesize_voxcpm2", fake_engine)
    cfg = Config(tts_engine="voxcpm2", voice="indian_male")
    for _ in range(3):
        t = threading.Thread(target=lambda: list(services.synthesize("नमस्ते", cfg)))
        t.start()
        t.join()
    assert len(threads) == 3 and len(set(threads)) == 1 and threads[0] != threading.get_ident()


def test_mlx_thread_passes_errors_through_and_stops_when_the_caller_does(monkeypatch):
    made = []

    def endless(text, voice, cfg):
        while True:
            made.append(1)
            yield np.zeros(2, dtype=np.float32)

    def broken(text, voice, cfg):
        raise RuntimeError("model failed")
        yield

    monkeypatch.setattr(services, "APPLE_SILICON", True)
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxcpm2": ["indian_male"], "kokoro": ["hm_omega"]})
    cfg = Config(tts_engine="voxcpm2", voice="indian_male")
    monkeypatch.setattr(services, "_synthesize_voxcpm2", endless)
    gen = services.synthesize("नमस्ते", cfg)
    next(gen)
    gen.close()
    # the endless generator stopped, or the MLX thread would never be free again
    services._mlx_thread.submit(lambda: None).result(timeout=5)
    assert made
    monkeypatch.setattr(services, "_synthesize_voxcpm2", broken)
    with pytest.raises(RuntimeError, match="model failed"):
        list(services.synthesize("नमस्ते", cfg))
