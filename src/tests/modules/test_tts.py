import threading

import numpy as np
import pytest

from src.core.config.settings import VOICES, Config
from src.modules.tts import services


def test_empty_text_yields_nothing(monkeypatch):
    monkeypatch.setattr(services, "_synthesize", lambda text, voice, cfg: 1 / 0)
    assert list(services.synthesize("   ", Config())) == []


def test_resolve_voice_falls_back_to_the_default(monkeypatch):
    monkeypatch.setattr(services, "VOICES", ["indian_male", "other"])
    assert services.resolve_voice(Config(voice="other")) == "other"
    assert services.resolve_voice(Config(voice="missing")) == "indian_male"


def test_every_voice_has_a_reference_clip_and_transcript():
    for voice in VOICES:
        assert (services.VOICES_DIR / f"{voice}.wav").is_file()
        assert (services.VOICES_DIR / f"{voice}.txt").read_text().strip()


def test_mlx_work_runs_on_one_thread_whichever_thread_asks(monkeypatch):
    threads = []

    def fake_engine(text, voice, cfg):
        threads.append(threading.get_ident())
        yield np.zeros(2, dtype=np.float32)

    monkeypatch.setattr(services, "APPLE_SILICON", True)
    monkeypatch.setattr(services, "_synthesize", fake_engine)
    cfg = Config()
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
    cfg = Config()
    monkeypatch.setattr(services, "_synthesize", endless)
    gen = services.synthesize("नमस्ते", cfg)
    next(gen)
    gen.close()
    # the endless generator stopped, or the MLX thread would never be free again
    services._mlx_thread.submit(lambda: None).result(timeout=5)
    assert made
    monkeypatch.setattr(services, "_synthesize", broken)
    with pytest.raises(RuntimeError, match="model failed"):
        list(services.synthesize("नमस्ते", cfg))


def test_unload_frees_the_model_and_voices_but_not_while_a_request_runs(monkeypatch):
    monkeypatch.setattr(services, "APPLE_SILICON", False)
    monkeypatch.setitem(__import__("sys").modules, "torch", type("T", (), {"cuda": type("C", (), {"empty_cache": staticmethod(lambda: None)})})())
    monkeypatch.setattr(services, "_backend", ("backend", "model"))
    monkeypatch.setitem(services._voices, "indian_male", "voice")
    assert services.loaded()
    with services._use:
        assert services.unload() is False and services.loaded()
    assert services.unload() is True
    assert not services.loaded() and services._voices == {}
    assert services.unload() is False  # already unloaded
