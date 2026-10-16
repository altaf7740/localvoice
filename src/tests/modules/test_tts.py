from src.core.config.settings import Config
from src.modules.tts import services


def test_empty_text_yields_nothing(monkeypatch):
    for name in ("_synthesize_voxtral", "_synthesize_kokoro_mlx", "_synthesize_kokoro_torch"):
        monkeypatch.setattr(services, name, lambda text, voice, cfg: 1 / 0)
    assert list(services.synthesize("   ", Config())) == []


def test_kokoro_backend_selection(monkeypatch):
    monkeypatch.setattr(services, "APPLE_SILICON", True)
    assert services._kokoro_on_mlx(Config())
    assert not services._kokoro_on_mlx(Config(kokoro_backend="torch"))
    monkeypatch.setattr(services, "APPLE_SILICON", False)
    assert not services._kokoro_on_mlx(Config())
    assert services._kokoro_on_mlx(Config(kokoro_backend="mlx"))


def test_resolve_voice_keeps_valid_pairs(monkeypatch):
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxtral": ["hi_male"], "kokoro": ["hm_omega", "hm_psi"]})
    assert services.resolve_voice(Config(tts_engine="voxtral", voice="hi_male")) == ("voxtral", "hi_male")
    assert services.resolve_voice(Config(tts_engine="kokoro", voice="hm_psi")) == ("kokoro", "hm_psi")


def test_resolve_voice_fixes_mismatched_voice(monkeypatch):
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxtral": ["hi_male"], "kokoro": ["hm_omega", "hm_psi"]})
    assert services.resolve_voice(Config(tts_engine="voxtral", voice="hm_psi")) == ("voxtral", "hi_male")
    assert services.resolve_voice(Config(tts_engine="kokoro", voice="hf_alpha")) == ("kokoro", "hm_omega")


def test_voxtral_falls_back_to_kokoro_where_it_cannot_run(monkeypatch):
    monkeypatch.setattr(services, "ENGINE_VOICES", {"kokoro": ["hm_omega", "hm_psi"]})
    assert services.resolve_voice(Config(tts_engine="voxtral", voice="hi_male")) == ("kokoro", "hm_omega")


def test_dispatches_to_configured_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(services, "ENGINE_VOICES", {"voxtral": ["hi_male"], "kokoro": ["hm_omega", "hm_psi"]})
    monkeypatch.setattr(services, "APPLE_SILICON", True)
    monkeypatch.setattr(
        services, "_synthesize_voxtral", lambda text, voice, cfg: calls.append(("voxtral", voice)) or []
    )
    monkeypatch.setattr(
        services, "_synthesize_kokoro_mlx", lambda text, voice, cfg: calls.append(("kokoro", voice)) or []
    )
    list(services.synthesize("नमस्ते", Config(tts_engine="voxtral", voice="hi_male")))
    list(services.synthesize("नमस्ते", Config(tts_engine="kokoro", voice="hm_psi")))
    assert calls == [("voxtral", "hi_male"), ("kokoro", "hm_psi")]
