from src.core.config.settings import Config
from src.modules.tts import services


def test_empty_text_yields_nothing(monkeypatch):
    monkeypatch.setattr(services, "_synthesize_mlx", lambda text, cfg: 1 / 0)
    monkeypatch.setattr(services, "_synthesize_torch", lambda text, cfg: 1 / 0)
    assert list(services.synthesize("   ", Config())) == []


def test_backend_selection(monkeypatch):
    monkeypatch.setattr(services, "APPLE_SILICON", True)
    assert services._use_mlx(Config())
    assert not services._use_mlx(Config(tts_backend="torch"))
    monkeypatch.setattr(services, "APPLE_SILICON", False)
    assert not services._use_mlx(Config())
    assert services._use_mlx(Config(tts_backend="mlx"))
