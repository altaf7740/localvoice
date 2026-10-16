import json

from src.core.config import settings
from src.core.config.settings import VOICES, Config


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "missing.json")
    cfg = Config.load()
    assert cfg.voice in VOICES
    assert cfg.speed == 1.0
    assert cfg.transliterate_llm is True


def test_config_file_overrides_and_ignores_unknown_keys(tmp_path, monkeypatch):
    path = tmp_path / "localvoice.json"
    path.write_text(json.dumps({"hotkey": "cmd_r", "voice": "hm_omega", "not_a_field": 1}))
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    cfg = Config.load()
    assert cfg.hotkey == "cmd_r"
    assert cfg.voice == "hm_omega"
    assert not hasattr(cfg, "not_a_field")


def test_save_persists_only_given_keys_and_keeps_existing(tmp_path, monkeypatch):
    path = tmp_path / "localvoice.json"
    path.write_text(json.dumps({"hotkey": "cmd_r"}))
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    cfg = Config.load()
    cfg.speed = 1.25
    cfg.voice = "hm_psi"
    cfg.save("speed")
    assert json.loads(path.read_text()) == {"hotkey": "cmd_r", "speed": 1.25}
