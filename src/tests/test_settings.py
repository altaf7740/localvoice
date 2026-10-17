import argparse
import json

import pytest

from src.core.config import settings
from src.core.config.settings import ENGINE_VOICES, Config, add_run_args, run_overrides


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "missing.json")
    cfg = Config.load()
    assert cfg.voice in ENGINE_VOICES[cfg.tts_engine]
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


def test_overrides_beat_config_file_and_none_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "localvoice.json"
    path.write_text(json.dumps({"speed": 1.25, "voxtral_steps": 8}))
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    cfg = Config.load(speed=1.4, voxtral_steps=None)
    assert cfg.speed == 1.4
    assert cfg.voxtral_steps == 8


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    return parser.parse_args(argv)


def test_run_args():
    args = _parse("--speed", "1.3", "--steps", "4")
    assert (args.speed, args.steps) == (1.3, 4)
    assert (_parse().speed, _parse().steps) == (None, None)


def test_steps_flag_applies_to_both_diffusion_engines(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "missing.json")
    cfg = Config.load(**run_overrides(_parse("--steps", "5")))
    assert (cfg.voxcpm2_steps, cfg.voxtral_steps) == (5, 5)
    cfg = Config.load(**run_overrides(_parse()))
    assert (cfg.voxcpm2_steps, cfg.voxtral_steps) == (4, 6)


@pytest.mark.parametrize("argv", [("--speed", "3"), ("--speed", "fast"), ("--steps", "1"), ("--steps", "12")])
def test_run_args_reject_out_of_range(argv):
    with pytest.raises(SystemExit):
        _parse(*argv)
