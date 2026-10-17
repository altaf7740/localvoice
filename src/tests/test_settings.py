import argparse
import json
import sys

import pytest

from src.core.config import settings
from src.core.config.settings import VOICES, Config, add_run_args, run_overrides


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "missing.json")
    cfg = Config.load()
    assert cfg.voice in VOICES
    assert cfg.speed == 1.0


def test_config_file_overrides_and_ignores_unknown_keys(tmp_path, monkeypatch):
    path = tmp_path / "localvoice.json"
    path.write_text(json.dumps({"hotkey": "cmd_r", "voice": "other_voice", "not_a_field": 1}))
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    cfg = Config.load()
    assert cfg.hotkey == "cmd_r"
    assert cfg.voice == "other_voice"
    assert not hasattr(cfg, "not_a_field")


def test_save_persists_only_given_keys_and_keeps_existing(tmp_path, monkeypatch):
    path = tmp_path / "localvoice.json"
    path.write_text(json.dumps({"hotkey": "cmd_r"}))
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    cfg = Config.load()
    cfg.speed = 1.25
    cfg.voice = "other_voice"
    cfg.save("speed")
    assert json.loads(path.read_text()) == {"hotkey": "cmd_r", "speed": 1.25}


def test_overrides_beat_config_file_and_none_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "localvoice.json"
    path.write_text(json.dumps({"speed": 1.25, "voxcpm2_steps": 8}))
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    cfg = Config.load(speed=1.4, voxcpm2_steps=None)
    assert cfg.speed == 1.4
    assert cfg.voxcpm2_steps == 8


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    return parser.parse_args(argv)


def test_run_args():
    args = _parse("--speed", "1.3", "--steps", "4")
    assert (args.speed, args.steps) == (1.3, 4)
    assert (_parse().speed, _parse().steps) == (None, None)


def test_steps_flag_sets_voxcpm2_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "missing.json")
    assert Config.load(**run_overrides(_parse("--steps", "5"))).voxcpm2_steps == 5
    assert Config.load(**run_overrides(_parse())).voxcpm2_steps == 4


@pytest.mark.parametrize("argv", [("--speed", "3"), ("--speed", "fast"), ("--steps", "1"), ("--steps", "12")])
def test_run_args_reject_out_of_range(argv):
    with pytest.raises(SystemExit):
        _parse(*argv)


def test_cuda_check_skips_macos_and_missing_torch(monkeypatch):
    monkeypatch.setattr(settings.sys, "platform", "darwin")
    assert settings._has_cuda() is False
    monkeypatch.setattr(settings.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch raises ImportError
    assert settings._has_cuda() is False
