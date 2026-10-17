import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".localvoice.json"
BASE_DIR = Path(__file__).resolve().parents[2]  # src/
STATIC_DIR = BASE_DIR / "static"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

# Male voices per engine (Kokoro's hf_alpha / hf_beta and Voxtral's hi_female are left out).
# VoxCPM2's indian_male is cloned from static/voices/indian_male.wav. VoxCPM2 and Voxtral run on
# MLX only, so other machines get Kokoro alone.
ENGINE_VOICES = {"voxcpm2": ["indian_male"], "voxtral": ["hi_male"], "kokoro": ["hm_omega", "hm_psi"]}
if not APPLE_SILICON:
    del ENGINE_VOICES["voxcpm2"], ENGINE_VOICES["voxtral"]
ENGINE_NAMES = {"voxcpm2": "VoxCPM2", "voxtral": "Voxtral", "kokoro": "Kokoro"}  # for menus
SPEEDS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]


@dataclass
class Config:
    # "voxcpm2": deep Indian male voice, Apache 2.0, ~1.5 s to first audio; only just keeps up
    #   with 1× playback on an M1 Pro.
    # "voxtral": natural Hinglish, ~0.7 s to first audio, CC BY-NC (non-commercial) weights.
    # "kokoro": flatter but ~0.2 s to first audio, Apache 2.0, runs everywhere.
    tts_engine: str = "voxcpm2" if APPLE_SILICON else "kokoro"
    voice: str = "indian_male" if APPLE_SILICON else "hm_omega"
    # Kokoro speeds up natively; VoxCPM2 and Voxtral output is time-stretched (pitch unchanged)
    speed: float = 1.0
    voxcpm2_model: str = "mlx-community/VoxCPM2-8bit"
    # diffusion steps per 0.16 s patch (model default 10). 4 sounds as clean as 10 in Whisper
    # round-trips and is the most an M1 Pro can run at 1× speed; 3 garbles words.
    voxcpm2_steps: int = 4
    voxtral_model: str = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
    # flow-matching steps per 80 ms frame (model default 8). On an M1 Pro, 8 only just keeps up
    # with 1× playback; 6 sounds the same and generates fast enough for up to ~1.4×.
    voxtral_steps: int = 6
    # "auto" = MLX on Apple Silicon, PyTorch (CUDA or CPU) everywhere else
    kokoro_backend: str = "auto"
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    kokoro_torch_model: str = "hexgrad/Kokoro-82M"
    # Both engines pronounce Hindi best from Devanagari. Roman Hindi words go through the
    # built-in lexicon always, and through the Ollama model first when this is on.
    transliterate_llm: bool = True
    transliterate_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    # pynput key tapped (pressed and released alone) to read the selection or stop.
    # Right Alt is AltGr on many Windows/Linux layouts, so those use Right Shift.
    hotkey: str = "alt_r" if sys.platform == "darwin" else "shift_r"
    # output device name; empty = system default
    output_device: str = ""

    @classmethod
    def load(cls, **overrides) -> "Config":
        """Defaults, then ~/.localvoice.json, then `overrides` that aren't None (command-line flags)."""
        data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**known)

    def save(self, *keys: str):
        """Persist the given fields to ~/.localvoice.json, keeping whatever else is in the file."""
        data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        for key in keys:
            data[key] = getattr(self, key)
        CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def add_run_args(parser: argparse.ArgumentParser):
    """Per-run overrides shared by `make start` and `make say` (SPEED=… STEPS=…)."""
    parser.add_argument("--speed", type=_speed, help="speaking speed, 0.5–2.0 (e.g. 1.25)")
    parser.add_argument("--steps", type=_steps, help="VoxCPM2/Voxtral steps, 2–10: fewer = faster, rougher")


def run_overrides(args: argparse.Namespace) -> dict:
    """Config.load() overrides from add_run_args(); --steps applies to whichever engine is active."""
    return {"speed": args.speed, "voxcpm2_steps": args.steps, "voxtral_steps": args.steps}


def _speed(value: str) -> float:
    speed = float(value)
    if not 0.5 <= speed <= 2.0:
        raise argparse.ArgumentTypeError(f"speed must be between 0.5 and 2.0, got {value}")
    return speed


def _steps(value: str) -> int:
    steps = int(value)
    if not 2 <= steps <= 10:
        raise argparse.ArgumentTypeError(f"steps must be between 2 and 10, got {value}")
    return steps
