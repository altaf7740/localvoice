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


def _has_cuda() -> bool:
    """An NVIDIA GPU PyTorch can use. VoxCPM2 is installed with PyTorch only on Windows and Linux."""
    if sys.platform not in ("win32", "linux"):
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


CUDA = _has_cuda()

# Male voices per engine (Kokoro's hf_alpha / hf_beta are left out). VoxCPM2's indian_male is
# cloned from static/voices/indian_male.wav. VoxCPM2 runs on MLX (Apple Silicon) or an NVIDIA GPU
# (it's far too slow for a CPU); elsewhere only Kokoro.
ENGINE_VOICES = {"voxcpm2": ["indian_male"], "kokoro": ["hm_omega", "hm_psi"]}
if not (APPLE_SILICON or CUDA):
    del ENGINE_VOICES["voxcpm2"]
ENGINE_NAMES = {"voxcpm2": "VoxCPM2", "kokoro": "Kokoro"}  # for menus
SPEEDS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]


@dataclass
class Config:
    # "voxcpm2": deep Indian male voice, Apache 2.0, ~0.8 s to first audio; only just keeps up
    #   with 1× playback on an M1 Pro, ~3× faster on an RTX 4090.
    # "kokoro": flatter but ~0.2 s to first audio, Apache 2.0, runs everywhere.
    tts_engine: str = "voxcpm2" if APPLE_SILICON or CUDA else "kokoro"
    voice: str = "indian_male" if APPLE_SILICON or CUDA else "hm_omega"
    # Kokoro speeds up natively; VoxCPM2 output is time-stretched (pitch unchanged)
    speed: float = 1.0
    voxcpm2_model: str = "mlx-community/VoxCPM2-8bit"  # Apple Silicon
    voxcpm2_torch_model: str = "openbmb/VoxCPM2"  # NVIDIA GPUs (bf16, ~8 GB of VRAM)
    # diffusion steps per 0.16 s patch (model default 10). 4 sounds as clean as 10 in Whisper
    # round-trips and is the most an M1 Pro can run at 1× speed; 3 garbles words.
    voxcpm2_steps: int = 4
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"  # Apple Silicon
    kokoro_torch_model: str = "hexgrad/Kokoro-82M"  # elsewhere (CUDA or CPU)
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
    parser.add_argument("--steps", type=_steps, help="VoxCPM2 diffusion steps, 2–10: fewer = faster, rougher")


def run_overrides(args: argparse.Namespace) -> dict:
    """Config.load() overrides from add_run_args()."""
    return {"speed": args.speed, "voxcpm2_steps": args.steps}


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
