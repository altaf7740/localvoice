import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".localvoice.json"
BASE_DIR = Path(__file__).resolve().parents[2]  # src/
STATIC_DIR = BASE_DIR / "static"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

# Male Hindi voices per engine (Kokoro's hf_alpha / hf_beta and Voxtral's hi_female are left out).
# Voxtral runs on MLX only, so other machines get Kokoro alone.
ENGINE_VOICES = {"voxtral": ["hi_male"], "kokoro": ["hm_omega", "hm_psi"]}
if not APPLE_SILICON:
    del ENGINE_VOICES["voxtral"]
SPEEDS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]


@dataclass
class Config:
    # "voxtral": natural Hinglish, ~0.7 s to first audio, CC BY-NC (non-commercial) weights.
    # "kokoro": flatter but ~0.2 s to first audio, Apache 2.0, runs everywhere.
    tts_engine: str = "voxtral" if APPLE_SILICON else "kokoro"
    voice: str = "hi_male" if APPLE_SILICON else "hm_omega"
    # Kokoro speeds up natively; Voxtral's output is time-stretched (pitch unchanged)
    speed: float = 1.1
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
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self, *keys: str):
        """Persist the given fields to ~/.localvoice.json, keeping whatever else is in the file."""
        data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        for key in keys:
            data[key] = getattr(self, key)
        CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
