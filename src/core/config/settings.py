import json
import sys
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".localvoice.json"
BASE_DIR = Path(__file__).resolve().parents[2]  # src/
STATIC_DIR = BASE_DIR / "static"

# Kokoro-82M's male Hindi voices (its hf_alpha / hf_beta are female and left out)
VOICES = ["hm_omega", "hm_psi"]
SPEEDS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]


@dataclass
class Config:
    # "auto" = MLX on Apple Silicon, PyTorch (CUDA or CPU) everywhere else
    tts_backend: str = "auto"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    torch_model: str = "hexgrad/Kokoro-82M"
    voice: str = "hm_omega"
    speed: float = 1.0
    # Kokoro pronounces Hindi only from Devanagari. Roman Hindi words go through the
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
