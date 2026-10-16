"""Download Kokoro-82M and the male Hindi voices into the HuggingFace cache.

Usage: make model
   or: uv run python -m scripts.download_model
"""

from dataclasses import replace

from src.core.config.settings import VOICES, Config
from src.modules.tts import services as tts


def main():
    cfg = Config.load()
    for voice in VOICES:
        tts.warmup(replace(cfg, voice=voice))
        print(f"[model] {voice} ready", flush=True)


if __name__ == "__main__":
    main()
