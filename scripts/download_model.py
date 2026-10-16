"""Download Voxtral (Apple Silicon), Kokoro-82M and their male Hindi voices into the HuggingFace cache.

Usage: make model
   or: uv run python -m scripts.download_model
"""

from dataclasses import replace

from src.core.config.settings import ENGINE_VOICES, Config
from src.modules.tts import services as tts


def main():
    cfg = Config.load()
    for engine, voices in ENGINE_VOICES.items():
        for voice in voices:
            tts.warmup(replace(cfg, tts_engine=engine, voice=voice))
            print(f"[model] {engine} {voice} ready", flush=True)


if __name__ == "__main__":
    main()
