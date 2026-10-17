"""Download VoxCPM2 and prepare its voices.

Usage: make model
   or: uv run python -m scripts.download_model
"""

from dataclasses import replace

from src.core.config.settings import VOICES, Config, exit_if_unsupported
from src.modules.tts import services as tts


def main():
    exit_if_unsupported()
    cfg = Config.load()
    for voice in VOICES:
        tts.warmup(replace(cfg, voice=voice))
        print(f"[model] {voice} ready", flush=True)


if __name__ == "__main__":
    main()
