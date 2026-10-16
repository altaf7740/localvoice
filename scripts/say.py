"""Speak text from the command line, without the hotkey or menu bar.

Usage: make say TEXT="Kal ki meeting mein timeline discuss karni hai"
   or: uv run python -m scripts.say "<text>"
"""

import sys
import time

from src.core.config.settings import Config
from src.modules.transliterate import services as transliterate
from src.modules.tts import services as tts
from src.shared.audio import Player
from src.shared.text import split_sentences


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        sys.exit(__doc__)
    cfg = Config.load()
    tts.warmup(cfg)
    player = Player()
    player.start(cfg.output_device)
    t0 = time.time()
    first = None
    for sentence in split_sentences(text):
        prepared = transliterate.to_devanagari(sentence, cfg)
        print(prepared, flush=True)
        for audio in tts.synthesize(prepared, cfg):
            first = first or time.time() - t0
            player.feed(audio)
    print(f"[first audio after {first:.2f}s]", flush=True)
    player.finish()


if __name__ == "__main__":
    main()
