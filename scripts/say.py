"""Speak text from the command line, without the hotkey or menu bar.

Usage: make say TEXT="Kal ki meeting mein timeline discuss karni hai" [SPEED=1.25] [STEPS=4]
   or: uv run python -m scripts.say [--speed 1.25] [--steps 4] "<text>"
"""

import argparse
import time

from src.core.config.settings import Config, add_run_args
from src.modules.transliterate import services as transliterate
from src.modules.tts import services as tts
from src.shared.audio import Player
from src.shared.text import split_sentences


def main():
    parser = argparse.ArgumentParser(prog="say", description="Speak text with the configured voice.")
    parser.add_argument("text", nargs="*")
    add_run_args(parser)
    args = parser.parse_args()
    text = " ".join(args.text).strip()
    if not text:
        parser.error('no text given (make say TEXT="...")')
    cfg = Config.load(speed=args.speed, voxtral_steps=args.voxtral_steps)
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
