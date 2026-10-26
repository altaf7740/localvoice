"""Speak text from the command line, without the hotkey or menu bar.

Usage: make say TEXT="Kal ki meeting mein timeline discuss karni hai" [SPEED=1.25] [STEPS=4]
   or: uv run python -m scripts.say [--speed 1.25] [--steps 4] "<text>"
"""

import argparse
import time

from src.core.config.settings import Config, add_run_args, exit_if_unsupported, run_overrides
from src.modules.transliterate import services as transliterate
from src.modules.tts import services as tts
from src.shared.audio import Player, pause
from src.shared.text import split_chunks, strip_markup


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(prog="say", description="Speak text with the configured voice.")
    parser.add_argument("text", nargs="*")
    add_run_args(parser)
    args = parser.parse_args()
    text = " ".join(args.text).strip()
    if not text:
        parser.error('no text given (make say TEXT="...")')
    exit_if_unsupported()
    cfg = Config.load(**run_overrides(args))
    # no tts.warmup(): it loads the model like the first sentence would, then spends ~1 s
    # speaking a throwaway word, which here is just a longer wait
    player = Player()
    player.start(cfg.output_device)
    first = None
    chunks = split_chunks(strip_markup(text) if cfg.strip_markup else text)
    speakable = transliterate.prepare_chunks(chunks, cfg)
    for i, (_, ends_sentence) in enumerate(chunks):
        prepared = next(speakable)
        print(prepared, flush=True)
        for audio in tts.synthesize(prepared, cfg):
            first = first or time.time() - t0
            player.feed(audio)
        if i < len(chunks) - 1:
            player.feed(pause(ends_sentence, cfg.speed))
    print(f"[first audio {first:.2f}s after launch]", flush=True)
    player.finish()


if __name__ == "__main__":
    main()
