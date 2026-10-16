"""Score TTS intelligibility and speed on src/static/eval/sentences.tsv.

Each sentence is synthesised, transcribed back with Whisper (auto language, which writes
Hinglish as Devanagari + Latin English; forcing "hi" garbles pure-English clips) and
compared to the mixed-script reference.

Usage: make eval [ARGS="--llm --save"]
   or: uv run --group eval python -m scripts.run_eval [--llm] [--save]
       --llm   also score the Ollama transliteration pass (needs Ollama running)
       --save  write each clip to src/static/eval/out/<mode>-<n>.wav
"""

import sys
import time
import wave
from dataclasses import replace

import numpy as np

from src.core.config.settings import STATIC_DIR, Config
from src.modules.evaluation.metrics import cer
from src.modules.transliterate import services as transliterate
from src.modules.tts import services as tts

EVAL_DIR = STATIC_DIR / "eval"
WHISPER = "mlx-community/whisper-large-v3-turbo"


def main():
    import mlx_whisper

    rows = [line.split("\t") for line in (EVAL_DIR / "sentences.tsv").read_text().splitlines() if line]
    base = Config.load()
    modes = {
        "raw": lambda text: text,
        "lexicon": transliterate.apply_lexicon,
    }
    if "--llm" in sys.argv:
        llm_cfg = replace(base, transliterate_llm=True)
        modes["llm"] = lambda text: transliterate.to_devanagari(text, llm_cfg)
    save = "--save" in sys.argv
    tts.warmup(base)
    for name, prepare in modes.items():
        total, t_first, rtf = 0.0, 0.0, 0.0
        print(f"\n=== {name} ===")
        for n, (text, ref) in enumerate(rows, 1):
            t0 = time.time()
            prepared = prepare(text)
            chunks, first = [], None
            for chunk in tts.synthesize(prepared, base):
                first = first or time.time() - t0
                chunks.append(chunk)
            audio = np.concatenate(chunks)
            elapsed = time.time() - t0
            heard = mlx_whisper.transcribe(_to_16k(audio), path_or_hf_repo=WHISPER, temperature=0.0)
            score = cer(ref, heard["text"])
            total, t_first, rtf = total + score, t_first + first, rtf + elapsed / (len(audio) / tts.SAMPLE_RATE)
            print(f"{n} CER={score:.2f} first={first:.2f}s\n  in:    {prepared}\n  heard: {heard['text'].strip()}")
            if save:
                _write_wav(EVAL_DIR / "out" / f"{name}-{n}.wav", audio)
        k = len(rows)
        print(f"-> mean CER {total / k:.3f} | first audio {t_first / k:.2f}s | RTF {rtf / k:.3f}")


def _to_16k(audio: np.ndarray) -> np.ndarray:
    steps = np.arange(0, len(audio), tts.SAMPLE_RATE / 16000)
    return np.interp(steps, np.arange(len(audio)), audio).astype(np.float32)


def _write_wav(path, audio: np.ndarray):
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(tts.SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
