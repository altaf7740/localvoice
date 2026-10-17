"""Try VoxCPM2 in a browser: type text, pick a voice, hear it, see the transliteration and timings.

Usage: make web [SPEED=1.25] [STEPS=4]   (then open http://localhost:8765)
   or: uv run python -m scripts.web [--port 8765] [--speed 1.25] [--steps 4]
"""

import argparse
import dataclasses
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from src.core.config.settings import ENGINE_VOICES, SPEEDS, Config, add_run_args, run_overrides
from src.modules.transliterate import services as transliterate
from src.modules.tts import services as tts
from src.shared.text import split_sentences

PAGE = Path(__file__).with_name("web.html")
MAX_TEXT = 5000  # characters per request
ENGINE = "voxcpm2"  # the only engine this UI offers

# one synthesis at a time: the engines keep a single loaded model and aren't thread-safe
_tts_lock = threading.Lock()


def to_wav(audio: np.ndarray, sample_rate: int = tts.SAMPLE_RATE) -> bytes:
    """16-bit mono PCM WAV, which every browser plays."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()


def request_config(base: Config, body: dict) -> Config:
    """`base` switched to VoxCPM2, with the voice, speed, steps and AI-transliteration choices from the page."""
    cfg = dataclasses.replace(base, tts_engine=ENGINE)
    voices = ENGINE_VOICES[ENGINE]
    cfg.voice = body.get("voice") if body.get("voice") in voices else (cfg.voice if cfg.voice in voices else voices[0])
    if "speed" in body:
        cfg.speed = min(max(float(body["speed"]), 0.5), 2.0)
    if "steps" in body:
        cfg.voxcpm2_steps = min(max(int(body["steps"]), 2), 10)
    if "transliterate_llm" in body:
        cfg.transliterate_llm = bool(body["transliterate_llm"])
    return cfg


def options(cfg: Config) -> dict:
    cfg = request_config(cfg, {})
    return {
        "voices": ENGINE_VOICES[ENGINE],
        "speeds": SPEEDS,
        "defaults": {
            "voice": cfg.voice,
            "speed": cfg.speed,
            "steps": cfg.voxcpm2_steps,
            "transliterate_llm": cfg.transliterate_llm,
        },
    }


def prepare(text: str, cfg: Config) -> list[dict]:
    """Split into sentences and transliterate each, timing the transliteration."""
    out = []
    for sentence in split_sentences(text[:MAX_TEXT]):
        t0 = time.time()
        prepared = transliterate.to_devanagari(sentence, cfg)
        out.append({"original": sentence, "prepared": prepared, "prep_seconds": round(time.time() - t0, 3)})
    return out


def speak(text: str, cfg: Config) -> tuple[bytes, dict]:
    """WAV of one (already transliterated) sentence, and how long it took."""
    with _tts_lock:
        t0 = time.time()
        first = None
        chunks = []
        for chunk in tts.synthesize(text[:MAX_TEXT], cfg):
            first = first or time.time() - t0
            chunks.append(chunk)
        total = time.time() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    seconds = len(audio) / tts.SAMPLE_RATE
    stats = {
        "first_chunk": round(first or 0, 3),
        "synth": round(total, 3),
        "audio": round(seconds, 3),
        "rtf": round(total / seconds, 3) if seconds else None,
    }
    return to_wav(audio), stats


def warmup(cfg: Config):
    """Load the model and voice at startup, so the first Speak doesn't wait ~4 s for them.
    Requests that arrive meanwhile queue on the lock instead of loading it a second time."""
    t0 = time.time()
    with _tts_lock:
        tts.warmup(cfg)
    print(f"[web] ready in {time.time() - t0:.1f}s", flush=True)


def make_handler(base: Config):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/options":
                self._json(options(base))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                cfg = request_config(base, body)
                text = str(body.get("text", "")).strip()
            except (ValueError, TypeError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            if not text:
                return self._json({"error": "no text"}, 400)
            if self.path == "/api/prepare":
                self._json({"sentences": prepare(text, cfg)})
            elif self.path == "/api/speak":
                wav, stats = speak(text, cfg)
                self._send(200, wav, "audio/wav", {"X-Stats": json.dumps(stats)})
            else:
                self._send(404, b"not found", "text/plain")

        def _json(self, data: dict, status: int = 200):
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # the pipeline prints its own [tts] lines

    return Handler


def main():
    parser = argparse.ArgumentParser(prog="web", description="Test LocalVoice's VoxCPM2 voice in the browser.")
    parser.add_argument("--port", type=int, default=8765)
    add_run_args(parser)
    args = parser.parse_args()
    if ENGINE not in ENGINE_VOICES:
        parser.exit(1, "VoxCPM2 needs Apple Silicon or an NVIDIA GPU, and this machine has neither\n")
    cfg = Config.load(**run_overrides(args))
    # localhost only: nothing leaves the machine
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cfg))
    except OSError as e:
        parser.exit(1, f"can't listen on port {args.port} ({e.strerror}): is the UI already running? Try --port\n")
    print(f"LocalVoice test UI on http://localhost:{args.port} (Ctrl+C to quit)", flush=True)
    threading.Thread(target=warmup, args=(request_config(cfg, {}),), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
