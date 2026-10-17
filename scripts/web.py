"""Try VoxCPM2 in a browser: type text, pick a voice, hear it, see the transliteration and timings.

Usage: make web [SPEED=1.25] [STEPS=4]   (then open http://localhost:8765)
   or: uv run python -m scripts.web [--port 8765] [--speed 1.25] [--steps 4]
"""

import argparse
import base64
import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import numpy as np

from src.core.config.settings import VOICES, Config, add_run_args, exit_if_unsupported, run_overrides
from src.modules.transliterate import services as transliterate
from src.modules.tts import services as tts
from src.shared.text import split_sentences

PAGE = Path(__file__).with_name("web.html")
MAX_TEXT = 5000  # characters per request

# one synthesis at a time: there's a single loaded model and it isn't thread-safe
_tts_lock = threading.Lock()


def to_pcm16(audio: np.ndarray) -> str:
    """Base64 of 16-bit little-endian mono samples, which the page decodes and plays."""
    return base64.b64encode((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()).decode()


def request_config(base: Config, body: dict) -> Config:
    """`base` with the voice, speed and steps chosen on the page."""
    cfg = dataclasses.replace(base)
    cfg.voice = body["voice"] if body.get("voice") in VOICES else tts.resolve_voice(cfg)
    if "speed" in body:
        cfg.speed = min(max(float(body["speed"]), 0.5), 2.0)
    if "steps" in body:
        cfg.voxcpm2_steps = min(max(int(body["steps"]), 2), 10)
    return cfg


def options(cfg: Config) -> dict:
    cfg = request_config(cfg, {})
    return {
        "voices": VOICES,
        "sample_rate": tts.SAMPLE_RATE,
        "defaults": {
            "voice": cfg.voice,
            "speed": cfg.speed,
            "steps": cfg.voxcpm2_steps,
        },
    }


def prepare(text: str) -> list[dict]:
    """Split into sentences and transliterate each."""
    return [{"original": s, "prepared": transliterate.to_devanagari(s)} for s in split_sentences(text[:MAX_TEXT])]


def speak(text: str, cfg: Config) -> Iterator[dict]:
    """One (already transliterated) sentence as {"audio": …} messages, sent as each chunk is made,
    then {"stats": …} with how long it took. Closing it early stops the synthesis."""
    with _tts_lock:
        t0 = time.time()
        first = None
        samples = 0
        for chunk in tts.synthesize(text[:MAX_TEXT], cfg):
            first = first or time.time() - t0
            samples += len(chunk)
            yield {"audio": to_pcm16(chunk)}
        total = time.time() - t0
    seconds = samples / tts.SAMPLE_RATE
    yield {
        "stats": {
            "first_chunk": round(first or 0, 3),
            "synth": round(total, 3),
            "audio": round(seconds, 3),
            "rtf": round(total / seconds, 3) if seconds else None,
        }
    }


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
                self._json({"sentences": prepare(text)})
            elif self.path == "/api/speak":
                self._stream(speak(text, cfg))
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self, messages: Iterator[dict]):
            """One JSON object per line, each written as soon as it's ready; the body ends when
            the connection closes (HTTP/1.0)."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for message in messages:
                    self.wfile.write(json.dumps(message).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page pressed Stop; closing `messages` below stops the synthesis
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode() + b"\n")
            finally:
                messages.close()

        def _json(self, data: dict, status: int = 200):
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
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
    exit_if_unsupported()
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
