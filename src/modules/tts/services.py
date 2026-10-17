import hashlib
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.config.settings import APPLE_SILICON, STATIC_DIR, VOICES, Config
from ...shared.hub import cached
from .resample import HalfRate
from .stretch import TimeStretcher

SAMPLE_RATE = 24000  # VoxCPM2's 48 kHz, halved
VOICES_DIR = STATIC_DIR / "voices"  # per VoxCPM2 voice: a reference clip (.wav) and what it says (.txt)
VOICE_CACHE = Path.home() / ".cache" / "localvoice" / "voices"  # those voices prepared by the MLX backend
# MLX keeps freed GPU buffers for reuse and never hands them back, so VoxCPM2's footprint grew from
# 3 GB of weights to 9+ GB. Capped at 1 GB it stays ~4.3 GB at the same speed (0.5 GB is slower).
MLX_CACHE_LIMIT = 1 << 30

# VoxCPM2's tokenizer loads its config.json as a generic one, and transformers warns that the model
# types differ on every start. Read when transformers is first imported, i.e. when a model loads.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_backend = None  # (backend module, model) once loaded
_voices: dict = {}  # voice -> reference clip already run through the model (prepare_voice)
# MLX runs a model only on the thread that loaded it ("There is no Stream(cpu, 1) in current thread"),
# so all MLX work happens on this one thread, whichever thread asks for speech
_mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_DONE = object()


def synthesize(text: str, cfg: Config) -> Iterator[np.ndarray]:
    """Yield float32 mono chunks at SAMPLE_RATE as VoxCPM2 produces them."""
    if not text.strip():
        return
    chunks = _synthesize(text, resolve_voice(cfg), cfg)
    yield from _on_mlx_thread(chunks) if APPLE_SILICON else chunks


def _on_mlx_thread(chunks: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    """Run the `chunks` generator on the MLX thread, yielding each chunk here as soon as it's made.
    If the caller stops early (e.g. the reader was stopped), generation stops after the current chunk."""
    out: queue.Queue = queue.Queue()
    stopped = threading.Event()

    def pump():
        try:
            for chunk in chunks:
                if stopped.is_set():
                    break
                out.put(chunk)
        except Exception as e:
            out.put(e)
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()  # here, not later on whatever thread garbage-collects it
            out.put(_DONE)

    _mlx_thread.submit(pump)
    try:
        while (item := out.get()) is not _DONE:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stopped.set()


def warmup(cfg: Config):
    """Load the model and voice so the first real request doesn't pay for it. Long enough for
    VoxCPM2 to decode every chunk size once, as compiling each costs ~0.2 s the first time."""
    for _ in synthesize("नमस्ते, आज का दिन बहुत अच्छा है।", cfg):
        pass


def resolve_voice(cfg: Config) -> str:
    """The configured voice, or the default if it doesn't exist."""
    return cfg.voice if cfg.voice in VOICES else VOICES[0]


def _load(cfg: Config):
    """(backend module, model): MLX on Apple Silicon, OpenBMB's PyTorch package on NVIDIA GPUs.
    Both modules offer prepare_voice(model, clip, transcript) and stream(model, text, voice, steps);
    the MLX one's prepare_voice can also save the voice to a file."""
    global _backend
    if _backend is None:
        if APPLE_SILICON:
            import mlx.core as mx
            from mlx_audio.tts.utils import load
            from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS

            from . import voxcpm2 as backend

            mx.set_cache_limit(MLX_CACHE_LIMIT)
            model = load(cached(cfg.voxcpm2_model, DEFAULT_ALLOW_PATTERNS))  # the files mlx-audio downloads
            backend.compile_model(model)
            # every voice now (from its saved file after the first run), so the encoder can go
            for voice in VOICES:
                _voices[voice] = _prepare_voice(backend, model, voice, cfg)
            backend.drop_encoder(model)
            print(f"[tts] {cfg.voxcpm2_model} on mlx", flush=True)
        else:
            from . import voxcpm2_torch as backend

            model = backend.load(cfg.voxcpm2_torch_model)
        _backend = backend, model
    return _backend


def _prepare_voice(backend, model, voice: str, cfg: Config):
    clip = VOICES_DIR / f"{voice}.wav"
    transcript = (VOICES_DIR / f"{voice}.txt").read_text().strip()
    if not APPLE_SILICON:
        return backend.prepare_voice(model, str(clip), transcript)
    # named after the model and what's in the clip and transcript, so editing either or switching
    # models never reuses a stale voice
    key = hashlib.sha256(clip.read_bytes() + transcript.encode()).hexdigest()[:16]
    cache = VOICE_CACHE / cfg.voxcpm2_model.replace("/", "--") / f"{voice}-{key}.safetensors"
    return backend.prepare_voice(model, str(clip), transcript, cache)


def _synthesize(text: str, voice: str, cfg: Config) -> Iterator[np.ndarray]:
    backend, model = _load(cfg)
    if voice not in _voices:
        _voices[voice] = _prepare_voice(backend, model, voice, cfg)
    # VoxCPM2 has no speed control, so stretch its output (pitch stays the same)
    stretcher = TimeStretcher(cfg.speed) if cfg.speed != 1.0 else None
    halve = HalfRate()  # 48 kHz -> the player's 24 kHz
    for chunk in backend.stream(model, text, _voices[voice], cfg.voxcpm2_steps):
        audio = halve.process(chunk)
        yield stretcher.process(audio) if stretcher else audio
    if stretcher:
        yield stretcher.flush()
