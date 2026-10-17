import hashlib
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.config.settings import APPLE_SILICON, ENGINE_VOICES, STATIC_DIR, Config
from ...shared.hub import cached
from .resample import HalfRate
from .stretch import TimeStretcher

SAMPLE_RATE = 24000  # Kokoro's; VoxCPM2's 48 kHz is halved to match
VOICES_DIR = STATIC_DIR / "voices"  # per VoxCPM2 voice: a reference clip (.wav) and what it says (.txt)
VOICE_CACHE = Path.home() / ".cache" / "localvoice" / "voices"  # those voices prepared by the MLX backend
KOKORO_LANG = "h"  # Kokoro's Hindi pipeline: espeak-ng "hi" G2P, which also reads Latin-script English

# espeak-ng warns on every sentence that mixes Devanagari and English. phonemizer resets
# the level whenever a backend is created, so stop the records reaching the root handler.
logging.getLogger("phonemizer").propagate = False
# VoxCPM2's tokenizer loads its config.json as a generic one, and transformers warns that the model
# types differ on every start. Read when transformers is first imported, i.e. when a model loads.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_mlx_models: dict = {}
_voxcpm2_voices: dict = {}  # voice -> reference clip already run through the model (voxcpm2.prepare_voice)
_torch_pipeline = None
# MLX runs a model only on the thread that loaded it ("There is no Stream(cpu, 1) in current thread"),
# so all MLX work happens on this one thread, whichever thread asks for speech
_mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_DONE = object()


def synthesize(text: str, cfg: Config) -> Iterator[np.ndarray]:
    """Yield float32 mono chunks at SAMPLE_RATE as the engine produces them."""
    if not text.strip():
        return
    engine, voice = resolve_voice(cfg)
    if engine == "voxcpm2":
        chunks = _synthesize_voxcpm2(text, voice, cfg)
    elif APPLE_SILICON:
        chunks = _synthesize_kokoro_mlx(text, voice, cfg)
    else:
        chunks = _synthesize_kokoro_torch(text, voice, cfg)
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


def resolve_voice(cfg: Config) -> tuple[str, str]:
    """The configured engine and voice, falling back to what this machine can run."""
    engine = cfg.tts_engine if cfg.tts_engine in ENGINE_VOICES else "kokoro"
    voices = ENGINE_VOICES[engine]
    return engine, cfg.voice if cfg.voice in voices else voices[0]


def _load_mlx(repo: str):
    if repo not in _mlx_models:
        import mlx.core as mx
        from mlx_audio.tts.utils import load
        from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS

        # one engine at a time: VoxCPM2 needs 6 GB
        _mlx_models.clear()
        _voxcpm2_voices.clear()
        mx.clear_cache()
        _mlx_models[repo] = load(cached(repo, DEFAULT_ALLOW_PATTERNS))  # the files mlx-audio downloads
        print(f"[tts] {repo} on mlx", flush=True)
    return _mlx_models[repo]


def _voxcpm2_backend(cfg: Config):
    """(backend module, model): MLX on Apple Silicon, OpenBMB's PyTorch package on NVIDIA GPUs.
    Both modules offer prepare_voice(model, clip, transcript) and stream(model, text, voice, steps);
    the MLX one's prepare_voice can also save the voice to a file."""
    if APPLE_SILICON:
        from . import voxcpm2 as backend

        fresh = cfg.voxcpm2_model not in _mlx_models
        model = _load_mlx(cfg.voxcpm2_model)
        if fresh:
            backend.compile_model(model)
        return backend, model
    from . import voxcpm2_torch as backend

    return backend, backend.load(cfg.voxcpm2_torch_model)


def _prepare_voxcpm2_voice(backend, model, voice: str, cfg: Config):
    clip = VOICES_DIR / f"{voice}.wav"
    transcript = (VOICES_DIR / f"{voice}.txt").read_text().strip()
    if not APPLE_SILICON:
        return backend.prepare_voice(model, str(clip), transcript)
    # named after the model and what's in the clip and transcript, so editing either or switching
    # models never reuses a stale voice
    key = hashlib.sha256(clip.read_bytes() + transcript.encode()).hexdigest()[:16]
    cache = VOICE_CACHE / cfg.voxcpm2_model.replace("/", "--") / f"{voice}-{key}.safetensors"
    return backend.prepare_voice(model, str(clip), transcript, cache)


def _synthesize_voxcpm2(text: str, voice: str, cfg: Config) -> Iterator[np.ndarray]:
    backend, model = _voxcpm2_backend(cfg)
    if voice not in _voxcpm2_voices:
        _voxcpm2_voices[voice] = _prepare_voxcpm2_voice(backend, model, voice, cfg)
    # VoxCPM2 has no speed control, so stretch its output (pitch stays the same)
    stretcher = TimeStretcher(cfg.speed) if cfg.speed != 1.0 else None
    halve = HalfRate()  # 48 kHz -> the player's 24 kHz
    for chunk in backend.stream(model, text, _voxcpm2_voices[voice], cfg.voxcpm2_steps):
        audio = halve.process(chunk)
        yield stretcher.process(audio) if stretcher else audio
    if stretcher:
        yield stretcher.flush()


def _synthesize_kokoro_mlx(text: str, voice: str, cfg: Config) -> Iterator[np.ndarray]:
    model = _load_mlx(cfg.kokoro_model)
    for result in model.generate(text, voice=voice, speed=cfg.speed, lang_code=KOKORO_LANG):
        yield np.asarray(result.audio, dtype=np.float32)


def _synthesize_kokoro_torch(text: str, voice: str, cfg: Config) -> Iterator[np.ndarray]:
    global _torch_pipeline
    if _torch_pipeline is None:
        import torch
        from kokoro import KPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _torch_pipeline = KPipeline(lang_code=KOKORO_LANG, repo_id=cfg.kokoro_torch_model, device=device)
        print(f"[tts] {cfg.kokoro_torch_model} on {device}", flush=True)
    for result in _torch_pipeline(text, voice=voice, speed=cfg.speed):
        if result.audio is not None:
            yield result.audio.cpu().numpy().astype(np.float32)
