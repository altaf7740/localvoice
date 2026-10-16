import logging
import platform
import sys
from typing import Iterator

import numpy as np

from ...core.config.settings import Config

SAMPLE_RATE = 24000
LANG_CODE = "h"  # Kokoro's Hindi pipeline: espeak-ng "hi" G2P, which also reads Latin-script English
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

# espeak-ng warns on every sentence that mixes Devanagari and English. phonemizer resets
# the level whenever a backend is created, so stop the records reaching the root handler.
logging.getLogger("phonemizer").propagate = False

_mlx_model = None
_torch_pipeline = None


def synthesize(text: str, cfg: Config) -> Iterator[np.ndarray]:
    """Yield float32 mono chunks at SAMPLE_RATE as Kokoro produces them."""
    if not text.strip():
        return
    if _use_mlx(cfg):
        yield from _synthesize_mlx(text, cfg)
    else:
        yield from _synthesize_torch(text, cfg)


def warmup(cfg: Config):
    """Load the model and voice so the first real request doesn't pay for it."""
    for _ in synthesize("नमस्ते", cfg):
        pass


def _use_mlx(cfg: Config) -> bool:
    if cfg.tts_backend == "auto":
        return APPLE_SILICON
    return cfg.tts_backend == "mlx"


def _synthesize_mlx(text: str, cfg: Config) -> Iterator[np.ndarray]:
    global _mlx_model
    if _mlx_model is None:
        from mlx_audio.tts.utils import load

        _mlx_model = load(cfg.tts_model)
        print(f"[tts] {cfg.tts_model} on mlx", flush=True)
    for result in _mlx_model.generate(text, voice=cfg.voice, speed=cfg.speed, lang_code=LANG_CODE):
        yield np.asarray(result.audio, dtype=np.float32)


def _synthesize_torch(text: str, cfg: Config) -> Iterator[np.ndarray]:
    global _torch_pipeline
    if _torch_pipeline is None:
        import torch
        from kokoro import KPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _torch_pipeline = KPipeline(lang_code=LANG_CODE, repo_id=cfg.torch_model, device=device)
        print(f"[tts] {cfg.torch_model} on {device}", flush=True)
    for result in _torch_pipeline(text, voice=cfg.voice, speed=cfg.speed):
        if result.audio is not None:
            yield result.audio.cpu().numpy().astype(np.float32)
