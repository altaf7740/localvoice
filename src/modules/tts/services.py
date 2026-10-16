import logging
from typing import Iterator

import numpy as np

from ...core.config.settings import APPLE_SILICON, ENGINE_VOICES, Config
from .stretch import TimeStretcher

SAMPLE_RATE = 24000  # both Kokoro and Voxtral
KOKORO_LANG = "h"  # Kokoro's Hindi pipeline: espeak-ng "hi" G2P, which also reads Latin-script English
VOXTRAL_CHUNK = 0.5  # seconds of audio per streamed Voxtral chunk; smaller = sooner first audio

# espeak-ng warns on every sentence that mixes Devanagari and English. phonemizer resets
# the level whenever a backend is created, so stop the records reaching the root handler.
logging.getLogger("phonemizer").propagate = False

_mlx_models: dict = {}
_torch_pipeline = None


def synthesize(text: str, cfg: Config) -> Iterator[np.ndarray]:
    """Yield float32 mono chunks at SAMPLE_RATE as the engine produces them."""
    if not text.strip():
        return
    engine, voice = resolve_voice(cfg)
    if engine == "voxtral":
        yield from _synthesize_voxtral(text, voice, cfg)
    elif _kokoro_on_mlx(cfg):
        yield from _synthesize_kokoro_mlx(text, voice, cfg)
    else:
        yield from _synthesize_kokoro_torch(text, voice, cfg)


def warmup(cfg: Config):
    """Load the model and voice so the first real request doesn't pay for it."""
    for _ in synthesize("नमस्ते", cfg):
        pass


def resolve_voice(cfg: Config) -> tuple[str, str]:
    """The configured engine and voice, falling back to what this machine can run."""
    engine = cfg.tts_engine if cfg.tts_engine in ENGINE_VOICES else "kokoro"
    voices = ENGINE_VOICES[engine]
    return engine, cfg.voice if cfg.voice in voices else voices[0]


def _kokoro_on_mlx(cfg: Config) -> bool:
    if cfg.kokoro_backend == "auto":
        return APPLE_SILICON
    return cfg.kokoro_backend == "mlx"


def _load_mlx(repo: str):
    if repo not in _mlx_models:
        from mlx_audio.tts.utils import load

        _mlx_models[repo] = load(repo)
        print(f"[tts] {repo} on mlx", flush=True)
    return _mlx_models[repo]


def _synthesize_voxtral(text: str, voice: str, cfg: Config) -> Iterator[np.ndarray]:
    model = _load_mlx(cfg.voxtral_model)
    model.acoustic_transformer.args.n_denoising_steps = cfg.voxtral_steps
    # Voxtral has no speed control, so stretch its output (pitch stays the same)
    stretcher = TimeStretcher(cfg.speed) if cfg.speed != 1.0 else None
    for result in model.generate(text, voice=voice, stream=True, streaming_interval=VOXTRAL_CHUNK):
        audio = np.asarray(result.audio, dtype=np.float32)
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
