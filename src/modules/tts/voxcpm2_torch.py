"""VoxCPM2 on NVIDIA GPUs (Windows, Linux) with OpenBMB's own `voxcpm` package.

Same voice and prompt as the MLX backend in voxcpm2.py: each sentence continues the voice's
reference clip. The package already streams (a stateful decoder, one 0.16 s patch at a time) and
caches the encoded clip, so this is a thin wrapper. On CUDA it also compiles the model with
torch.compile + CUDA graphs when Triton is installed (Linux); without it it runs eagerly.
"""

from typing import Iterator

import numpy as np

from ...shared.hub import cached
from .limits import max_patches

_models: dict = {}


def load(repo: str):
    if repo not in _models:
        from voxcpm import VoxCPM

        _models.clear()
        # the denoiser and text normaliser pull in ModelScope and WeText, which aren't installed
        _models[repo] = VoxCPM.from_pretrained(str(cached(repo)), load_denoiser=False, optimize=True).tts_model
        print(f"[tts] {repo} on {_models[repo].device}", flush=True)
    return _models[repo]


def prepare_voice(model, clip: str, transcript: str) -> dict:
    # the package joins prompt_text + text with nothing between; the MLX backend's prompt is
    # f"{transcript} {text}", so match it
    return model.build_prompt_cache(prompt_text=f"{transcript} ", prompt_wav_path=clip)


def stream(model, text: str, voice: dict, steps: int, cfg_value: float = 2.0) -> Iterator[np.ndarray]:
    """Yield float32 audio at 48 kHz for `text`, in the voice from prepare_voice()."""
    chunks = model.generate_with_prompt_cache_streaming(
        target_text=text,
        prompt_cache=voice,
        inference_timesteps=steps,
        cfg_value=cfg_value,
        max_len=max_patches(text),
    )
    try:
        for audio, _, _ in chunks:
            yield audio.squeeze(0).float().cpu().numpy()
    finally:
        chunks.close()  # stopping mid-sentence must end the generation too
