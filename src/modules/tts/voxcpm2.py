"""Streaming VoxCPM2 voice cloning by continuation.

Each sentence is generated as a continuation of the voice's reference clip: the prompt is
[clip transcript, sentence, audio_start, clip audio] and the model carries on speaking after the
clip. That's mlx-audio's "continuation" mode (prompt_audio + prompt_text). Its plain
reference-cloning mode often fumbled a sentence's first word (in 12 tries "सुबह दस बजे" came out
as "3-3-10-10-1", "Duh, ds bajay"…), which continuation never did.

Two changes from mlx-audio 0.5.9's generate(), which this otherwise mirrors:
- Streaming: it decodes a sentence only once all of it is generated, so a 5 s sentence started
  playing ~5 s after the tap. This decodes every few patches (CHUNKS). The audio decoder is causal, so a
  window with CONTEXT patches of history in front reproduces the whole-sentence decode (relative
  error 2e-4 with 3 patches).
- The transcript comes first in the prompt and both LMs are causal, so it's run through them once
  per voice (prepare_voice) instead of once per sentence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .limits import PATCH_SECONDS, max_patches

# patches decoded at a time (× 0.16 s of audio each), the last size repeating. Small at first so
# audio starts sooner, then growing so decoding keeps ahead of playback.
CHUNKS = (2, 3, 4, 6)
CONTEXT = 3  # patches of history in front of each decoded chunk (mlx-audio's streaming_prefix_len - 1)
MIN_PATCHES = 2  # mlx-audio's min_tokens: don't stop before this many


@dataclass
class Voice:
    lm_cache: list  # both LMs' caches after reading the clip's transcript
    res_cache: list
    feats: mx.array  # (1, T, P, D): the clip as latent patches
    feat_embed: mx.array  # (1, T, H): those patches as LM inputs


def compile_model(model):
    """Compile the parts that always see the same shapes: the DiT's ODE solve (one patch at a time)
    and the audio decoder (windows of at most CONTEXT + max(CHUNKS) patches). Together ~20% faster on an
    M1 Pro (RTF 1.75 → 1.38 under load). The LMs can't be: their KV caches grow every step, and
    4-bit LMs were no faster than 8-bit, so they're bound by per-op overhead, not memory."""
    model.feat_decoder.solve_euler = mx.compile(model.feat_decoder.solve_euler)
    # the decoder draws noise, so the compiled graph must carry the random state
    model.audio_vae.decode = mx.compile(model.audio_vae.decode, inputs=[mx.random.state], outputs=[mx.random.state])


def prepare_voice(model, clip: str, transcript: str, cache: Path | None = None) -> Voice:
    """Encode the reference clip and run its transcript through both LMs, once per voice.
    That takes ~1 s, so with `cache` it's saved there and loaded on later starts instead."""
    if cache and cache.exists():
        arrays = mx.load(str(cache))
        return Voice(_layers(arrays, "lm"), _layers(arrays, "res"), arrays["feats"], arrays["feat_embed"])
    voice = _prepare_voice(model, clip, transcript)
    if cache:
        arrays = {"feats": voice.feats, "feat_embed": voice.feat_embed}
        for lm, layers in (("lm", voice.lm_cache), ("res", voice.res_cache)):
            arrays[f"{lm}.n"] = mx.array(len(layers))
            for i, (k, v) in enumerate(layers):
                arrays[f"{lm}.{i}.k"], arrays[f"{lm}.{i}.v"] = k, v
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix(".partial.safetensors")  # renamed once complete, so a crash can't leave half a file
        mx.save_safetensors(str(partial), arrays)
        partial.replace(cache)
    return voice


def _layers(arrays: dict, lm: str) -> list:
    return [(arrays[f"{lm}.{i}.k"], arrays[f"{lm}.{i}.v"]) for i in range(int(arrays[f"{lm}.n"]))]


def _prepare_voice(model, clip: str, transcript: str) -> Voice:
    tokens = mx.array(model._tokenize(transcript), dtype=mx.int32)[None]
    enc, lm_cache = model.base_lm(model.base_lm.embed_tokens(tokens) * _scale_emb(model))
    # text positions have no audio features, so the residual LM sees zeros beside the text encoding
    _, res_cache = model.residual_lm(model.fusion_concat_proj(mx.concatenate([enc, mx.zeros_like(enc)], -1)))
    feats = model._encode_wav(clip, padding_mode="left")[None]
    feat_embed = model.enc_to_lm_proj(model.feat_encoder(feats))
    mx.eval(lm_cache, res_cache, feats, feat_embed)
    return Voice(lm_cache, res_cache, feats, feat_embed)


def stream(model, text: str, voice: Voice, steps: int, cfg_value: float = 2.0) -> Iterator[np.ndarray]:
    """Yield float32 audio at model.sample_rate for `text`, in the voice from prepare_voice()."""
    # Tokenised on its own, the sentence's first word gets SentencePiece's "▁", so this is the
    # prompt mlx-audio builds from f"{transcript} {text}".
    tokens = mx.array(model._tokenize(text) + [model.audio_start_token], dtype=mx.int32)[None]
    n = tokens.shape[1]
    text_embed = model.base_lm.embed_tokens(tokens) * _scale_emb(model)
    embed = mx.concatenate([text_embed, voice.feat_embed], axis=1)  # sentence, then the clip's audio
    enc, lm_cache = model.base_lm(embed, cache=voice.lm_cache, mask=_mask(embed.shape[1], voice.lm_cache))
    enc = mx.concatenate([enc[:, :n], model.fsq_layer(enc[:, n:])], axis=1)  # FSQ on audio positions only
    audio_in = mx.concatenate([mx.zeros_like(text_embed), voice.feat_embed], axis=1)
    res, res_cache = model.residual_lm(
        model.fusion_concat_proj(mx.concatenate([enc, audio_in], -1)),
        cache=voice.res_cache,
        mask=_mask(embed.shape[1], voice.res_cache),
    )
    lm_hidden, res_hidden = enc[:, -1, :], res[:, -1, :]
    cond = voice.feats[:, -1]  # (1, P, D): the previous patch conditions the next

    limit = max_patches(text)
    # the clip's last patches go first as decoding context, so the sentence joins on smoothly
    patches: list[mx.array] = [voice.feats[:, i] for i in range(voice.feats.shape[1] - CONTEXT, voice.feats.shape[1])]
    done = len(patches)  # patches already decoded and yielded (or context only)
    chunk = 0  # chunks yielded so far
    for i in range(limit):
        dit_h = mx.concatenate([model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(res_hidden)], axis=-1)
        pred = model.feat_decoder.sample(
            mu=dit_h, n_timesteps=steps, patch_size=model.patch_size, cond=cond.transpose(0, 2, 1), cfg_value=cfg_value
        ).transpose(0, 2, 1)  # (1, P, D)
        patches.append(pred)
        stop = mx.argmax(model.stop_head(nn.silu(model.stop_proj(lm_hidden))), axis=-1).item() == 1
        if i > MIN_PATCHES and stop:
            break
        if len(patches) - done >= CHUNKS[min(chunk, len(CHUNKS) - 1)]:
            yield _decode(model, patches, done, last=False)
            done = len(patches)
            chunk += 1

        curr = model.enc_to_lm_proj(model.feat_encoder(pred[:, None, :, :]))
        out, lm_cache = model.base_lm(inputs_embeds=curr, cache=lm_cache)
        lm_hidden = model.fsq_layer(out[:, -1, :])
        res, res_cache = model.residual_lm(
            inputs_embeds=model.fusion_concat_proj(mx.concatenate([lm_hidden[:, None, :], curr], axis=-1)),
            cache=res_cache,
        )
        res_hidden = res[:, -1, :]
        cond = pred
    else:
        print(f"[tts] voxcpm2 didn't stop; cut off after {limit * PATCH_SECONDS:.1f}s", flush=True)
    if len(patches) > done:
        yield _decode(model, patches, done, last=True)


def _scale_emb(model) -> float:
    lm_cfg = model.args.lm_config
    return lm_cfg.scale_emb if lm_cfg.use_mup else 1.0


def _mask(n: int, cache: list) -> mx.array:
    """Causal mask for n new positions after a cache, every cached position visible. (MiniCPMModel's
    own mask is only right without a cache.)"""
    cached = cache[0][0].shape[1]
    causal = mx.triu(mx.full((n, n), float("-inf")), k=1)
    return mx.concatenate([mx.zeros((n, cached)), causal], axis=1)[None, None]


def _decode(model, patches: list[mx.array], start: int, last: bool) -> np.ndarray:
    """Audio for patches[start:], decoded with up to CONTEXT patches of history."""
    lo = max(0, start - CONTEXT)
    audio = np.asarray(model.audio_vae.decode(mx.concatenate(patches[lo:], axis=1)).flatten(), dtype=np.float32)
    per_patch = model.patch_size * model.audio_vae.decode_chunk_size
    skip = (start - lo) * per_patch
    # the decoder emits a few samples past the last patch; keep them only at the very end
    return audio[skip:] if last else audio[skip : skip + (len(patches) - start) * per_patch]
