from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from src.modules.tts import voxcpm2  # noqa: E402

PER_PATCH = 4 * 5  # patch_size × decode_chunk_size in the fake model


class FakeVAE:
    """Causal 'decoder': sample t of the output is the running sum of everything up to t,
    plus 3 extra samples at the end like the real decoder emits."""

    decode_chunk_size = 5

    def decode(self, z):  # z: (1, frames, D)
        frames = np.asarray(z)[0, :, 0]
        audio = np.cumsum(np.repeat(frames, self.decode_chunk_size))
        return mx.array(np.concatenate([audio, audio[-1:].repeat(3)])[None])


def test_chunked_decode_with_enough_context_matches_whole_decode(monkeypatch):
    # the fake decoder's memory is unbounded, so give it all the history it needs
    monkeypatch.setattr(voxcpm2, "CONTEXT", 100)
    model = SimpleNamespace(patch_size=4, audio_vae=FakeVAE())
    patches = [mx.array(np.full((1, 4, 1), i + 1.0, dtype=np.float32)) for i in range(11)]
    whole = voxcpm2._decode(model, patches, 0, last=True)
    parts, done = [], 0
    for end in (3, 6, 9):
        parts.append(voxcpm2._decode(model, patches[:end], done, last=False))
        done = end
    parts.append(voxcpm2._decode(model, patches, done, last=True))
    assert [len(p) for p in parts[:-1]] == [3 * PER_PATCH] * 3
    np.testing.assert_allclose(np.concatenate(parts), whole)
