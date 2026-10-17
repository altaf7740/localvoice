import numpy as np

from src.modules.tts import limits, voxcpm2_torch


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def squeeze(self, dim):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values.astype(np.float32)


class _FakeModel:
    def __init__(self, chunks=3):
        self.chunks, self.closed, self.kwargs, self.cache_args = chunks, False, None, None

    def build_prompt_cache(self, **kwargs):
        self.cache_args = kwargs
        return {"cached": True}

    def generate_with_prompt_cache_streaming(self, **kwargs):
        self.kwargs = kwargs

        def gen():
            try:
                for i in range(self.chunks):
                    yield _FakeTensor([i, i]), None, None
            finally:
                self.closed = True

        return gen()


def test_prompt_matches_the_mlx_backend():
    model = _FakeModel()
    assert voxcpm2_torch.prepare_voice(model, "clip.wav", "Good evening.") == {"cached": True}
    # the package concatenates prompt_text + text, the MLX backend reads f"{transcript} {text}"
    assert model.cache_args == {"prompt_text": "Good evening. ", "prompt_wav_path": "clip.wav"}


def test_stream_yields_float32_and_caps_length():
    model = _FakeModel()
    out = list(voxcpm2_torch.stream(model, "चार शब्दों का वाक्य", {"cached": True}, steps=4))
    assert [o.tolist() for o in out] == [[0, 0], [1, 1], [2, 2]] and out[0].dtype == np.float32
    assert model.kwargs["inference_timesteps"] == 4
    assert model.kwargs["max_len"] == limits.max_patches("चार शब्दों का वाक्य")


def test_stopping_early_ends_the_generation():
    model = _FakeModel(chunks=100)
    stream = voxcpm2_torch.stream(model, "text", {}, steps=4)
    next(stream)
    stream.close()
    assert model.closed


def test_length_cap_grows_with_words():
    assert limits.max_patches("") == int(2.0 / 0.16)
    assert limits.max_patches("a b c d e f g h i j") == int((2.0 + 7.0) / 0.16)
