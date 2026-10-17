"""Shared by both VoxCPM2 backends (MLX in voxcpm2.py, PyTorch in voxcpm2_torch.py)."""

PATCH_SECONDS = 0.16  # audio per generated patch (4 latent frames × 1920 samples at 48 kHz)
# About 1 sentence in 30 the stop head never fires and the model rambles on for 20 s or more.
# Normal speech here is ~0.3 s per Hindi word and ~0.45 s per English word, so cut it off well past that.
MAX_SECONDS_BASE = 2.0
MAX_SECONDS_PER_WORD = 0.7


def max_patches(text: str) -> int:
    return int((MAX_SECONDS_BASE + MAX_SECONDS_PER_WORD * len(text.split())) / PATCH_SECONDS)
