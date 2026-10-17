import numpy as np

TAPS = 63
CUTOFF = 11_000 / 48_000  # low-pass just under the 12 kHz Nyquist of the 24 kHz output


class HalfRate:
    """Streaming 2:1 decimation (VoxCPM2's 48 kHz to the player's 24 kHz) with a windowed-sinc
    low-pass. Filter history carries over between chunks, so chunk joins are seamless."""

    def __init__(self):
        n = np.arange(TAPS) - (TAPS - 1) / 2
        h = 2 * CUTOFF * np.sinc(2 * CUTOFF * n) * np.blackman(TAPS)
        self.h = (h / h.sum()).astype(np.float32)
        self.history = np.zeros(TAPS - 1, dtype=np.float32)
        self.seen = 0  # input samples processed so far, to keep every other sample across chunks

    def process(self, chunk: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.history, chunk.astype(np.float32)])
        filtered = np.convolve(buf, self.h, "valid")  # one output per input sample
        out = np.ascontiguousarray(filtered[self.seen % 2 :: 2])
        self.seen += len(chunk)
        self.history = buf[len(buf) - (TAPS - 1) :]
        return out
