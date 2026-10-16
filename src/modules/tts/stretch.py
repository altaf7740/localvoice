import numpy as np

FRAME = 960  # 40 ms at 24 kHz: a couple of pitch periods, the usual size for speech WSOLA
HOP_OUT = FRAME // 2
TOLERANCE = 240  # ±10 ms search for the best-aligned frame


class TimeStretcher:
    """Streaming WSOLA: plays speech at `speed`× without changing its pitch.

    Output frames are taken from the input `speed` times further apart than they are
    written, each nudged by up to TOLERANCE samples so its waveform lines up with the
    previous frame, then overlap-added with a Hann window (which sums to 1 at 50% overlap).
    """

    def __init__(self, speed: float):
        self.hop_in = HOP_OUT * speed
        # periodic Hann; float32 like the audio, or every frame is silently promoted to float64
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)).astype(np.float32)
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_start = 0  # absolute input index of buf[0]
        self.k = 0  # next output frame
        self.prev: int | None = None  # absolute input start of the previous frame
        self.tail = np.zeros(FRAME - HOP_OUT, dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, chunk.astype(np.float32)])
        out = []
        while True:
            nominal = round(self.k * self.hop_in)
            end = self.buf_start + len(self.buf)
            if self.prev is None:
                if nominal + FRAME > end:
                    break
                start = nominal
            else:
                lo = max(nominal - TOLERANCE, self.buf_start)
                natural = self.prev + HOP_OUT  # where the previous frame would have continued
                if max(nominal + TOLERANCE, lo) + FRAME > end or natural + FRAME > end:
                    break
                target = self._slice(natural)
                region = self.buf[lo - self.buf_start : nominal + TOLERANCE + FRAME - self.buf_start]
                start = lo + int(np.argmax(np.correlate(region, target, "valid")))
            frame = self._slice(start) * self.window
            frame[: len(self.tail)] += self.tail
            out.append(frame[:HOP_OUT])
            self.tail = frame[HOP_OUT:]
            self.prev, self.k = start, self.k + 1
            keep = min(round(self.k * self.hop_in) - TOLERANCE, self.prev + HOP_OUT)
            if keep > self.buf_start:
                self.buf = self.buf[keep - self.buf_start :]
                self.buf_start = keep
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """The overlap tail of the last frame; call once the input has ended."""
        tail, self.tail = self.tail, np.zeros(0, dtype=np.float32)
        return tail

    def _slice(self, start: int) -> np.ndarray:
        return self.buf[start - self.buf_start : start - self.buf_start + FRAME]
