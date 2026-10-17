import queue
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000  # Kokoro's output rate
BLOCK = SAMPLE_RATE // 10  # write 100 ms at a time so stop() is near-instant


class Player:
    """Plays chunks back to back on one output stream while the producer synthesises ahead."""

    def __init__(self):
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, device: str = ""):
        self._queue = queue.Queue()
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, args=(device,), daemon=True)
        self._thread.start()

    def feed(self, audio: np.ndarray):
        # the stream rejects any other dtype, and strided views (e.g. every other sample)
        self._queue.put(np.ascontiguousarray(audio, dtype=np.float32))

    def finish(self):
        """Block until everything fed so far has played (or stop() was called)."""
        self._queue.put(None)
        if self._thread:
            self._thread.join()

    def stop(self):
        self._stopped.set()
        self._queue.put(None)

    def _run(self, device: str):
        if device and device not in output_devices():
            device = ""  # unplugged since it was chosen: fall back to the system default
        gaps = 0  # times the speaker ran dry waiting for synthesis
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device or None) as stream:
            while not self._stopped.is_set():
                chunk = self._queue.get()
                if chunk is None:
                    break
                for i in range(0, len(chunk), BLOCK):
                    if self._stopped.is_set():
                        break
                    if stream.write(chunk[i : i + BLOCK]):  # True when the device underflowed
                        gaps += 1
        if gaps > 1:  # the stream opens before the first audio exists, so waiting for it is always one
            print(f"[audio] {gaps - 1} gap(s): synthesis fell behind playback (lower SPEED or STEPS)", flush=True)


def output_devices() -> list[str]:
    return [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
