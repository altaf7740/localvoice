import threading

import numpy as np

from src.shared import audio
from src.shared.audio import BLOCK, Player


class _FakeStream:
    instances: list["_FakeStream"] = []

    def __init__(self, device, **kwargs):
        self.device = device
        self.written: list[np.ndarray] = []
        self.release = threading.Event()
        self.release.set()
        _FakeStream.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, chunk):
        if chunk.dtype != np.float32:  # like sounddevice: "dtype mismatch: 'float64' vs 'float32'"
            raise TypeError(f"dtype mismatch: {chunk.dtype}")
        self.release.wait()
        self.written.append(chunk)


def _player(monkeypatch, devices=("MacBook Pro Speakers",)):
    _FakeStream.instances = []
    monkeypatch.setattr(audio.sd, "OutputStream", _FakeStream)
    monkeypatch.setattr(audio, "output_devices", lambda: list(devices))
    return Player()


def test_plays_every_chunk_in_order(monkeypatch):
    player = _player(monkeypatch)
    player.start()
    player.feed(np.ones(BLOCK * 2 + 10, dtype=np.float32))
    player.feed(np.zeros(5, dtype=np.float32))
    player.finish()
    written = _FakeStream.instances[0].written
    assert [len(c) for c in written] == [BLOCK, BLOCK, 10, 5]


def test_stop_drops_queued_audio(monkeypatch):
    player = _player(monkeypatch)
    player.start()
    while not _FakeStream.instances:
        pass
    stream = _FakeStream.instances[0]
    stream.release.clear()  # hold playback inside the first write
    player.feed(np.ones(BLOCK * 10, dtype=np.float32))
    player.stop()
    stream.release.set()
    player.finish()
    assert len(stream.written) <= 1


def test_unplugged_device_falls_back_to_default(monkeypatch):
    player = _player(monkeypatch)
    player.start("AirPods Pro")
    player.finish()
    assert _FakeStream.instances[0].device is None


def test_float64_chunks_are_converted(monkeypatch):
    player = _player(monkeypatch)
    player.start()
    player.feed(np.ones(10, dtype=np.float64))
    player.finish()
    assert _FakeStream.instances[0].written[0].dtype == np.float32
