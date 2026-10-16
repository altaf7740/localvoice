import numpy as np
import pytest

from src.modules.tts.stretch import TimeStretcher

SR = 24000


def _tone(seconds: float, hz: float = 180.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _stretch(audio: np.ndarray, speed: float, chunk: int = 12000) -> np.ndarray:
    s = TimeStretcher(speed)
    parts = [s.process(audio[i : i + chunk]) for i in range(0, len(audio), chunk)]
    return np.concatenate(parts + [s.flush()])


def _peak_hz(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.argmax(spectrum) * SR / len(audio)


@pytest.mark.parametrize("speed", [0.8, 1.1, 1.25, 1.5])
def test_duration_scales_and_pitch_stays(speed):
    audio = _tone(3.0)
    out = _stretch(audio, speed)
    assert len(out) / SR == pytest.approx(3.0 / speed, abs=0.06)
    assert _peak_hz(out) == pytest.approx(180.0, abs=2.0)


def test_chunk_size_does_not_change_the_result():
    audio = _tone(2.0)
    np.testing.assert_allclose(_stretch(audio, 1.25, chunk=1000), _stretch(audio, 1.25, chunk=24000), atol=1e-5)


def test_steady_tone_has_no_dropouts():
    out = _stretch(_tone(2.0), 1.25)
    body = out[2000:-2000]  # skip the fade-in/out of the first and last frame
    body = body[: len(body) // 240 * 240]
    envelope = np.abs(body).reshape(-1, 240).max(axis=1)  # 10 ms blocks
    assert envelope.min() > 0.4


def test_output_stays_float32():
    s = TimeStretcher(1.1)
    assert s.process(_tone(0.5)).dtype == np.float32
    assert s.flush().dtype == np.float32
