import numpy as np
import pytest

from src.modules.tts.resample import HalfRate


def _tone(hz: float, seconds: float = 1.0, sr: int = 48000) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * hz * np.arange(int(sr * seconds)) / sr)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x**2).mean()))


def test_halves_the_length():
    assert len(HalfRate().process(_tone(440))) == 24000


@pytest.mark.parametrize("chunk", [1, 7, 960, 7680])
def test_chunked_matches_one_shot(chunk):
    audio = _tone(440) + _tone(3000)
    whole = HalfRate().process(audio)
    stream = HalfRate()
    parts = np.concatenate([stream.process(audio[i : i + chunk]) for i in range(0, len(audio), chunk)])
    np.testing.assert_allclose(parts, whole, atol=1e-5)


def test_keeps_speech_band_and_removes_what_would_alias():
    speech = HalfRate().process(_tone(3000))[1000:]
    ultrasonic = HalfRate().process(_tone(18000))[1000:]  # would fold back to 6 kHz
    assert _rms(speech) == pytest.approx(_rms(_tone(3000)), rel=0.02)
    assert _rms(ultrasonic) < 0.01
