import threading

import numpy as np
from pynput import keyboard

from src.core import reader
from src.core.config.settings import Config
from src.core.reader import IDLE, LOADING, SPEAKING, WORKING, Reader


class FakePlayer:
    def __init__(self):
        self.fed: list[np.ndarray] = []
        self.stopped = False

    def start(self, device=""):
        self.stopped = False

    def feed(self, audio):
        self.fed.append(audio)

    def finish(self):
        pass

    def stop(self):
        self.stopped = True


def _make(monkeypatch, selection="Kal milte hain. Theek hai?"):
    states, results, empty = [], [], []
    monkeypatch.setattr(reader, "Player", FakePlayer)
    monkeypatch.setattr(reader, "selected_text", lambda: selection)
    monkeypatch.setattr(reader.transliterate, "to_devanagari", lambda text: text.upper())
    monkeypatch.setattr(reader.tts, "synthesize", lambda text, cfg: iter([np.full(3, len(text), dtype=np.float32)]))
    monkeypatch.setattr(reader.tts, "loaded", lambda: True)
    r = Reader(Config(hotkey="alt_r"), states.append, results.append, lambda: empty.append(1))
    r.ready = True
    return r, states, results, empty


def _join_workers():
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(timeout=1)


def _tap(r, *between):
    r._on_press(keyboard.Key.alt_r)
    for key in between:
        r._on_press(key)
        r._on_release(key)
    r._on_release(keyboard.Key.alt_r)


def test_tap_reads_selection_sentence_by_sentence(monkeypatch):
    r, states, results, empty = _make(monkeypatch)
    _tap(r)
    _join_workers()
    assert results == ["Kal milte hain. Theek hai?"]
    sound, silence = r.player.fed[0::2], r.player.fed[1::2]
    assert [a[0] for a in sound] == [len("Kal milte hain."), len("Theek hai?")]
    assert len(silence) == 1 and not silence[0].any()  # a pause between the two, none after the last
    assert states == [WORKING, SPEAKING, SPEAKING, IDLE]
    assert empty == []


def test_hotkey_used_as_modifier_does_nothing(monkeypatch):
    r, states, results, empty = _make(monkeypatch)
    _tap(r, keyboard.KeyCode.from_char("e"))  # ⌥E accent
    _join_workers()
    assert states == []
    assert results == []


def test_nothing_selected_reports_empty(monkeypatch):
    r, states, results, empty = _make(monkeypatch, selection="  ")
    _tap(r)
    _join_workers()
    assert empty == [1]
    assert states == []


def test_tap_while_speaking_stops(monkeypatch):
    r, states, results, empty = _make(monkeypatch)
    gate = threading.Event()

    def slow_synthesize(text, cfg):
        gate.wait(timeout=1)
        yield np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(reader.tts, "synthesize", slow_synthesize)
    _tap(r)
    _tap(r)  # second tap while the first sentence is still synthesising
    gate.set()
    _join_workers()
    assert r.player.stopped
    assert len(r.player.fed) == 0
    assert states[-1] == IDLE


def test_taps_ignored_while_loading(monkeypatch):
    r, states, results, empty = _make(monkeypatch)
    r.ready = False
    _tap(r)
    _join_workers()
    assert results == []


def test_set_voice_saves_and_reloads(monkeypatch, tmp_path):
    from src.core.config import settings

    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "localvoice.json")
    r, states, results, empty = _make(monkeypatch)
    warmed = []
    monkeypatch.setattr(reader.tts, "warmup", lambda cfg: warmed.append(cfg.voice))
    r.set_voice("other_voice")
    _join_workers()
    assert warmed == ["other_voice"]
    assert states == [LOADING, IDLE]
    assert r.ready
    assert (tmp_path / "localvoice.json").read_text().count("other_voice") == 1


def test_idle_model_is_unloaded_only_after_the_limit_and_when_not_reading(monkeypatch):
    r, *_ = _make(monkeypatch)
    unloads = []
    monkeypatch.setattr(reader.tts, "unload", lambda: unloads.append(1) or True)
    r.cfg.idle_unload_minutes = 10
    assert not r.unload_if_idle()  # just used
    r.last_active -= 601
    r.cfg.idle_unload_minutes = 0
    assert not r.unload_if_idle()  # 0 keeps it loaded
    r.cfg.idle_unload_minutes = 10
    r.ready = False
    assert not r.unload_if_idle()  # still warming up
    r.ready = True
    r.worker = threading.Thread(target=threading.Event().wait, daemon=True)
    r.worker.start()
    assert not r.unload_if_idle()  # reading
    r.worker = None
    assert r.unload_if_idle() and unloads == [1]


def test_tap_after_an_unload_shows_loading_until_audio_starts(monkeypatch):
    r, states, *_ = _make(monkeypatch)
    monkeypatch.setattr(reader.tts, "loaded", lambda: False)
    _tap(r)
    _join_workers()
    assert states[:3] == [WORKING, LOADING, SPEAKING]


def test_cut_mid_sentence_is_spoken_with_a_comma_and_a_short_pause(monkeypatch):
    r, *_ = _make(
        monkeypatch, selection="Meeting kal subah ko office mein hogi lekin location abhi tak confirm nahi hui hai."
    )
    spoken = []
    monkeypatch.setattr(
        reader.tts, "synthesize", lambda text, cfg: spoken.append(text) or iter([np.ones(3, dtype=np.float32)])
    )
    _tap(r)
    _join_workers()
    assert spoken == ["MEETING KAL SUBAH KO OFFICE MEIN HOGI,", "LEKIN LOCATION ABHI TAK CONFIRM NAHI HUI HAI."]
    assert len(r.player.fed[1]) == len(reader.pause(False))
