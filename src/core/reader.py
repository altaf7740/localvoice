import threading
import time
from typing import Callable

from pynput import keyboard

from ..modules.transliterate import services as transliterate
from ..modules.tts import services as tts
from ..shared.audio import Player
from ..shared.selection import selected_text
from ..shared.text import split_sentences
from .config.settings import Config

LOADING, IDLE, WORKING, SPEAKING = "loading", "idle", "working", "speaking"


class Reader:
    """Tap-to-read loop shared by every UI. The UI only renders state changes."""

    def __init__(
        self,
        cfg: Config,
        on_state: Callable[[str], None],
        on_result: Callable[[str], None],
        on_empty: Callable[[], None],
    ):
        self.cfg = cfg
        self.on_state = on_state
        self.on_result = on_result
        self.on_empty = on_empty  # called when the hotkey is tapped with nothing selected
        self.player = Player()
        self.hotkey = getattr(keyboard.Key, cfg.hotkey)
        self.ready = False
        self.tapping = False
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self):
        self.on_state(LOADING)
        threading.Thread(target=self._warmup, daemon=True).start()
        keyboard.Listener(on_press=self._on_press, on_release=self._on_release).start()

    def _warmup(self):
        tts.warmup(self.cfg)
        self.ready = True
        self.on_state(IDLE)

    def set_voice(self, engine: str, voice: str):
        """Switch engine/voice and load it now, so the next tap doesn't wait for the model."""
        self.stop()
        self.cfg.tts_engine, self.cfg.voice = engine, voice
        self.cfg.save("tts_engine", "voice")
        self.ready = False
        self.on_state(LOADING)
        threading.Thread(target=self._warmup, daemon=True).start()

    def _on_press(self, key):
        # any other key while the hotkey is down means it's being used as a modifier (⌥E etc.)
        self.tapping = key == self.hotkey

    def _on_release(self, key):
        if key == self.hotkey and self.tapping:
            self.tapping = False
            self.toggle()

    def toggle(self):
        """Read the current selection, or stop if already reading."""
        if self.worker and self.worker.is_alive():
            self.stop()
        elif self.ready:
            self.cancel.clear()
            self.worker = threading.Thread(target=self._read_selection, daemon=True)
            self.worker.start()

    def stop(self):
        self.cancel.set()
        self.player.stop()

    def _read_selection(self):
        text = selected_text().strip()
        if not text:
            self.on_empty()
            return
        self.speak(text)

    def speak(self, text: str):
        self.on_state(WORKING)
        self.on_result(text)
        self.player.start(self.cfg.output_device)
        try:
            for i, sentence in enumerate(split_sentences(text)):
                if self.cancel.is_set():
                    break
                t0 = time.time()
                prepared = transliterate.to_devanagari(sentence)
                t1 = time.time()
                for audio in tts.synthesize(prepared, self.cfg):
                    if self.cancel.is_set():
                        break
                    self.player.feed(audio)
                    self.on_state(SPEAKING)
                print(f"[prep {t1 - t0:.2f}s] {prepared}\n[tts {time.time() - t1:.2f}s] sentence {i + 1}", flush=True)
        finally:
            self.player.finish()
            self.on_state(IDLE)
