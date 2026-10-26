import threading
import time
from typing import Callable

from pynput import keyboard

from ..modules.transliterate import services as transliterate
from ..modules.tts import services as tts
from ..shared.audio import Player, pause
from ..shared.selection import selected_text
from ..shared.text import split_chunks, strip_markup
from .config.settings import Config

LOADING, IDLE, WORKING, SPEAKING = "loading", "idle", "working", "speaking"
IDLE_CHECK_SECONDS = 30


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
        self.last_active = time.time()

    def start(self):
        self.on_state(LOADING)
        threading.Thread(target=self._warmup, daemon=True).start()
        keyboard.Listener(on_press=self._on_press, on_release=self._on_release).start()
        threading.Thread(target=self._unload_when_idle, daemon=True).start()

    def _warmup(self):
        tts.warmup(self.cfg)
        self.last_active = time.time()
        self.ready = True
        self.on_state(IDLE)

    def _unload_when_idle(self):
        while True:
            time.sleep(IDLE_CHECK_SECONDS)
            self.unload_if_idle()

    def unload_if_idle(self) -> bool:
        """Free the model once nothing has been read for `idle_unload_minutes`; speak() reloads it."""
        limit = self.cfg.idle_unload_minutes * 60
        busy = self.worker is not None and self.worker.is_alive()
        if limit <= 0 or not self.ready or busy or time.time() - self.last_active < limit:
            return False
        return tts.unload()

    def set_voice(self, voice: str):
        """Switch voice and prepare it now, so the next tap doesn't wait for it."""
        self.stop()
        self.cfg.voice = voice
        self.cfg.save("voice")
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
        if not tts.loaded():
            self.on_state(LOADING)  # woken from an idle unload
        self.player.start(self.cfg.output_device)
        try:
            chunks = split_chunks(strip_markup(text) if self.cfg.strip_markup else text)
            speakable = transliterate.prepare_chunks(chunks, self.cfg)
            for i, (_, ends_sentence) in enumerate(chunks):
                if self.cancel.is_set():
                    break
                t0 = time.time()
                prepared = next(speakable)
                t1 = time.time()
                for audio in tts.synthesize(prepared, self.cfg):
                    if self.cancel.is_set():
                        break
                    self.player.feed(audio)
                    self.on_state(SPEAKING)
                if i < len(chunks) - 1 and not self.cancel.is_set():
                    self.player.feed(pause(ends_sentence, self.cfg.speed))
                print(f"[prep {t1 - t0:.2f}s] {prepared}\n[tts {time.time() - t1:.2f}s] chunk {i + 1}", flush=True)
        finally:
            self.player.finish()
            self.last_active = time.time()
            self.on_state(IDLE)
