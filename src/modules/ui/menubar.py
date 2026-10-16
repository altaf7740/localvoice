import threading

import rumps

from ...core.config.settings import ENGINE_VOICES, SPEEDS, Config
from ...core.reader import IDLE, LOADING, SPEAKING, WORKING, Reader
from ..tts.services import resolve_voice

TITLES = {LOADING: "…", IDLE: "🔈", WORKING: "⏳", SPEAKING: "🔊"}
NOTHING_SELECTED = "🔇"


class MenuBarApp(rumps.App):
    def __init__(self, cfg: Config):
        super().__init__("LocalVoice", title=TITLES[LOADING], quit_button="Quit")
        self.cfg = cfg
        self.reader = Reader(cfg, self.set_state, self.set_last, self.on_empty)
        self.llm_item = rumps.MenuItem("AI transliteration", callback=self.toggle_llm)
        self.llm_item.state = cfg.transliterate_llm
        self.voice_menu = rumps.MenuItem("Voice")
        self.voices: dict[str, tuple[str, str]] = {}  # menu title -> (engine, voice)
        for engine, voices in ENGINE_VOICES.items():
            for voice in voices:
                title = f"{engine.capitalize()} · {voice}"
                self.voices[title] = (engine, voice)
                item = rumps.MenuItem(title, callback=self.choose_voice)
                item.state = (engine, voice) == resolve_voice(cfg)
                self.voice_menu.add(item)
        self.speed_menu = rumps.MenuItem("Speed")
        for speed in sorted({*SPEEDS, cfg.speed}):  # include a custom SPEED= so it shows as ticked
            item = rumps.MenuItem(f"{speed}×", callback=self.choose_speed)
            item.state = speed == cfg.speed
            self.speed_menu.add(item)
        self.stop_item = rumps.MenuItem("Stop", callback=lambda _: self.reader.stop())
        self.last_item = rumps.MenuItem("Last: —")
        self.menu = [self.llm_item, self.voice_menu, self.speed_menu, self.stop_item, self.last_item, None]
        self.reader.start()

    def set_state(self, state: str):
        self.title = TITLES[state]

    def set_last(self, text: str):
        self.last_item.title = f"Last: {text[:60]}"

    def on_empty(self):
        # rumps notifications need a bundled app, so flash the icon instead
        self.title = NOTHING_SELECTED
        threading.Timer(1.5, self.set_state, args=(IDLE,)).start()

    def toggle_llm(self, item):
        item.state = not item.state
        self.cfg.transliterate_llm = bool(item.state)
        self.cfg.save("transliterate_llm")

    def choose_voice(self, item):
        for other in self.voice_menu.values():
            other.state = other is item
        self.reader.set_voice(*self.voices[item.title])

    def choose_speed(self, item):
        for other in self.speed_menu.values():
            other.state = other is item
        self.cfg.speed = float(item.title.rstrip("×"))
        self.cfg.save("speed")


def run(cfg: Config):
    MenuBarApp(cfg).run()
