"""System-tray UI for Windows and Linux."""

import pystray
from PIL import Image, ImageDraw

from ...core.config.settings import ENGINE_VOICES, SPEEDS, Config
from ...core.reader import IDLE, LOADING, SPEAKING, WORKING, Reader
from ..tts.services import resolve_voice

COLORS = {LOADING: "#9e9e9e", IDLE: "#2e7d32", WORKING: "#f9a825", SPEAKING: "#1565c0"}
TOOLTIPS = {
    LOADING: "LocalVoice: loading models…",
    IDLE: "LocalVoice: ready",
    WORKING: "LocalVoice: preparing",
    SPEAKING: "LocalVoice: speaking",
}


def _dot(color: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((8, 8, 56, 56), fill=color)
    return img


ICONS = {state: _dot(color) for state, color in COLORS.items()}


def run(cfg: Config):
    last = {"text": "—"}

    def set_state(state: str):
        icon.icon = ICONS[state]
        icon.title = TOOLTIPS[state]

    def set_last(text: str):
        last["text"] = text[:60]
        icon.update_menu()

    def on_empty():
        icon.notify("Select some text, then tap the hotkey.", "LocalVoice")

    def toggle_llm(_icon, _item):
        cfg.transliterate_llm = not cfg.transliterate_llm
        cfg.save("transliterate_llm")

    def set_voice(engine: str, voice: str):
        return lambda _icon, _item: reader.set_voice(engine, voice)

    def set_field(key: str, value):
        def choose(_icon, _item):
            setattr(cfg, key, value)
            cfg.save(key)

        return choose

    reader = Reader(cfg, set_state, set_last, on_empty)
    icon = pystray.Icon(
        "LocalVoice",
        ICONS[LOADING],
        TOOLTIPS[LOADING],
        menu=pystray.Menu(
            pystray.MenuItem("AI transliteration", toggle_llm, checked=lambda _: cfg.transliterate_llm),
            pystray.MenuItem(
                "Voice",
                pystray.Menu(
                    *(
                        pystray.MenuItem(
                            f"{e.capitalize()} · {v}",
                            set_voice(e, v),
                            checked=lambda _, e=e, v=v: resolve_voice(cfg) == (e, v),
                            radio=True,
                        )
                        for e, voices in ENGINE_VOICES.items()
                        for v in voices
                    )
                ),
            ),
            pystray.MenuItem(
                "Speed",
                pystray.Menu(
                    *(
                        pystray.MenuItem(
                            f"{s}×", set_field("speed", s), checked=lambda _, s=s: cfg.speed == s, radio=True
                        )
                        for s in SPEEDS
                    )
                ),
            ),
            pystray.MenuItem("Stop", lambda _i, _: reader.stop()),
            pystray.MenuItem(lambda _: f"Last: {last['text']}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda i, _: i.stop()),
        ),
    )

    def setup(i: pystray.Icon):
        i.visible = True
        reader.start()

    icon.run(setup=setup)
