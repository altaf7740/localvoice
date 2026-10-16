"""LocalVoice entry point: menu-bar app on macOS, system tray elsewhere."""

import sys

from .core.config.settings import Config


def main():
    cfg = Config.load()
    if sys.platform == "darwin":
        from .modules.ui.menubar import run
    else:
        from .modules.ui.tray import run
    run(cfg)


if __name__ == "__main__":
    main()
