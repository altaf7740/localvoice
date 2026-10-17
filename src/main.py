"""LocalVoice entry point: menu-bar app on macOS, system tray elsewhere."""

import argparse
import sys

from .core.config.settings import Config, add_run_args, run_overrides


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="localvoice", description="Select text, tap the hotkey, hear it.")
    add_run_args(parser)
    args = parser.parse_args(argv)
    cfg = Config.load(**run_overrides(args))
    if sys.platform == "darwin":
        from .modules.ui.menubar import run
    else:
        from .modules.ui.tray import run
    run(cfg)


if __name__ == "__main__":
    main()
