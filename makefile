.PHONY: help install model start stop status restart say eval test lint lint-fix check clean

TRANSLITERATE_MODEL := qwen2.5:7b-instruct

# Colors
C := \033[36m
G := \033[32m
Y := \033[33m
R := \033[0m

help:
	@echo "$(Y)Usage:$(R) make [target]\n"
	@echo "$(C)Setup:$(R)"
	@echo "  $(G)install$(R)      Install dependencies (incl. dev tools)"
	@echo "  $(G)model$(R)        Download Kokoro-82M + male Hindi voices and pull the Ollama transliteration model"
	@echo ""
	@echo "$(C)App:$(R)"
	@echo "  $(G)start$(R)        Run LocalVoice (menu bar on macOS, tray on Windows/Linux)"
	@echo "  $(G)stop$(R)         Stop a running LocalVoice"
	@echo "  $(G)status$(R)       Check if LocalVoice is running"
	@echo "  $(G)restart$(R)      Stop, then start"
	@echo "  $(G)say$(R)          Speak TEXT=\"...\" from the terminal"
	@echo ""
	@echo "$(C)Evaluation:$(R)"
	@echo "  $(G)eval$(R)         Whisper round-trip CER + latency on src/static/eval (ARGS=\"--llm --save\")"
	@echo ""
	@echo "$(C)Testing & Code Quality:$(R)"
	@echo "  $(G)test$(R)         Run tests"
	@echo "  $(G)lint$(R)         Show linting errors"
	@echo "  $(G)lint-fix$(R)     Fix linting errors and format"
	@echo "  $(G)check$(R)        Lint, format, and test"
	@echo ""
	@echo "$(C)Cleanup:$(R)"
	@echo "  $(G)clean$(R)        Remove cache files"

install:
	uv sync --all-groups

model:
	uv run python -m scripts.download_model
	@command -v ollama > /dev/null && ollama pull $(TRANSLITERATE_MODEL) \
		|| echo "Ollama not found: AI transliteration is off until you install it (the lexicon still works)"

start:
	uv run python -m src.main

stop:
	@pkill -f "python -m src.main" 2>/dev/null && echo "LocalVoice stopped" || echo "LocalVoice not running"

status:
	@pgrep -f "python -m src.main" > /dev/null && echo "LocalVoice is running" || echo "LocalVoice is NOT running"

restart: stop start

say:
	uv run python -m scripts.say "$(TEXT)"

eval:
	uv run --group eval python -m scripts.run_eval $(ARGS)

test:
	uv run pytest

lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix
	uv run ruff format

check: lint-fix test

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
