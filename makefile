.PHONY: help install model start stop status restart say web test lint lint-fix check clean

# Optional per-run overrides: make start SPEED=1.25 STEPS=4 (they beat ~/.localvoice.json)
RUN_FLAGS = $(if $(SPEED),--speed $(SPEED)) $(if $(STEPS),--steps $(STEPS))
# this repo's interpreter only: LocalFlow and other projects also run "python -m src.main"
APP_PROCESS := $(CURDIR)/.venv/bin/python.* -m src.main

# Colors
C := \033[36m
G := \033[32m
Y := \033[33m
R := \033[0m

help:
	@echo "$(Y)Usage:$(R) make [target]\n"
	@echo "$(C)Setup:$(R)"
	@echo "  $(G)install$(R)      Install dependencies (incl. dev tools)"
	@echo "  $(G)model$(R)        Download VoxCPM2 and prepare its voice"
	@echo ""
	@echo "$(C)App:$(R)"
	@echo "  $(G)start$(R)        Run LocalVoice (menu bar on macOS, tray on Windows/Linux) [SPEED=1.25 STEPS=4]"
	@echo "  $(G)stop$(R)         Stop a running LocalVoice"
	@echo "  $(G)status$(R)       Check if LocalVoice is running"
	@echo "  $(G)restart$(R)      Stop, then start"
	@echo "  $(G)say$(R)          Speak TEXT=\"...\" from the terminal [SPEED=1.25 STEPS=4]"
	@echo "  $(G)web$(R)          Test VoxCPM2 in the browser at http://localhost:8765 [SPEED=1.25 STEPS=4]"
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

start:
	uv run python -m src.main $(RUN_FLAGS)

stop:
	@pkill -f "$(APP_PROCESS)" 2>/dev/null && echo "LocalVoice stopped" || echo "LocalVoice not running"

status:
	@pgrep -f "$(APP_PROCESS)" > /dev/null && echo "LocalVoice is running" || echo "LocalVoice is NOT running"

restart: stop start

say:
	uv run python -m scripts.say $(RUN_FLAGS) "$(TEXT)"

web:
	uv run python -m scripts.web $(RUN_FLAGS)

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
