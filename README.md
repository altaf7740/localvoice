# LocalVoice

Fully local text-to-speech for macOS, Windows and Linux that reads Hinglish aloud.
Select text in any app, tap the hotkey (**Right ⌥** on macOS, **Right Shift** on Windows/Linux), and it speaks.
Tap again to stop. The companion to [LocalFlow](../localflow), which goes the other way.

## Pipeline

1. **Grab** — ⌘C / Ctrl+C copies the selection, then the previous clipboard is restored.
2. **Split** — into sentences, so the first one plays while the rest are synthesised.
3. **Transliterate** — Kokoro pronounces Hindi only from Devanagari, so Roman Hindi words are converted
   ("kal ki meeting" → "कल की meeting") and English words stay in Latin script:
   - `qwen2.5:7b-instruct` via Ollama, when *AI transliteration* is on and Ollama is running;
   - then a built-in lexicon (`src/static/lexicon/`) catches common words the model missed, or does
     the whole job when Ollama is off. Words that are also English ("main", "me", "to") are converted
     only next to Hindi words, so "tell me kya" keeps "me".
4. **Speak** — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) with its male Hindi voices
   (`hm_omega`, the default, or `hm_psi`): MLX on Apple Silicon, PyTorch elsewhere
   (CUDA if available, else CPU). espeak-ng's Hindi G2P reads the Latin-script English words too.

Nothing leaves the machine.

## Speed and quality (M1 Pro, `make eval`)

| | first audio | real-time factor | Whisper round-trip CER |
|---|---|---|---|
| raw Roman Hinglish | 0.37 s | 0.095 | 0.40 |
| + lexicon | 0.27 s | 0.073 | 0.21 |

CER is pessimistic: Whisper writes some correctly spoken English loanwords in Devanagari (मीटिंग).
Numbers are for `hm_omega` (`hm_psi`: 0.25). Kokoro's Hindi voices are graded C by their author (1–10 h of training data), so speech is clear but flat.

## Setup

| | macOS (Apple Silicon) | Windows | Linux |
|---|---|---|---|
| TTS | MLX (GPU) | PyTorch (CUDA or CPU) | PyTorch (CUDA or CPU) |
| UI | menu bar | system tray | system tray |
| Hotkey | Right ⌥ | Right Shift | Right Shift |

Common to all: install [uv](https://docs.astral.sh/uv/) and, optionally, [Ollama](https://ollama.com), then

```sh
make install   # uv sync --all-groups
make model     # download Kokoro-82M + male Hindi voices, ollama pull qwen2.5:7b-instruct
make start
```

Try it without the hotkey: `make say TEXT="Kal ki meeting mein timeline discuss karni hai"`.
Run `make` to list every target.

Settings live in `~/.localvoice.json` (voice and speed are also in the menu); e.g.
`{"hotkey": "cmd_r", "transliterate_llm": false, "output_device": "External Headphones"}`.

### macOS (Apple Silicon)

Grant your terminal **Accessibility** and **Input Monitoring** in
System Settings → Privacy & Security (for the hotkey and ⌘C), then restart the app.
espeak-ng is bundled; no Homebrew install needed.

### Windows

- Kokoro (~330 MB) and the voices download on first launch.
- NVIDIA GPU: install a CUDA build of PyTorch for GPU speed; without it it runs on CPU (still faster than real time).
- Reading from apps running as Administrator requires running LocalVoice as Administrator.

### Linux

- **X11 only.** Wayland blocks global hotkeys and simulated keystrokes; log in to an X11 session.
- System packages (Debian/Ubuntu): `sudo apt install libportaudio2 xclip gir1.2-ayatanaappindicator3-0.1`
- GNOME needs the *AppIndicator* extension to show the tray icon.
- Terminals copy with Ctrl+Shift+C, so selections in a terminal aren't read; other apps work.
