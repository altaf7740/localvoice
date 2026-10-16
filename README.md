# LocalVoice

Fully local text-to-speech for macOS, Windows and Linux that reads Hinglish aloud.
Select text in any app, tap the hotkey (**Right ⌥** on macOS, **Right Shift** on Windows/Linux), and it speaks.
Tap again to stop. The companion to [LocalFlow](../localflow), which goes the other way.

## Pipeline

1. **Grab** — ⌘C / Ctrl+C copies the selection, then the previous clipboard is restored.
2. **Split** — into sentences, so the first one plays while the rest are synthesised.
3. **Transliterate** — both engines pronounce Hindi best from Devanagari, so Roman Hindi words are
   converted ("kal ki meeting" → "कल की meeting") and English words stay in Latin script:
   - `qwen2.5:7b-instruct` via Ollama, when *AI transliteration* is on and Ollama is running;
   - then a built-in lexicon (`src/static/lexicon/`) catches common words the model missed, or does
     the whole job when Ollama is off. Words that are also English ("main", "me", "to") are converted
     only next to Hindi words, so "tell me kya" keeps "me".
4. **Speak** — streamed to the speakers chunk by chunk, with one of two engines (Voice menu):
   - **[Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)** `hi_male` (default, Apple Silicon
     only): Mistral's 4B model, 4-bit on MLX. Natural Hinglish; English words sound English.
   - **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** `hm_omega` / `hm_psi`: tiny and instant but
     flat; MLX on Apple Silicon, PyTorch (CUDA or CPU) elsewhere.

   Speed (default 1.1×) is native in Kokoro; Voxtral's output is time-stretched with WSOLA, so pitch
   stays the same.

Nothing leaves the machine.

## Engines (M1 Pro)

| | Voxtral `hi_male` | Kokoro `hm_omega` |
|---|---|---|
| Pronunciation | natural, handles code-mixing | flat; garbles some English ("server restart") |
| First audio | ~0.6 s | ~0.2 s |
| Real-time factor | 0.70 (keeps up with playback up to ~1.4×) | 0.07 |
| Memory | 3.4 GB | < 1 GB |
| Weights licence | **CC BY-NC 4.0 (non-commercial)** | Apache 2.0 |

Voxtral runs 6 flow-matching steps per frame instead of its default 8 (`voxtral_steps`): Whisper
transcripts were identical and it generates ~30% faster, which is what makes faster speeds stutter-free.

`make eval` scores the configured engine: each sentence is synthesised, transcribed back with
Whisper and compared to a mixed-script reference (CER). For Kokoro the lexicon cuts CER from 0.40 to
0.21. CER overstates errors: Whisper writes correctly spoken English loanwords in Devanagari (मीटिंग).

## Setup

| | macOS (Apple Silicon) | Windows | Linux |
|---|---|---|---|
| TTS | MLX (GPU) | PyTorch (CUDA or CPU) | PyTorch (CUDA or CPU) |
| UI | menu bar | system tray | system tray |
| Hotkey | Right ⌥ | Right Shift | Right Shift |

Common to all: install [uv](https://docs.astral.sh/uv/) and, optionally, [Ollama](https://ollama.com), then

```sh
make install   # uv sync --all-groups
make model     # download Voxtral + Kokoro and their male voices, ollama pull qwen2.5:7b-instruct
make start
```

Try it without the hotkey: `make say TEXT="Kal ki meeting mein timeline discuss karni hai"`.
Run `make` to list every target.

Settings live in `~/.localvoice.json` (voice and speed are also in the menu); e.g.
`{"speed": 1.25, "hotkey": "cmd_r", "transliterate_llm": false, "output_device": "External Headphones"}`.

### macOS (Apple Silicon)

Grant your terminal **Accessibility** and **Input Monitoring** in
System Settings → Privacy & Security (for the hotkey and ⌘C), then restart the app.
espeak-ng is bundled; no Homebrew install needed.

### Windows

- Windows and Linux use Kokoro only (Voxtral needs MLX). Kokoro (~330 MB) and the voices download on first launch.
- NVIDIA GPU: install a CUDA build of PyTorch for GPU speed; without it it runs on CPU (still faster than real time).
- Reading from apps running as Administrator requires running LocalVoice as Administrator.

### Linux

- **X11 only.** Wayland blocks global hotkeys and simulated keystrokes; log in to an X11 session.
- System packages (Debian/Ubuntu): `sudo apt install libportaudio2 xclip gir1.2-ayatanaappindicator3-0.1`
- GNOME needs the *AppIndicator* extension to show the tray icon.
- Terminals copy with Ctrl+Shift+C, so selections in a terminal aren't read; other apps work.
