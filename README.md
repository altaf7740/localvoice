# LocalVoice

Fully local text-to-speech for macOS, Windows and Linux that reads Hinglish aloud.
Select text in any app, tap the hotkey (**Right ⌥** on macOS, **Right Shift** on Windows/Linux), and it speaks.
Tap again to stop. The companion to [LocalFlow](https://github.com/altaf7740/localflow), which goes the other
way (speak Hinglish, get text). The two use different hotkeys (Left ⌥ hold vs Right ⌥ tap; Right Ctrl vs Right
Shift on Windows/Linux) and can run at the same time; `make stop` / `make status` in each repo only touch that
repo's app.

## Pipeline

1. **Grab** — ⌘C / Ctrl+C copies the selection, then the previous clipboard is restored.
2. **Split** — into sentences, so the first one plays while the rest are synthesised.
3. **Transliterate** — the engines pronounce Hindi best from Devanagari, so Roman Hindi words are
   converted ("kal ki meeting" → "कल की meeting") and English words stay in Latin script:
   - `qwen2.5:7b-instruct` via Ollama, when *AI transliteration* is on and Ollama is running;
   - then a built-in lexicon (`src/static/lexicon/`) catches common words the model missed, or does
     the whole job when Ollama is off. Words that are also English ("main", "me", "to") are converted
     only next to Hindi words, so "tell me kya" keeps "me".
4. **Speak** — streamed to the speakers chunk by chunk, with one of three engines (Voice menu):
   - **[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)** `indian_male` (default, Apple Silicon only):
     OpenBMB's 2B model, 8-bit on MLX. A deep, calm Indian male voice, cloned from
     `src/static/voices/indian_male.wav` (see [Voices](#voices)).
   - **[Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)** `hi_male` (Apple Silicon only):
     Mistral's 4B model, 4-bit on MLX. Natural Hinglish, lighter voice; English words sound English.
   - **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** `hm_omega` / `hm_psi`: tiny and instant but
     flat; MLX on Apple Silicon, PyTorch (CUDA or CPU) elsewhere.

   Speed (default 1.0×) is native in Kokoro; VoxCPM2 and Voxtral output is time-stretched with WSOLA,
   so pitch stays the same.

Nothing leaves the machine.

## Engines (M1 Pro)

| | VoxCPM2 `indian_male` | Voxtral `hi_male` | Kokoro `hm_omega` |
|---|---|---|---|
| Voice | deep (~100 Hz), calm Indian male | Indian male (~130 Hz) | flat Indian male |
| Pronunciation | natural, handles code-mixing | natural, handles code-mixing | flat; garbles some English ("server restart") |
| First audio | ~0.9 s | ~0.6 s | ~0.2 s |
| Real-time factor | 0.84 (only just keeps up at 1×) | 0.70 (keeps up to ~1.4×) | 0.07 |
| Memory | 6 GB | 3.4 GB | < 1 GB |
| Download | 3 GB | 2.4 GB | 0.3 GB |
| Weights licence | Apache 2.0 | **CC BY-NC 4.0 (non-commercial)** | Apache 2.0 |

Only one engine is loaded at a time; switching voices frees the previous model.

VoxCPM2 runs 4 diffusion steps per 0.16 s patch instead of its default 10 (`voxcpm2_steps`): Whisper
transcripts were as clean as at 10, and 4 is the most an M1 Pro can run at 1× (3 garbles words).
mlx-audio's VoxCPM2 only returns a sentence once it's fully generated, so `src/modules/tts/voxcpm2.py`
re-implements its generation loop to decode every ~1 s, and reads the voice's transcript into the
model once instead of per sentence. It pins `mlx-audio==0.5.9`, whose internals it uses. About one
sentence in 30 the model doesn't stop on its own; it's cut off at 2 s + 0.7 s per word.

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
make model     # download VoxCPM2 + Voxtral + Kokoro and their male voices, ollama pull qwen2.5:7b-instruct
make start
```

Try it without the hotkey: `make say TEXT="Kal ki meeting mein timeline discuss karni hai"`.
Run `make` to list every target.

### Speed

`SPEED` (0.5–2.0, default 1.0) and `STEPS` (diffusion steps for VoxCPM2 / Voxtral, 2–10, default 4 / 6)
override `~/.localvoice.json` for one run:

```sh
make start SPEED=1.25
make say SPEED=1.5 STEPS=4 TEXT="Main abhi busy hoon, baad mein call karta hoon."
```

The menu's Speed submenu changes it while running and saves it. VoxCPM2 only just keeps up at 1× on an
M1 Pro, so faster speeds make it pause between chunks (the terminal prints `[audio] N gap(s)`); pick
Voxtral for fast listening. Voxtral keeps up to ~1.4× at 6 steps and ~1.8× at 4.

### Voices

VoxCPM2 speaks in the voice of a short reference clip, continuing on from it: each voice is a `.wav`
and a `.txt` with exactly what the clip says, in `src/static/voices/`. `indian_male.wav` was itself
generated by VoxCPM2 from the description "A calm, confident middle-aged Indian man with a deep
voice, speaking clearly and professionally like a news anchor", so no real person's voice is cloned.
To add a voice, drop in `<name>.wav` (5–10 s, clean speech) and `<name>.txt`, and add `<name>` to
`ENGINE_VOICES["voxcpm2"]` in `src/core/config/settings.py`. Only use recordings of yourself or
someone who agreed to it.

Settings live in `~/.localvoice.json` (voice and speed are also in the menu); e.g.
`{"speed": 1.25, "hotkey": "cmd_r", "transliterate_llm": false, "output_device": "External Headphones"}`.

### macOS (Apple Silicon)

Grant your terminal **Accessibility** and **Input Monitoring** in
System Settings → Privacy & Security (for the hotkey and ⌘C), then restart the app.
espeak-ng is bundled; no Homebrew install needed.

### Windows

- Windows and Linux use Kokoro only (VoxCPM2 and Voxtral need MLX here). Kokoro (~330 MB) and the voices download on first launch.
- NVIDIA GPU: install a CUDA build of PyTorch for GPU speed; without it it runs on CPU (still faster than real time).
- Reading from apps running as Administrator requires running LocalVoice as Administrator.

### Linux

- **X11 only.** Wayland blocks global hotkeys and simulated keystrokes; log in to an X11 session.
- System packages (Debian/Ubuntu): `sudo apt install libportaudio2 xclip gir1.2-ayatanaappindicator3-0.1`
- GNOME needs the *AppIndicator* extension to show the tray icon.
- Terminals copy with Ctrl+Shift+C, so selections in a terminal aren't read; other apps work.
