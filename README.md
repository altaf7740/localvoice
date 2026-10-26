# LocalVoice

Fully local text-to-speech that reads Hinglish aloud, on Apple Silicon Macs and on Windows or Linux
with an NVIDIA GPU. Select text in any app, tap the hotkey (**Right ⌥** on macOS, **Right Shift** on
Windows/Linux), and it speaks. Tap again to stop. The companion to
[LocalFlow](https://github.com/altaf7740/localflow), which goes the other way (speak Hinglish, get
text). The two use different hotkeys (Left ⌥ hold vs Right ⌥ tap; Right Ctrl vs Right Shift on
Windows/Linux) and can run at the same time; `make stop` / `make status` in each repo only touch
that repo's app.

## Pipeline

1. **Grab** — ⌘C / Ctrl+C copies the selection, then the previous clipboard is restored.
2. **Split** — into phrases, so the first one plays while the rest are synthesised: at sentence ends,
   at commas, and before connectors like "lekin" and "but" (not "aur" or "and", which join words inside a
   phrase). A cut mid-sentence is spoken with a trailing comma and a shorter pause than a sentence end.
3. **Transliterate** — VoxCPM2 pronounces Hindi best from Devanagari, so Roman Hindi words are
   converted ("kal ki meeting" → "कल की meeting") by a built-in lexicon (`src/static/lexicon/`) and
   English words stay in Latin script. Words that are also English ("main", "me", "to") are converted
   only next to Hindi words, so "tell me kya" keeps "me". Hindi words missing from the lexicon are
   read as written; add them to `hinglish.tsv`.
4. **Speak** — streamed to the speakers chunk by chunk with
   **[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)**, OpenBMB's 2B model: 4-bit on MLX (its
   audio-feature parts at 8-bit), or bf16 on CUDA with OpenBMB's `voxcpm` package. The `indian_male`
   voice is a deep, calm Indian male, cloned from `src/static/voices/indian_male.wav` (see
   [Voices](#voices)). Speed (default 1.0×) is applied by time-stretching with WSOLA, so pitch stays
   the same.

Nothing leaves the machine.

## Languages

LocalVoice reads Hinglish: English, Hindi in Devanagari, and Roman Hindi. VoxCPM2 itself speaks 30
languages, per its [model card](https://huggingface.co/openbmb/VoxCPM2), with no language tag needed:

Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi,
Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian,
Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese

plus nine Chinese dialects (Sichuanese, Cantonese, Wu, Northeastern, Henan, Shaanxi, Shandong,
Tianjin, Hokkien). Quality varies with each language's training data.

The rest of the pipeline assumes Hinglish, so other languages need some changes first:
transliteration would turn Latin words that are also in the Hinglish lexicon ("ja", "wo", "ab") into
Devanagari; sentences only split on `. ! ? ।` followed by a space, so Chinese and Japanese text isn't
split; the runaway cut-off counts space-separated words, so long sentences in Chinese, Japanese, Thai,
Lao, Khmer and Burmese would be cut short; and `indian_male` is a Hinglish clip, so other languages
would likely get its accent; add a native-speaker voice for each (see [Voices](#voices)).

## Performance (M1 Pro)

| | VoxCPM2 `indian_male` |
|---|---|
| First audio | ~0.25 s (~0.4 s for a long sentence) |
| Real-time factor | ~0.42 (1 s of speech in 0.42 s) |
| Memory | ~2.3 GB idle, ~3.2 GB peak while speaking |
| CPU / GPU while speaking | ~35% of one core / ~95%; both idle otherwise |
| Download | 2.1 GB |
| Weights licence | Apache 2.0 |

On an NVIDIA GPU OpenBMB reports VoxCPM2 at RTF ~0.30 on an RTX 4090 (bf16, ~8 GB of VRAM); their
package streams it 0.16 s at a time.

VoxCPM2 runs 4 diffusion steps per 0.16 s patch instead of its default 10 (`voxcpm2_steps`): Whisper
transcripts were as clean as at 10, and fewer (3) garbles words. mlx-audio's VoxCPM2 only returns a
sentence once it's fully generated, so `src/modules/tts/voxcpm2.py` re-implements its generation loop
to decode as it goes (0.3 s of audio after the first 2 patches, growing to ~1 s chunks), and reads
the voice's transcript into the model once instead of per sentence. It pins `mlx-audio==0.5.9`,
whose internals it uses. About one sentence in 30 the model doesn't stop on its own; it's cut off at
2 s + 0.7 s per word.

Memory on a Mac: the `mlx-community/VoxCPM2-4bit` checkpoint quantizes only the two LMs, so the
audio-feature parts (the DiT and feature encoder) are converted from 16-bit to 8-bit at load, which
also makes them ~17% faster. The clip encoder is freed once the voice is prepared, and MLX's buffer
cache is capped at 256 MB (uncapped, it grew past 9 GB). All MLX work runs on one thread, since MLX
can only use a model on the thread that loaded it.

## Setup

| | macOS (Apple Silicon) | Windows | Linux |
|---|---|---|---|
| VoxCPM2 | MLX (GPU) | NVIDIA GPU (PyTorch CUDA) | NVIDIA GPU (PyTorch CUDA) |
| UI | menu bar | system tray | system tray |
| Hotkey | Right ⌥ | Right Shift | Right Shift |

Common to all: install [uv](https://docs.astral.sh/uv/), then

```sh
make install   # uv sync --all-groups
make model     # download VoxCPM2 (2.1 GB on a Mac) and prepare its voice
make start
```

Try it without the hotkey: `make say TEXT="Kal ki meeting mein timeline discuss karni hai"`, or
`make web` and open http://localhost:8765 to type text, set the speed and steps, and hear it as it's
generated. Each sentence shows its transliteration and timings (first chunk, synthesis time,
real-time factor, any gaps), with replay and download. It loads the model at startup and only
listens on localhost. Run `make` to list every target.

### Speed

`SPEED` (0.5–2.0, default 1.0) and `STEPS` (VoxCPM2 diffusion steps, 2–10, default 4) override
`~/.localvoice.json` for one run:

```sh
make start SPEED=1.25
make say SPEED=1.5 STEPS=4 TEXT="Main abhi busy hoon, baad mein call karta hoon."
```

The menu's Speed submenu changes it while running and saves it. Generation runs only ~2.4× faster
than playback on an M1 Pro, so at high speeds or with more steps it can pause between chunks (the
terminal prints `[audio] N gap(s)`).

### Voices

VoxCPM2 speaks in the voice of a short reference clip, continuing on from it: each voice is a `.wav`
and a `.txt` with exactly what the clip says, in `src/static/voices/`. `indian_male.wav` was itself
generated by VoxCPM2 from the description "A calm, confident middle-aged Indian man with a deep
voice, speaking clearly and professionally like a news anchor", so no real person's voice is cloned.
To add a voice, drop in `<name>.wav` (5–10 s, clean speech) and `<name>.txt`, and add `<name>` to
`VOICES` in `src/core/config/settings.py`. Every sentence re-reads the whole clip, so a longer clip
delays each sentence's first audio (a 7 s clip costs ~0.1 s). On a Mac, each voice is prepared once
(~1 s) and saved in `~/.cache/localvoice/voices`, so later starts skip it; editing the clip or its
transcript prepares it again. Only use recordings of yourself or someone who agreed to it.

After 10 idle minutes LocalVoice frees the model, taking a Mac from ~2.3 GB to ~0.2 GB; the next tap
shows "…" in the menu bar while it reloads (~1.5 s with the files cached, longer if macOS has dropped
them). Set `"idle_unload_minutes"` to another number of minutes, or `0` to keep the model loaded.

Settings live in `~/.localvoice.json` (voice and speed are also in the menu); e.g.
`{"speed": 1.25, "hotkey": "cmd_r", "output_device": "External Headphones"}`. Setting
`"voxcpm2_model": "mlx-community/VoxCPM2-8bit"` uses the 8-bit checkpoint instead (~1 GB more memory).

### macOS (Apple Silicon)

Grant your terminal **Accessibility** and **Input Monitoring** in
System Settings → Privacy & Security (for the hotkey and ⌘C), then restart the app.
Intel Macs aren't supported: there's no MLX for them, nor PyTorch 2.5+, which VoxCPM2 needs.

### Windows

- VoxCPM2 needs an NVIDIA GPU, RTX 20-series or newer with ~8 GB of VRAM, and driver 580+:
  `make install` fetches PyTorch's CUDA 13.0 build (a few GB). It's far too slow on a CPU, so
  LocalVoice won't start without one. `make model` downloads VoxCPM2 (~5 GB).
- Triton isn't available on Windows, so VoxCPM2 runs without `torch.compile` there (it prints a
  warning); Linux gets the compiled, faster path.
- Reading from apps running as Administrator requires running LocalVoice as Administrator.

### Linux

- **X11 only.** Wayland blocks global hotkeys and simulated keystrokes; log in to an X11 session.
- VoxCPM2 needs an NVIDIA GPU as on Windows (PyPI's Linux PyTorch already includes CUDA).
- System packages (Debian/Ubuntu): `sudo apt install libportaudio2 xclip gir1.2-ayatanaappindicator3-0.1`
- GNOME needs the *AppIndicator* extension to show the tray icon.
- Terminals copy with Ctrl+Shift+C, so selections in a terminal aren't read; other apps work.
