# Whisperer

A tiny Windows GUI wrapper around the [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) CLI.
Pick one or more audio/video files (or drag & drop them into the file list), choose a model and options, click **Whisper** — each transcript (txt/srt/vtt/...) lands in the output folder, named after its input file. Files are processed one at a time: a toast notification fires as each finishes, and a broken file only fails itself. If whisper-ctranslate2 is missing, the app offers to install it via pipx.

The model list shows each model's size and strengths and marks the ones already present in the local HuggingFace cache, so you can prefer a model that won't need a download. **Download selected model** fetches a missing model ahead of time (with progress), so the first real transcription doesn't stall on a multi-GB download.

## Run from source

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python whisperer.py
```

## Build a single exe

```
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller Whisperer.spec
```

The exe lands in `dist\Whisperer-1.1.exe`. Settings persist in `%APPDATA%\Whisperer\config.json`.
