# Whisperer

A tiny Windows GUI wrapper around the [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) CLI.
Pick an audio/video file, choose a model and options, click **Whisper** — get transcripts (txt/srt/vtt/...) in the output folder. If whisper-ctranslate2 is missing, the app offers to install it via pipx.

## Run from source

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python whisperer.py
```

## Build a single exe

```
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --noconsole --onefile --name Whisperer whisperer.py
```

The exe lands in `dist\Whisperer.exe`. Settings persist in `%APPDATA%\Whisperer\config.json`.
