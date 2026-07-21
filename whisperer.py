"""Whisperer - a tiny GUI wrapper around the whisper-ctranslate2 CLI."""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import wx
import wx.adv

APP_NAME = "Whisperer"
APP_VERSION = "1.1"
EXE_NAME = "whisper-ctranslate2"
CONFIG_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME / "config.json"

MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3",
          "large-v3-turbo", "distil-large-v3")
LANGUAGES = ("Auto", "en", "vi", "ja", "ko", "zh", "fr", "de", "es", "ru", "id", "th")
TASKS = ("transcribe", "translate")
FORMATS = ("txt", "srt", "vtt", "tsv", "json", "all")

# HuggingFace repos faster-whisper downloads each model from; used to detect
# an already-cached model so runs can skip the online version check
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

# approximate download size in MiB + one-line characterization; the size also
# drives the download progress estimate (the CLI prints no progress into pipes)
MODEL_INFO = {
    "tiny": (75, "Fastest, lowest accuracy"),
    "base": (145, "Very fast, basic accuracy"),
    "small": (484, "Good balance of speed and accuracy"),
    "medium": (1536, "Accurate, noticeably slower"),
    "large-v2": (3175, "High accuracy (previous generation)"),
    "large-v3": (3175, "Most accurate"),
    "large-v3-turbo": (1638, "Near large-v3 accuracy, much faster"),
    "distil-large-v3": (1536, "Fast, English only"),
}

MEDIA_WILDCARD = (
    "Audio/Video files|*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.opus;*.aac;*.wma;"
    "*.mp4;*.mkv;*.webm;*.mov;*.avi|All files|*.*"
)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PCT_TQDM = re.compile(r"(\d{1,3})%\|")
PCT_ANY = re.compile(r"\b(\d{1,3})%")

MANUAL_INSTALL = (
    "To install whisper-ctranslate2 manually, run in a terminal:\n\n"
    "    pip install --user pipx\n"
    "    python -m pipx install whisper-ctranslate2\n\n"
    "then click Whisper again."
)


# --------------------------------------------------------------------------- helpers

def utf8_env():
    # whisper-ctranslate2 prints CJK characters and crashes on cp1252 pipes
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def find_whisper():
    exe = shutil.which(EXE_NAME)
    if exe:
        return exe
    # a freshly pipx-installed exe is not on this process's PATH yet
    candidates = [Path.home() / ".local" / "bin" / (EXE_NAME + ".exe")]
    pipx_bin = os.environ.get("PIPX_BIN_DIR")
    if pipx_bin:
        candidates.append(Path(pipx_bin) / (EXE_NAME + ".exe"))
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def find_python():
    # sys.executable is Whisperer.exe when frozen by PyInstaller
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    for name, extra in (("python", []), ("py", ["-3"])):
        path = shutil.which(name)
        if path:
            return [path, *extra]
    return None


def read_config():
    try:
        return json.loads(CONFIG_PATH.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def write_config(cfg):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), "utf-8")
    except OSError:
        pass


def hf_model_dir(model):
    """The model's HuggingFace cache directory (may not exist yet)."""
    repo = MODEL_REPOS.get(model)
    if not repo:
        return None
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        home = os.environ.get("HF_HOME")
        base = Path(home) if home else Path.home() / ".cache" / "huggingface"
        cache = str(base / "hub")
    return Path(cache) / ("models--" + repo.replace("/", "--"))


def model_is_cached(model):
    """True when the model already sits complete in the HuggingFace cache, so
    the run can pass --local_files_only and skip the online version check.

    A snapshot entry for model.bin only appears once the weights finished
    downloading; an interrupted download leaves only *.incomplete blobs."""
    root = hf_model_dir(model)
    if root is None:
        return False
    try:
        return any((snap / "model.bin").is_file()
                   for snap in (root / "snapshots").iterdir())
    except OSError:
        return False


def model_disk_size(model):
    """Bytes the model occupies in the cache (hard-linked files count once)."""
    root = hf_model_dir(model)
    if root is None:
        return 0
    total, seen = 0, set()
    try:
        for path in root.rglob("*"):
            try:
                st = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            key = (st.st_dev, st.st_ino)
            if st.st_ino and key in seen:
                continue
            seen.add(key)
            total += st.st_size
    except OSError:
        pass
    return total


def fmt_size(n):
    if n >= 1 << 30:
        return "%.1f GB" % (n / (1 << 30))
    return "%d MB" % round(n / (1 << 20))


def write_silent_wav(path, seconds=0.5):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


def output_written(job, since):
    """True if the run produced its output file(s).

    whisper-ctranslate2 swallows per-file decode errors and still exits 0,
    so a missing/stale output is the only reliable failure signal."""
    stem = Path(job["audio"]).stem
    fmt = job["output_format"]
    exts = ("txt", "srt", "vtt", "tsv", "json") if fmt == "all" else (fmt,)
    for ext in exts:
        path = Path(job["output_dir"]) / ("%s.%s" % (stem, ext))
        try:
            if path.stat().st_mtime >= since - 2:
                return True
        except OSError:
            pass
    return False


def cache_incomplete(tail):
    # a --local_files_only run tripped over a hole in the HF cache
    text = "\n".join(tail).lower()
    return any(s in text for s in ("local_files_only", "cached snapshot",
                                   "disk cache", "offline"))


def build_cmd(exe, job):
    cmd = [
        exe, job["audio"],
        "--model", job["model"],
        "--task", job["task"],
        "--output_format", job["output_format"],
        "--output_dir", job["output_dir"],
        "--vad_filter", str(bool(job["vad"])),
        "--word_timestamps", str(bool(job["word_timestamps"])),
        "--verbose", "False",
    ]
    if job["language"] != "Auto":
        cmd += ["--language", job["language"]]
    if job.get("local_only"):
        cmd += ["--local_files_only", "True"]
    if job.get("force_cpu"):
        cmd += ["--device", "cpu"]
    return cmd


def classify_error(rc, tail, on_cpu):
    """Map an exit code + last output lines to ('cuda'|'error', friendly message)."""
    text = "\n".join(tail).lower()

    def has(*needles):
        return any(n in text for n in needles)

    if not on_cpu and (rc == 100 or has("cublas", "cudnn", "cuda")):
        return "cuda", ("GPU (CUDA) isn't available on this machine.\n\n"
                        "Retry on CPU?")
    if has("out of memory", "bad_alloc", "failed to allocate"):
        return "error", "Ran out of memory. Try a smaller model."
    if has("huggingface.co", "connection", "getaddrinfo", "timed out",
           "max retries", "ssl"):
        return "error", ("Couldn't download the model. "
                         "Check your internet connection and try again.")
    if has("invalid data", "error opening", "does not contain any stream", "av."):
        return "error", ("Couldn't read this audio/video file. "
                         "It may be corrupt or unsupported.")
    if has("permission denied", "errno 13", "winerror 5"):
        return "error", "Can't write to the output folder. Pick a different one."
    detail = "\n".join(tail[-3:])
    return "error", "Transcription failed (exit code %s).\n\n%s" % (rc, detail)


def popen_stream(cmd):
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
        env=utf8_env(),
    )


def stream_process(proc, on_line):
    """Blocking: relay output to on_line, return (rc, tail).

    tqdm progress updates are separated by bare \\r, so the buffer is split on
    both \\r and \\n; readline() would stay silent until the process exits.
    Progress-bar lines are excluded from the tail kept for error diagnosis.
    """
    tail = collections.deque(maxlen=40)

    def emit(raw):
        line = raw.decode("utf-8", "replace").strip()
        if line:
            if not PCT_TQDM.search(line):
                tail.append(line)
            on_line(line)

    buf = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        *lines, buf = re.split(rb"[\r\n]", buf)
        for raw in lines:
            emit(raw)
    emit(buf)
    return proc.wait(), list(tail)


def run_quiet(cmd):
    try:
        return subprocess.run(cmd, capture_output=True,
                              creationflags=CREATE_NO_WINDOW,
                              env=utf8_env()).returncode
    except OSError:
        return 1


def kill_tree(pid):
    # the pipx shim exe spawns a python.exe child; plain kill() would orphan it
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


# --------------------------------------------------------------------------- UI

class FileDrop(wx.FileDropTarget):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def OnDropFiles(self, x, y, filenames):
        return self.frame.on_drop_files(filenames)


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="%s %s" % (APP_NAME, APP_VERSION))
        self.exe = find_whisper()
        self.cfg = read_config()
        self.proc = None
        self.busy = None            # None | "run" | "install"
        self.cancelled = False
        self.pct_seen = False
        self.files = []
        self.batch = None
        self.download = None        # (model, tmpdir) while busy == "download"
        self._auto_outdir = ""

        panel = wx.Panel(self)
        outer = wx.BoxSizer(wx.VERTICAL)
        PAD = wx.EXPAND | wx.LEFT | wx.RIGHT

        outer.Add(wx.StaticText(panel, label="Input files (browse or drop files here)"),
                  0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.file_list = wx.ListCtrl(panel, style=wx.LC_REPORT)
        self.file_list.InsertColumn(0, "File", width=350)
        self.file_list.InsertColumn(1, "Status", width=120)
        self.file_list.SetMinSize(wx.Size(-1, 110))
        btn_audio = wx.Button(panel, label="Browse...")
        row.Add(self.file_list, 1, wx.EXPAND | wx.RIGHT, 6)
        row.Add(btn_audio, 0)
        outer.Add(row, 1, PAD | wx.TOP, 4)
        self.file_list.SetDropTarget(FileDrop(self))

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(panel, label="Model"), 0, wx.ALIGN_CENTER_VERTICAL)
        row.AddStretchSpacer()
        self.dl_btn = wx.Button(panel, label="Download selected model")
        row.Add(self.dl_btn, 0)
        outer.Add(row, 0, PAD | wx.TOP, 12)
        self.model_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for col, (label, width) in enumerate(
                (("Model", 130), ("Size", 70), ("Notes", 250), ("Status", 110))):
            self.model_list.InsertColumn(col, label, width=width)
        self.model_list.SetMinSize(wx.Size(-1, 178))
        outer.Add(self.model_list, 0, PAD | wx.TOP, 4)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.lang_cb = wx.ComboBox(panel, value="Auto", choices=list(LANGUAGES))
        self.task_ch = wx.Choice(panel, choices=list(TASKS))
        for label, ctrl in (("Language", self.lang_cb),
                            ("Task", self.task_ch)):
            row.Add(wx.StaticText(panel, label=label), 0,
                    wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            row.Add(ctrl, 0, wx.RIGHT, 12)
        outer.Add(row, 0, PAD | wx.TOP, 12)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.fmt_ch = wx.Choice(panel, choices=list(FORMATS))
        row.Add(wx.StaticText(panel, label="Format"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        row.Add(self.fmt_ch, 0, wx.RIGHT, 16)
        self.vad_cb = wx.CheckBox(panel, label="VAD filter")
        self.words_cb = wx.CheckBox(panel, label="Word timestamps")
        row.Add(self.vad_cb, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 16)
        row.Add(self.words_cb, 0, wx.ALIGN_CENTER_VERTICAL)
        outer.Add(row, 0, PAD | wx.TOP, 8)

        outer.Add(wx.StaticText(panel, label="Output folder"), 0,
                  wx.LEFT | wx.RIGHT | wx.TOP, 12)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.out_txt = wx.TextCtrl(panel)
        btn_out = wx.Button(panel, label="Browse...")
        row.Add(self.out_txt, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        row.Add(btn_out, 0)
        outer.Add(row, 0, PAD | wx.TOP, 4)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.whisper_btn = wx.Button(panel, label="Whisper")
        font = self.whisper_btn.GetFont()
        font.SetPointSize(font.GetPointSize() + 2)
        font = font.Bold()
        self.whisper_btn.SetFont(font)
        self.whisper_btn.SetMinSize(wx.Size(180, 40))
        self.whisper_btn.SetDefault()
        self.cancel_btn = wx.Button(panel, label="Cancel")
        self.cancel_btn.Disable()
        row.AddStretchSpacer()
        row.Add(self.whisper_btn, 0, wx.RIGHT, 8)
        row.Add(self.cancel_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        row.AddStretchSpacer()
        outer.Add(row, 0, PAD | wx.TOP, 16)

        self.gauge = wx.Gauge(panel, range=100)
        outer.Add(self.gauge, 0, PAD | wx.TOP, 12)
        self.status = wx.StaticText(panel, label="Ready")
        outer.Add(self.status, 0, PAD | wx.TOP | wx.BOTTOM, 8)

        panel.SetSizer(outer)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizerAndFit(sizer)
        self.SetMinSize(self.GetSize())
        self.SetSize(wx.Size(620, self.GetSize().height))

        self.pulse = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda e: self.gauge.Pulse(), self.pulse)
        self.dl_poll = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_dl_poll, self.dl_poll)
        btn_audio.Bind(wx.EVT_BUTTON, self.on_browse_audio)
        btn_out.Bind(wx.EVT_BUTTON, self.on_browse_out)
        self.whisper_btn.Bind(wx.EVT_BUTTON, self.on_whisper)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.dl_btn.Bind(wx.EVT_BUTTON, self.on_download)
        self.model_list.Bind(wx.EVT_LIST_ITEM_SELECTED,
                             lambda e: self.update_dl_btn())
        self.model_list.Bind(wx.EVT_LIST_ITEM_DESELECTED,
                             lambda e: self.update_dl_btn())
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.refresh_models()
        self.apply_config()

    # ------------------------------------------------------------------ config

    def apply_config(self):
        def pick(ctrl, value, default):
            if not (value and ctrl.SetStringSelection(value)):
                ctrl.SetStringSelection(default)
        self.select_model(self.cfg.get("model"))
        pick(self.task_ch, self.cfg.get("task"), "transcribe")
        pick(self.fmt_ch, self.cfg.get("output_format"), "txt")
        self.lang_cb.SetValue(self.cfg.get("language") or "Auto")
        self.vad_cb.SetValue(bool(self.cfg.get("vad")))
        self.words_cb.SetValue(bool(self.cfg.get("word_timestamps")))

    def save_config(self):
        self.cfg.update(
            model=self.selected_model(),
            language=self.lang_cb.GetValue().strip() or "Auto",
            task=self.task_ch.GetStringSelection(),
            output_format=self.fmt_ch.GetStringSelection(),
            vad=self.vad_cb.GetValue(),
            word_timestamps=self.words_cb.GetValue(),
        )
        write_config(self.cfg)

    # ------------------------------------------------------------------ events

    def on_browse_audio(self, _):
        if self.busy:
            return
        with wx.FileDialog(self, "Choose audio or video files",
                           defaultDir=self.cfg.get("last_dir", ""),
                           wildcard=MEDIA_WILDCARD,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
                           | wx.FD_MULTIPLE) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            paths = dlg.GetPaths()
        self.set_files(paths)

    def on_drop_files(self, paths):
        if self.busy:
            return False
        return self.set_files(paths)

    def set_files(self, paths):
        files = [p for p in paths if Path(p).is_file()]
        if not files:
            return False
        self.files = files
        self.file_list.DeleteAllItems()
        for i, path in enumerate(files):
            self.file_list.InsertItem(i, os.path.basename(path))
        self.cfg["last_dir"] = os.path.dirname(files[0])
        out = self.out_txt.GetValue().strip()
        if not out or out == self._auto_outdir:
            self._auto_outdir = os.path.dirname(files[0])
            self.out_txt.SetValue(self._auto_outdir)
        self.set_status("%d file(s) selected." % len(files))
        return True

    def on_browse_out(self, _):
        with wx.DirDialog(self, "Choose the output folder",
                          defaultPath=self.out_txt.GetValue().strip()) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.out_txt.SetValue(dlg.GetPath())

    def on_whisper(self, _):
        if self.busy:
            return
        if not (self.exe and Path(self.exe).is_file()):
            self.exe = find_whisper()
        if not self.exe:
            self.prompt_install()
            return
        files = [f for f in self.files if Path(f).is_file()]
        if not files:
            wx.MessageBox("Please choose one or more audio files first.",
                          APP_NAME, wx.ICON_WARNING, self)
            return
        if len(files) != len(self.files):
            self.set_files(files)       # some inputs disappeared since selection
        outdir = self.out_txt.GetValue().strip() or os.path.dirname(files[0]) or "."
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError:
            wx.MessageBox("Can't create the output folder. Pick a different one.",
                          APP_NAME, wx.ICON_ERROR, self)
            return
        self.save_config()
        model = self.selected_model()
        local_only = model_is_cached(model)
        self.start_batch([{
            "audio": f,
            "output_dir": outdir,
            "model": model,
            "language": self.lang_cb.GetValue().strip() or "Auto",
            "task": self.task_ch.GetStringSelection(),
            "output_format": self.fmt_ch.GetStringSelection(),
            "vad": self.vad_cb.GetValue(),
            "word_timestamps": self.words_cb.GetValue(),
            "local_only": local_only,
        } for f in files])

    def on_cancel(self, _):
        if self.busy:
            self.cancelled = True
            self.set_status("Cancelling...")
            if self.proc:
                kill_tree(self.proc.pid)

    def on_close(self, event):
        if self.busy and event.CanVeto():
            if wx.MessageBox("A task is still running. Stop it and quit?",
                             APP_NAME, wx.YES_NO | wx.ICON_QUESTION, self) != wx.YES:
                event.Veto()
                return
        self.cancelled = True
        if self.proc:
            kill_tree(self.proc.pid)
        if self.download:
            shutil.rmtree(self.download[1], ignore_errors=True)
        self.save_config()
        self.Destroy()

    # ------------------------------------------------------------------ running

    def start_batch(self, jobs):
        # files run one at a time, each in its own CLI process: a broken file
        # only fails itself, and finished transcripts are on disk immediately
        self.batch = {"jobs": jobs, "index": 0, "ok": 0, "failed": []}
        self.set_busy("run")
        for i in range(len(jobs)):
            self.file_status(i, "Waiting")
        self.start_file()

    def start_file(self):
        b = self.batch
        i, job = b["index"], b["jobs"][b["index"]]
        job["started_at"] = time.time()
        try:
            self.proc = popen_stream(build_cmd(self.exe, job))
        except OSError:
            self.batch = None
            self.set_idle()
            self.exe = None
            self.prompt_install()
            return
        self.file_status(i, "Starting...")
        note = "" if job.get("local_only") else " (may download the model first)"
        self.set_status("[%d/%d] %s — starting...%s"
                        % (i + 1, len(b["jobs"]),
                           os.path.basename(job["audio"]), note))
        threading.Thread(target=self._run_worker, args=(self.proc,),
                         daemon=True).start()

    def _run_worker(self, proc):
        rc, tail = stream_process(
            proc, lambda line: wx.CallAfter(self.on_output_line, line))
        wx.CallAfter(self.on_run_finished, rc, tail)

    def on_output_line(self, line):
        b = self.batch
        if not b:
            return
        i, total = b["index"], len(b["jobs"])
        m = PCT_TQDM.search(line) or PCT_ANY.search(line)
        if m:
            pct = min(100, int(m.group(1)))
            if not self.pct_seen:
                self.pct_seen = True
                self.pulse.Stop()
            self.gauge.SetValue(int((i * 100 + pct) / total))
            self.file_status(i, "%d%%" % pct)
            self.set_status("[%d/%d] %s — %d%%"
                            % (i + 1, total,
                               os.path.basename(b["jobs"][i]["audio"]), pct))
        else:
            self.set_status("[%d/%d] %s" % (i + 1, total, line))

    def on_run_finished(self, rc, tail):
        b = self.batch
        self.proc = None
        if b is None:
            self.set_idle()
            return
        i = b["index"]
        job = b["jobs"][i]
        name = os.path.basename(job["audio"])
        if self.cancelled:
            for row in range(i, len(b["jobs"])):
                self.file_status(row, "Cancelled")
            self.batch = None
            self.set_idle()
            self.set_status("Cancelled.")
            return
        if rc == 0:
            if output_written(job, job["started_at"]):
                b["ok"] += 1
                self.file_status(i, "Done")
                self.notify("Done: %s" % name)
            else:
                # exit 0 but no transcript: the CLI swallowed a per-file error
                _, msg = classify_error(rc, tail, True)
                b["failed"].append((name, msg))
                self.file_status(i, "Failed")
            self.advance()
            return
        if job.get("local_only") and cache_incomplete(tail):
            # the cached model was incomplete after all: redo with network access
            for j in b["jobs"]:
                j["local_only"] = False
            self.start_file()
            return
        kind, msg = classify_error(rc, tail, job.get("force_cpu", False))
        if kind == "cuda":
            if wx.MessageBox(msg, APP_NAME,
                             wx.YES_NO | wx.ICON_WARNING, self) == wx.YES:
                for j in b["jobs"]:
                    j["force_cpu"] = True
                self.start_file()
                return
            # no GPU and CPU declined: every remaining file would fail the same way
            b["failed"].append((name, "GPU (CUDA) isn't available."))
            self.file_status(i, "Failed")
            for row in range(i + 1, len(b["jobs"])):
                self.file_status(row, "Skipped")
            self.finish_batch()
            return
        b["failed"].append((name, msg))
        self.file_status(i, "Failed")
        self.advance()

    def advance(self):
        b = self.batch
        b["index"] += 1
        if not self.pct_seen:
            self.pct_seen = True
            self.pulse.Stop()
        self.gauge.SetValue(int(b["index"] * 100 / len(b["jobs"])))
        if b["index"] < len(b["jobs"]):
            self.start_file()
        else:
            self.finish_batch()

    def finish_batch(self):
        b = self.batch
        self.batch = None
        self.set_idle()
        total, ok, failed = len(b["jobs"]), b["ok"], b["failed"]
        outdir = b["jobs"][0]["output_dir"]
        if ok == total:
            self.gauge.SetValue(100)
            self.set_status("Done.")
            msg = ("Done. %d files transcribed.\n\nOpen the output folder?" % total
                   if total > 1 else "Done. Open the output folder?")
            if wx.MessageBox(msg, APP_NAME,
                             wx.YES_NO | wx.ICON_INFORMATION, self) == wx.YES:
                self.open_folder(outdir)
            return
        detail = "\n".join("%s — %s" % (n, m.splitlines()[0])
                           for n, m in failed[:5])
        if len(failed) > 5:
            detail += "\n(and %d more)" % (len(failed) - 5)
        if ok:
            self.set_status("Done with errors.")
            if wx.MessageBox("Finished: %d ok, %d failed.\n\n%s\n\n"
                             "Open the output folder?" % (ok, len(failed), detail),
                             APP_NAME, wx.YES_NO | wx.ICON_WARNING, self) == wx.YES:
                self.open_folder(outdir)
        else:
            self.set_status("Failed.")
            if len(failed) == 1:
                wx.MessageBox(failed[0][1], APP_NAME, wx.ICON_ERROR, self)
            else:
                wx.MessageBox("All files failed.\n\n%s" % detail,
                              APP_NAME, wx.ICON_ERROR, self)

    def open_folder(self, path):
        try:
            os.startfile(path)
        except OSError:
            pass

    def notify(self, message):
        try:
            wx.adv.NotificationMessage(APP_NAME, message).Show()
        except Exception:
            pass

    def file_status(self, row, text):
        if 0 <= row < self.file_list.GetItemCount():
            self.file_list.SetItem(row, 1, text)

    # ------------------------------------------------------------------ models

    def selected_model(self):
        i = self.model_list.GetFirstSelected()
        return MODELS[i] if i != -1 else "small"

    def select_model(self, name):
        i = MODELS.index(name if name in MODELS else "small")
        self.model_list.Select(i)
        self.model_list.Focus(i)
        self.model_list.EnsureVisible(i)

    def refresh_models(self):
        for i, model in enumerate(MODELS):
            if self.model_list.GetItemCount() <= i:
                self.model_list.InsertItem(i, model)
                self.model_list.SetItem(i, 2, MODEL_INFO[model][1])
            approx = fmt_size(MODEL_INFO[model][0] << 20)
            if model_is_cached(model):
                size = model_disk_size(model)
                self.model_list.SetItem(i, 1, fmt_size(size) if size else approx)
                self.model_list.SetItem(i, 3, "✓ Downloaded")
            else:
                self.model_list.SetItem(i, 1, approx)
                self.model_list.SetItem(i, 3, "")
        self.update_dl_btn()

    def update_dl_btn(self):
        self.dl_btn.Enable(not self.busy
                           and not model_is_cached(self.selected_model()))

    def on_download(self, _):
        if self.busy:
            return
        model = self.selected_model()
        if model_is_cached(model):
            return
        if not (self.exe and Path(self.exe).is_file()):
            self.exe = find_whisper()
        if not self.exe:
            self.prompt_install()
            return
        tmpdir = tempfile.mkdtemp(prefix="whisperer-dl-")
        wav = Path(tmpdir) / "silence.wav"
        try:
            write_silent_wav(wav)
        except OSError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            wx.MessageBox("Couldn't create a temporary file.", APP_NAME,
                          wx.ICON_ERROR, self)
            return
        # transcribing half a second of silence on CPU makes the CLI fetch the
        # model with its own downloader, straight into the right cache
        cmd = [self.exe, str(wav), "--model", model, "--device", "cpu",
               "--language", "en", "--output_dir", tmpdir,
               "--output_format", "txt", "--verbose", "False"]
        try:
            self.proc = popen_stream(cmd)
        except OSError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self.exe = None
            self.prompt_install()
            return
        self.download = (model, tmpdir)
        self.set_busy("download")
        self.set_status("Downloading model %s..." % model)
        self.dl_poll.Start(1000)
        threading.Thread(target=self._download_worker, args=(self.proc,),
                         daemon=True).start()

    def _download_worker(self, proc):
        # the CLI prints no download progress into a pipe; the poll timer
        # tracks the growing cache instead, so lines are only kept for errors
        rc, tail = stream_process(proc, lambda line: None)
        wx.CallAfter(self.on_download_finished, rc, tail)

    def on_dl_poll(self, _):
        if self.busy != "download" or not self.download:
            return
        model = self.download[0]
        done = model_disk_size(model)
        if not done:
            return
        if not self.pct_seen:
            self.pct_seen = True
            self.pulse.Stop()
        pct = min(99, done * 100 // (MODEL_INFO[model][0] << 20))
        self.gauge.SetValue(pct)
        self.set_status("Downloading model %s — %d%% (%s of ~%s)"
                        % (model, pct, fmt_size(done),
                           fmt_size(MODEL_INFO[model][0] << 20)))

    def on_download_finished(self, rc, tail):
        self.dl_poll.Stop()
        self.proc = None
        model, tmpdir = self.download
        self.download = None
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.set_idle()
        if self.cancelled:
            self.set_status("Cancelled.")
            return
        # the temp transcription may fail (e.g. out of memory) after the
        # download itself succeeded; the cache is the ground truth
        if model_is_cached(model):
            self.set_status("Model %s is ready." % model)
            return
        _, msg = classify_error(rc, tail, True)
        self.set_status("Download failed.")
        wx.MessageBox(msg, APP_NAME, wx.ICON_ERROR, self)

    # ------------------------------------------------------------------ install

    def prompt_install(self):
        dlg = wx.MessageDialog(
            self,
            "whisper-ctranslate2 was not found on this computer.\n\n"
            "Whisperer can install it for you with pipx "
            "(needs internet, takes a few minutes).",
            APP_NAME, wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION)
        dlg.SetYesNoCancelLabels("Install automatically", "Show instructions",
                                 "Not now")
        choice = dlg.ShowModal()
        dlg.Destroy()
        if choice == wx.ID_YES:
            self.set_busy("install")
            self.set_status("Looking for pipx...")
            threading.Thread(target=self._install_worker, daemon=True).start()
        elif choice == wx.ID_NO:
            wx.MessageBox(MANUAL_INSTALL, APP_NAME, wx.ICON_INFORMATION, self)

    def _install_worker(self):
        runner = ["pipx"] if shutil.which("pipx") else None
        if runner is None:
            py = find_python()
            if py is None:
                wx.CallAfter(self.on_install_done,
                             "Python was not found on this computer.")
                return
            if run_quiet([*py, "-m", "pipx", "--version"]) != 0:
                wx.CallAfter(self.set_status, "Installing pipx...")
                run_quiet([*py, "-m", "pip", "install", "--user", "pipx"])
                if run_quiet([*py, "-m", "pipx", "--version"]) != 0:
                    wx.CallAfter(self.on_install_done, "Could not install pipx.")
                    return
            runner = [*py, "-m", "pipx"]
        if self.cancelled:
            wx.CallAfter(self.on_install_done, "cancelled")
            return
        wx.CallAfter(self.set_status,
                     "Installing whisper-ctranslate2 (this may take a few minutes)...")
        try:
            proc = popen_stream([*runner, "install", "--force", EXE_NAME])
        except OSError as exc:
            wx.CallAfter(self.on_install_done, str(exc))
            return
        self.proc = proc
        rc, tail = stream_process(
            proc, lambda line: wx.CallAfter(self.set_status, line))
        wx.CallAfter(self.on_install_done,
                     None if rc == 0 else "\n".join(tail[-3:]))

    def on_install_done(self, error):
        self.proc = None
        self.set_idle()
        if self.cancelled:
            self.set_status("Cancelled.")
            return
        self.exe = find_whisper()
        if self.exe and error is None:
            self.set_status("whisper-ctranslate2 installed.")
            wx.MessageBox("whisper-ctranslate2 was installed successfully.\n"
                          "You can start transcribing now.",
                          APP_NAME, wx.ICON_INFORMATION, self)
        else:
            self.set_status("Install failed.")
            detail = "\n\nDetails:\n%s" % error if error else ""
            wx.MessageBox("Automatic install didn't work.%s\n\n%s"
                          % (detail, MANUAL_INSTALL),
                          APP_NAME, wx.ICON_ERROR, self)

    # ------------------------------------------------------------------ state

    def set_busy(self, mode):
        self.busy = mode
        self.cancelled = False
        self.pct_seen = False
        self.whisper_btn.Disable()
        self.dl_btn.Disable()
        self.cancel_btn.Enable()
        self.gauge.SetValue(0)
        self.pulse.Start(120)

    def set_idle(self):
        self.busy = None
        self.pulse.Stop()
        self.whisper_btn.Enable()
        self.cancel_btn.Disable()
        self.gauge.SetValue(0)
        self.refresh_models()   # a run or download may have cached a model

    def set_status(self, text):
        text = " ".join(text.split())
        if len(text) > 90:
            text = text[:87] + "..."
        self.status.SetLabel(text.replace("&", "&&"))


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except OSError:
            pass
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    if frame.exe is None:
        wx.CallAfter(frame.prompt_install)
    app.MainLoop()


if __name__ == "__main__":
    main()
