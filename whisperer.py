"""Whisperer - a tiny GUI wrapper around the whisper-ctranslate2 CLI."""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import wx

APP_NAME = "Whisperer"
EXE_NAME = "whisper-ctranslate2"
CONFIG_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME / "config.json"

MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3",
          "large-v3-turbo", "distil-large-v3")
LANGUAGES = ("Auto", "en", "vi", "ja", "ko", "zh", "fr", "de", "es", "ru", "id", "th")
TASKS = ("transcribe", "translate")
FORMATS = ("txt", "srt", "vtt", "tsv", "json", "all")

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

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=APP_NAME)
        self.exe = find_whisper()
        self.cfg = read_config()
        self.proc = None
        self.busy = None            # None | "run" | "install"
        self.cancelled = False
        self.pct_seen = False
        self.last_job = None
        self._auto_outdir = ""

        panel = wx.Panel(self)
        outer = wx.BoxSizer(wx.VERTICAL)
        PAD = wx.EXPAND | wx.LEFT | wx.RIGHT

        outer.Add(wx.StaticText(panel, label="Audio file"), 0,
                  wx.LEFT | wx.RIGHT | wx.TOP, 12)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.audio_txt = wx.TextCtrl(panel)
        btn_audio = wx.Button(panel, label="Browse...")
        row.Add(self.audio_txt, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        row.Add(btn_audio, 0)
        outer.Add(row, 0, PAD | wx.TOP, 4)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.model_ch = wx.Choice(panel, choices=list(MODELS))
        self.lang_cb = wx.ComboBox(panel, value="Auto", choices=list(LANGUAGES))
        self.task_ch = wx.Choice(panel, choices=list(TASKS))
        for label, ctrl in (("Model", self.model_ch), ("Language", self.lang_cb),
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
        btn_audio.Bind(wx.EVT_BUTTON, self.on_browse_audio)
        btn_out.Bind(wx.EVT_BUTTON, self.on_browse_out)
        self.whisper_btn.Bind(wx.EVT_BUTTON, self.on_whisper)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.apply_config()

    # ------------------------------------------------------------------ config

    def apply_config(self):
        def pick(ctrl, value, default):
            if not (value and ctrl.SetStringSelection(value)):
                ctrl.SetStringSelection(default)
        pick(self.model_ch, self.cfg.get("model"), "small")
        pick(self.task_ch, self.cfg.get("task"), "transcribe")
        pick(self.fmt_ch, self.cfg.get("output_format"), "txt")
        self.lang_cb.SetValue(self.cfg.get("language") or "Auto")
        self.vad_cb.SetValue(bool(self.cfg.get("vad")))
        self.words_cb.SetValue(bool(self.cfg.get("word_timestamps")))

    def save_config(self):
        self.cfg.update(
            model=self.model_ch.GetStringSelection(),
            language=self.lang_cb.GetValue().strip() or "Auto",
            task=self.task_ch.GetStringSelection(),
            output_format=self.fmt_ch.GetStringSelection(),
            vad=self.vad_cb.GetValue(),
            word_timestamps=self.words_cb.GetValue(),
        )
        write_config(self.cfg)

    # ------------------------------------------------------------------ events

    def on_browse_audio(self, _):
        with wx.FileDialog(self, "Choose an audio or video file",
                           defaultDir=self.cfg.get("last_dir", ""),
                           wildcard=MEDIA_WILDCARD,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        self.audio_txt.SetValue(path)
        self.cfg["last_dir"] = os.path.dirname(path)
        out = self.out_txt.GetValue().strip()
        if not out or out == self._auto_outdir:
            self._auto_outdir = os.path.dirname(path)
            self.out_txt.SetValue(self._auto_outdir)

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
        audio = self.audio_txt.GetValue().strip().strip('"')
        if not (audio and Path(audio).is_file()):
            wx.MessageBox("Please choose an audio file first.", APP_NAME,
                          wx.ICON_WARNING, self)
            return
        outdir = self.out_txt.GetValue().strip() or os.path.dirname(audio) or "."
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError:
            wx.MessageBox("Can't create the output folder. Pick a different one.",
                          APP_NAME, wx.ICON_ERROR, self)
            return
        self.save_config()
        self.start_run({
            "audio": audio,
            "output_dir": outdir,
            "model": self.model_ch.GetStringSelection(),
            "language": self.lang_cb.GetValue().strip() or "Auto",
            "task": self.task_ch.GetStringSelection(),
            "output_format": self.fmt_ch.GetStringSelection(),
            "vad": self.vad_cb.GetValue(),
            "word_timestamps": self.words_cb.GetValue(),
        })

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
        self.save_config()
        self.Destroy()

    # ------------------------------------------------------------------ running

    def start_run(self, job):
        try:
            self.proc = popen_stream(build_cmd(self.exe, job))
        except OSError:
            self.exe = None
            self.prompt_install()
            return
        self.last_job = job
        self.set_busy("run")
        self.set_status("Starting... (the first run may download the model)")
        threading.Thread(target=self._run_worker, args=(self.proc,),
                         daemon=True).start()

    def _run_worker(self, proc):
        rc, tail = stream_process(
            proc, lambda line: wx.CallAfter(self.on_output_line, line))
        wx.CallAfter(self.on_run_finished, rc, tail)

    def on_output_line(self, line):
        m = PCT_TQDM.search(line) or PCT_ANY.search(line)
        if m:
            if not self.pct_seen:
                self.pct_seen = True
                self.pulse.Stop()
            self.gauge.SetValue(min(100, int(m.group(1))))
        else:
            self.set_status(line)

    def on_run_finished(self, rc, tail):
        job = self.last_job
        self.proc = None
        self.set_idle()
        if self.cancelled:
            self.set_status("Cancelled.")
            return
        if rc == 0:
            self.gauge.SetValue(100)
            self.set_status("Done.")
            if wx.MessageBox("Done. Open the output folder?", APP_NAME,
                             wx.YES_NO | wx.ICON_INFORMATION, self) == wx.YES:
                try:
                    os.startfile(job["output_dir"])
                except OSError:
                    pass
            return
        kind, msg = classify_error(rc, tail, job.get("force_cpu", False))
        self.set_status("Failed.")
        if kind == "cuda":
            if wx.MessageBox(msg, APP_NAME,
                             wx.YES_NO | wx.ICON_WARNING, self) == wx.YES:
                self.start_run({**job, "force_cpu": True})
        else:
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
        self.cancel_btn.Enable()
        self.gauge.SetValue(0)
        self.pulse.Start(120)

    def set_idle(self):
        self.busy = None
        self.pulse.Stop()
        self.whisper_btn.Enable()
        self.cancel_btn.Disable()
        self.gauge.SetValue(0)

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
