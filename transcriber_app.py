# LocalScribe - desktop app
# Author: limmieo (github.com/limmieo)

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, END, LEFT, RIGHT, StringVar, Text, Tk, filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import transcribe_core

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None


DEFAULT_OUTPUT = ROOT / "output"
MEDIA_TYPES = {
    ".mp3", ".mp4", ".m4a", ".mov", ".mkv", ".wav", ".webm",
    ".avi", ".aac", ".flac", ".ogg",
}


def _positive_int(value: str) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


class TranscriberApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("LocalScribe")
        self.root.geometry("980x720")
        self.root.minsize(820, 620)
        self.files: list[Path] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.model = StringVar(value="turbo")
        self.language = StringVar(value="English")
        self.output_dir = StringVar(value=str(DEFAULT_OUTPUT))
        self.word_timestamps = BooleanVar(value=False)
        self.exports = {
            "plain_text": BooleanVar(value=True),
            "timestamped_text": BooleanVar(value=True),
            "srt": BooleanVar(value=True),
            "vtt": BooleanVar(value=False),
            "csv_file": BooleanVar(value=False),
            "json_file": BooleanVar(value=True),
            "markdown": BooleanVar(value=True),
            "highlights": BooleanVar(value=True),
        }
        self.diarize = BooleanVar(value=False)
        self.min_speakers = StringVar(value="")
        self.max_speakers = StringVar(value="")
        self.hf_token = StringVar(value=transcribe_core.resolve_hf_token() or "")
        self._configure_style()
        self._build()
        self.root.after(100, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", foreground="#56616f")
        style.configure("Drop.TLabel", background="#eef3f8", foreground="#31465a", padding=28)
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11), padding=(18, 10))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="LocalScribe", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Drop in audio or video, choose what you need, and get publishing-ready files.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 18))

        self.drop = ttk.Label(
            outer,
            text="Drop files here\nor click to browse",
            style="Drop.TLabel",
            anchor="center",
            justify="center",
            cursor="hand2",
        )
        self.drop.pack(fill="x")
        self.drop.bind("<Button-1>", lambda _event: self.browse())
        if DND_FILES and hasattr(self.drop, "drop_target_register"):
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop.configure(text="Click to browse for audio or video\nInstall tkinterdnd2 to enable drag-and-drop")

        queue_frame = ttk.Frame(outer)
        queue_frame.pack(fill="x", pady=(12, 18))
        self.file_list = ttk.Treeview(queue_frame, columns=("file",), show="headings", height=4)
        self.file_list.heading("file", text="Queued files")
        self.file_list.column("file", anchor="w")
        self.file_list.pack(side=LEFT, fill="x", expand=True)
        ttk.Button(queue_frame, text="Remove", command=self.remove_selected).pack(side=RIGHT, padx=(10, 0))

        settings = ttk.LabelFrame(outer, text="Settings", padding=16)
        settings.pack(fill="x")
        ttk.Label(settings, text="Whisper model").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.model,
            values=("tiny", "base", "small", "medium", "turbo", "large"),
            state="readonly",
            width=14,
        ).grid(row=1, column=0, sticky="w", padx=(0, 22), pady=(4, 12))
        ttk.Label(settings, text="Language").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.language,
            values=("Auto detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Thai"),
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=(0, 22), pady=(4, 12))
        ttk.Checkbutton(
            settings,
            text="Word-level timestamps (slower)",
            variable=self.word_timestamps,
        ).grid(row=1, column=2, sticky="w", pady=(4, 12))

        ttk.Label(settings, text="Exports").grid(row=2, column=0, sticky="w", pady=(4, 5))
        export_labels = [
            ("plain_text", "Plain text"),
            ("timestamped_text", "Timestamped text"),
            ("srt", "SRT captions"),
            ("vtt", "VTT captions"),
            ("csv_file", "CSV"),
            ("json_file", "JSON data"),
            ("markdown", "Markdown"),
            ("highlights", "Suggested highlights"),
        ]
        for index, (key, label) in enumerate(export_labels):
            ttk.Checkbutton(settings, text=label, variable=self.exports[key]).grid(
                row=3 + index // 4,
                column=index % 4,
                sticky="w",
                padx=(0, 22),
                pady=3,
            )

        ttk.Label(settings, text="Output folder").grid(row=5, column=0, sticky="w", pady=(12, 4))
        ttk.Entry(settings, textvariable=self.output_dir).grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=(0, 10)
        )
        ttk.Button(settings, text="Browse", command=self.choose_output).grid(row=6, column=3)
        settings.columnconfigure(2, weight=1)

        speakers = ttk.LabelFrame(outer, text="Speakers", padding=16)
        speakers.pack(fill="x", pady=(14, 0))
        ttk.Checkbutton(
            speakers,
            text="Identify speakers (diarization)",
            variable=self.diarize,
            command=self._update_speaker_fields,
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(speakers, text="Min speakers (optional)").grid(row=1, column=0, sticky="w", pady=(10, 4))
        ttk.Label(speakers, text="Max speakers (optional)").grid(row=1, column=1, sticky="w", pady=(10, 4), padx=(12, 0))
        ttk.Label(speakers, text="Hugging Face token").grid(row=1, column=2, sticky="w", pady=(10, 4), padx=(12, 0))

        self.min_speakers_entry = ttk.Entry(speakers, textvariable=self.min_speakers, width=10)
        self.min_speakers_entry.grid(row=2, column=0, sticky="w")
        self.max_speakers_entry = ttk.Entry(speakers, textvariable=self.max_speakers, width=10)
        self.max_speakers_entry.grid(row=2, column=1, sticky="w", padx=(12, 0))
        self.hf_token_entry = ttk.Entry(speakers, textvariable=self.hf_token, show="*", width=40)
        self.hf_token_entry.grid(row=2, column=2, sticky="ew", padx=(12, 0))
        speakers.columnconfigure(2, weight=1)

        self.speaker_hint = ttk.Label(
            speakers,
            text=(
                "One-time setup: accept the terms for pyannote/segmentation-3.0 and "
                "pyannote/speaker-diarization-3.1 on huggingface.co, then paste a read "
                "token above. It's saved locally in config.json."
            ),
            style="Subtitle.TLabel",
            wraplength=850,
        )
        self.speaker_hint.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        self._update_speaker_fields()

        action = ttk.Frame(outer)
        action.pack(fill="x", pady=(16, 8))
        self.start_button = ttk.Button(
            action, text="Transcribe queued files", style="Accent.TButton", command=self.start
        )
        self.start_button.pack(side=LEFT)
        self.progress = ttk.Progressbar(action, mode="indeterminate")
        self.progress.pack(side=LEFT, fill="x", expand=True, padx=(14, 0))
        self.status = StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w")
        self.log = Text(outer, height=8, wrap="word", font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        ttk.Label(
            outer,
            text="LocalScribe · Built by limmieo (github.com/limmieo)",
            style="Subtitle.TLabel",
        ).pack(anchor="center", pady=(8, 0))

    def browse(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose audio or video",
            filetypes=[("Audio and video", " ".join(f"*{ext}" for ext in sorted(MEDIA_TYPES))), ("All files", "*.*")],
        )
        self.add_files(paths)

    def _on_drop(self, event) -> None:
        self.add_files(self.root.tk.splitlist(event.data))

    def add_files(self, paths) -> None:
        for value in paths:
            path = Path(value)
            if path.is_file() and path.suffix.lower() in MEDIA_TYPES and path not in self.files:
                self.files.append(path)
                self.file_list.insert("", END, values=(str(path),))
        self.status.set(f"{len(self.files)} file(s) queued")

    def remove_selected(self) -> None:
        selected = self.file_list.selection()
        values = {self.file_list.item(item, "values")[0] for item in selected}
        self.files = [path for path in self.files if str(path) not in values]
        for item in selected:
            self.file_list.delete(item)
        self.status.set(f"{len(self.files)} file(s) queued")

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_dir.get())
        if selected:
            self.output_dir.set(selected)

    def _update_speaker_fields(self) -> None:
        state = "normal" if self.diarize.get() else "disabled"
        self.min_speakers_entry.configure(state=state)
        self.max_speakers_entry.configure(state=state)
        self.hf_token_entry.configure(state=state)

    def start(self) -> None:
        if self.running:
            return
        if not self.files:
            messagebox.showinfo("No files queued", "Add at least one audio or video file first.")
            return
        if not any(value.get() for value in self.exports.values()):
            messagebox.showinfo("No exports selected", "Choose at least one output format.")
            return
        destination = Path(self.output_dir.get()).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        options = transcribe_core.ExportOptions(**{key: value.get() for key, value in self.exports.items()})
        if self.diarize.get():
            hf_token = self.hf_token.get().strip()
            if hf_token:
                transcribe_core.save_config({"hf_token": hf_token})
        files = self.files.copy()
        self.running = True
        self.start_button.configure(state="disabled")
        self.progress.start(12)
        self._append_log(f"Starting {len(files)} file(s) with Whisper {self.model.get()}...\n")
        threading.Thread(target=self._worker, args=(files, destination, options), daemon=True).start()

    def _worker(self, files: list[Path], destination: Path, options: transcribe_core.ExportOptions) -> None:
        try:
            for number, media_path in enumerate(files, 1):
                self.events.put(("status", f"Transcribing {number}/{len(files)}: {media_path.name}"))
                self.events.put(("log", f"\n[{number}/{len(files)}] {media_path.name}\n"))
                command = [
                    sys.executable, "-m", "whisper", str(media_path),
                    "--model", self.model.get(),
                    "--output_format", "json",
                    "--output_dir", str(destination),
                    "--verbose", "False",
                ]
                language = self.language.get().strip()
                if language and language != "Auto detect":
                    command.extend(["--language", language])
                if self.word_timestamps.get():
                    command.extend(["--word_timestamps", "True"])
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    if line.strip():
                        self.events.put(("log", line))
                if process.wait():
                    raise RuntimeError(f"Whisper failed while processing {media_path.name}")
                json_path = destination / f"{media_path.stem}.json"
                if not json_path.exists():
                    raise FileNotFoundError(f"Whisper did not create {json_path.name}")

                if self.diarize.get():
                    hf_token = transcribe_core.resolve_hf_token(self.hf_token.get().strip() or None)
                    if not hf_token:
                        self.events.put(("log", "Speaker labels skipped: no Hugging Face token configured.\n"))
                    else:
                        self.events.put(("status", f"Identifying speakers in {media_path.name}..."))
                        try:
                            data = json.loads(json_path.read_text(encoding="utf-8"))
                            segments = data.get("segments", [])
                            transcribe_core.diarize_segments(
                                media_path,
                                segments,
                                hf_token,
                                min_speakers=_positive_int(self.min_speakers.get()),
                                max_speakers=_positive_int(self.max_speakers.get()),
                            )
                            data["segments"] = segments
                            json_path.write_text(json.dumps(data), encoding="utf-8")
                        except Exception as exc:
                            self.events.put(("log", f"Speaker labels skipped: {exc}\n"))

                created = transcribe_core.write_exports(json_path, destination, options, source_name=media_path.name)
                self.events.put(("log", f"Created {len(created)} export(s) in {destination}\n"))
            self.events.put(("done", destination))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text)
        self.log.see(END)
        self.log.configure(state="disabled")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "status":
                    self.status.set(str(payload))
                elif kind == "done":
                    self._finish()
                    self.status.set(f"Done. Files saved to {payload}")
                    messagebox.showinfo("Transcription complete", f"Your files are ready in:\n{payload}")
                elif kind == "error":
                    self._finish()
                    self.status.set("Transcription failed")
                    self._append_log(f"\nERROR: {payload}\n")
                    messagebox.showerror("Transcription failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish(self) -> None:
        self.running = False
        self.progress.stop()
        self.start_button.configure(state="normal")


def main() -> None:
    root = TkinterDnD.Tk() if TkinterDnD else Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
