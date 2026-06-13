from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import transcribe_core

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
INDEX_FILE = ROOT / "index.html"
HISTORY_PATH = ROOT / "history.json"
HOST = "127.0.0.1"
PORT = 8765

ALLOWED_FORMATS = {"txt", "timestamped_txt", "srt", "vtt", "tsv", "json", "highlights"}
MAX_HISTORY = 20

history_lock = threading.Lock()


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def add_history_entry(entry: dict) -> None:
    with history_lock:
        history = load_history()
        history.insert(0, entry)
        HISTORY_PATH.write_text(json.dumps(history[:MAX_HISTORY], indent=2), encoding="utf-8")


@dataclass
class Job:
    state: str = "idle"
    filename: str = ""
    message: str = "Ready"
    started_at: float | None = None
    finished_at: float | None = None
    outputs: list[str] = field(default_factory=list)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "filename": self.filename,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outputs": self.outputs,
            "error": self.error,
        }


job = Job()
job_lock = threading.Lock()


def safe_filename(value: str) -> str:
    name = Path(unquote(value)).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or f"upload-{int(time.time())}.media"


def _positive_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def run_transcription(source: Path, options: dict) -> None:
    selected = set(options.get("formats", [])) & ALLOWED_FORMATS
    selected = selected or {"txt", "srt"}
    model = options.get("model", "turbo")
    language = options.get("language", "auto")
    task = options.get("task", "transcribe")
    diarize = bool(options.get("diarize"))
    min_speakers = _positive_int(options.get("min_speakers"))
    max_speakers = _positive_int(options.get("max_speakers"))
    hf_token_input = (options.get("hf_token") or "").strip()
    highlight_preset = options.get("highlight_preset", transcribe_core.DEFAULT_HIGHLIGHT_PRESET)
    if highlight_preset not in transcribe_core.HIGHLIGHT_PRESETS:
        highlight_preset = transcribe_core.DEFAULT_HIGHLIGHT_PRESET

    with job_lock:
        job.state = "running"
        job.message = f"Transcribing {source.name} with Whisper {model}..."
        job.started_at = time.time()
        job.finished_at = None
        job.outputs = []
        job.error = ""

    command = [
        sys.executable,
        "-m",
        "whisper",
        str(source),
        "--model",
        model,
        "--task",
        task,
        "--output_format",
        "json",
        "--output_dir",
        str(OUTPUT_DIR),
        "--verbose",
        "False",
    ]
    if language != "auto":
        command.extend(["--language", language])

    try:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(details[-4000:] or f"Whisper exited with code {result.returncode}")

        stem = source.stem
        json_path = OUTPUT_DIR / f"{stem}.json"
        if not json_path.exists():
            raise RuntimeError("Whisper finished, but its JSON transcript was not found.")

        note = ""
        if diarize:
            if hf_token_input:
                transcribe_core.save_config({"hf_token": hf_token_input})
            hf_token = transcribe_core.resolve_hf_token(hf_token_input or None)
            if not hf_token:
                note = " Speaker labels skipped: no Hugging Face token configured."
            else:
                with job_lock:
                    job.message = f"Identifying speakers in {source.name}..."
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    segments = data.get("segments", [])
                    transcribe_core.diarize_segments(
                        source,
                        segments,
                        hf_token,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    data["segments"] = segments
                    json_path.write_text(json.dumps(data), encoding="utf-8")
                except Exception as exc:
                    note = f" Speaker labels skipped: {exc}"

        export_options = transcribe_core.ExportOptions(
            plain_text="txt" in selected,
            timestamped_text="timestamped_txt" in selected,
            srt="srt" in selected,
            vtt="vtt" in selected,
            tsv="tsv" in selected,
            json_file="json" in selected,
            highlights="highlights" in selected,
            highlight_preset=highlight_preset,
        )
        generated = transcribe_core.write_exports(json_path, OUTPUT_DIR, export_options, source_name=source.name)

        finished_at = time.time()
        outputs = sorted(path.name for path in generated)
        with job_lock:
            job.state = "complete"
            job.message = f"Finished {source.name}.{note}"
            job.finished_at = finished_at
            job.outputs = outputs
        add_history_entry({
            "filename": source.name,
            "finished_at": finished_at,
            "state": "complete",
            "outputs": outputs,
            "note": note.strip(),
        })
    except Exception as exc:
        finished_at = time.time()
        with job_lock:
            job.state = "error"
            job.message = "Transcription failed"
            job.finished_at = finished_at
            job.error = str(exc)
        add_history_entry({
            "filename": source.name,
            "finished_at": finished_at,
            "state": "error",
            "error": str(exc),
        })


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalTranscriber/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {format % args}")

    def send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            body = INDEX_FILE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            with job_lock:
                self.send_json(job.as_dict())
        elif path == "/api/config":
            self.send_json({"hf_token_configured": transcribe_core.has_hf_token()})
        elif path == "/api/history":
            self.send_json({"history": load_history()})
        elif path == "/api/output":
            name = Path(unquote(parse_qs(parsed.query).get("name", [""])[0])).name
            file_path = OUTPUT_DIR / name
            if not name or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = file_path.read_text(encoding="utf-8", errors="replace").encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/files":
            files = [
                {
                    "name": item.name,
                    "size": item.stat().st_size,
                    "modified": item.stat().st_mtime,
                }
                for item in sorted(INPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                if item.is_file()
            ]
            self.send_json({"files": files})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload":
            self.handle_upload()
        elif path == "/api/transcribe":
            self.handle_transcribe()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def handle_upload(self) -> None:
        filename = safe_filename(self.headers.get("X-Filename", ""))
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_json({"error": "The uploaded file was empty."}, HTTPStatus.BAD_REQUEST)
            return

        destination = INPUT_DIR / filename
        remaining = length
        with destination.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)

        if remaining:
            destination.unlink(missing_ok=True)
            self.send_json({"error": "The upload was interrupted."}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"filename": destination.name})

    def handle_transcribe(self) -> None:
        with job_lock:
            if job.state == "running":
                self.send_json({"error": "A transcription is already running."}, HTTPStatus.CONFLICT)
                return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            options = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid transcription options."}, HTTPStatus.BAD_REQUEST)
            return

        filename = safe_filename(options.get("filename", ""))
        source = INPUT_DIR / filename
        if not source.is_file():
            self.send_json({"error": "Choose or upload a valid input file."}, HTTPStatus.BAD_REQUEST)
            return

        with job_lock:
            job.filename = filename
            job.state = "queued"
            job.message = f"Starting {filename}..."
            job.error = ""
            job.outputs = []

        thread = threading.Thread(target=run_transcription, args=(source, options), daemon=True)
        thread.start()
        self.send_json({"ok": True, "filename": filename}, HTTPStatus.ACCEPTED)


def main() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not INDEX_FILE.exists():
        raise SystemExit(f"Missing UI file: {INDEX_FILE}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    address = f"http://{HOST}:{PORT}"
    print(f"LocalScribe is running at {address}")
    print("Press Ctrl+C in this window to stop it.")
    threading.Timer(0.7, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LocalScribe.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
