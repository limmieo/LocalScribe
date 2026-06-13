from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def timestamp(seconds: float, milliseconds: bool = True, separator: str = ".") -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}{separator}{ms:03d}" if milliseconds else base


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_segments(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        segment
        for segment in data.get("segments", [])
        if segment.get("text", "").strip()
    ]


# ---------------------------------------------------------------------------
# Config / token storage
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(data: dict) -> None:
    config = load_config()
    config.update(data)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def resolve_hf_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    token = load_config().get("hf_token")
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


def has_hf_token() -> bool:
    return bool(resolve_hf_token())


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def assign_speakers(segments: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """Assign a human-readable speaker label to each segment.

    ``turns`` is a list of ``(start, end, raw_label)`` diarization turns.
    Labels are renumbered to ``Speaker 1``, ``Speaker 2``, ... in order of
    first appearance (by turn start time).
    """
    if not turns:
        return segments

    mapping: dict[str, str] = {}
    for _, _, raw_label in sorted(turns, key=lambda turn: turn[0]):
        if raw_label not in mapping:
            mapping[raw_label] = f"Speaker {len(mapping) + 1}"

    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        best_label = None
        best_overlap = 0.0
        for start, end, raw_label in turns:
            overlap = min(seg_end, end) - max(seg_start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = raw_label
        if best_label is None:
            def gap(turn: tuple[float, float, str]) -> float:
                start, end, _ = turn
                if seg_end <= start:
                    return start - seg_end
                if seg_start >= end:
                    return seg_start - end
                return 0.0
            best_label = min(turns, key=gap)[2]
        segment["speaker"] = mapping[best_label]

    return segments


def _patch_hf_hub_download_for_legacy_token() -> None:
    """pyannote.audio 3.3.x calls ``hf_hub_download(..., use_auth_token=...)``,
    a kwarg removed in newer ``huggingface_hub`` releases in favor of
    ``token``. Patch it in place so both old pyannote and new
    huggingface_hub versions work together."""
    import sys

    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_legacy_token_patch", False):
        return

    def compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            value = kwargs.pop("use_auth_token")
            if value is not None:
                kwargs.setdefault("token", value)
        return original(*args, **kwargs)

    compat_hf_hub_download._legacy_token_patch = True
    huggingface_hub.hf_hub_download = compat_hf_hub_download

    for module_name in ("pyannote.audio.core.pipeline", "pyannote.audio.core.model"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "hf_hub_download"):
            module.hf_hub_download = compat_hf_hub_download


def _patch_speechbrain_lazy_module_windows() -> None:
    """speechbrain's LazyModule.ensure_module() is supposed to raise
    AttributeError (instead of importing its target) when the call came from
    `inspect.py`, since `inspect.stack()` (used by pytorch_lightning) probes
    `__file__` on every module on the call stack. The check is
    ``filename.endswith("/inspect.py")``, which never matches on Windows
    (paths use backslashes), so the probe falls through to a real import of
    optional integrations like ``speechbrain.integrations.k2_fsa`` and fails
    with ImportError when the optional ``k2`` package isn't installed. Patch
    the check to be path-separator-agnostic."""
    import os

    try:
        from speechbrain.utils import importutils
    except ImportError:
        return

    if getattr(importutils.LazyModule.ensure_module, "_windows_path_patch", False):
        return

    original = importutils.LazyModule.ensure_module

    def patched_ensure_module(self, stacklevel: int = 1):
        import inspect
        import sys

        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            pass

        if importer_frame is not None and os.path.basename(importer_frame.filename) == "inspect.py":
            raise AttributeError()

        return original(self, stacklevel)

    patched_ensure_module._windows_path_patch = True
    importutils.LazyModule.ensure_module = patched_ensure_module


def _allow_pyannote_checkpoint_globals() -> None:
    """pyannote 3.x checkpoints embed a handful of small support objects
    (TorchVersion, pyannote's Specifications/Problem/Resolution enums) that
    torch >= 2.6 rejects under its new default ``torch.load(weights_only=True)``.
    Allow-list just those specific classes rather than disabling the safety
    check entirely."""
    import torch
    from torch.torch_version import TorchVersion
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    torch.serialization.add_safe_globals([TorchVersion, Specifications, Problem, Resolution])


def _extract_audio_to_wav(media_path: Path, destination: Path) -> None:
    """pyannote.audio loads files via torchaudio/soundfile, which can't read
    container formats like .mkv/.mp4. Extract a 16kHz mono WAV with ffmpeg
    (already required by Whisper) for the diarization pipeline to consume."""
    import subprocess

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-2000:] or "ffmpeg audio extraction failed")


def diarize_segments(
    media_path: Path,
    segments: list[dict],
    hf_token: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str | None = None,
) -> list[dict]:
    """Run pyannote diarization on ``media_path`` and tag ``segments`` with speakers."""
    _patch_hf_hub_download_for_legacy_token()
    _allow_pyannote_checkpoint_globals()
    _patch_speechbrain_lazy_module_windows()

    import tempfile

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    kwargs: dict[str, int] = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / "audio.wav"
        _extract_audio_to_wav(Path(media_path), audio_path)
        diarization = pipeline(str(audio_path), **kwargs)

    turns = [
        (turn.start, turn.end, label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]
    return assign_speakers(segments, turns)


# ---------------------------------------------------------------------------
# Export writers
# ---------------------------------------------------------------------------

def _speaker_prefix(segment: dict) -> str:
    speaker = segment.get("speaker")
    return f"{speaker}: " if speaker else ""


def write_plain_text(segments: list[dict], destination: Path) -> None:
    if any(segment.get("speaker") for segment in segments):
        blocks: list[str] = []
        current_speaker: object = object()
        buffer: list[str] = []
        for segment in segments:
            speaker = segment.get("speaker")
            if speaker != current_speaker:
                if buffer:
                    blocks.append(" ".join(buffer))
                buffer = []
                current_speaker = speaker
                if speaker:
                    buffer.append(f"{speaker}:")
            buffer.append(clean_text(segment["text"]))
        if buffer:
            blocks.append(" ".join(buffer))
        destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    else:
        text = clean_text(" ".join(segment["text"] for segment in segments))
        destination.write_text(text + "\n", encoding="utf-8")


def write_timestamped_text(segments: list[dict], destination: Path) -> None:
    lines = [
        f"[{timestamp(segment['start'])} - {timestamp(segment['end'])}] "
        f"{_speaker_prefix(segment)}{clean_text(segment['text'])}"
        for segment in segments
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments: list[dict], destination: Path) -> None:
    blocks = []
    for number, segment in enumerate(segments, 1):
        blocks.append(
            f"{number}\n"
            f"{timestamp(segment['start'], True, ',')} --> {timestamp(segment['end'], True, ',')}\n"
            f"{_speaker_prefix(segment)}{clean_text(segment['text'])}"
        )
    destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_vtt(segments: list[dict], destination: Path) -> None:
    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            f"{timestamp(segment['start'], True, '.')} --> {timestamp(segment['end'], True, '.')}\n"
            f"{_speaker_prefix(segment)}{clean_text(segment['text'])}"
        )
    destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_tsv(segments: list[dict], destination: Path) -> None:
    lines = ["start\tend\ttext"]
    for segment in segments:
        start_ms = round(float(segment["start"]) * 1000)
        end_ms = round(float(segment["end"]) * 1000)
        text = f"{_speaker_prefix(segment)}{clean_text(segment['text'])}"
        lines.append(f"{start_ms}\t{end_ms}\t{text}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(segments: list[dict], destination: Path) -> None:
    has_speaker = any(segment.get("speaker") for segment in segments)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        header = ["start", "end", "duration_seconds", "text"]
        if has_speaker:
            header.append("speaker")
        writer.writerow(header)
        for segment in segments:
            row = [
                timestamp(segment["start"], True, "."),
                timestamp(segment["end"], True, "."),
                round(segment["end"] - segment["start"], 3),
                clean_text(segment["text"]),
            ]
            if has_speaker:
                row.append(segment.get("speaker", ""))
            writer.writerow(row)


def write_markdown(segments: list[dict], destination: Path, title: str, language: str = "unknown") -> None:
    lines = [
        f"# {title}",
        "",
        f"Language: {language}",
        "",
        "## Transcript",
        "",
    ]
    for segment in segments:
        speaker = segment.get("speaker")
        prefix = f"**{speaker}:** " if speaker else ""
        lines.append(f"**[{timestamp(segment['start'])}]** {prefix}{clean_text(segment['text'])}")
    destination.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------

HOOK_TERMS = {
    "actually": 1.0,
    "amazing": 1.3,
    "best": 1.0,
    "biggest": 1.3,
    "changed": 1.0,
    "crazy": 1.2,
    "important": 1.1,
    "learned": 1.0,
    "mistake": 1.4,
    "never": 1.0,
    "problem": 0.8,
    "secret": 1.5,
    "surprised": 1.2,
    "truth": 1.3,
    "why": 0.8,
}


HIGHLIGHT_PRESETS = {
    "standard": {
        "label": "Standard (15-60s)",
        "min_duration": 15,
        "max_duration": 60,
        "ideal_min": 18,
        "ideal_max": 60,
    },
    "shorts": {
        "label": "Shorts / Reels / TikTok (15-45s)",
        "min_duration": 15,
        "max_duration": 45,
        "ideal_min": 15,
        "ideal_max": 35,
    },
}
DEFAULT_HIGHLIGHT_PRESET = "standard"


def highlight_score(text: str, duration: float, ideal_min: float = 18, ideal_max: float = 60) -> float:
    lowered = text.lower()
    words = re.findall(r"\b[\w']+\b", lowered)
    score = sum(weight for term, weight in HOOK_TERMS.items() if term in lowered)
    score += min(text.count("?") * 0.8, 1.6)
    score += min(text.count("!") * 0.5, 1.0)
    score += 0.8 if re.search(r"\b\d+(?:\.\d+)?%?\b", text) else 0
    score += 0.8 if ideal_min <= duration <= ideal_max else 0
    score += min(len(words) / 80, 1.0)
    return score


def build_highlight_windows(
    segments: list[dict],
    min_duration: float = 15,
    max_duration: float = 60,
    ideal_min: float = 18,
    ideal_max: float = 60,
) -> list[dict]:
    windows = []
    for start_index in range(len(segments)):
        text_parts = []
        speakers = []
        start = float(segments[start_index]["start"])
        for end_index in range(start_index, min(start_index + 12, len(segments))):
            segment = segments[end_index]
            text_parts.append(segment["text"].strip())
            if segment.get("speaker"):
                speakers.append(segment["speaker"])
            end = float(segment["end"])
            duration = end - start
            if duration >= min_duration:
                text = " ".join(text_parts)
                windows.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                        "score": highlight_score(text, duration, ideal_min, ideal_max),
                        "speaker": Counter(speakers).most_common(1)[0][0] if speakers else None,
                    }
                )
            if duration >= max_duration:
                break

    selected = []
    for candidate in sorted(windows, key=lambda item: item["score"], reverse=True):
        overlaps = any(
            candidate["start"] < current["end"] and candidate["end"] > current["start"]
            for current in selected
        )
        if not overlaps:
            selected.append(candidate)
        if len(selected) == 10:
            break
    return sorted(selected, key=lambda item: item["start"])


def write_highlights(segments: list[dict], destination: Path, source_name: str, preset: str = DEFAULT_HIGHLIGHT_PRESET) -> None:
    settings = HIGHLIGHT_PRESETS.get(preset, HIGHLIGHT_PRESETS[DEFAULT_HIGHLIGHT_PRESET])
    highlights = build_highlight_windows(
        segments,
        min_duration=settings["min_duration"],
        max_duration=settings["max_duration"],
        ideal_min=settings["ideal_min"],
        ideal_max=settings["ideal_max"],
    )
    lines = [
        f"# Highlight candidates: {source_name}",
        "",
        f"Style: {settings['label']}. These are automatically ranked starting points. Review them against the video before publishing.",
        "",
    ]
    if not highlights:
        lines.append("No highlight windows were found.")
    for index, item in enumerate(highlights, 1):
        heading = f"## {index}. {timestamp(item['start'])} - {timestamp(item['end'])}"
        if item.get("speaker"):
            heading += f" ({item['speaker']})"
        lines.extend([heading, "", item["text"], ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Export options / driver
# ---------------------------------------------------------------------------

@dataclass
class ExportOptions:
    plain_text: bool = False
    timestamped_text: bool = False
    srt: bool = False
    vtt: bool = False
    tsv: bool = False
    csv_file: bool = False
    json_file: bool = False
    markdown: bool = False
    highlights: bool = False
    highlight_preset: str = DEFAULT_HIGHLIGHT_PRESET


def write_exports(json_path: Path, destination: Path, options: ExportOptions, source_name: str | None = None) -> list[Path]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = [segment for segment in data.get("segments", []) if segment.get("text", "").strip()]
    stem = json_path.stem
    title = source_name or stem
    created: list[Path] = []

    if options.plain_text:
        path = destination / f"{stem}.txt"
        write_plain_text(segments, path)
        created.append(path)

    if options.timestamped_text:
        path = destination / f"{stem}.timestamps.txt"
        write_timestamped_text(segments, path)
        created.append(path)

    if options.srt:
        path = destination / f"{stem}.srt"
        write_srt(segments, path)
        created.append(path)

    if options.vtt:
        path = destination / f"{stem}.vtt"
        write_vtt(segments, path)
        created.append(path)

    if options.tsv:
        path = destination / f"{stem}.tsv"
        write_tsv(segments, path)
        created.append(path)

    if options.csv_file:
        path = destination / f"{stem}.csv"
        write_csv(segments, path)
        created.append(path)

    if options.markdown:
        path = destination / f"{stem}.md"
        write_markdown(segments, path, title, data.get("language", "unknown"))
        created.append(path)

    if options.highlights:
        path = destination / f"{stem}.highlights.md"
        write_highlights(segments, path, title, options.highlight_preset)
        created.append(path)

    if options.json_file:
        created.append(json_path)
    else:
        json_path.unlink(missing_ok=True)

    return created
