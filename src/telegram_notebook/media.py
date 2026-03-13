from __future__ import annotations

import mimetypes
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".opus", ".aac", ".flac"}
MAX_TRANSCRIBE_BYTES = 24 * 1024 * 1024
SEGMENT_SECONDS = 15 * 60


def detect_media_kind(file_name: str | None, mime_type: str | None) -> str | None:
    suffix = Path(file_name or "").suffix.lower()
    if suffix in VIDEO_EXTENSIONS or (mime_type and mime_type.startswith("video/")):
        return "video"
    if suffix in AUDIO_EXTENSIONS or (mime_type and mime_type.startswith("audio/")):
        return "audio"
    return None


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH")


def normalize_audio_input(source_path: Path, work_dir: Path) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg()

    normalized = work_dir / f"{source_path.stem}.normalized.mp3"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "32k",
        str(normalized),
    ]
    subprocess.run(command, check=True, capture_output=True)

    if normalized.stat().st_size <= MAX_TRANSCRIBE_BYTES:
        return [normalized]

    segment_pattern = work_dir / f"{source_path.stem}.part%03d.mp3"
    segment_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(normalized),
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_SECONDS),
        "-c",
        "copy",
        str(segment_pattern),
    ]
    subprocess.run(segment_command, check=True, capture_output=True)
    return sorted(work_dir.glob(f"{source_path.stem}.part*.mp3"))


def guess_mime_type(path: Path) -> str | None:
    return mimetypes.guess_type(path.name)[0]
