"""Parse a Telegram Desktop "Export chat history" backup into searchable items.

Telegram Desktop can export a chat (or a whole account) as *Machine-readable
JSON*. A single-chat export is a JSON object with a top-level ``messages`` list;
a full-account export nests one such object per chat under ``chats.list``. Users
often zip the export folder (the JSON plus media subfolders), so this module also
knows how to pull ``result.json`` out of a ``.zip``.

Everything here is a pure function over bytes/dicts (only the standard library),
so the parser and the Markdown renderer are trivially unit-testable, and the bot
and web layers just feed the parsed chats into the existing ingestion pipeline.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ZIP_MAGIC = b"PK\x03\x04"


@dataclass(slots=True)
class ParsedMessage:
    id: int
    date: str | None
    sender: str | None
    text: str
    media_label: str | None = None
    forwarded_from: str | None = None
    reply_to: int | None = None
    # Relative path of an attached media file inside the export (zip), plus the
    # info needed to route it to OCR/transcription when the media is available.
    media_path: str | None = None
    media_kind: str | None = None
    media_mime: str | None = None

    @property
    def searchable_text(self) -> str:
        """Media tag + text, mirroring how the Forwarded Inbox stores items."""
        return "\n".join(part for part in (self.media_label, self.text) if part).strip()


@dataclass(slots=True)
class ParsedChat:
    name: str
    type: str | None
    id: object
    messages: list[ParsedMessage] = field(default_factory=list)


# --- reading the upload (zip or raw json) ----------------------------------


def read_export(data: bytes, filename: str | None = None) -> dict:
    """Return the export's JSON object from raw upload ``data``.

    Handles both a raw ``result.json`` and a ``.zip`` of the export folder
    (detected by the zip magic bytes or a ``.zip`` filename).
    """
    if not data:
        raise ValueError("The uploaded file is empty.")
    is_zip = data[:4] == ZIP_MAGIC or (filename or "").lower().endswith(".zip")
    raw = _read_result_json_from_zip(data) if is_zip else data
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Telegram export JSON must be an object.")
    return parsed


def _read_result_json_from_zip(data: bytes) -> bytes:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"The file looks like a zip but could not be opened ({exc}).") from exc
    with archive as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.rsplit("/", 1)[-1].lower() == "result.json"]
        if not candidates:
            candidates = [n for n in names if n.lower().endswith(".json")]
        if not candidates:
            raise ValueError(
                "No result.json found in the zip. Export your chat as "
                "'Machine-readable JSON' in Telegram Desktop before zipping it."
            )
        # Prefer the shallowest / shortest path (the top-level result.json).
        candidates.sort(key=lambda n: (n.count("/"), len(n)))
        return zf.read(candidates[0])


# --- parsing ---------------------------------------------------------------


def parse_export(data: dict) -> list[ParsedChat]:
    """Normalize a single-chat or full-account export into ``ParsedChat`` list."""
    if not isinstance(data, dict):
        raise ValueError("Telegram export must be a JSON object.")
    if isinstance(data.get("messages"), list):
        return [_parse_chat(data)]
    chats = data.get("chats")
    if isinstance(chats, dict) and isinstance(chats.get("list"), list):
        return [_parse_chat(c) for c in chats["list"] if isinstance(c, dict)]
    if isinstance(chats, list):
        return [_parse_chat(c) for c in chats if isinstance(c, dict)]
    raise ValueError(
        "Unrecognized Telegram export. Expected a chat export (with 'messages') "
        "or a full export (with 'chats.list'). Export as 'Machine-readable JSON' "
        "from Telegram Desktop."
    )


def _parse_chat(raw: dict) -> ParsedChat:
    messages: list[ParsedMessage] = []
    for item in raw.get("messages", []) or []:
        parsed = _parse_message(item)
        if parsed is not None:
            messages.append(parsed)
    name = (raw.get("name") or "").strip() or "Telegram chat"
    return ParsedChat(name=name, type=raw.get("type"), id=raw.get("id"), messages=messages)


def _parse_message(raw: object) -> ParsedMessage | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("type") == "service":  # joined/pinned/etc. — no real content to index
        return None
    try:
        msg_id = int(raw.get("id"))
    except (TypeError, ValueError):
        return None
    reply_to = raw.get("reply_to_message_id")
    return ParsedMessage(
        id=msg_id,
        date=raw.get("date") or None,
        sender=(raw.get("from") or None),
        text=message_text(raw),
        media_label=media_label(raw),
        forwarded_from=(raw.get("forwarded_from") or None),
        reply_to=reply_to if isinstance(reply_to, int) else None,
        media_path=media_path(raw),
        media_kind=media_kind(raw),
        media_mime=(raw.get("mime_type") or None),
    )


def media_path(raw: dict) -> str | None:
    """Relative path of an attached media file inside the export, or None."""
    path = raw.get("photo") or raw.get("file")
    return str(path) if isinstance(path, str) and path and not path.startswith("(") else None


def media_kind(raw: dict) -> str | None:
    """Coarse kind of a message's attachment ('photo'/media_type/'file')."""
    if raw.get("photo"):
        return "photo"
    if raw.get("media_type"):
        return str(raw["media_type"])
    if raw.get("file"):
        return "file"
    return None


def backup_media_route(kind: str | None, mime: str | None) -> str | None:
    """Which processor handles a backup media file: 'transcribe', 'extract', or None.

    Mirrors the Forwarded-Inbox routing: audio/video -> transcription, images/PDF
    -> OCR/extraction. Unknown kinds (stickers, etc.) are left as a label only.
    """
    if kind in ("voice_message", "video_message", "video_file", "audio_file"):
        return "transcribe"
    mt = (mime or "").lower()
    if kind == "photo":
        return "extract"
    if kind == "file":
        if mt == "application/pdf" or mt.startswith("image/"):
            return "extract"
        if mt.startswith(("audio/", "video/")):
            return "transcribe"
    return None


def message_text(raw: dict) -> str:
    """Flatten a message's text, preferring ``text_entities`` when present.

    Telegram stores text either as a plain string or as a list of fragments
    (strings and ``{"type", "text", "href"}`` entity dicts) that concatenate to
    the full message. For ``text_link`` entities the visible text and the URL
    differ, so we inline the URL to keep links searchable.
    """
    entities = raw.get("text_entities")
    if isinstance(entities, list) and entities:
        flattened = _entities_to_text(entities).strip()
        if flattened:
            return flattened
    value = raw.get("text")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return _entities_to_text(value).strip()
    return ""


def _entities_to_text(entities: list) -> str:
    parts: list[str] = []
    for entity in entities:
        if isinstance(entity, str):
            parts.append(entity)
        elif isinstance(entity, dict):
            text = str(entity.get("text") or "")
            href = entity.get("href")
            if entity.get("type") == "text_link" and href and href not in text:
                parts.append(f"{text} ({href})")
            else:
                parts.append(text)
    return "".join(parts)


def media_label(raw: dict) -> str | None:
    """A short ``[kind]`` tag for a message's attachment, or None for text-only."""
    if raw.get("photo"):
        return "[photo]"
    media_type = raw.get("media_type")
    if media_type:
        if media_type == "sticker":
            emoji = raw.get("sticker_emoji")
            return f"[sticker {emoji}]" if emoji else "[sticker]"
        names = {
            "voice_message": "voice message",
            "video_message": "video message",
            "video_file": "video",
            "audio_file": "audio",
            "animation": "GIF",
        }
        base = names.get(media_type, str(media_type).replace("_", " "))
        if media_type == "audio_file":
            title = raw.get("title") or raw.get("file_name")
            return f"[audio: {title}]" if title else "[audio]"
        duration = raw.get("duration_seconds")
        if media_type in ("voice_message", "video_message", "video_file") and duration:
            return f"[{base}: {duration}s]"
        return f"[{base}]"
    if raw.get("file"):
        return f"[file: {raw.get('file_name') or 'document'}]"
    if raw.get("poll"):
        question = (raw["poll"] or {}).get("question")
        return f"[poll: {question}]" if question else "[poll]"
    if raw.get("location_information"):
        return "[location]"
    if raw.get("contact_information"):
        return "[contact]"
    return None


# --- markdown rendering ----------------------------------------------------


def count_messages(chats: list[ParsedChat]) -> int:
    return sum(len(chat.messages) for chat in chats)


def render_markdown(chats: list[ParsedChat], *, generated_at: str | None = None) -> str:
    """Render parsed chats as a single, readable Markdown chat log."""
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = count_messages(chats)
    lines = [
        f"# Telegram Backup — {len(chats)} chat(s)",
        "",
        f"_Converted {stamp} · {total} message(s)_",
        "",
    ]
    for chat in chats:
        lines.append(f"## {chat.name}")
        lines.append("")
        for msg in chat.messages:
            header = []
            if msg.sender:
                header.append(f"**{msg.sender}**")
            if msg.date:
                header.append(msg.date.replace("T", " "))
            if header:
                lines.append(" · ".join(header))
            if msg.forwarded_from:
                lines.append(f"_forwarded from {msg.forwarded_from}_")
            if msg.media_label:
                lines.append(msg.media_label)
            if msg.text:
                lines.append(msg.text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- media inside the zip (OCR / transcription) ----------------------------

# Cap how many media files we OCR/transcribe per import, since each is an LLM
# call. Bulk exports can hold thousands of photos; this keeps cost bounded.
DEFAULT_MEDIA_LIMIT = 200


def open_backup_zip(data: bytes, filename: str | None = None) -> zipfile.ZipFile | None:
    """Return an open ZipFile when the upload is a zip (else None), for media access."""
    is_zip = data[:4] == ZIP_MAGIC or (filename or "").lower().endswith(".zip")
    if not is_zip:
        return None
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None


def resolve_zip_member(zf: zipfile.ZipFile, rel_path: str) -> str | None:
    """Find the zip member for a media path from result.json.

    Telegram stores media paths relative to the export root (e.g.
    ``photos/photo_1.jpg``), but a zipped export usually nests everything under a
    top folder, so we match by exact name first and fall back to a suffix match.
    """
    if not rel_path:
        return None
    names = zf.namelist()
    if rel_path in names:
        return rel_path
    needle = rel_path.lstrip("./")
    for name in names:
        if name == needle or name.endswith("/" + needle):
            return name
    return None


def make_zip_extractor(zf, tmpdir: Path, *, transcribe=None, ocr=None):
    """Build an ``extract(rel_path, route) -> str | None`` over a backup zip.

    ``transcribe(path)`` and ``ocr(path)`` are injected (the per-user services),
    so this stays testable without real LLM calls. The referenced media file is
    written to ``tmpdir`` before being handed to the chosen processor.
    """
    def extract(rel_path: str, route: str) -> str | None:
        member = resolve_zip_member(zf, rel_path)
        if not member:
            return None
        dest = tmpdir / f"m{abs(hash(rel_path))}{Path(rel_path).suffix}"
        dest.write_bytes(zf.read(member))
        if route == "transcribe" and transcribe is not None:
            return transcribe(dest)
        if route == "extract" and ocr is not None:
            return ocr(dest)
        return None

    return extract


def enrich_with_media(chats: list[ParsedChat], *, extract, limit: int = DEFAULT_MEDIA_LIMIT) -> int:
    """Append OCR/transcription text to messages that carry a media file.

    ``extract(rel_path, route)`` returns text (or None). Errors per item are
    swallowed so one bad file never aborts a whole import. Returns the number of
    messages enriched; processing stops once ``limit`` items have been enriched.
    """
    enriched = 0
    for chat in chats:
        for msg in chat.messages:
            if enriched >= limit:
                return enriched
            if not msg.media_path:
                continue
            route = backup_media_route(msg.media_kind, msg.media_mime)
            if route is None:
                continue
            try:
                text = (extract(msg.media_path, route) or "").strip()
            except Exception:
                text = ""
            if text:
                msg.text = f"{msg.text}\n{text}".strip() if msg.text else text
                enriched += 1
    return enriched
