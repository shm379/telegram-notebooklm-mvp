"""Backup attachments: extract DOCX/XLSX content bundled in a Telegram export zip."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.office import extract_office_bytes
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.telegram_backup import (
    make_zip_file_resolver,
    parse_export,
    read_export,
    render_markdown,
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _docx_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W_NS}"><w:body>'
            f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buf.getvalue()


def _xlsx_bytes(cell: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", f'<sst xmlns="{SS_NS}"><si><t>{cell}</t></si></sst>')
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{SS_NS}"><sheetData><row><c t="s"><v>0</v></c></row></sheetData></worksheet>',
        )
    return buf.getvalue()


def _export_zip(messages, files):
    """A Telegram Desktop export zip: nested result.json + attachment files."""
    result = {"name": "My Chat", "type": "personal_chat", "id": 99, "messages": messages}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ChatExport_2026/result.json", json.dumps(result))
        for path, content in files.items():
            zf.writestr(f"ChatExport_2026/{path}", content)
    return buf.getvalue()


# --- extract_office_bytes ---

def test_extract_office_bytes_docx_and_xlsx():
    assert extract_office_bytes(_docx_bytes("from bytes"), "docx") == "from bytes"
    assert extract_office_bytes(_xlsx_bytes("CellValue"), "xlsx") == "CellValue"


def test_extract_office_bytes_rejects_unknown_kind():
    with pytest.raises(ValueError):
        extract_office_bytes(b"x", "pdf")


# --- make_zip_file_resolver ---

def test_resolver_none_for_non_zip():
    assert make_zip_file_resolver(b'{"not": "a zip"}') is None


def test_resolver_matches_by_suffix_and_basename():
    data = _export_zip([], {"files/report.docx": b"DOCXDATA", "photos/p.jpg": b"IMG"})
    resolve = make_zip_file_resolver(data)
    assert resolve("files/report.docx") == b"DOCXDATA"  # suffix match (nested under ChatExport_2026/)
    assert resolve("report.docx") == b"DOCXDATA"          # basename fallback
    assert resolve("missing.docx") is None


# --- parse_export with attachments ---

def _message(**kw):
    base = {"id": 1, "type": "message", "date": "2026-06-01T10:00:00", "from": "Alice", "text": ""}
    base.update(kw)
    return base


def test_parse_export_extracts_attached_docx():
    msgs = [_message(id=10, text="see attached", file="files/report.docx",
                     file_name="report.docx", mime_type=DOCX_MIME)]
    data = _export_zip(msgs, {"files/report.docx": _docx_bytes("quarterly revenue summary")})
    chats = parse_export(read_export(data, "export.zip"), file_resolver=make_zip_file_resolver(data))
    msg = chats[0].messages[0]
    assert msg.document_text == "quarterly revenue summary"
    assert "quarterly revenue summary" in msg.searchable_text
    assert "see attached" in msg.searchable_text
    assert "[file: report.docx]" in msg.searchable_text


def test_parse_export_without_resolver_keeps_label_only():
    # backward compatible: no resolver -> no document text, just the media label
    msgs = [_message(id=10, file="files/report.docx", file_name="report.docx", mime_type=DOCX_MIME)]
    chat = parse_export({"name": "c", "id": 1, "messages": msgs})[0]
    assert chat.messages[0].document_text == ""
    assert chat.messages[0].media_label == "[file: report.docx]"


def test_parse_export_ignores_non_office_attachments():
    msgs = [_message(id=10, file="files/archive.zip", file_name="archive.zip")]
    data = _export_zip(msgs, {"files/archive.zip": b"PK\x03\x04junk"})
    chat = parse_export(read_export(data, "e.zip"), file_resolver=make_zip_file_resolver(data))[0]
    assert chat.messages[0].document_text == ""


def test_render_markdown_includes_document_text():
    msgs = [_message(id=10, text="note", file="files/r.docx", file_name="r.docx", mime_type=DOCX_MIME)]
    data = _export_zip(msgs, {"files/r.docx": _docx_bytes("contract terms")})
    chats = parse_export(read_export(data, "e.zip"), file_resolver=make_zip_file_resolver(data))
    md = render_markdown(chats, generated_at="2026-06-17 10:00 UTC")
    assert "contract terms" in md


# --- end to end through ingest_backup ---

def _pipeline(repo):
    return IngestionPipeline(
        settings=SimpleNamespace(chunk_size=900, chunk_overlap=120),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="x"),
    )


def test_ingest_backup_makes_attached_docx_searchable(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    msgs = [
        _message(id=10, text="agenda", file="files/notes.docx", file_name="notes.docx", mime_type=DOCX_MIME),
        _message(id=11, file="files/budget.xlsx", file_name="budget.xlsx",
                 mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ]
    data = _export_zip(msgs, {
        "files/notes.docx": _docx_bytes("roadmap discussion about pricing"),
        "files/budget.xlsx": _xlsx_bytes("Q3Forecast"),
    })
    chats = parse_export(read_export(data, "e.zip"), file_resolver=make_zip_file_resolver(data))

    asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))

    # content from inside the attached documents is now searchable
    assert len(repo.keyword_candidates(owner_id=1, query="roadmap", top_k=5, channel_url=None)) == 1
    assert len(repo.keyword_candidates(owner_id=1, query="Q3Forecast", top_k=5, channel_url=None)) == 1
