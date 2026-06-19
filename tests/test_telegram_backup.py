import asyncio
import io
import json
import zipfile

import pytest

from telegram_notebook.config import get_settings
from telegram_notebook.db import Repository
from telegram_notebook.embeddings import EmbeddingService
from telegram_notebook.pipeline import IngestionPipeline
from telegram_notebook.telegram_backup import (
    _html_date_to_iso,
    backup_media_route,
    count_messages,
    enrich_with_media,
    load_backup,
    make_zip_extractor,
    media_kind,
    media_label,
    media_path,
    message_text,
    open_backup_zip,
    parse_export,
    parse_html_chat,
    read_export,
    render_markdown,
    resolve_zip_member,
)

SINGLE_CHAT = {
    "name": "My Channel",
    "type": "public_channel",
    "id": 123456,
    "messages": [
        {
            "id": 1,
            "type": "message",
            "date": "2021-05-12T10:11:12",
            "from": "Alice",
            "text": "hello world",
            "text_entities": [{"type": "plain", "text": "hello world"}],
        },
        {"id": 2, "type": "service", "date": "2021-05-12T10:12:00", "action": "pin_message"},
        {
            "id": 3,
            "type": "message",
            "date": "2021-05-12T10:13:00",
            "from": "Bob",
            "text": [
                {"type": "plain", "text": "check "},
                {"type": "text_link", "text": "this", "href": "https://example.com"},
            ],
        },
        {
            "id": 4,
            "type": "message",
            "date": "2021-05-12T10:14:00",
            "from": "Bob",
            "photo": "photos/1.jpg",
            "text": "",
        },
    ],
}

FULL_EXPORT = {
    "about": "exported data",
    "chats": {
        "about": "chats",
        "list": [
            SINGLE_CHAT,
            {"name": "Notes", "type": "saved_messages", "id": 777, "messages": [
                {"id": 1, "type": "message", "date": "2022-01-01T00:00:00", "from": "Me", "text": "remember this"},
            ]},
        ],
    },
}


# --- text flattening -------------------------------------------------------

def test_message_text_plain_string():
    assert message_text({"text": "hi there"}) == "hi there"


def test_message_text_prefers_entities_and_inlines_links():
    raw = SINGLE_CHAT["messages"][2]
    assert message_text(raw) == "check this (https://example.com)"


def test_message_text_empty():
    assert message_text({"text": ""}) == ""
    assert message_text({}) == ""


# --- media labels ----------------------------------------------------------

def test_media_label_variants():
    assert media_label({"photo": "p.jpg"}) == "[photo]"
    assert media_label({"media_type": "voice_message", "duration_seconds": 12}) == "[voice message: 12s]"
    assert media_label({"media_type": "sticker", "sticker_emoji": "🔥"}) == "[sticker 🔥]"
    assert media_label({"media_type": "audio_file", "title": "song"}) == "[audio: song]"
    assert media_label({"file": "x", "file_name": "report.pdf"}) == "[file: report.pdf]"
    assert media_label({"text": "just text"}) is None


# --- parsing ---------------------------------------------------------------

def test_parse_single_chat_skips_service_messages():
    chats = parse_export(SINGLE_CHAT)
    assert len(chats) == 1
    chat = chats[0]
    assert chat.name == "My Channel" and chat.id == 123456
    assert [m.id for m in chat.messages] == [1, 3, 4]  # service id 2 dropped


def test_parse_media_only_message_is_searchable_via_label():
    chat = parse_export(SINGLE_CHAT)[0]
    photo_msg = next(m for m in chat.messages if m.id == 4)
    assert photo_msg.text == ""
    assert photo_msg.searchable_text == "[photo]"


def test_parse_full_export_lists_all_chats():
    chats = parse_export(FULL_EXPORT)
    assert {c.name for c in chats} == {"My Channel", "Notes"}
    assert count_messages(chats) == 4  # 3 from My Channel + 1 from Notes


def test_parse_unrecognized_structure_raises():
    with pytest.raises(ValueError):
        parse_export({"something": "else"})


# --- reading uploads (json / zip) -----------------------------------------

def test_read_export_from_json_bytes():
    data = json.dumps(SINGLE_CHAT).encode("utf-8")
    assert read_export(data, "result.json")["id"] == 123456


def test_read_export_from_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ChatExport/result.json", json.dumps(SINGLE_CHAT))
        zf.writestr("ChatExport/photos/1.jpg", b"\xff\xd8stub")
    parsed = read_export(buf.getvalue(), "export.zip")
    assert parsed["name"] == "My Channel"


def test_read_export_empty_raises():
    with pytest.raises(ValueError):
        read_export(b"", "x.json")


def test_read_export_zip_without_json_raises():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("photos/1.jpg", b"data")
    with pytest.raises(ValueError):
        read_export(buf.getvalue(), "export.zip")


# --- markdown rendering ----------------------------------------------------

def test_render_markdown_structure():
    md = render_markdown(parse_export(SINGLE_CHAT), generated_at="2026-06-17 10:00 UTC")
    assert md.startswith("# Telegram Backup — 1 chat(s)")
    assert "3 message(s)" in md
    assert "## My Channel" in md
    assert "**Alice**" in md and "hello world" in md
    assert "[photo]" in md
    assert md.endswith("\n")


# --- pipeline ingestion ----------------------------------------------------

def _pipeline(repo):
    return IngestionPipeline(
        settings=get_settings(),
        repository=repo,
        transcription=None,
        embeddings=EmbeddingService(provider="openai", api_key=None, model="text-embedding-3-small"),
    )


def test_ingest_backup_makes_content_searchable(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    chats = parse_export(SINGLE_CHAT)

    result = asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))
    assert result["chats"] == 1
    assert result["messages"] == 3  # 3 text/media items stored

    rows = repo.keyword_candidates(owner_id=1, query="hello", top_k=5, channel_url=None)
    assert rows and "hello world" in rows[0]["chunk_text"]

    channels = repo.list_channels(owner_id=1)
    assert channels[0]["channel_url"] == "backup://123456"


def test_ingest_backup_is_idempotent(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    chats = parse_export(SINGLE_CHAT)

    first = asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))
    second = asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=chats))
    assert first["messages"] == 3
    assert second["messages"] == 0  # re-import stores nothing new


def test_ingest_backup_isolated_per_owner(tmp_path):
    repo = Repository(tmp_path / "store.db")
    repo.init()
    asyncio.run(_pipeline(repo).ingest_backup(owner_id=1, chats=parse_export(SINGLE_CHAT)))
    assert repo.keyword_candidates(owner_id=2, query="hello", top_k=5, channel_url=None) == []


# --- media inside the zip (OCR / transcription) ----------------------------

MEDIA_CHAT = {
    "name": "C",
    "type": "personal_chat",
    "id": 1,
    "messages": [
        {"id": 1, "type": "message", "date": "2021-01-01T00:00:00", "from": "A", "photo": "photos/photo_1.jpg", "text": ""},
        {
            "id": 2, "type": "message", "date": "2021-01-01T00:01:00", "from": "A",
            "file": "voice_messages/audio_1.ogg", "media_type": "voice_message",
            "mime_type": "audio/ogg", "text": "",
        },
        {"id": 3, "type": "message", "date": "2021-01-01T00:02:00", "from": "A", "text": "plain"},
    ],
}


def test_media_path_and_kind():
    assert media_path({"photo": "photos/p.jpg"}) == "photos/p.jpg"
    assert media_path({"file": "files/x.pdf"}) == "files/x.pdf"
    assert media_path({"photo": "(File not included. Change data exporting settings to download.)"}) is None
    assert media_kind({"photo": "p.jpg"}) == "photo"
    assert media_kind({"media_type": "voice_message"}) == "voice_message"
    assert media_kind({"file": "x"}) == "file"


def test_backup_media_route():
    assert backup_media_route("photo", None) == "extract"
    assert backup_media_route("voice_message", "audio/ogg") == "transcribe"
    assert backup_media_route("file", "application/pdf") == "extract"
    assert backup_media_route("file", "image/png") == "extract"
    assert backup_media_route("file", "audio/mpeg") == "transcribe"
    assert backup_media_route("sticker", None) is None
    assert backup_media_route("file", "text/plain") is None


def test_open_backup_zip_returns_none_for_json():
    assert open_backup_zip(b'{"messages": []}', "result.json") is None


def test_resolve_zip_member_matches_nested_path():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ChatExport/photos/photo_1.jpg", b"x")
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        assert resolve_zip_member(zf, "photos/photo_1.jpg") == "ChatExport/photos/photo_1.jpg"
        assert resolve_zip_member(zf, "missing.jpg") is None


def test_enrich_with_media_from_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Export/result.json", json.dumps(MEDIA_CHAT))
        zf.writestr("Export/photos/photo_1.jpg", b"OCRTEXT")
        zf.writestr("Export/voice_messages/audio_1.ogg", b"VOICETEXT")
    data = buf.getvalue()
    chats = parse_export(read_export(data, "x.zip"))

    def transcribe(path):
        return "transcribed:" + path.read_bytes().decode()

    def ocr(path):
        return "ocr:" + path.read_bytes().decode()

    media_dir = tmp_path / "m"
    media_dir.mkdir()
    zf = open_backup_zip(data, "x.zip")
    extractor = make_zip_extractor(zf, media_dir, transcribe=transcribe, ocr=ocr)
    count = enrich_with_media(chats, extract=extractor)
    zf.close()

    assert count == 2
    msgs = {m.id: m for m in chats[0].messages}
    assert "ocr:OCRTEXT" in msgs[1].text
    assert "transcribed:VOICETEXT" in msgs[2].text
    assert msgs[3].text == "plain"  # text-only message untouched
    assert "ocr:OCRTEXT" in msgs[1].searchable_text  # now indexable


def test_enrich_with_media_respects_limit():
    chats = parse_export(MEDIA_CHAT)

    def extract(path, route):
        return "X"

    assert enrich_with_media(chats, extract=extract, limit=1) == 1


def test_enrich_with_media_swallows_extractor_errors():
    chats = parse_export(MEDIA_CHAT)

    def boom(path, route):
        raise RuntimeError("nope")

    assert enrich_with_media(chats, extract=boom) == 0


# --- HTML export parsing ---------------------------------------------------

HTML_EXPORT = """<!DOCTYPE html><html><head></head><body>
<div class="page_header"><div class="content"><div class="text bold">My Chat</div></div></div>
<div class="history">
<div class="message default clearfix" id="message1"><div class="body">
  <div class="pull_right date details" title="26.05.2021 18:03:43 UTC+03:00">18:03</div>
  <div class="from_name">Alice</div>
  <div class="text">hello <a href="https://example.com">link</a></div>
</div></div>
<div class="message default clearfix joined" id="message2"><div class="body">
  <div class="pull_right date details" title="26.05.2021 18:04:00 UTC+03:00">18:04</div>
  <div class="text">second message</div>
</div></div>
<div class="message service" id="message3"><div class="body details">Channel created</div></div>
<div class="message default clearfix" id="message4"><div class="body">
  <div class="pull_right date details" title="26.05.2021 18:05:00 UTC+03:00">18:05</div>
  <div class="from_name">Bob</div>
  <div class="media_wrap clearfix"><a class="media clearfix pull_left block_link media_photo"><div class="body"><div class="title bold">Photo</div><div class="status details">Not included</div></div></a></div>
</div></div>
</div></body></html>"""


def test_html_date_to_iso():
    assert _html_date_to_iso("26.05.2021 18:03:43 UTC+03:00") == "2021-05-26T18:03:43"
    assert _html_date_to_iso("garbage") is None


def test_parse_html_chat():
    chat = parse_html_chat([HTML_EXPORT])
    assert chat.name == "My Chat"
    by_id = {m.id: m for m in chat.messages}
    assert sorted(by_id) == [1, 2, 4]  # service message skipped
    assert by_id[1].sender == "Alice"
    assert by_id[1].date == "2021-05-26T18:03:43"
    assert "hello link (https://example.com)" in by_id[1].text
    assert by_id[2].sender == "Alice"  # "joined" reuses the previous sender
    assert by_id[2].text == "second message"
    assert by_id[4].sender == "Bob"
    assert by_id[4].media_label == "[Photo]"
    assert by_id[4].searchable_text == "[Photo]"


def test_load_backup_html_bytes():
    chats = load_backup(HTML_EXPORT.encode("utf-8"), "messages.html")
    assert len(chats) == 1 and chats[0].name == "My Chat"


def test_load_backup_html_in_zip_merges_pages():
    page2 = (
        HTML_EXPORT.replace('id="message1"', 'id="message10"')
        .replace('id="message2"', 'id="message20"')
        .replace('id="message4"', 'id="message40"')
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ChatExport/messages.html", HTML_EXPORT)
        zf.writestr("ChatExport/messages2.html", page2)
    chats = load_backup(buf.getvalue(), "export.zip")
    assert len(chats) == 1  # two pages merged into one chat
    assert {m.id for m in chats[0].messages} >= {1, 2, 4, 10, 20, 40}


def test_load_backup_json_still_works():
    chats = load_backup(json.dumps(SINGLE_CHAT).encode("utf-8"), "result.json")
    assert chats[0].name == "My Channel"


def test_load_backup_prefers_result_json_in_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X/result.json", json.dumps(SINGLE_CHAT))
        zf.writestr("X/messages.html", HTML_EXPORT)
    chats = load_backup(buf.getvalue(), "export.zip")
    assert chats[0].name == "My Channel"  # JSON preferred over HTML
