from pathlib import Path

from telegram_notebook.db import Repository


def _repo(tmp_path: Path) -> Repository:
    repo = Repository(tmp_path / "store.db")
    repo.init()
    return repo


def test_channel_message_media_chunk_flow_and_search(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.upsert_channel(telegram_id=1, channel_url="https://t.me/c", title="C", username="c")
    # upsert is idempotent on channel_url
    assert repo.upsert_channel(telegram_id=1, channel_url="https://t.me/c", title="C2", username="c") == cid

    mid = repo.create_or_get_message(
        channel_id=cid, telegram_message_id=10, message_date=None,
        message_url="https://t.me/c/10", caption="cap",
    )
    assert repo.create_or_get_message(
        channel_id=cid, telegram_message_id=10, message_date=None,
        message_url="https://t.me/c/10", caption="cap",
    ) == mid

    media_id = repo.create_or_get_media(
        message_id=mid, file_name="text", file_path="", mime_type="text/plain",
        media_kind="text", duration_seconds=None, file_size_bytes=3,
    )
    repo.replace_chunks(media_item_id=media_id, chunks=[
        {"chunk_index": 0, "text": "hello world", "embedding": None, "start_char": 0, "end_char": 11},
    ])

    rows = repo.keyword_candidates(query="hello", top_k=5, channel_url=None)
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "hello world"
    assert rows[0]["channel_url"] == "https://t.me/c"

    chunk = repo.get_chunk_by_media_and_index(media_id, 0)
    assert chunk and chunk["chunk_text"] == "hello world"

    assert len(repo.list_channels()) == 1
    assert repo.delete_channel_data(channel_url="https://t.me/c") is True
    assert repo.list_channels() == []
    assert repo.delete_channel_data(channel_url="https://t.me/c") is False


def test_keyword_candidates_filters_by_channel(tmp_path):
    repo = _repo(tmp_path)
    for url in ("https://t.me/a", "https://t.me/b"):
        cid = repo.upsert_channel(telegram_id=hash(url) % 1000, channel_url=url, title=url, username=None)
        mid = repo.create_or_get_message(channel_id=cid, telegram_message_id=1, message_date=None, message_url=url + "/1", caption=None)
        media_id = repo.create_or_get_media(message_id=mid, file_name="t", file_path="", mime_type="text/plain", media_kind="text", duration_seconds=None, file_size_bytes=1)
        repo.replace_chunks(media_item_id=media_id, chunks=[{"chunk_index": 0, "text": "shared keyword", "embedding": None, "start_char": 0, "end_char": 14}])

    assert len(repo.keyword_candidates(query="keyword", top_k=10, channel_url=None)) == 2
    only_a = repo.keyword_candidates(query="keyword", top_k=10, channel_url="https://t.me/a")
    assert len(only_a) == 1 and only_a[0]["channel_url"] == "https://t.me/a"


def test_disconnect_bot_user(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_bot_user(bot_user_id=7, chat_id=7, username="u", first_name="f")
    repo.save_bot_user_session(bot_user_id=7, phone="+100", api_id=1, api_hash="h", session_string="SESS", connected_at="now")
    assert repo.get_bot_user(bot_user_id=7)["session_string"] == "SESS"

    assert repo.disconnect_bot_user(bot_user_id=7) is True
    user = repo.get_bot_user(bot_user_id=7)
    assert user["session_string"] is None
    assert user["api_hash"] is None
    assert user["bot_user_id"] == 7  # row preserved
    # second disconnect is a no-op
    assert repo.disconnect_bot_user(bot_user_id=7) is False
