from __future__ import annotations

import sqlite3
import json
import threading
from pathlib import Path
from typing import Any


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self.lock = threading.RLock()

    def init(self) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS channels (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        telegram_id INTEGER,
                        channel_url TEXT UNIQUE,
                        channel_title TEXT,
                        channel_username TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        channel_id INTEGER,
                        telegram_message_id INTEGER,
                        message_date TEXT,
                        message_url TEXT,
                        caption TEXT,
                        FOREIGN KEY(channel_id) REFERENCES channels(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS media_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id INTEGER,
                        file_name TEXT,
                        file_path TEXT,
                        mime_type TEXT,
                        media_kind TEXT,
                        duration_seconds INTEGER,
                        file_size_bytes INTEGER,
                        transcript_text TEXT,
                        transcript_status TEXT DEFAULT 'pending',
                        transcript_error TEXT,
                        FOREIGN KEY(message_id) REFERENCES messages(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        media_item_id INTEGER,
                        chunk_index INTEGER,
                        text TEXT,
                        embedding BLOB,
                        start_char INTEGER,
                        end_char INTEGER,
                        FOREIGN KEY(media_item_id) REFERENCES media_items(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bot_users (
                        bot_user_id INTEGER PRIMARY KEY,
                        chat_id INTEGER,
                        username TEXT,
                        first_name TEXT,
                        phone TEXT,
                        api_id INTEGER,
                        api_hash TEXT,
                        session_string TEXT,
                        connected_at TEXT,
                        preferred_transcription_model TEXT DEFAULT 'gemini-2.0-flash-lite',
                        preferred_embedding_model TEXT DEFAULT 'text-embedding-004',
                        gemini_api_key TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS auth_flows (
                        bot_user_id INTEGER PRIMARY KEY,
                        chat_id INTEGER,
                        phone TEXT,
                        api_id INTEGER,
                        api_hash TEXT,
                        session_string TEXT,
                        phone_code_hash TEXT,
                        status TEXT
                    )
                """)
                conn.commit()

    def upsert_channel(self, *, telegram_id: int, channel_url: str, title: str | None, username: str | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO channels (telegram_id, channel_url, channel_title, channel_username)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(channel_url) DO UPDATE SET
                        telegram_id=excluded.telegram_id,
                        channel_title=excluded.channel_title,
                        channel_username=excluded.channel_username
                """, (telegram_id, channel_url, title, username))
                res = conn.execute("SELECT id FROM channels WHERE channel_url = ?", (channel_url,)).fetchone()
                return res[0]

    def create_or_get_message(self, *, channel_id: int, telegram_message_id: int, message_date: str | None, message_url: str | None, caption: str | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT id FROM messages WHERE channel_id = ? AND telegram_message_id = ?", (channel_id, telegram_message_id)).fetchone()
                if res:
                    conn.execute("UPDATE messages SET caption = ?, message_url = ? WHERE id = ?", (caption, message_url, res[0]))
                    return res[0]
                
                cursor = conn.execute("""
                    INSERT INTO messages (channel_id, telegram_message_id, message_date, message_url, caption)
                    VALUES (?, ?, ?, ?, ?)
                """, (channel_id, telegram_message_id, message_date, message_url, caption))
                return cursor.lastrowid

    def create_or_get_media(self, *, message_id: int, file_name: str | None, file_path: str, mime_type: str | None, media_kind: str, duration_seconds: int | None, file_size_bytes: int | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT id FROM media_items WHERE message_id = ?", (message_id,)).fetchone()
                if res: return res[0]
                
                cursor = conn.execute("""
                    INSERT INTO media_items (message_id, file_name, file_path, mime_type, media_kind, duration_seconds, file_size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (message_id, file_name, file_path, mime_type, media_kind, duration_seconds, file_size_bytes))
                return cursor.lastrowid

    def replace_chunks(self, *, media_item_id: int, chunks: list[dict[str, Any]]) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("DELETE FROM chunks WHERE media_item_id = ?", (media_item_id,))
                for chunk in chunks:
                    embedding_blob = json.dumps(chunk["embedding"]).encode('utf-8') if chunk.get("embedding") else None
                    conn.execute("""
                        INSERT INTO chunks (media_item_id, chunk_index, text, embedding, start_char, end_char)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (media_item_id, chunk["chunk_index"], chunk["text"], embedding_blob, chunk["start_char"], chunk["end_char"]))
                conn.commit()

    def mark_media_transcribed(self, *, media_item_id: int, transcript_text: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE media_items SET transcript_text = ?, transcript_status = 'done' WHERE id = ?", (transcript_text, media_item_id))

    def media_already_transcribed(self, media_item_id: int) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT transcript_status FROM media_items WHERE id = ?", (media_item_id,)).fetchone()
                return res and res[0] == 'done'

    def keyword_candidates(self, *, query: str, top_k: int, channel_url: str | None) -> list[dict[str, Any]]:
        # جستجوی متنی سریع در SQLite
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT c.text as chunk_text, m.message_url, ch.channel_title, ch.channel_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE c.text LIKE ?
                """
                params = [f"%{query}%"]
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                
                rows = conn.execute(sql + f" LIMIT {top_k}", params).fetchall()
                return [dict(r) for r in rows]

    def embedding_candidates(self, *, channel_url: str | None) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                sql = """
                    SELECT c.id as chunk_id, c.text as chunk_text, c.embedding as embedding_json,
                           mi.media_kind, m.message_url, ch.channel_title, ch.channel_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE c.embedding IS NOT NULL
                """
                params = []
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)
                
                rows = conn.execute(sql, params).fetchall()
                results = []
                for r in rows:
                    d = dict(r)
                    if d["embedding_json"]:
                        d["embedding_json"] = d["embedding_json"].decode('utf-8')
                    results.append(d)
                return results

    def list_channels(self) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute("SELECT * FROM channels").fetchall()]

    def delete_channel_data(self, *, channel_url: str) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT id FROM channels WHERE channel_url = ?", (channel_url,)).fetchone()
                if not res: return False
                cid = res[0]
                conn.execute("DELETE FROM chunks WHERE media_item_id IN (SELECT id FROM media_items WHERE message_id IN (SELECT id FROM messages WHERE channel_id = ?))", (cid,))
                conn.execute("DELETE FROM media_items WHERE message_id IN (SELECT id FROM messages WHERE channel_id = ?)", (cid,))
                conn.execute("DELETE FROM messages WHERE channel_id = ?", (cid,))
                conn.execute("DELETE FROM channels WHERE id = ?", (cid,))
                conn.commit()
                return True

    def upsert_bot_user(self, *, bot_user_id: int, chat_id: int, username: str | None, first_name: str | None) -> dict[str, Any]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO bot_users (bot_user_id, chat_id, username, first_name)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(bot_user_id) DO UPDATE SET
                        chat_id=excluded.chat_id, username=excluded.username, first_name=excluded.first_name
                """, (bot_user_id, chat_id, username, first_name))
                conn.commit()
                conn.row_factory = sqlite3.Row
                return dict(conn.execute("SELECT * FROM bot_users WHERE bot_user_id = ?", (bot_user_id,)).fetchone())

    def get_bot_user(self, *, bot_user_id: int) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                res = conn.execute("SELECT * FROM bot_users WHERE bot_user_id = ?", (bot_user_id,)).fetchone()
                return dict(res) if res else None

    def save_bot_user_phone(self, *, bot_user_id: int, phone: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET phone = ? WHERE bot_user_id = ?", (phone, bot_user_id))

    def save_bot_user_session(self, *, bot_user_id: int, phone: str, api_id: int | None, api_hash: str | None, session_string: str, connected_at: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    UPDATE bot_users SET phone=?, api_id=?, api_hash=?, session_string=?, connected_at=?
                    WHERE bot_user_id=?
                """, (phone, api_id, api_hash, session_string, connected_at, bot_user_id))

    def update_user_gemini_key(self, *, bot_user_id: int, api_key: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET gemini_api_key = ? WHERE bot_user_id = ?", (api_key, bot_user_id))

    def update_user_models(self, bot_user_id: int, transcription_model: str, embedding_model: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET preferred_transcription_model=?, preferred_embedding_model=? WHERE bot_user_id=?", (transcription_model, embedding_model, bot_user_id))

    def upsert_auth_flow(self, *, bot_user_id: int, chat_id: int, phone: str, api_id: int | None, api_hash: str | None, session_string: str, phone_code_hash: str, status: str) -> dict[str, Any]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO auth_flows (bot_user_id, chat_id, phone, api_id, api_hash, session_string, phone_code_hash, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bot_user_id) DO UPDATE SET
                        chat_id=excluded.chat_id, phone=excluded.phone, api_id=excluded.api_id,
                        api_hash=excluded.api_hash, session_string=excluded.session_string,
                        phone_code_hash=excluded.phone_code_hash, status=excluded.status
                """, (bot_user_id, chat_id, phone, api_id, api_hash, session_string, phone_code_hash, status))
                conn.commit()
                conn.row_factory = sqlite3.Row
                return dict(conn.execute("SELECT * FROM auth_flows WHERE bot_user_id = ?", (bot_user_id,)).fetchone())

    def update_auth_flow_status(self, *, bot_user_id: int, status: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE auth_flows SET status = ? WHERE bot_user_id = ?", (status, bot_user_id))

    def get_auth_flow(self, *, bot_user_id: int) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                res = conn.execute("SELECT * FROM auth_flows WHERE bot_user_id = ?", (bot_user_id,)).fetchone()
                return dict(res) if res else None

    def clear_auth_flow(self, *, bot_user_id: int) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("DELETE FROM auth_flows WHERE bot_user_id = ?", (bot_user_id,))
