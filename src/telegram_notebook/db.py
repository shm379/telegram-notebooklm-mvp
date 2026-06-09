from __future__ import annotations

import sqlite3
import json
import threading
from pathlib import Path
from typing import Any


def connect(db_path: Path) -> Path:
    path = Path(db_path)
    if path.suffix.lower() != ".json":
        return path

    target = path.with_suffix(".db")
    if target.exists():
        return target

    _migrate_json_store(path, target)
    return target


def _migrate_json_store(source: Path, target: Path) -> None:
    repo = Repository(target)
    repo.init()

    if not source.exists():
        return

    raw = source.read_text(encoding="utf-8").strip()
    if not raw:
        return

    data = json.loads(raw)
    if not isinstance(data, dict):
        return

    with sqlite3.connect(target) as conn:
        for table in ("channels", "messages", "media_items", "chunks", "bot_users", "auth_flows"):
            rows = data.get(table, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or not row:
                    continue
                columns = list(row.keys())
                placeholders = ", ".join("?" for _ in columns)
                col_sql = ", ".join(columns)
                values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()]
                conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})",
                    values,
                )
        conn.commit()


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
                        owner_id INTEGER,
                        telegram_id INTEGER,
                        channel_url TEXT,
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
                        preferred_transcription_model TEXT DEFAULT 'gemini-2.5-flash-lite',
                        preferred_embedding_model TEXT DEFAULT 'text-embedding-004',
                        gemini_api_key TEXT,
                        vertex_project_id TEXT,
                        vertex_region TEXT,
                        vertex_index_id TEXT,
                        vertex_endpoint_id TEXT,
                        vertex_deployed_index_id TEXT
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
                        status TEXT,
                        vertex_project_id TEXT,
                        vertex_region TEXT,
                        vertex_index_id TEXT,
                        vertex_endpoint_id TEXT,
                        vertex_deployed_index_id TEXT
                    )
                """)
                self._ensure_channel_owner(conn)
                conn.commit()

    @staticmethod
    def _ensure_channel_owner(conn: sqlite3.Connection) -> None:
        """Migrate older databases to per-user channel ownership.

        Adds the ``owner_id`` column and replaces the global ``UNIQUE(channel_url)``
        constraint with a composite ``UNIQUE(owner_id, channel_url)`` index so that
        two users can independently ingest the same channel URL without sharing rows.
        Legacy rows (created before ownership existed) keep ``owner_id = NULL`` and are
        therefore invisible to per-user queries rather than leaking across users.
        """
        cols = [row[1] for row in conn.execute("PRAGMA table_info(channels)").fetchall()]
        if "owner_id" not in cols:
            conn.execute("ALTER TABLE channels RENAME TO channels_legacy")
            conn.execute("""
                CREATE TABLE channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER,
                    telegram_id INTEGER,
                    channel_url TEXT,
                    channel_title TEXT,
                    channel_username TEXT
                )
            """)
            conn.execute("""
                INSERT INTO channels (id, owner_id, telegram_id, channel_url, channel_title, channel_username)
                SELECT id, NULL, telegram_id, channel_url, channel_title, channel_username FROM channels_legacy
            """)
            conn.execute("DROP TABLE channels_legacy")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_channels_owner_url ON channels(owner_id, channel_url)"
        )

    def upsert_channel(self, *, owner_id: int, telegram_id: int, channel_url: str, title: str | None, username: str | None) -> int:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO channels (owner_id, telegram_id, channel_url, channel_title, channel_username)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(owner_id, channel_url) DO UPDATE SET
                        telegram_id=excluded.telegram_id,
                        channel_title=excluded.channel_title,
                        channel_username=excluded.channel_username
                """, (owner_id, telegram_id, channel_url, title, username))
                res = conn.execute(
                    "SELECT id FROM channels WHERE owner_id = ? AND channel_url = ?",
                    (owner_id, channel_url),
                ).fetchone()
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
                conn.commit()

    def mark_media_failed(self, *, media_item_id: int, error: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "UPDATE media_items SET transcript_status = 'error', transcript_error = ? WHERE id = ?",
                    (error, media_item_id),
                )
                conn.commit()

    def media_already_transcribed(self, media_item_id: int) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute("SELECT transcript_status FROM media_items WHERE id = ?", (media_item_id,)).fetchone()
                return res and res[0] == 'done'

    def keyword_candidates(self, *, owner_id: int, query: str, top_k: int, channel_url: str | None) -> list[dict[str, Any]]:
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
                    WHERE ch.owner_id = ? AND c.text LIKE ?
                """
                params: list[Any] = [owner_id, f"%{query}%"]
                if channel_url:
                    sql += " AND ch.channel_url = ?"
                    params.append(channel_url)

                rows = conn.execute(sql + " LIMIT ?", params + [top_k]).fetchall()
                return [dict(r) for r in rows]

    def embedding_candidates(self, *, owner_id: int, channel_url: str | None) -> list[dict[str, Any]]:
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
                    WHERE c.embedding IS NOT NULL AND ch.owner_id = ?
                """
                params: list[Any] = [owner_id]
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

    def list_channels(self, *, owner_id: int) -> list[dict[str, Any]]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM channels WHERE owner_id = ?", (owner_id,)
                ).fetchall()]

    def delete_channel_data(self, *, owner_id: int, channel_url: str) -> bool:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute(
                    "SELECT id FROM channels WHERE owner_id = ? AND channel_url = ?",
                    (owner_id, channel_url),
                ).fetchone()
                if not res: return False
                cid = res[0]
                conn.execute("DELETE FROM chunks WHERE media_item_id IN (SELECT id FROM media_items WHERE message_id IN (SELECT id FROM messages WHERE channel_id = ?))", (cid,))
                conn.execute("DELETE FROM media_items WHERE message_id IN (SELECT id FROM messages WHERE channel_id = ?)", (cid,))
                conn.execute("DELETE FROM messages WHERE channel_id = ?", (cid,))
                conn.execute("DELETE FROM channels WHERE id = ?", (cid,))
                conn.commit()
                return True

    def get_chunk_by_media_and_index(self, *, owner_id: int, media_item_id: int, chunk_index: int) -> dict[str, Any] | None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("""
                    SELECT c.text as chunk_text, mi.media_kind, mi.file_name,
                           m.message_url, m.caption, ch.channel_title, ch.channel_url
                    FROM chunks c
                    JOIN media_items mi ON c.media_item_id = mi.id
                    JOIN messages m ON mi.message_id = m.id
                    JOIN channels ch ON m.channel_id = ch.id
                    WHERE c.media_item_id = ? AND c.chunk_index = ? AND ch.owner_id = ?
                """, (media_item_id, chunk_index, owner_id)).fetchone()
                return dict(row) if row else None

    def upsert_bot_user(
self, *, bot_user_id: int, chat_id: int, username: str | None, first_name: str | None) -> dict[str, Any]:
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

    def disconnect_bot_user(self, *, bot_user_id: int) -> bool:
        """Remove a user's session and credentials. Returns True if a session existed."""
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                res = conn.execute(
                    "SELECT session_string FROM bot_users WHERE bot_user_id = ?",
                    (bot_user_id,),
                ).fetchone()
                had_session = bool(res and res[0])
                conn.execute(
                    """
                    UPDATE bot_users SET
                        session_string = NULL, api_id = NULL, api_hash = NULL,
                        phone = NULL, connected_at = NULL, gemini_api_key = NULL,
                        vertex_project_id = NULL, vertex_region = NULL, vertex_index_id = NULL,
                        vertex_endpoint_id = NULL, vertex_deployed_index_id = NULL
                    WHERE bot_user_id = ?
                    """,
                    (bot_user_id,),
                )
                conn.commit()
                return had_session

    def save_bot_user_phone(self, *, bot_user_id: int, phone: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET phone = ? WHERE bot_user_id = ?", (phone, bot_user_id))

    def save_bot_user_session(self, *, bot_user_id: int, phone: str, api_id: int | None, api_hash: str | None, session_string: str, connected_at: str, 
                              v_project: str | None = None, v_region: str | None = None, v_index: str | None = None, v_endpoint: str | None = None, v_deployed: str | None = None) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    UPDATE bot_users SET phone=?, api_id=?, api_hash=?, session_string=?, connected_at=?,
                                       vertex_project_id=?, vertex_region=?, vertex_index_id=?, vertex_endpoint_id=?, vertex_deployed_index_id=?
                    WHERE bot_user_id=?
                """, (phone, api_id, api_hash, session_string, connected_at, v_project, v_region, v_index, v_endpoint, v_deployed, bot_user_id))

    def update_user_gemini_key(self, *, bot_user_id: int, api_key: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET gemini_api_key = ? WHERE bot_user_id = ?", (api_key, bot_user_id))

    def update_user_models(self, bot_user_id: int, transcription_model: str, embedding_model: str) -> None:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE bot_users SET preferred_transcription_model=?, preferred_embedding_model=? WHERE bot_user_id=?", (transcription_model, embedding_model, bot_user_id))

    def upsert_auth_flow(self, *, bot_user_id: int, chat_id: int, phone: str, api_id: int | None, api_hash: str | None, session_string: str, phone_code_hash: str, status: str,
                        v_project: str | None = None, v_region: str | None = None, v_index: str | None = None, v_endpoint: str | None = None, v_deployed: str | None = None) -> dict[str, Any]:
        with self.lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute("""
                    INSERT INTO auth_flows (bot_user_id, chat_id, phone, api_id, api_hash, session_string, phone_code_hash, status,
                                          vertex_project_id, vertex_region, vertex_index_id, vertex_endpoint_id, vertex_deployed_index_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bot_user_id) DO UPDATE SET
                        chat_id=excluded.chat_id, phone=excluded.phone, api_id=excluded.api_id,
                        api_hash=excluded.api_hash, session_string=excluded.session_string,
                        phone_code_hash=excluded.phone_code_hash, status=excluded.status,
                        vertex_project_id=excluded.vertex_project_id, vertex_region=excluded.vertex_region,
                        vertex_index_id=excluded.vertex_index_id, vertex_endpoint_id=excluded.vertex_endpoint_id,
                        vertex_deployed_index_id=excluded.vertex_deployed_index_id
                """, (bot_user_id, chat_id, phone, api_id, api_hash, session_string, phone_code_hash, status, v_project, v_region, v_index, v_endpoint, v_deployed))
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
