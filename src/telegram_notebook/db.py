from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


def connect(db_path: Path) -> Path:
    return db_path


class Repository:
    def __init__(self, connection: Path) -> None:
        self.path = connection
        self.lock = threading.RLock()

    def init(self) -> None:
        with self.lock:
            data = self._load()
            changed = False
            for key in (
                "channels",
                "messages",
                "media_items",
                "chunks",
                "bot_users",
                "auth_flows",
            ):
                if key not in data:
                    data[key] = []
                    changed = True
            if changed or not self.path.exists():
                self._save(data)

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {
                "channels": [],
                "messages": [],
                "media_items": [],
                "chunks": [],
                "bot_users": [],
                "auth_flows": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _next_id(records: list[dict[str, Any]]) -> int:
        return max((int(record["id"]) for record in records), default=0) + 1

    def upsert_channel(
        self,
        *,
        telegram_id: int,
        channel_url: str,
        title: str | None,
        username: str | None,
    ) -> int:
        with self.lock:
            data = self._load()
            for channel in data["channels"]:
                if channel["channel_url"] == channel_url:
                    channel.update(
                        {
                            "telegram_id": telegram_id,
                            "channel_title": title,
                            "channel_username": username,
                        }
                    )
                    self._save(data)
                    return int(channel["id"])

            channel_id = self._next_id(data["channels"])
            data["channels"].append(
                {
                    "id": channel_id,
                    "telegram_id": telegram_id,
                    "channel_url": channel_url,
                    "channel_title": title,
                    "channel_username": username,
                }
            )
            self._save(data)
            return channel_id

    def create_or_get_message(
        self,
        *,
        channel_id: int,
        telegram_message_id: int,
        message_date: str | None,
        message_url: str | None,
        caption: str | None,
    ) -> int:
        with self.lock:
            data = self._load()
            for message in data["messages"]:
                if (
                    int(message["channel_id"]) == channel_id
                    and int(message["telegram_message_id"]) == telegram_message_id
                ):
                    message.update(
                        {
                            "message_date": message_date,
                            "message_url": message_url,
                            "caption": caption,
                        }
                    )
                    self._save(data)
                    return int(message["id"])

            message_id = self._next_id(data["messages"])
            data["messages"].append(
                {
                    "id": message_id,
                    "channel_id": channel_id,
                    "telegram_message_id": telegram_message_id,
                    "message_date": message_date,
                    "message_url": message_url,
                    "caption": caption,
                }
            )
            self._save(data)
            return message_id

    def create_or_get_media(
        self,
        *,
        message_id: int,
        file_name: str | None,
        file_path: str,
        mime_type: str | None,
        media_kind: str,
        duration_seconds: int | None,
        file_size_bytes: int | None,
    ) -> int:
        with self.lock:
            data = self._load()
            for item in data["media_items"]:
                if int(item["message_id"]) == message_id:
                    item.update(
                        {
                            "file_name": file_name,
                            "file_path": file_path,
                            "mime_type": mime_type,
                            "media_kind": media_kind,
                            "duration_seconds": duration_seconds,
                            "file_size_bytes": file_size_bytes,
                        }
                    )
                    self._save(data)
                    return int(item["id"])

            media_item_id = self._next_id(data["media_items"])
            data["media_items"].append(
                {
                    "id": media_item_id,
                    "message_id": message_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "mime_type": mime_type,
                    "media_kind": media_kind,
                    "duration_seconds": duration_seconds,
                    "file_size_bytes": file_size_bytes,
                    "transcript_text": None,
                    "transcript_status": "pending",
                    "transcript_error": None,
                }
            )
            self._save(data)
            return media_item_id

    def media_already_transcribed(self, media_item_id: int) -> bool:
        with self.lock:
            data = self._load()
            for item in data["media_items"]:
                if int(item["id"]) == media_item_id:
                    return item.get("transcript_status") == "done"
            return False

    def replace_chunks(
        self,
        *,
        media_item_id: int,
        chunks: list[dict[str, Any]],
    ) -> None:
        with self.lock:
            data = self._load()
            data["chunks"] = [
                chunk
                for chunk in data["chunks"]
                if int(chunk["media_item_id"]) != media_item_id
            ]
            next_id = self._next_id(data["chunks"])
            for offset, chunk in enumerate(chunks):
                data["chunks"].append(
                    {
                        "id": next_id + offset,
                        "media_item_id": media_item_id,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "embedding": chunk["embedding"],
                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                    }
                )
            self._save(data)

    def mark_media_transcribed(
        self,
        *,
        media_item_id: int,
        transcript_text: str,
    ) -> None:
        with self.lock:
            data = self._load()
            for item in data["media_items"]:
                if int(item["id"]) == media_item_id:
                    item["transcript_text"] = transcript_text
                    item["transcript_status"] = "done"
                    item["transcript_error"] = None
                    break
            self._save(data)

    def mark_media_failed(self, *, media_item_id: int, error: str) -> None:
        with self.lock:
            data = self._load()
            for item in data["media_items"]:
                if int(item["id"]) == media_item_id:
                    item["transcript_status"] = "failed"
                    item["transcript_error"] = error
                    break
            self._save(data)

    def keyword_candidates(
        self,
        *,
        query: str,
        top_k: int,
        channel_url: str | None,
    ) -> list[dict[str, Any]]:
        tokens = [token.strip().lower() for token in query.split() if token.strip()]
        if not tokens:
            return []
        rows = self._joined_rows(channel_url=channel_url)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            haystack = row["chunk_text"].lower()
            matches = sum(haystack.count(token) for token in tokens)
            if matches > 0:
                scored.append((float(matches), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    def embedding_candidates(
        self,
        *,
        channel_url: str | None,
    ) -> list[dict[str, Any]]:
        rows = self._joined_rows(channel_url=channel_url)
        return [row for row in rows if row.get("embedding_json") is not None]

    def _joined_rows(self, *, channel_url: str | None) -> list[dict[str, Any]]:
        with self.lock:
            data = self._load()

        channels = {int(item["id"]): item for item in data["channels"]}
        messages = {int(item["id"]): item for item in data["messages"]}
        media_items = {int(item["id"]): item for item in data["media_items"]}
        rows: list[dict[str, Any]] = []
        for chunk in data["chunks"]:
            media_item = media_items.get(int(chunk["media_item_id"]))
            if not media_item:
                continue
            message = messages.get(int(media_item["message_id"]))
            if not message:
                continue
            channel = channels.get(int(message["channel_id"]))
            if not channel:
                continue
            if channel_url and channel["channel_url"] != channel_url:
                continue
            rows.append(
                {
                    "chunk_id": int(chunk["id"]),
                    "chunk_text": chunk["text"],
                    "embedding_json": json.dumps(chunk["embedding"])
                    if chunk.get("embedding") is not None
                    else None,
                    "media_kind": media_item["media_kind"],
                    "file_name": media_item.get("file_name"),
                    "caption": message.get("caption"),
                    "message_url": message.get("message_url"),
                    "channel_title": channel.get("channel_title"),
                    "channel_url": channel["channel_url"],
                }
            )
        return rows

    def list_channels(self) -> list[dict[str, Any]]:
        with self.lock:
            data = self._load()
            return [dict(c) for c in data["channels"]]

    def delete_channel_data(self, *, channel_url: str) -> bool:
        with self.lock:
            data = self._load()
            channel = next((c for c in data["channels"] if c["channel_url"] == channel_url), None)
            if not channel:
                return False
            
            channel_id = int(channel["id"])
            
            # پیدا کردن تمام پیام‌های این کانال
            message_ids = [int(m["id"]) for m in data["messages"] if int(m["channel_id"]) == channel_id]
            
            # پیدا کردن تمام مدیاهای این پیام‌ها
            media_ids = [int(md["id"]) for md in data["media_items"] if int(md["message_id"]) in message_ids]
            
            # حذف چانک‌ها
            data["chunks"] = [ch for ch in data["chunks"] if int(ch["media_item_id"]) not in media_ids]
            
            # حذف مدیاها
            data["media_items"] = [md for md in data["media_items"] if int(md["id"]) not in media_ids]
            
            # حذف پیام‌ها
            data["messages"] = [m for m in data["messages"] if int(m["id"]) not in message_ids]
            
            # حذف خود کانال
            data["channels"] = [c for c in data["channels"] if int(c["id"]) != channel_id]
            
            self._save(data)
            return True

    def upsert_bot_user(
        self,
        *,
        bot_user_id: int,
        chat_id: int,
        username: str | None,
        first_name: str | None,
    ) -> dict[str, Any]:
        with self.lock:
            data = self._load()
            for user in data["bot_users"]:
                if int(user["bot_user_id"]) == bot_user_id:
                    user.update(
                        {
                            "chat_id": chat_id,
                            "username": username,
                            "first_name": first_name,
                        }
                    )
                    self._save(data)
                    return dict(user)

            record = {
                "id": self._next_id(data["bot_users"]),
                "bot_user_id": bot_user_id,
                "chat_id": chat_id,
                "username": username,
                "first_name": first_name,
                "phone": None,
                "api_id": None,
                "api_hash": None,
                "session_string": None,
                "connected_at": None,
                "preferred_transcription_model": "gemini-2.0-flash-lite",
                "preferred_embedding_model": "text-embedding-004",
            }
            data["bot_users"].append(record)
            self._save(data)
            return dict(record)

    def update_user_models(self, *, bot_user_id: int, transcription_model: str | None = None, embedding_model: str | None = None) -> None:
        with self.lock:
            data = self._load()
            for user in data["bot_users"]:
                if int(user["bot_user_id"]) == bot_user_id:
                    if transcription_model:
                        user["preferred_transcription_model"] = transcription_model
                    if embedding_model:
                        user["preferred_embedding_model"] = embedding_model
                    break
            self._save(data)

    def get_bot_user(self, *, bot_user_id: int) -> dict[str, Any] | None:
        with self.lock:
            data = self._load()
            for user in data["bot_users"]:
                if int(user["bot_user_id"]) == bot_user_id:
                    return dict(user)
        return None

    def save_bot_user_phone(self, *, bot_user_id: int, phone: str) -> None:
        with self.lock:
            data = self._load()
            for user in data["bot_users"]:
                if int(user["bot_user_id"]) == bot_user_id:
                    user["phone"] = phone
                    break
            self._save(data)

    def save_bot_user_session(
        self,
        *,
        bot_user_id: int,
        phone: str,
        api_id: int | None,
        api_hash: str | None,
        session_string: str,
        connected_at: str,
    ) -> None:
        with self.lock:
            data = self._load()
            for user in data["bot_users"]:
                if int(user["bot_user_id"]) == bot_user_id:
                    user["phone"] = phone
                    user["api_id"] = api_id
                    user["api_hash"] = api_hash
                    user["session_string"] = session_string
                    user["connected_at"] = connected_at
                    break
            self._save(data)

    def upsert_auth_flow(
        self,
        *,
        bot_user_id: int,
        chat_id: int,
        phone: str,
        api_id: int | None,
        api_hash: str | None,
        session_string: str,
        phone_code_hash: str,
        status: str,
    ) -> dict[str, Any]:
        with self.lock:
            data = self._load()
            for flow in data["auth_flows"]:
                if int(flow["bot_user_id"]) == bot_user_id:
                    flow.update(
                        {
                            "chat_id": chat_id,
                            "phone": phone,
                            "api_id": api_id,
                            "api_hash": api_hash,
                            "session_string": session_string,
                            "phone_code_hash": phone_code_hash,
                            "status": status,
                        }
                    )
                    self._save(data)
                    return dict(flow)

            record = {
                "id": self._next_id(data["auth_flows"]),
                "bot_user_id": bot_user_id,
                "chat_id": chat_id,
                "phone": phone,
                "api_id": api_id,
                "api_hash": api_hash,
                "session_string": session_string,
                "phone_code_hash": phone_code_hash,
                "status": status,
            }
            data["auth_flows"].append(record)
            self._save(data)
            return dict(record)

    def update_auth_flow_status(self, *, bot_user_id: int, status: str) -> None:
        with self.lock:
            data = self._load()
            for flow in data["auth_flows"]:
                if int(flow["bot_user_id"]) == bot_user_id:
                    flow["status"] = status
                    break
            self._save(data)

    def get_auth_flow(self, *, bot_user_id: int) -> dict[str, Any] | None:
        with self.lock:
            data = self._load()
            for flow in data["auth_flows"]:
                if int(flow["bot_user_id"]) == bot_user_id:
                    return dict(flow)
        return None

    def clear_auth_flow(self, *, bot_user_id: int) -> None:
        with self.lock:
            data = self._load()
            data["auth_flows"] = [
                flow
                for flow in data["auth_flows"]
                if int(flow["bot_user_id"]) != bot_user_id
            ]
            self._save(data)
