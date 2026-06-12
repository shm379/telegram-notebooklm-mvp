from __future__ import annotations

import json
import ssl
from pathlib import Path
from urllib import request

import certifi


class TelegramBotApi:
    def __init__(self, token: str) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}"
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def call(self, method: str, payload: dict[str, object] | None = None, files: dict | None = None) -> dict[str, object]:
        if files:
            # استفاده از requests برای ارسال فایل (ساده‌تر است)
            import requests
            url = f"{self.base_url}/{method}"
            res = requests.post(url, data=payload, files=files)
            return res.json()
            
        raw = None
        if payload is not None:
            raw = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/{method}",
            data=raw,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=90, context=self.ssl_context) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(str(data))
        return data

    def get_me(self) -> dict[str, object]:
        return self.call("getMe")

    def get_updates(self, *, offset: int | None = None, timeout: int = 30) -> list[dict[str, object]]:
        payload: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        return list(self.call("getUpdates", payload).get("result", []))

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: dict[str, object] | None = None,
        disable_web_page_preview: bool | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        if disable_web_page_preview is not None:
            payload["disable_web_page_preview"] = disable_web_page_preview
        self.call("sendMessage", payload)

    def send_photo(
        self,
        *,
        chat_id: int,
        photo_path: Path,
        caption: str | None = None,
    ) -> None:
        payload = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
        
        with open(photo_path, "rb") as f:
            self.call("sendPhoto", payload=payload, files={"photo": f})

    def get_file(self, file_id: str) -> dict[str, object]:
        """Resolve a file_id to its metadata (including ``file_path``) via getFile."""
        return dict(self.call("getFile", {"file_id": file_id}).get("result", {}))

    def download_file(self, file_path: str, dest: Path) -> Path:
        """Download a Bot API file (``file_path`` from getFile) to ``dest``."""
        url = f"{self.file_base_url}/{file_path}"
        req = request.Request(url)
        with request.urlopen(req, timeout=120, context=self.ssl_context) as response:
            dest.write_bytes(response.read())
        return dest

    def answer_callback_query(self, callback_query_id: str) -> None:
        self.call("answerCallbackQuery", {"callback_query_id": callback_query_id})

    def delete_message(self, chat_id: int, message_id: int) -> None:
        self.call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    @staticmethod
    def contact_keyboard() -> dict[str, object]:
        return {
            "keyboard": [
                [
                    {
                        "text": "Share Phone Number",
                        "request_contact": True,
                    }
                ]
            ],
            "resize_keyboard": True,
            "one_time_keyboard": True,
        }

    @staticmethod
    def remove_keyboard() -> dict[str, object]:
        return {"remove_keyboard": True}
