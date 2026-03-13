from __future__ import annotations

import json
import ssl
from urllib import parse, request
from pathlib import Path

import certifi


class TelegramBotApi:
    def __init__(self, token: str) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
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
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
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
