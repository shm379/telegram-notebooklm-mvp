from __future__ import annotations

import json
import ssl
from urllib import parse, request

import certifi


class TelegramBotApi:
    def __init__(self, token: str) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def call(self, method: str, payload: dict[str, object] | None = None) -> dict[str, object]:
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
        payload: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message"]}
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
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self.call("sendMessage", payload)

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
