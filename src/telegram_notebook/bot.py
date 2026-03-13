from __future__ import annotations

import asyncio
import re
import time
import os
from dataclasses import dataclass
from pathlib import Path

from .bot_api import TelegramBotApi
from .config import get_settings
from .db import Repository, connect
from .embeddings import EmbeddingService
from .pipeline import IngestionPipeline
from .search import SearchService
from .telegram_client import request_login_code, sign_in_with_code, sign_in_with_password
from .transcription import TranscriptionService


PHONE_RE = re.compile(r"^\+?\d{10,15}$")
CODE_RE = re.compile(r"^\d[\d\-\s]{2,12}\d$")


@dataclass(slots=True)
class BotServices:
    api: TelegramBotApi
    repository: Repository
    search_service: SearchService
    settings: object


def build_services() -> BotServices:
    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the bot")

    repository = Repository(settings.db_path)
    repository.init()
    
    embeddings = EmbeddingService(
        provider="gemini",
        api_key=settings.gemini_api_key,
        model="text-embedding-004",
    )
    
    search_service = SearchService(repository, embeddings)
    api = TelegramBotApi(settings.telegram_bot_token)
    return BotServices(api=api, repository=repository, search_service=search_service, settings=settings)


def normalize_phone(raw: str) -> str | None:
    compact = raw.strip().replace(" ", "").replace("-", "")
    if not PHONE_RE.match(compact): return None
    return compact if compact.startswith("+") else f"+{compact}"


def normalize_code(raw: str) -> str | None:
    compact = raw.strip().replace(" ", "").replace("-", "")
    return compact if (compact.isdigit() and 3 <= len(compact) <= 12) or CODE_RE.match(raw.strip()) else None


class NotebookBot:
    def __init__(self, services: BotServices) -> None:
        self.services = services
        self.offset: int | None = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run_forever(self) -> None:
        print(f"Bot polling started...")
        while True:
            try:
                updates = self.services.api.get_updates(offset=self.offset, timeout=30)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    self.handle_update(update)
            except KeyboardInterrupt: break
            except Exception as exc:
                print(f"bot polling error: {exc}")
                time.sleep(3)

    def handle_update(self, update: dict[str, object]) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            self._handle_callback(callback)
            return

        message = update.get("message")
        if not isinstance(message, dict): return

        chat_id = int(message["chat"]["id"])
        sender = message.get("from") or {}
        bot_user_id = int(sender.get("id", 0))
        if not bot_user_id: return

        self.services.repository.upsert_bot_user(bot_user_id=bot_user_id, chat_id=chat_id, username=sender.get("username"), first_name=sender.get("first_name"))

        if "contact" in message:
            self._handle_contact(chat_id, bot_user_id, message["contact"])
            return

        text = str(message.get("text", "")).strip()
        if not text: return

        if text.startswith("/start"): self._send_welcome(chat_id)
        elif text.startswith("/connect"): self._begin_connect(chat_id, bot_user_id)
        elif text.startswith("/cancel"):
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="لغو شد.", reply_markup=TelegramBotApi.remove_keyboard())
        elif text.startswith("/search"):
            query = text.removeprefix("/search").strip()
            source = None
            if " --source " in query:
                parts = query.split(" --source ")
                query, source = parts[0].strip(), parts[1].strip()
            self._search(chat_id, query, source)
        elif text.startswith("/join"): self._handle_join(chat_id, bot_user_id, text.removeprefix("/join").strip())
        elif text.startswith("/ingest"): self._handle_ingest(chat_id, bot_user_id, text.removeprefix("/ingest").strip())
        elif text.startswith("/sources"): self._handle_sources(chat_id)
        elif text.startswith("/delete"): self._handle_delete(chat_id, text.removeprefix("/delete").strip())
        
        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow:
            st = flow["status"]
            if st == "awaiting_gemini_key": self._handle_gemini_key(chat_id, bot_user_id, text)
            elif st == "awaiting_api_id": self._handle_api_id(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_api_hash": self._handle_api_hash(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_code": self._handle_code(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_password": self._handle_password(chat_id, bot_user_id, text, flow)

    def _send_welcome(self, chat_id: int) -> None:
        self.services.api.send_message(chat_id=chat_id, text="سلام! به دستیار هوشمند تحقیق خوش آمدید.\nبرای شروع اتصال اکانت /connect را بزنید.")

    def _begin_connect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone="", api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_phone_initial")
        self.services.api.send_message(chat_id=chat_id, text="۱. اشتراک‌گذاری شماره\nلطفاً دکمه زیر را بزنید:", reply_markup=TelegramBotApi.contact_keyboard())

    def _handle_contact(self, chat_id: int, bot_user_id: int, contact: dict) -> None:
        phone = normalize_phone(str(contact.get("phone_number", "")))
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=phone)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_gemini_key")
        self.services.api.send_message(chat_id=chat_id, text="۲. کلید Vertex AI (Gemini)\nبه لینک زیر بروید و کلید خود را بسازید:\nhttps://aistudio.google.com/app/apikey\nسپس کلید را اینجا بفرستید (شروع با AIza).", reply_markup=TelegramBotApi.remove_keyboard())

    def _handle_gemini_key(self, chat_id: int, bot_user_id: int, text: str) -> None:
        if not text.startswith("AIza"):
            self.services.api.send_message(chat_id=chat_id, text="کلید نامعتبر.")
            return
        self.services.repository.update_user_gemini_key(bot_user_id=bot_user_id, api_key=text)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_api_id")
        self.services.api.send_message(chat_id=chat_id, text="۳. تنظیمات تلگرام\nبه my.telegram.org بروید و در بخش API Development Tools یک اپ بسازید.\nحالا API_ID را بفرستید.")

    def _handle_api_id(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        if not text.isdigit(): return
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=int(text), api_hash=None, session_string="", phone_code_hash="", status="awaiting_api_hash")
        self.services.api.send_message(chat_id=chat_id, text="حالا API_HASH را بفرستید.")

    def _handle_api_hash(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        self.services.api.send_message(chat_id=chat_id, text="در حال درخواست کد... (صبر کنید)")
        try:
            res = self.loop.run_until_complete(request_login_code(self.services.settings, flow["phone"], api_id=flow["api_id"], api_hash=text))
            self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=text, session_string=res["session_string"], phone_code_hash=res["phone_code_hash"], status="awaiting_code")
            self.services.api.send_message(chat_id=chat_id, text="کد تلگرام ارسال شد. آن را وارد کنید.")
        except Exception as e:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا: {e}")

    def _handle_code(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        code = normalize_code(text)
        try:
            res = self.loop.run_until_complete(sign_in_with_code(self.services.settings, phone=flow["phone"], session_string=flow["session_string"], code=code, phone_code_hash=flow["phone_code_hash"], api_id=flow["api_id"], api_hash=flow["api_hash"]))
            if res["status"] == "password_required":
                self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_password")
                self.services.api.send_message(chat_id=chat_id, text="رمز دو مرحله‌ای را وارد کنید.")
                return
            self.services.repository.save_bot_user_session(bot_user_id=bot_user_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], connected_at=res["connected_at"])
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="متصل شد! 🎉")
        except Exception as e: self.services.api.send_message(chat_id=chat_id, text=f"خطا: {e}")

    def _search(self, chat_id: int, query: str, source: str | None) -> None:
        results = self.services.search_service.search(query=query, channel_url=source, top_k=5)
        if not results:
            self.services.api.send_message(chat_id=chat_id, text="نتیجه‌ای یافت نشد.")
            return
        resp = [f"📍 {r.channel_title or 'منبع'}\n{r.chunk_text[:300]}\n🔗 {r.message_url or ''}" for r in results]
        self.services.api.send_message(chat_id=chat_id, text="\n\n".join(resp))

    def _handle_sources(self, chat_id: int) -> None:
        ch = self.services.repository.list_channels()
        if not ch: self.services.api.send_message(chat_id, "منبعی نیست."); return
        self.services.api.send_message(chat_id, "\n".join([f"📚 {c.get('channel_title') or 'بدون نام'}: {c['channel_url']}" for c in ch]))

    def _handle_delete(self, chat_id: int, link: str) -> None:
        if self.services.repository.delete_channel_data(channel_url=link): self.services.api.send_message(chat_id, "پاک شد.")
        else: self.services.api.send_message(chat_id, "یافت نشد.")

    def _handle_join(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        from .telegram_client import build_client_from_session_string, join_chat
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))
        async def _do():
            async with client: return await join_chat(client, link)
        try: self.services.api.send_message(chat_id, self.loop.run_until_complete(_do()))
        except Exception as e: self.services.api.send_message(chat_id, f"خطا: {e}")

    def _handle_ingest(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        from .telegram_client import build_client_from_session_string, iter_all_messages, fetch_channel_info
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))
        async def _do():
            async with client:
                info = await fetch_channel_info(client, link)
                channel_id = self.services.repository.upsert_channel(telegram_id=info.telegram_id, channel_url=info.canonical_url, title=info.title, username=info.username)
                messages = await iter_all_messages(client, channel_url=info.canonical_url, limit=100)
                from .embeddings import EmbeddingService
                u_emb = EmbeddingService(provider="gemini", api_key=user.get("gemini_api_key") or self.services.settings.gemini_api_key, model="text-embedding-004")
                from .pipeline import IngestionPipeline
                pipeline = IngestionPipeline(settings=self.services.settings, repository=self.services.repository, transcription=None, embeddings=u_emb)
                count = 0
                for msg in messages:
                    mid = self.services.repository.create_or_get_message(channel_id=channel_id, telegram_message_id=msg.telegram_message_id, message_date=msg.message_date, message_url=msg.message_url, caption=msg.caption)
                    if msg.media_kind == "text" and msg.caption:
                        await pipeline._process_text_message(mid, msg.caption); count += 1
                return f"تعداد {count} پیام ایندکس شد."
        try: self.services.api.send_message(chat_id, self.loop.run_until_complete(_do()))
        except Exception as e: self.services.api.send_message(chat_id, f"خطا: {e}")

    def _handle_callback(self, callback: dict) -> None:
        self.services.api.answer_callback_query(callback["id"])

def main() -> None:
    bot = NotebookBot(build_services())
    bot.run_forever()

if __name__ == "__main__":
    main()
