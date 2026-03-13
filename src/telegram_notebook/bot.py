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

    repository = Repository(connect(settings.db_path))
    repository.init()
    
    # سرویس‌های پیش‌فرض (سیستمی)
    embeddings = EmbeddingService(
        provider=settings.embedding_provider,
        api_key=settings.gemini_api_key if settings.embedding_provider == "gemini" else settings.openai_api_key,
        model=settings.embedding_model,
    )
    transcription = TranscriptionService(
        provider=settings.transcription_provider,
        api_key=settings.gemini_api_key if settings.transcription_provider == "gemini" else settings.openai_api_key,
        model=settings.transcription_model,
    )
    
    search_service = SearchService(repository, embeddings)
    api = TelegramBotApi(settings.telegram_bot_token)
    return BotServices(
        api=api,
        repository=repository,
        search_service=search_service,
        settings=settings,
    )


def normalize_phone(raw: str) -> str | None:
    compact = raw.strip().replace(" ", "").replace("-", "")
    if not PHONE_RE.match(compact):
        return None
    return compact if compact.startswith("+") else f"+{compact}"


def normalize_code(raw: str) -> str | None:
    compact = raw.strip().replace(" ", "").replace("-", "")
    if compact.isdigit() and 3 <= len(compact) <= 12:
        return compact
    if CODE_RE.match(raw.strip()):
        return compact
    return None


class NotebookBot:
    def __init__(self, services: BotServices) -> None:
        self.services = services
        self.offset: int | None = None
        # استفاده از یک لوپ مشخص برای جلوگیری از گیر کردن
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def run_forever(self) -> None:
        me = self.services.api.get_me().get("result", {})
        username = me.get("username", "unknown")
        print(f"Bot polling started for @{username}")
        while True:
            try:
                updates = self.services.api.get_updates(offset=self.offset, timeout=30)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    self.handle_update(update)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"bot polling error: {exc}")
                time.sleep(3)

    def _safe_run(self, coro):
        """اجرای ایمن توابع اسینک برای جلوگیری از deadlocks"""
        return self.loop.run_until_complete(coro)

    def handle_update(self, update: dict[str, object]) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            self._handle_callback(callback)
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        if chat.get("type") != "private":
            return

        chat_id = int(chat["id"])
        bot_user_id = int(sender["id"])
        
        self.services.repository.upsert_bot_user(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            username=sender.get("username"),
            first_name=sender.get("first_name"),
        )

        if "contact" in message:
            self._handle_contact(chat_id=chat_id, bot_user_id=bot_user_id, contact=message["contact"])
            return

        text = str(message.get("text", "")).strip()
        if not text:
            return

        if text.startswith("/start"):
            self._send_welcome(chat_id, bot_user_id)
            return
        if text.startswith("/connect"):
            self._begin_connect(chat_id, bot_user_id)
            return
        if text.startswith("/settings"):
            self._send_settings(chat_id, bot_user_id)
            return
        if text.startswith("/status"):
            self._send_status(chat_id, bot_user_id)
            return
        if text.startswith("/cancel"):
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(
                chat_id=chat_id,
                text="عملیات لغو شد.",
                reply_markup=TelegramBotApi.remove_keyboard(),
            )
            return
        if text.startswith("/search"):
            source_url = None
            query = text.removeprefix("/search").strip()
            if " --source " in query:
                parts = query.split(" --source ")
                query = parts[0].strip()
                source_url = parts[1].strip()
            self._search(chat_id, query, source_url)
            return
        if text.startswith("/join"):
            link = text.removeprefix("/join").strip()
            self._handle_join(chat_id, bot_user_id, link)
            return
        if text.startswith("/ingest"):
            link = text.removeprefix("/ingest").strip()
            self._handle_ingest(chat_id, bot_user_id, link)
            return
        if text.startswith("/sources"):
            self._handle_sources(chat_id)
            return
        if text.startswith("/delete"):
            link = text.removeprefix("/delete").strip()
            self._handle_delete(chat_id, link)
            return

        # مدیریت جریان‌های ورودی (States)
        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow:
            status = flow["status"]
            if status == "awaiting_gemini_key":
                self._handle_gemini_key(chat_id, bot_user_id, text, flow)
            elif status == "awaiting_api_id":
                self._handle_api_id(chat_id, bot_user_id, text, flow)
            elif status == "awaiting_api_hash":
                self._handle_api_hash(chat_id, bot_user_id, text, flow)
            elif status == "awaiting_code":
                self._handle_code(chat_id, bot_user_id, text, flow)
            elif status == "awaiting_password":
                self._handle_password(chat_id, bot_user_id, text, flow)
            return

        self.services.api.send_message(chat_id=chat_id, text="دستور نامعتبر. /start یا /connect")

    def _send_welcome(self, chat_id: int, bot_user_id: int) -> None:
        self.services.api.send_message(
            chat_id=chat_id,
            text=(
                "سلام! خوش آمدید. این ربات دستیار هوشمند شما برای تحقیق در تلگرام است.\n\n"
                "برای شروع ابتدا باید اکانت خود را متصل کنید:\n"
                "/connect - شروع فرآیند اتصال و تنظیمات\n"
                "/settings - مدیریت مدل‌های هوش مصنوعی\n"
                "/search - جست‌وجو در منابع ایندکس شده"
            ),
        )

    def _begin_connect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id, chat_id=chat_id, phone="",
            api_id=None, api_hash=None, session_string="", phone_code_hash="",
            status="awaiting_phone_initial"
        )
        self.services.api.send_message(
            chat_id=chat_id,
            text="قدم اول: اشتراک‌گذاری شماره موبایل\nلطفاً دکمه زیر را بزنید تا شماره شما در سیستم ثبت شود.",
            reply_markup=TelegramBotApi.contact_keyboard()
        )

    def _handle_contact(self, chat_id: int, bot_user_id: int, contact: dict) -> None:
        phone = normalize_phone(str(contact.get("phone_number", "")))
        if not phone:
            self.services.api.send_message(chat_id=chat_id, text="خطا در دریافت شماره.")
            return
        
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=phone)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_gemini_key")
        
        # ارسال راهنمای Gemini (با عکس اگر موجود باشد)
        guide_text = (
            "قدم دوم: تنظیم کلید Gemini\n\n"
            "برای استفاده از هوش مصنوعی، نیاز به یک کلید رایگان دارید:\n"
            "۱. به سایت https://aistudio.google.com/app/apikey بروید.\n"
            "۲. روی دکمه Create API Key کلیک کنید.\n"
            "۳. کلید ساخته شده را کپی کرده و اینجا بفرستید."
        )
        
        photo_path = Path("data/media/gemini_guide.jpg")
        if photo_path.exists():
            # در این نسخه ساده متد ارسال عکس نداریم، پس فعلاً فقط متن
            pass
        
        self.services.api.send_message(chat_id=chat_id, text=guide_text, reply_markup=TelegramBotApi.remove_keyboard())

    def _handle_gemini_key(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        if not text.startswith("AIza"):
            self.services.api.send_message(chat_id=chat_id, text="کلید Gemini نامعتبر است. باید با AIza شروع شود.")
            return
        
        self.services.repository.update_user_gemini_key(bot_user_id=bot_user_id, api_key=text)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_api_id")
        
        guide_text = (
            "قدم سوم: دریافت API تلگرام\n\n"
            "۱. به سایت https://my.telegram.org بروید.\n"
            "۲. وارد شوید و به بخش API Development Tools بروید.\n"
            "۳. یک اپلیکیشن بسازید (نام دلخواه).\n"
            "۴. مقدار api_id را کپی کرده و اینجا بفرستید."
        )
        self.services.api.send_message(chat_id=chat_id, text=guide_text)

    def _handle_api_id(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        if not text.isdigit():
            self.services.api.send_message(chat_id=chat_id, text="فقط عدد بفرستید.")
            return
        
        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"],
            api_id=int(text), api_hash=None, session_string="", phone_code_hash="",
            status="awaiting_api_hash"
        )
        self.services.api.send_message(chat_id=chat_id, text="حالا api_hash را بفرستید.")

    def _handle_api_hash(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        if len(text) < 10:
            self.services.api.send_message(chat_id=chat_id, text="api_hash نامعتبر.")
            return
        
        self.services.api.send_message(chat_id=chat_id, text="در حال درخواست کد تایید... (کمی صبر کنید)")
        
        try:
            # استفاده از متد جدید برای جلوگیری از فریز شدن
            res = self._safe_run(request_login_code(self.services.settings, flow["phone"], api_id=flow["api_id"], api_hash=text))
            
            self.services.repository.upsert_auth_flow(
                bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"],
                api_id=flow["api_id"], api_hash=text,
                session_string=res["session_string"], phone_code_hash=res["phone_code_hash"],
                status="awaiting_code"
            )
            self.services.api.send_message(chat_id=chat_id, text="کد تلگرام ارسال شد. آن را وارد کنید.")
        except Exception as e:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا در ارتباط با تلگرام: {e}\nممکن است API_ID یا HASH اشتباه باشد.")

    def _handle_code(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        code = normalize_code(text)
        if not code:
            self.services.api.send_message(chat_id=chat_id, text="کد نامعتبر.")
            return
        
        try:
            res = self._safe_run(sign_in_with_code(
                self.services.settings, phone=flow["phone"], session_string=flow["session_string"],
                code=code, phone_code_hash=flow["phone_code_hash"],
                api_id=flow["api_id"], api_hash=flow["api_hash"]
            ))
            
            if res["status"] == "password_required":
                self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_password")
                self.services.api.send_message(chat_id=chat_id, text="رمز دو مرحله‌ای (Two-step verification) اکانت را وارد کنید.")
                return

            self.services.repository.save_bot_user_session(
                bot_user_id=bot_user_id, phone=flow["phone"],
                api_id=flow["api_id"], api_hash=flow["api_hash"],
                session_string=res["session_string"], connected_at=res["connected_at"]
            )
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="اکانت با موفقیت متصل شد! 🎉")
        except Exception as e:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا در تایید کد: {e}")

    def _handle_password(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        try:
            res = self._safe_run(sign_in_with_password(
                self.services.settings, session_string=flow["session_string"], password=text,
                api_id=flow["api_id"], api_hash=flow["api_hash"]
            ))
            self.services.repository.save_bot_user_session(
                bot_user_id=bot_user_id, phone=flow["phone"],
                api_id=flow["api_id"], api_hash=flow["api_hash"],
                session_string=res["session_string"], connected_at=res["connected_at"]
            )
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="اکانت با موفقیت متصل شد!")
        except Exception as e:
            self.services.api.send_message(chat_id=chat_id, text=f"رمز اشتباه است: {e}")

    def _handle_callback(self, callback: dict) -> None:
        chat_id = int(callback["message"]["chat"]["id"])
        bot_user_id = int(callback["from"]["id"])
        data = str(callback.get("data", ""))
        
        if data == "set_model_gemini":
            self.services.repository.update_user_models(bot_user_id, "gemini-2.0-flash-lite", "text-embedding-004")
            self.services.api.send_message(chat_id, "مدل به Gemini تغییر یافت.")
        elif data == "set_model_openai":
            self.services.repository.update_user_models(bot_user_id, "whisper-1", "text-embedding-3-small")
            self.services.api.send_message(chat_id, "مدل به OpenAI تغییر یافت.")
        
        self.services.api.answer_callback_query(callback["id"])

    def _search(self, chat_id: int, query: str, source_url: str | None = None) -> None:
        if not query:
            self.services.api.send_message(chat_id, "عبارت جست‌وجو را بنویسید.")
            return
        # دریافت کلید اختصاصی کاربر برای جست‌وجو
        user = self.services.repository.get_bot_user(bot_user_id=chat_id) # ساده سازی
        api_key = user.get("gemini_api_key") if user else None
        
        results = self.services.search_service.search(query=query, channel_url=source_url, top_k=5)
        if not results:
            self.services.api.send_message(chat_id, "نتیجه‌ای یافت نشد.")
            return
        
        resp = []
        for item in results:
            resp.append(f"📍 {item.channel_title or 'منبع'}\n{item.chunk_text[:300]}\n🔗 {item.message_url or ''}")
        self.services.api.send_message(chat_id, "\n\n".join(resp))

    def _handle_sources(self, chat_id: int) -> None:
        channels = self.services.repository.list_channels()
        if not channels:
            self.services.api.send_message(chat_id, "منبعی ایندکس نشده.")
            return
        lines = [f"📚 {c.get('channel_title') or 'بدون نام'}: {c['channel_url']}" for c in channels]
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_delete(self, chat_id: int, link: str) -> None:
        if self.services.repository.delete_channel_data(channel_url=link):
            self.services.api.send_message(chat_id, "پاک شد.")
        else:
            self.services.api.send_message(chat_id, "یافت نشد.")

    def _handle_join(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "ابتدا /connect")
            return
        from .telegram_client import build_client_from_session_string, join_chat
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))
        async def _do():
            async with client: return await join_chat(client, link)
        try:
            self.services.api.send_message(chat_id, self._safe_run(_do()))
        except Exception as e:
            self.services.api.send_message(chat_id, f"خطا: {e}")

    def _handle_ingest(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "ابتدا /connect")
            return
        self.services.api.send_message(chat_id, "شروع دریافت پیام‌ها...")
        from .telegram_client import build_client_from_session_string, iter_all_messages, fetch_channel_info
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))
        
        async def _do():
            async with client:
                info = await fetch_channel_info(client, link)
                channel_id = self.services.repository.upsert_channel(telegram_id=info.telegram_id, channel_url=info.canonical_url, title=info.title, username=info.username)
                messages = await iter_all_messages(client, channel_url=info.canonical_url, limit=100)
                
                # استفاده از کلید کاربر در پایپ‌لاین
                from .embeddings import EmbeddingService
                user_embeddings = EmbeddingService(
                    provider="gemini", 
                    api_key=user.get("gemini_api_key") or self.services.settings.gemini_api_key,
                    model="text-embedding-004"
                )
                
                from .pipeline import IngestionPipeline
                pipeline = IngestionPipeline(
                    settings=self.services.settings, repository=self.services.repository,
                    transcription=None, embeddings=user_embeddings
                )
                
                count = 0
                for msg in messages:
                    db_msg_id = self.services.repository.create_or_get_message(channel_id=channel_id, telegram_message_id=msg.telegram_message_id, message_date=msg.message_date, message_url=msg.message_url, caption=msg.caption)
                    if msg.media_kind == "text" and msg.caption:
                        await pipeline._process_text_message(db_msg_id, msg.caption)
                        count += 1
                return f"تعداد {count} پیام ایندکس شد."
        
        try:
            self.services.api.send_message(chat_id, self._safe_run(_do()))
        except Exception as e:
            self.services.api.send_message(chat_id, f"خطا: {e}")

    def _send_status(self, chat_id: int, bot_user_id: int) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "متصل نیستید. /connect")
            return
        phone = user.get("phone", "نامعلوم")
        has_key = "✅ ثبت شده" if user.get("gemini_api_key") else "❌ ثبت نشده"
        self.services.api.send_message(chat_id, f"وضعیت اکانت:\nشماره: {phone}\nکلید Gemini: {has_key}")

    def _send_settings(self, chat_id: int, bot_user_id: int) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        current_trans = user.get("preferred_transcription_model", "Gemini")
        
        markup = {
            "inline_keyboard": [
                [
                    {"text": "Gemini (Lite)", "callback_data": "set_model_gemini"},
                    {"text": "OpenAI (Whisper)", "callback_data": "set_model_openai"}
                ],
                [{"text": "بستن", "callback_data": "close_settings"}]
            ]
        }
        self.services.api.send_message(
            chat_id=chat_id,
            text=f"مدل فعلی: {current_trans}\n\nانتخاب مدل:",
            reply_markup=markup
        )


def main() -> None:
    bot = NotebookBot(build_services())
    bot.run_forever()


if __name__ == "__main__":
    main()
