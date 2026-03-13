from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

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
    IngestionPipeline(
        settings=settings,
        repository=repository,
        transcription=transcription,
        embeddings=embeddings,
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
                raise
            except Exception as exc:
                print(f"bot polling error: {exc}")
                time.sleep(3)

    def handle_update(self, update: dict[str, object]) -> None:
        # مدیریت دکمه‌های شیشه‌ای (Callback Query)
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
        if text.startswith("/join"):
            link = text.removeprefix("/join").strip()
            self._handle_join(chat_id, bot_user_id, link)
            return
        if text.startswith("/ingest"):
            link = text.removeprefix("/ingest").strip()
            self._handle_ingest(chat_id, bot_user_id, link)
            return
        if text.startswith("/cancel"):
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(
                chat_id=chat_id,
                text="فرآیند اتصال لغو شد.",
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
        if text.startswith("/sources"):
            self._handle_sources(chat_id)
            return
        if text.startswith("/delete"):
            link = text.removeprefix("/delete").strip()
            self._handle_delete(chat_id, link)
            return

        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow:
            if flow["status"] == "awaiting_api_id":
                self._handle_api_id(chat_id=chat_id, bot_user_id=bot_user_id, text=text, flow=flow)
            elif flow["status"] == "awaiting_api_hash":
                self._handle_api_hash(chat_id=chat_id, bot_user_id=bot_user_id, text=text, flow=flow)
            elif flow["status"] == "awaiting_phone":
                self._handle_phone(chat_id=chat_id, bot_user_id=bot_user_id, raw_phone=text, flow=flow)
            elif flow["status"] == "awaiting_code":
                self._handle_code(chat_id=chat_id, bot_user_id=bot_user_id, raw_code=text, flow=flow)
            elif flow["status"] == "awaiting_password":
                self._handle_password(chat_id=chat_id, bot_user_id=bot_user_id, password=text, flow=flow)
            return

        self.services.api.send_message(
            chat_id=chat_id,
            text="دستور را نشناختم. /start یا /connect را بزن.",
        )

    def _send_welcome(self, chat_id: int, bot_user_id: int) -> None:
        self.services.api.send_message(
            chat_id=chat_id,
            text=(
                "سلام. این ربات برای مدیریت جست‌وجو در کانال‌های تلگرام و ساخت Notebook شخصی شماست.\n\n"
                "دستورها:\n"
                "/connect برای اتصال اکانت واقعی\n"
                "/settings برای انتخاب مدل هوش مصنوعی (Gemini/OpenAI)\n"
                "/join <link> برای عضویت در کانال/گروه\n"
                "/ingest <link> برای دریافت پیام‌ها\n"
                "/search <query> برای جست‌وجو\n"
                "/status برای دیدن وضعیت\n"
                "/cancel برای لغو عملیات فعلی"
            ),
        )

    def _send_settings(self, chat_id: int, bot_user_id: int) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        trans = user.get("preferred_transcription_model") or "Gemini"
        
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
            text=f"تنظیمات مدل هوش مصنوعی:\nمدل فعلی: {trans}\n\nیک مدل را انتخاب کنید:",
            reply_markup=markup
        )

    def _handle_callback(self, callback: dict[str, object]) -> None:
        chat_id = int(callback["message"]["chat"]["id"])
        bot_user_id = int(callback["from"]["id"])
        data = str(callback.get("data", ""))
        
        if data == "set_model_gemini":
            self.services.repository.update_user_models(
                bot_user_id=bot_user_id,
                transcription_model="gemini-2.0-flash-lite",
                embedding_model="text-embedding-004"
            )
            self.services.api.send_message(chat_id=chat_id, text="مدل به Gemini تغییر یافت.")
        elif data == "set_model_openai":
            self.services.repository.update_user_models(
                bot_user_id=bot_user_id,
                transcription_model="whisper-1",
                embedding_model="text-embedding-3-small"
            )
            self.services.api.send_message(chat_id=chat_id, text="مدل به OpenAI تغییر یافت.")
        elif data == "close_settings":
            self.services.api.delete_message(chat_id=chat_id, message_id=callback["message"]["message_id"])

        self.services.api.answer_callback_query(callback["id"])

    def _begin_connect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            phone="",
            api_id=None,
            api_hash=None,
            session_string="",
            phone_code_hash="",
            status="awaiting_phone_initial",
        )
        self.services.api.send_message(
            chat_id=chat_id,
            text=(
                "برای اتصال اکانت واقعی تلگرام، نیاز به API_ID و API_HASH دارید.\n\n"
                "۱. به سایت my.telegram.org بروید.\n"
                "۲. وارد شوید و در بخش API development tools یک اپلیکیشن بسازید.\n"
                "۳. کدهای نمایش داده شده را یادداشت کنید.\n\n"
                "اگر آماده‌اید، دکمه زیر را برای اشتراک‌گذاری شماره موبایل بزنید."
            ),
            reply_markup=TelegramBotApi.contact_keyboard(),
        )

    def _handle_contact(self, *, chat_id: int, bot_user_id: int, contact: dict[str, object]) -> None:
        raw_phone = str(contact.get("phone_number", ""))
        normalized = normalize_phone(raw_phone)
        if not normalized:
            self.services.api.send_message(chat_id=chat_id, text="شماره معتبر نبود.")
            return
        
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=normalized)
        
        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if not flow or flow["status"] != "awaiting_phone_initial":
            return

        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            phone=normalized,
            api_id=None,
            api_hash=None,
            session_string="",
            phone_code_hash="",
            status="awaiting_api_id",
        )
        self.services.api.send_message(
            chat_id=chat_id,
            text="شماره شما ثبت شد. حالا TELEGRAM_API_ID خود را بفرست.\n(می‌توانی از my.telegram.org بگیری)",
            reply_markup=TelegramBotApi.remove_keyboard(),
        )

    def _handle_api_id(self, *, chat_id: int, bot_user_id: int, text: str, flow: dict[str, object]) -> None:
        if not text.isdigit():
            self.services.api.send_message(chat_id=chat_id, text="API_ID باید فقط عدد باشد.")
            return
        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            phone=str(flow.get("phone", "")),
            api_id=int(text),
            api_hash=None,
            session_string="",
            phone_code_hash="",
            status="awaiting_api_hash",
        )
        self.services.api.send_message(chat_id=chat_id, text="حالا TELEGRAM_API_HASH را بفرست.")

    def _handle_api_hash(self, *, chat_id: int, bot_user_id: int, text: str, flow: dict[str, object]) -> None:
        if len(text) < 10:
            self.services.api.send_message(chat_id=chat_id, text="API_HASH معتبر به نظر نمی‌رسد.")
            return
        
        phone = str(flow.get("phone", ""))
        api_id = int(flow.get("api_id")) if flow.get("api_id") else None
        
        self.services.api.send_message(chat_id=chat_id, text="در حال درخواست کد تایید از تلگرام...")
        
        try:
            result = asyncio.run(request_login_code(self.services.settings, phone, api_id=api_id, api_hash=text))
        except Exception as exc:
            self.services.api.send_message(chat_id=chat_id, text=f"ارسال کد ناموفق بود: {exc}")
            return

        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            phone=phone,
            api_id=api_id,
            api_hash=text,
            session_string=result["session_string"],
            phone_code_hash=result["phone_code_hash"],
            status="awaiting_code",
        )
        self.services.api.send_message(chat_id=chat_id, text="کد تلگرام ارسال شد. همان کد را بفرست.")

    def _handle_phone(self, *, chat_id: int, bot_user_id: int, raw_phone: str, flow: dict[str, object]) -> None:
        normalized = normalize_phone(raw_phone)
        if not normalized:
            self.services.api.send_message(chat_id=chat_id, text="شماره معتبر نیست.")
            return
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=normalized)
        self.services.repository.upsert_auth_flow(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            phone=normalized,
            api_id=None,
            api_hash=None,
            session_string="",
            phone_code_hash="",
            status="awaiting_api_id",
        )
        self.services.api.send_message(chat_id=chat_id, text="شماره ثبت شد. حالا API_ID را بفرست.")

    def _handle_code(self, *, chat_id: int, bot_user_id: int, raw_code: str, flow: dict[str, object]) -> None:
        code = normalize_code(raw_code)
        if not code:
            self.services.api.send_message(chat_id=chat_id, text="کد معتبر نیست.")
            return

        api_id = int(flow["api_id"]) if flow.get("api_id") else None
        api_hash = str(flow["api_hash"]) if flow.get("api_hash") else None
        try:
            result = asyncio.run(
                sign_in_with_code(
                    self.services.settings,
                    phone=str(flow["phone"]),
                    session_string=str(flow["session_string"]),
                    code=code,
                    phone_code_hash=str(flow["phone_code_hash"]),
                    api_id=api_id,
                    api_hash=api_hash,
                )
            )
        except Exception as exc:
            self.services.api.send_message(chat_id=chat_id, text=f"تایید کد ناموفق بود: {exc}")
            return

        if result["status"] == "password_required":
            self.services.repository.upsert_auth_flow(
                bot_user_id=bot_user_id,
                chat_id=chat_id,
                phone=str(flow["phone"]),
                api_id=api_id,
                api_hash=api_hash,
                session_string=result["session_string"],
                phone_code_hash=str(flow["phone_code_hash"]),
                status="awaiting_password",
            )
            self.services.api.send_message(chat_id=chat_id, text="رمز دو مرحله‌ای را بفرست.")
            return

        self.services.repository.save_bot_user_session(
            bot_user_id=bot_user_id,
            phone=str(flow["phone"]),
            api_id=api_id,
            api_hash=api_hash,
            session_string=result["session_string"],
            connected_at=result["connected_at"],
        )
        self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
        self.services.api.send_message(chat_id=chat_id, text="اکانت با موفقیت متصل شد.")

    def _handle_password(self, *, chat_id: int, bot_user_id: int, password: str, flow: dict[str, object]) -> None:
        api_id = int(flow["api_id"]) if flow.get("api_id") else None
        api_hash = str(flow["api_hash"]) if flow.get("api_hash") else None
        try:
            result = asyncio.run(
                sign_in_with_password(
                    self.services.settings,
                    session_string=str(flow["session_string"]),
                    password=password,
                    api_id=api_id,
                    api_hash=api_hash,
                )
            )
        except Exception as exc:
            self.services.api.send_message(chat_id=chat_id, text=f"رمز دو مرحله‌ای قبول نشد: {exc}")
            return

        self.services.repository.save_bot_user_session(
            bot_user_id=bot_user_id,
            phone=str(flow["phone"]),
            api_id=api_id,
            api_hash=api_hash,
            session_string=result["session_string"],
            connected_at=result["connected_at"],
        )
        self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
        self.services.api.send_message(chat_id=chat_id, text="اکانت با موفقیت متصل شد.")

    def _send_status(self, chat_id: int, bot_user_id: int) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id=chat_id, text="اکانت متصل نشده است. /connect")
            return
        phone = str(user.get("phone") or "")
        self.services.api.send_message(chat_id=chat_id, text=f"اکانت متصل است: {phone}")

    def _search(self, chat_id: int, query: str, source_url: str | None = None) -> None:
        if not query:
            self.services.api.send_message(chat_id=chat_id, text="بعد از /search عبارت مورد نظر را بنویس.")
            return
        results = self.services.search_service.search(query=query, channel_url=source_url, top_k=5)
        if not results:
            self.services.api.send_message(chat_id=chat_id, text="نتیجه‌ای یافت نشد.")
            return
        lines = []
        for item in results:
            header = item.channel_title or item.channel_url
            link = f"\n{item.message_url}" if item.message_url else ""
            lines.append(f"- {header}\n{item.chunk_text[:280]}{link}")
        self.services.api.send_message(chat_id=chat_id, text="\n\n".join(lines))

    def _handle_sources(self, chat_id: int) -> None:
        channels = self.services.repository.list_channels()
        if not channels:
            self.services.api.send_message(chat_id=chat_id, text="منبعی ایندکس نشده.")
            return
        lines = ["منابع ایندکس شده:"]
        for c in channels:
            lines.append(f"- {c.get('channel_title') or 'بدون نام'}: {c['channel_url']}")
        self.services.api.send_message(chat_id=chat_id, text="\n".join(lines))

    def _handle_delete(self, chat_id: int, link: str) -> None:
        if not link:
            self.services.api.send_message(chat_id=chat_id, text="لینک منبع برای حذف؟ /delete <link>")
            return
        if self.services.repository.delete_channel_data(channel_url=link):
            self.services.api.send_message(chat_id=chat_id, text="با موفقیت پاک شد.")
        else:
            self.services.api.send_message(chat_id=chat_id, text="منبع یافت نشد.")

    def _handle_join(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id=chat_id, text="ابتدا متصل شوید. /connect")
            return
        from .telegram_client import build_client_from_session_string, join_chat
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))
        async def _do():
            async with client:
                return await join_chat(client, link)
        try:
            res = asyncio.run(_do())
            self.services.api.send_message(chat_id=chat_id, text=res)
        except Exception as e:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا: {e}")

    def _handle_ingest(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id=chat_id, text="ابتدا متصل شوید. /connect")
            return
        self.services.api.send_message(chat_id=chat_id, text="در حال پردازش...")
        from .telegram_client import build_client_from_session_string, iter_all_messages, fetch_channel_info
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))
        async def _do():
            async with client:
                info = await fetch_channel_info(client, link)
                channel_id = self.services.repository.upsert_channel(telegram_id=info.telegram_id, channel_url=info.canonical_url, title=info.title, username=info.username)
                messages = await iter_all_messages(client, channel_url=info.canonical_url, limit=100)
                from .pipeline import IngestionPipeline
                pipeline = IngestionPipeline(settings=self.services.settings, repository=self.services.repository, transcription=self.services.search_service.embeddings, embeddings=self.services.search_service.embeddings)
                count = 0
                for msg in messages:
                    db_msg_id = self.services.repository.create_or_get_message(channel_id=channel_id, telegram_message_id=msg.telegram_message_id, message_date=msg.message_date, message_url=msg.message_url, caption=msg.caption)
                    if msg.media_kind == "text" and msg.caption:
                        await pipeline._process_text_message(db_msg_id, msg.caption)
                        count += 1
                return f"تعداد {count} پیام ایندکس شد."
        try:
            res = asyncio.run(_do())
            self.services.api.send_message(chat_id=chat_id, text=res)
        except Exception as e:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا: {e}")


def main() -> None:
    bot = NotebookBot(build_services())
    bot.run_forever()


if __name__ == "__main__":
    main()
