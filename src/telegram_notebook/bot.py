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
            query = text.removeprefix("/search").strip()
            self._search(chat_id, query)
            return

        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow and flow["status"] == "awaiting_api_id":
            self._handle_api_id(chat_id=chat_id, bot_user_id=bot_user_id, text=text, flow=flow)
            return
        if flow and flow["status"] == "awaiting_api_hash":
            self._handle_api_hash(chat_id=chat_id, bot_user_id=bot_user_id, text=text, flow=flow)
            return
        if flow and flow["status"] == "awaiting_phone":
            self._handle_phone(chat_id=chat_id, bot_user_id=bot_user_id, raw_phone=text, flow=flow)
            return
        if flow and flow["status"] == "awaiting_code":
            self._handle_code(chat_id=chat_id, bot_user_id=bot_user_id, raw_code=text, flow=flow)
            return
        if flow and flow["status"] == "awaiting_password":
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
                "سلام. این ربات برای ساخت session اکانت واقعی تلگرام و بعدا مدیریت جست‌وجو استفاده می‌شود.\n\n"
                "دستورها:\n"
                "/connect برای اتصال اکانت واقعی\n"
                "/status برای دیدن وضعیت اتصال\n"
                "/search <query> برای جست‌وجو در archive\n"
                "/cancel برای لغو فرآیند فعلی"
            ),
        )
        self._begin_connect(chat_id, bot_user_id)

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
            text="لطفاً ابتدا شماره موبایل خود را با دکمه زیر به اشتراک بگذارید تا فرآیند اتصال شروع شود.",
            reply_markup=TelegramBotApi.contact_keyboard(),
        )

    def _handle_contact(self, *, chat_id: int, bot_user_id: int, contact: dict[str, object]) -> None:
        raw_phone = str(contact.get("phone_number", ""))
        normalized = normalize_phone(raw_phone)
        if not normalized:
            self.services.api.send_message(chat_id=chat_id, text="شماره معتبر نبود.")
            return
        
        # ذخیره شماره تلفن در مشخصات کاربر (قبل از هر چیز)
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=normalized)
        
        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if not flow or flow["status"] != "awaiting_phone_initial":
            return

        # تغییر وضعیت به مرحله بعد: دریافت API_ID
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
            text="شماره شما ثبت شد. حالا TELEGRAM_API_ID خود را بفرست (یک عدد چند رقمی).",
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
            self.services.api.send_message(chat_id=chat_id, text=f"ارسال کد ناموفق بود: {exc}\nمجدداً امتحان کنید یا فرآیند را لغو کنید.")
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
        self.services.api.send_message(
            chat_id=chat_id,
            text="کد تلگرام ارسال شد. همان کد را همینجا بفرست.",
        )

    def _handle_phone(self, *, chat_id: int, bot_user_id: int, raw_phone: str, flow: dict[str, object]) -> None:
        # در جریان جدید، شماره تلفن از طریق Contact Button گرفته شده است. 
        # اما برای احتیاط، اگر کاربر دستی وارد کرد (اگر اجازه بدیم):
        normalized = normalize_phone(raw_phone)
        if not normalized:
            self.services.api.send_message(chat_id=chat_id, text="شماره معتبر نیست. مثل +98912... بفرست.")
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

    def _handle_code(
        self,
        *,
        chat_id: int,
        bot_user_id: int,
        raw_code: str,
        flow: dict[str, object],
    ) -> None:
        code = normalize_code(raw_code)
        if not code:
            self.services.api.send_message(chat_id=chat_id, text="کد معتبر نیست. فقط رقم‌ها را بفرست.")
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
            self.services.api.send_message(chat_id=chat_id, text="رمز دو مرحله‌ای اکانت را بفرست.")
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

    def _handle_password(
        self,
        *,
        chat_id: int,
        bot_user_id: int,
        password: str,
        flow: dict[str, object],
    ) -> None:
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
        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow:
            self.services.api.send_message(
                chat_id=chat_id,
                text=f"وضعیت فعلی اتصال: {flow['status']}",
            )
            return
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id=chat_id, text="هنوز اکانت واقعی متصل نشده. /connect را بزن.")
            return
        phone = str(user.get("phone") or "")
        masked = f"{phone[:4]}***{phone[-2:]}" if len(phone) >= 6 else phone
        self.services.api.send_message(
            chat_id=chat_id,
            text=f"اکانت متصل است.\nPhone: {masked}\nConnected at: {user.get('connected_at')}",
        )

    def _search(self, chat_id: int, query: str) -> None:
        if not query:
            self.services.api.send_message(chat_id=chat_id, text="بعد از /search یک query بنویس.")
            return
        results = self.services.search_service.search(query=query, channel_url=None, top_k=5)
        if not results:
            self.services.api.send_message(chat_id=chat_id, text="نتیجه‌ای پیدا نشد.")
            return
        lines = []
        for item in results:
            header = item.channel_title or item.channel_url
            link = f"\n{item.message_url}" if item.message_url else ""
            lines.append(f"- {header}\n{item.chunk_text[:280]}{link}")
        self.services.api.send_message(chat_id=chat_id, text="\n\n".join(lines))

    def _handle_join(self, chat_id: int, bot_user_id: int, link: str) -> None:
        if not link:
            self.services.api.send_message(chat_id=chat_id, text="لینک گروه را بفرست: /join https://t.me/...")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id=chat_id, text="ابتدا با /connect متصل شو.")
            return

        from .telegram_client import build_client_from_session_string, join_chat
        client = build_client_from_session_string(
            self.services.settings,
            user["session_string"],
            api_id=user.get("api_id"),
            api_hash=user.get("api_hash")
        )

        async def _do_join():
            async with client:
                return await join_chat(client, link)

        try:
            msg = asyncio.run(_do_join())
            self.services.api.send_message(chat_id=chat_id, text=msg)
        except Exception as exc:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا در عضویت: {exc}")

    def _handle_ingest(self, chat_id: int, bot_user_id: int, link: str) -> None:
        if not link:
            self.services.api.send_message(chat_id=chat_id, text="لینک گروه را بفرست: /ingest https://t.me/...")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id=chat_id, text="ابتدا با /connect متصل شو.")
            return

        self.services.api.send_message(chat_id=chat_id, text="در حال دریافت و ایندکس کردن پیام‌ها... (کمی زمان می‌برد)")
        
        from .telegram_client import build_client_from_session_string, iter_all_messages, fetch_channel_info
        client = build_client_from_session_string(
            self.services.settings,
            user["session_string"],
            api_id=user.get("api_id"),
            api_hash=user.get("api_hash")
        )

        async def _do_ingest():
            async with client:
                info = await fetch_channel_info(client, link)
                channel_id = self.services.repository.upsert_channel(
                    telegram_id=info.telegram_id,
                    channel_url=info.canonical_url,
                    title=info.title,
                    username=info.username
                )
                messages = await iter_all_messages(client, channel_url=info.canonical_url, limit=100)
                
                # ارسال به پایپ‌لاین پردازش
                from .pipeline import IngestionPipeline
                pipeline = IngestionPipeline(
                    settings=self.services.settings,
                    repository=self.services.repository,
                    transcription=None, # اینجا برای متن نیازی نیست
                    embeddings=self.services.search_service.embeddings
                )
                
                count = 0
                for msg in messages:
                    # ذخیره پیام در دیتابیس
                    db_msg_id = self.services.repository.create_or_get_message(
                        channel_id=channel_id,
                        telegram_message_id=msg.telegram_message_id,
                        message_date=msg.message_date,
                        message_url=msg.message_url,
                        caption=msg.caption
                    )
                    
                    # اگر فقط متن بود، مستقیماً چانک‌بندی و ذخیره کن
                    if msg.media_kind == "text" and msg.caption:
                        await pipeline._process_text_message(db_msg_id, msg.caption)
                        count += 1
                    elif msg.media_kind in ("audio", "video", "voice"):
                        # مدیاها را بعداً پردازش کن یا همینجا استارت بزن
                        pass
                
                return f"تعداد {count} پیام متنی ایندکس شد. پردازش مدیاها در پس‌زمینه ادامه دارد."

        try:
            msg = asyncio.run(_do_ingest())
            self.services.api.send_message(chat_id=chat_id, text=msg)
        except Exception as exc:
            self.services.api.send_message(chat_id=chat_id, text=f"خطا در پردازش: {exc}")


def main() -> None:
    bot = NotebookBot(build_services())
    bot.run_forever()


if __name__ == "__main__":
    main()
