from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .bot_api import TelegramBotApi
from .config import Settings, get_settings
from .db import Repository, connect
from .embeddings import EmbeddingService
from .logging_config import setup_logging
from .pipeline import IngestionPipeline
from .rules import match_tags
from .search import SearchService
from .telegram_client import request_login_code, sign_in_with_code, sign_in_with_password
from .transcription import TranscriptionService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BotServices:
    api: TelegramBotApi
    repository: Repository
    search_service: SearchService
    pipeline: IngestionPipeline
    settings: Settings


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
    pipeline = IngestionPipeline(
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
        pipeline=pipeline,
        settings=settings,
    )


def normalize_phone(raw: str) -> str | None:
    table = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    text = raw.translate(table)
    compact = "".join(c for c in text if c.isdigit())
    if not (10 <= len(compact) <= 15):
        return None
    return f"+{compact}"


def normalize_code(raw: str) -> str | None:
    if not raw:
        return None
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"

    text = raw
    for i in range(10):
        text = text.replace(persian_digits[i], english_digits[i])
        text = text.replace(arabic_digits[i], english_digits[i])

    compact = "".join(c for c in text if c.isdigit())

    if 3 <= len(compact) <= 12:
        return compact
    return None


class NotebookBot:
    def __init__(self, services: BotServices) -> None:
        self.services = services
        self.offset: int | None = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run_forever(self) -> None:
        logger.info("Bot polling started")
        while True:
            try:
                updates = self.services.api.get_updates(offset=self.offset, timeout=30)
                for update in updates:
                    self.offset = int(update["update_id"]) + 1
                    try:
                        self.handle_update(update)
                    except Exception:
                        logger.exception("Failed to handle update %s", update.get("update_id"))
            except KeyboardInterrupt:
                logger.info("Bot stopped by KeyboardInterrupt")
                break
            except Exception:
                logger.exception("Bot polling error")
                time.sleep(3)

    def _async_to_sync(self, coro, timeout: int = 60):
        try:
            return self.loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
        except asyncio.TimeoutError:
            raise RuntimeError("Operation timed out. Please check your connection or try again.")

    def _api_key_for_user(self, user: dict | None, provider: str) -> str | None:
        if provider == "gemini":
            return (user.get("gemini_api_key") if user else None) or self.services.settings.gemini_api_key
        if provider == "openai":
            return self.services.settings.openai_api_key
        return None

    def _embedding_service_for_user(self, user: dict | None) -> EmbeddingService:
        provider = self.services.settings.embedding_provider
        model = (
            user.get("preferred_embedding_model")
            if user and user.get("preferred_embedding_model")
            else self.services.settings.embedding_model
        )
        return EmbeddingService(
            provider=provider,
            api_key=self._api_key_for_user(user, provider),
            model=model,
        )

    def _transcription_service_for_user(self, user: dict | None) -> TranscriptionService:
        provider = self.services.settings.transcription_provider
        model = (
            user.get("preferred_transcription_model")
            if user and user.get("preferred_transcription_model")
            else self.services.settings.transcription_model
        )
        return TranscriptionService(
            provider=provider,
            api_key=self._api_key_for_user(user, provider),
            model=model,
        )

    def _search_service_for_user(self, user: dict | None) -> SearchService:
        return SearchService(self.services.repository, self._embedding_service_for_user(user))

    def handle_update(self, update: dict[str, object]) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            self.services.api.answer_callback_query(callback["id"])
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat_id = int(message["chat"]["id"])
        sender = message.get("from") or {}
        bot_user_id = int(sender.get("id", 0))
        if not bot_user_id:
            return

        self.services.repository.upsert_bot_user(
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            username=sender.get("username"),
            first_name=sender.get("first_name"),
        )

        if "contact" in message:
            self._handle_contact(chat_id, bot_user_id, message["contact"])
            return

        text = str(message.get("text", "")).strip()

        if text.startswith("/"):
            self._handle_command(chat_id, bot_user_id, text)
            return

        if self._is_forwarded(message):
            self._handle_forwarded(chat_id, bot_user_id, message)
            return

        if not text:
            return

        flow = self.services.repository.get_auth_flow(bot_user_id=bot_user_id)
        if flow:
            st = flow["status"]
            if st == "awaiting_gemini_key":
                self._handle_gemini_key(chat_id, bot_user_id, text)
            elif st == "awaiting_v_project":
                self._handle_v_project(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_region":
                self._handle_v_region(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_index":
                self._handle_v_index(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_endpoint":
                self._handle_v_endpoint(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_v_deployed":
                self._handle_v_deployed(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_api_id":
                self._handle_api_id(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_api_hash":
                self._handle_api_hash(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_login_phone":
                self._handle_login_phone(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_code":
                self._handle_code(chat_id, bot_user_id, text, flow)
            elif st == "awaiting_password":
                self._handle_password(chat_id, bot_user_id, text, flow)

    def _handle_command(self, chat_id: int, bot_user_id: int, text: str) -> None:
        command = text.split()[0].split("@")[0].lower()
        if command == "/start":
            self._send_welcome(chat_id)
        elif command == "/help":
            self._send_help(chat_id)
        elif command == "/version":
            self.services.api.send_message(chat_id=chat_id, text="Bot Version: v5.0 (Stabilized Core)")
        elif command == "/connect":
            self._begin_connect(chat_id, bot_user_id)
        elif command == "/status":
            self._handle_status(chat_id, bot_user_id)
        elif command == "/disconnect":
            self._handle_disconnect(chat_id, bot_user_id)
        elif command == "/cancel":
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Operation cancelled.", reply_markup=TelegramBotApi.remove_keyboard())
        elif command == "/search":
            query, source, tag = self._split_filters(text.removeprefix("/search").strip())
            self._search(chat_id, bot_user_id, query, source, tag)
        elif command == "/ask":
            query, source, tag = self._split_filters(text.removeprefix("/ask").strip())
            self._ask_brain(chat_id, bot_user_id, query, source, tag)
        elif command == "/rule":
            self._handle_rule(chat_id, bot_user_id, text.removeprefix("/rule").strip())
        elif command == "/tags":
            self._handle_tags(chat_id, bot_user_id)
        elif command == "/join":
            self._handle_join(chat_id, bot_user_id, text.removeprefix("/join").strip())
        elif command == "/ingest":
            self._handle_ingest(chat_id, bot_user_id, text.removeprefix("/ingest").strip())
        elif command == "/sources":
            self._handle_sources(chat_id, bot_user_id)
        elif command == "/delete":
            self._handle_delete(chat_id, bot_user_id, text.removeprefix("/delete").strip())
        else:
            self.services.api.send_message(chat_id=chat_id, text="Unknown command. Send /help to see what I can do.")

    @staticmethod
    def _split_filters(text: str) -> tuple[str, str | None, str | None]:
        """Parse a query with optional ``--source <url>`` (single token) and
        ``--tag <tag>`` (rest of line, may contain spaces) filters."""
        tokens = text.split()
        source = None
        if "--source" in tokens:
            idx = tokens.index("--source")
            if idx + 1 < len(tokens):
                source = tokens[idx + 1]
                del tokens[idx:idx + 2]
        remaining = " ".join(tokens)
        tag = None
        head, sep, tail = remaining.partition("--tag")
        if sep:
            tag = tail.strip() or None
            remaining = head
        return remaining.strip(), source, tag

    def _send_welcome(self, chat_id: int) -> None:
        self.services.api.send_message(
            chat_id=chat_id,
            text=(
                "Welcome! I'm your AI Research Assistant.\n"
                "Use /connect to link your account and configure Vertex AI.\n"
                "Forward me any message to save it to your searchable inbox.\n"
                "Send /help to see all commands."
            ),
        )

    def _send_help(self, chat_id: int) -> None:
        help_text = (
            "<b>Available commands</b>\n"
            "/connect — link your Telegram account and configure AI\n"
            "/status — show your connection and indexing status\n"
            "/disconnect — remove your saved session and credentials\n"
            "/ingest &lt;channel_url&gt; — index a channel or source\n"
            "/search &lt;query&gt; [--source &lt;url&gt;] [--tag &lt;tag&gt;] — keyword/semantic search\n"
            "/ask &lt;question&gt; [--source &lt;url&gt;] [--tag &lt;tag&gt;] — ask the AI over your archive\n"
            "/sources — list indexed sources\n"
            "/delete &lt;channel_url&gt; — delete a source's data\n"
            "/cancel — cancel the current flow\n\n"
            "<b>Rules &amp; tags</b>\n"
            "/rule add &lt;keyword&gt; -&gt; &lt;tag&gt; — auto-tag matching content\n"
            "/rule list — show your rules\n"
            "/rule remove &lt;id&gt; — delete a rule\n"
            "/rule apply — re-tag existing content with current rules\n"
            "/tags — list your tags and their counts\n\n"
            "<b>Forwarded Inbox</b>\n"
            "Forward any message to me and I'll save its text/caption to your "
            "searchable inbox. Then use /search or /ask over it."
        )
        self.services.api.send_message(chat_id=chat_id, text=help_text)

    def _handle_status(self, chat_id: int, bot_user_id: int) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        connected = bool(user and user.get("session_string"))
        gemini_set = bool((user and user.get("gemini_api_key")) or self.services.settings.gemini_api_key)
        vertex_ready = bool(
            (user.get("vertex_project_id") if user else None) or self.services.settings.vertex_project_id
        )
        sources_count = len(self.services.repository.list_channels(owner_id=bot_user_id))
        lines = [
            "<b>Status</b>",
            f"• Account linked: {'✅ yes' if connected else '❌ no (use /connect)'}",
            f"• AI key configured: {'✅ yes' if gemini_set else '❌ no'}",
            f"• Vertex AI Search: {'✅ configured' if vertex_ready else 'local search only'}",
            f"• Indexed sources: {sources_count}",
        ]
        if connected and user.get("phone"):
            lines.append(f"• Linked phone: {user['phone']}")
        self.services.api.send_message(chat_id=chat_id, text="\n".join(lines))

    def _handle_disconnect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
        removed = self.services.repository.disconnect_bot_user(bot_user_id=bot_user_id)
        if removed:
            self.services.api.send_message(
                chat_id=chat_id,
                text="Disconnected. Your saved session and credentials were removed. Use /connect to link again.",
                reply_markup=TelegramBotApi.remove_keyboard(),
            )
        else:
            self.services.api.send_message(chat_id=chat_id, text="You have no linked account to disconnect.")

    def _begin_connect(self, chat_id: int, bot_user_id: int) -> None:
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone="", api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_phone_initial")
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 1: Your Profile</b>\nPlease share your phone number to register in the system:", reply_markup=TelegramBotApi.contact_keyboard())

    def _handle_contact(self, chat_id: int, bot_user_id: int, contact: dict) -> None:
        phone = normalize_phone(str(contact.get("phone_number", "")))
        self.services.repository.save_bot_user_phone(bot_user_id=bot_user_id, phone=phone)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_gemini_key")

        guide_text = (
            "<b>Step 2: Vertex AI API Key</b>\n"
            "Please provide your Vertex AI API Key (starts with AQ.) or AI Studio Key (starts with AIza).\n"
            "You can get it from <a href='https://aistudio.google.com/app/apikey'>Google AI Studio</a>.\n"
            "Then send it here:"
        )

        photo_path = Path("data/media/gemini_guide.jpg")
        if photo_path.exists():
            self.services.api.send_photo(chat_id=chat_id, photo_path=photo_path, caption=guide_text)
        else:
            self.services.api.send_message(chat_id=chat_id, text=guide_text, reply_markup=TelegramBotApi.remove_keyboard())

    def _handle_gemini_key(self, chat_id: int, bot_user_id: int, text: str) -> None:
        if not (text.startswith("AIza") or text.startswith("AQ.")):
            self.services.api.send_message(chat_id=chat_id, text="Invalid Key format. It should start with 'AIza' or 'AQ.'.")
            return
        self.services.repository.update_user_gemini_key(bot_user_id=bot_user_id, api_key=text)
        self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_v_project")
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 3: Vertex AI Search Config</b>\nPlease enter your Google Cloud <b>Project ID</b>:")

    def _handle_v_project(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_region", v_project=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Region</b> (e.g., us-central1):")

    def _handle_v_region(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_index", v_project=flow["vertex_project_id"], v_region=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Index ID</b>:")

    def _handle_v_index(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_endpoint", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Index Endpoint ID</b>:")

    def _handle_v_endpoint(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_v_deployed", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=text)
        self.services.api.send_message(chat_id=chat_id, text="Enter your <b>Deployed Index ID</b>:")

    def _handle_v_deployed(self, chat_id, bot_user_id, text, flow):
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=None, api_hash=None, session_string="", phone_code_hash="", status="awaiting_api_id", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=text)
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 4: Telegram API</b>\nGo to <a href='https://my.telegram.org'>my.telegram.org</a> and create an app.\nSend your <b>API_ID</b>:")

    def _handle_api_id(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        if not text.isdigit():
            return
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=int(text), api_hash=None, session_string="", phone_code_hash="", status="awaiting_api_hash", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
        self.services.api.send_message(chat_id=chat_id, text="Send your <b>API_HASH</b>:")

    def _handle_api_hash(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=text, session_string="", phone_code_hash="", status="awaiting_login_phone", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
        self.services.api.send_message(chat_id=chat_id, text="<b>Step 5: Phone Number to Link</b>\nSend the phone number of the account you want to link (+98...):")

    def _handle_login_phone(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        phone = normalize_phone(text)
        if not phone:
            self.services.api.send_message(chat_id=chat_id, text="Invalid phone number.")
            return
        self.services.api.send_message(chat_id=chat_id, text="Requesting code from Telegram... (Please wait up to 1 minute)")
        try:
            res = self._async_to_sync(request_login_code(self.services.settings, phone, api_id=flow["api_id"], api_hash=flow["api_hash"]), timeout=120)
            self.services.repository.upsert_auth_flow(bot_user_id=bot_user_id, chat_id=chat_id, phone=phone, api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], phone_code_hash=res["phone_code_hash"], status="awaiting_code", v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
            self.services.api.send_message(chat_id=chat_id, text="Code sent to your Telegram. Enter it here:")
        except Exception as e:
            logger.exception("Auth error while requesting login code")
            self.services.api.send_message(chat_id=chat_id, text=f"Error: {e}")

    def _handle_code(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        code = normalize_code(text)
        if not code:
            self.services.api.send_message(chat_id=chat_id, text="Invalid code format. Please try again.")
            return
        self.services.api.send_message(chat_id=chat_id, text="Verifying code with Telegram...")
        try:
            res = self._async_to_sync(sign_in_with_code(self.services.settings, phone=flow["phone"], session_string=flow["session_string"], code=code, phone_code_hash=flow["phone_code_hash"], api_id=flow["api_id"], api_hash=flow["api_hash"]), timeout=60)
            if res["status"] == "password_required":
                self.services.repository.update_auth_flow_status(bot_user_id=bot_user_id, status="awaiting_password")
                self.services.api.send_message(chat_id=chat_id, text="Two-step verification enabled. Enter your password:")
                return
            self.services.repository.save_bot_user_session(bot_user_id=bot_user_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], connected_at=res["connected_at"], v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Connected! Everything is ready.")
        except Exception as e:
            logger.exception("Code verification error during sign in")
            self.services.api.send_message(chat_id=chat_id, text=f"Error during sign in: {e}")

    def _handle_password(self, chat_id: int, bot_user_id: int, text: str, flow: dict) -> None:
        try:
            res = self._async_to_sync(sign_in_with_password(self.services.settings, session_string=flow["session_string"], password=text, api_id=flow["api_id"], api_hash=flow["api_hash"]))
            self.services.repository.save_bot_user_session(bot_user_id=bot_user_id, phone=flow["phone"], api_id=flow["api_id"], api_hash=flow["api_hash"], session_string=res["session_string"], connected_at=res["connected_at"], v_project=flow["vertex_project_id"], v_region=flow["vertex_region"], v_index=flow["vertex_index_id"], v_endpoint=flow["vertex_endpoint_id"], v_deployed=flow["vertex_deployed_index_id"])
            self.services.repository.clear_auth_flow(bot_user_id=bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Connected!")
        except Exception as e:
            logger.exception("Password sign-in error")
            self.services.api.send_message(chat_id=chat_id, text=f"Error: {e}")

    def _vertex_search_config(self, user: dict | None) -> dict | None:
        """Build a Vertex AI Search config from per-user data, falling back to settings."""
        s = self.services.settings
        v_project = (user.get("vertex_project_id") if user else None) or s.vertex_project_id
        v_region = (user.get("vertex_region") if user else None) or s.vertex_region
        v_endpoint = (user.get("vertex_endpoint_id") if user else None) or s.vertex_endpoint_id
        v_deployed = (user.get("vertex_deployed_index_id") if user else None) or s.vertex_deployed_index_id
        if v_project and v_region and v_endpoint and v_deployed:
            return {
                "api_key": (user.get("gemini_api_key") if user else None) or s.gemini_api_key,
                "project_id": v_project,
                "region": v_region,
                "index_endpoint_id": v_endpoint,
                "deployed_index_id": v_deployed,
            }
        return None

    def _vertex_ingest_config(self, user: dict | None) -> dict | None:
        """Build a Vertex AI upsert config (used while indexing) from per-user data."""
        s = self.services.settings
        v_project = (user.get("vertex_project_id") if user else None) or s.vertex_project_id
        v_region = (user.get("vertex_region") if user else None) or s.vertex_region
        v_index_id = (user.get("vertex_index_id") if user else None) or s.vertex_index_id
        if v_project and v_region and v_index_id:
            return {
                "api_key": (user.get("gemini_api_key") if user else None) or s.gemini_api_key,
                "project_id": v_project,
                "region": v_region,
                "index_id": v_index_id,
            }
        return None

    @staticmethod
    def _is_forwarded(message: dict) -> bool:
        return any(
            key in message
            for key in ("forward_origin", "forward_from", "forward_from_chat", "forward_sender_name", "forward_date")
        )

    @staticmethod
    def _forward_source_label(message: dict) -> str:
        origin = message.get("forward_origin")
        if isinstance(origin, dict):
            otype = origin.get("type")
            if otype == "user":
                u = origin.get("sender_user") or {}
                name = " ".join(p for p in (u.get("first_name"), u.get("last_name")) if p)
                return name or u.get("username") or "Unknown user"
            if otype == "hidden_user":
                return origin.get("sender_user_name") or "Hidden user"
            if otype == "chat":
                c = origin.get("sender_chat") or {}
                return c.get("title") or c.get("username") or "Unknown chat"
            if otype == "channel":
                c = origin.get("chat") or {}
                return c.get("title") or c.get("username") or "Unknown channel"
        ffc = message.get("forward_from_chat")
        if isinstance(ffc, dict):
            return ffc.get("title") or ffc.get("username") or "Unknown chat"
        ff = message.get("forward_from")
        if isinstance(ff, dict):
            name = " ".join(p for p in (ff.get("first_name"), ff.get("last_name")) if p)
            return name or ff.get("username") or "Unknown user"
        if message.get("forward_sender_name"):
            return str(message["forward_sender_name"])
        return "Forwarded"

    @staticmethod
    def _forward_message_url(message: dict) -> str | None:
        """Public t.me link when the forward came from a public channel, else None."""
        origin = message.get("forward_origin")
        if isinstance(origin, dict) and origin.get("type") == "channel":
            chat = origin.get("chat") or {}
            if chat.get("username") and origin.get("message_id"):
                return f"https://t.me/{chat['username']}/{origin['message_id']}"
        ffc = message.get("forward_from_chat")
        if isinstance(ffc, dict) and ffc.get("username") and message.get("forward_from_message_id"):
            return f"https://t.me/{ffc['username']}/{message['forward_from_message_id']}"
        return None

    @staticmethod
    def _forward_media_tag(message: dict) -> str | None:
        if "photo" in message:
            return "[Forwarded photo]"
        if "document" in message:
            name = (message.get("document") or {}).get("file_name")
            return f"[Forwarded document: {name}]" if name else "[Forwarded document]"
        if "video" in message:
            return "[Forwarded video]"
        if "audio" in message:
            audio = message.get("audio") or {}
            title = audio.get("title") or audio.get("file_name")
            return f"[Forwarded audio: {title}]" if title else "[Forwarded audio]"
        if "voice" in message:
            return "[Forwarded voice message]"
        return None

    def _handle_forwarded(self, chat_id: int, bot_user_id: int, message: dict) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        label = self._forward_source_label(message)
        raw_text = str(message.get("text") or message.get("caption") or "").strip()
        media_tag = self._forward_media_tag(message)
        body = "\n".join(part for part in (media_tag, raw_text) if part).strip()
        if not body:
            self.services.api.send_message(
                chat_id,
                "I couldn't extract any text from that forward yet. Media-only items "
                "(photos or files without a caption) aren't indexed in this version.",
            )
            return

        forward_key = int(message["message_id"])
        message_url = self._forward_message_url(message)
        vertex_config = self._vertex_ingest_config(user)

        async def _do():
            pipeline = IngestionPipeline(
                settings=self.services.settings,
                repository=self.services.repository,
                transcription=self._transcription_service_for_user(user),
                embeddings=self._embedding_service_for_user(user),
            )
            stored = await pipeline.ingest_forwarded_message(
                owner_id=bot_user_id,
                source_label=label,
                text=body,
                forward_key=forward_key,
                message_url=message_url,
                vertex_config=vertex_config,
            )
            if stored:
                return f"Saved to your inbox from “{label}”. Use /search or /ask to query it."
            return "You've already saved this forward."

        try:
            self.services.api.send_message(chat_id, self._async_to_sync(_do()))
        except Exception as e:
            logger.exception("Forward ingest failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error saving forward: {e}")

    def _search(self, chat_id: int, bot_user_id: int, query: str, source: str | None, tag: str | None = None) -> None:
        if not query:
            self.services.api.send_message(chat_id=chat_id, text="Usage: /search <query> [--source <url>] [--tag <tag>]")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        vertex_config = self._vertex_search_config(user)
        try:
            results = self._search_service_for_user(user).search(
                owner_id=bot_user_id, query=query, channel_url=source, tag=tag, top_k=5, vertex_config=vertex_config
            )
        except Exception:
            logger.exception("Search failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Search failed. Please try again later.")
            return
        if not results:
            self.services.api.send_message(chat_id=chat_id, text="No results found.")
        else:
            resp = [f"{r.channel_title}\n{r.chunk_text[:300]}\n{r.message_url}" for r in results]
            self.services.api.send_message(chat_id=chat_id, text="\n\n".join(resp))

    def _ask_brain(self, chat_id: int, bot_user_id: int, query: str, source: str | None, tag: str | None = None) -> None:
        if not query:
            self.services.api.send_message(chat_id=chat_id, text="Usage: /ask <question> [--source <url>] [--tag <tag>]")
            return
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        gemini_api_key = (user.get("gemini_api_key") if user else None) or self.services.settings.gemini_api_key
        vertex_config = self._vertex_search_config(user)
        v_project = vertex_config["project_id"] if vertex_config else self.services.settings.vertex_project_id
        v_region = vertex_config["region"] if vertex_config else self.services.settings.vertex_region

        self.services.api.send_message(chat_id=chat_id, text="AI Brain is thinking...")
        try:
            search_service = self._search_service_for_user(user)
            results = search_service.search(owner_id=bot_user_id, query=query, channel_url=source, tag=tag, top_k=5, vertex_config=vertex_config)
            answer = search_service.generate_answer(
                query=query,
                results=results,
                api_key=gemini_api_key,
                project_id=v_project,
                region=v_region,
            )
        except Exception:
            logger.exception("Ask failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id=chat_id, text="Sorry, I couldn't generate an answer right now. Please try again later.")
            return

        self.services.api.send_message(chat_id=chat_id, text=f"<b>AI Answer:</b>\n\n{answer}")
        if results:
            sources_text = "\n".join([f"- {r.channel_title or r.channel_url} ({r.message_url})" for r in results[:3]])
            self.services.api.send_message(chat_id=chat_id, text=f"<b>Sources:</b>\n{sources_text}", disable_web_page_preview=True)

    def _handle_sources(self, chat_id: int, bot_user_id: int) -> None:
        ch = self.services.repository.list_channels(owner_id=bot_user_id)
        if not ch:
            self.services.api.send_message(chat_id, "No sources indexed.")
        else:
            self.services.api.send_message(chat_id, "\n".join([f"{c.get('channel_title') or 'Unknown'}: {c['channel_url']}" for c in ch]))

    def _handle_delete(self, chat_id: int, bot_user_id: int, link: str) -> None:
        if not link:
            self.services.api.send_message(chat_id, "Usage: /delete <channel_url>")
            return
        if self.services.repository.delete_channel_data(owner_id=bot_user_id, channel_url=link):
            self.services.api.send_message(chat_id, "Deleted.")
        else:
            self.services.api.send_message(chat_id, "Not found.")

    @staticmethod
    def _parse_rule_add(rest: str) -> tuple[str, str] | None:
        if "->" not in rest:
            return None
        keyword, _, tag = rest.partition("->")
        keyword, tag = keyword.strip(), tag.strip()
        if not keyword or not tag:
            return None
        return keyword, tag

    def _handle_rule(self, chat_id: int, bot_user_id: int, args: str) -> None:
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "add":
            parsed = self._parse_rule_add(rest)
            if not parsed:
                self.services.api.send_message(chat_id, "Usage: /rule add <keyword> -> <tag>")
                return
            keyword, tag = parsed
            self.services.repository.add_rule(
                owner_id=bot_user_id, keyword=keyword, tag=tag,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self.services.api.send_message(chat_id, f"Rule added: “{keyword}” → {tag}\nRun /rule apply to tag existing items.")
        elif sub in ("", "list"):
            rules = self.services.repository.list_rules(owner_id=bot_user_id)
            if not rules:
                self.services.api.send_message(chat_id, "No rules yet. Add one with: /rule add <keyword> -> <tag>")
                return
            lines = ["<b>Your rules</b>"] + [f"#{r['id']}: “{r['keyword']}” → {r['tag']}" for r in rules]
            self.services.api.send_message(chat_id, "\n".join(lines))
        elif sub == "remove":
            try:
                rule_id = int(rest)
            except ValueError:
                self.services.api.send_message(chat_id, "Usage: /rule remove <rule_id>")
                return
            if self.services.repository.remove_rule(owner_id=bot_user_id, rule_id=rule_id):
                self.services.api.send_message(chat_id, f"Rule #{rule_id} removed. Run /rule apply to refresh tags.")
            else:
                self.services.api.send_message(chat_id, "Rule not found.")
        elif sub == "apply":
            self._retag_all(chat_id, bot_user_id)
        else:
            self.services.api.send_message(chat_id, "Usage: /rule add|list|remove|apply")

    def _retag_all(self, chat_id: int, bot_user_id: int) -> None:
        """Recompute tags for all of the user's stored content from the current rules."""
        rules = self.services.repository.list_rules(owner_id=bot_user_id)
        self.services.repository.clear_tags(owner_id=bot_user_id)
        tagged = 0
        for item in self.services.repository.media_texts(owner_id=bot_user_id):
            tags = match_tags(item.get("text"), rules)
            if tags:
                self.services.repository.tag_media(owner_id=bot_user_id, media_item_id=item["media_item_id"], tags=tags)
                tagged += 1
        self.services.api.send_message(chat_id, f"Re-tagged {tagged} item(s) using {len(rules)} rule(s).")

    def _handle_tags(self, chat_id: int, bot_user_id: int) -> None:
        tags = self.services.repository.list_tags(owner_id=bot_user_id)
        if not tags:
            self.services.api.send_message(chat_id, "No tags yet. Define rules with /rule add, then ingest or run /rule apply.")
            return
        lines = ["<b>Your tags</b>"] + [f"{t['tag']}: {t['count']}" for t in tags]
        self.services.api.send_message(chat_id, "\n".join(lines))

    def _handle_join(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user.get("session_string"):
            self.services.api.send_message(chat_id, "Please /connect first.")
            return
        if not link:
            self.services.api.send_message(chat_id, "Usage: /join <channel_or_invite_link>")
            return
        from .telegram_client import build_client_from_session_string, join_chat
        client = build_client_from_session_string(self.services.settings, user["session_string"], api_id=user.get("api_id"), api_hash=user.get("api_hash"))

        async def _do():
            async with client:
                return await join_chat(client, link)

        try:
            self.services.api.send_message(chat_id, self._async_to_sync(_do()))
        except Exception as e:
            logger.exception("Join failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")

    def _handle_ingest(self, chat_id: int, bot_user_id: int, link: str) -> None:
        user = self.services.repository.get_bot_user(bot_user_id=bot_user_id)
        if not user or not user["session_string"]:
            self.services.api.send_message(chat_id, "Please /connect first.")
            return
        if not link:
            self.services.api.send_message(chat_id, "Usage: /ingest <channel_url>")
            return

        self.services.api.send_message(chat_id, "Ingesting messages...")

        vertex_config = self._vertex_ingest_config(user)

        async def _do():
            pipeline = IngestionPipeline(
                settings=self.services.settings,
                repository=self.services.repository,
                transcription=self._transcription_service_for_user(user),
                embeddings=self._embedding_service_for_user(user),
            )
            stats = await pipeline.ingest_channel(
                owner_id=bot_user_id,
                channel_url=link,
                limit=100,
                api_id=user.get("api_id"),
                api_hash=user.get("api_hash"),
                session_string=user["session_string"],
                vertex_config=vertex_config,
            )
            return f"Indexed {stats.processed_media} items from {stats.channel_title or stats.channel_url}."

        try:
            self.services.api.send_message(chat_id, self._async_to_sync(_do()))
        except Exception as e:
            logger.exception("Ingest failed for user %s", bot_user_id)
            self.services.api.send_message(chat_id, f"Error: {e}")

    def _handle_callback(self, callback: dict) -> None:
        pass


def main() -> None:
    setup_logging()
    bot = NotebookBot(build_services())
    bot.run_forever()


if __name__ == "__main__":
    main()
