from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Settings
from .media import detect_media_kind


@dataclass(slots=True)
class ChannelInfo:
    telegram_id: int
    title: str | None
    username: str | None
    canonical_url: str


@dataclass(slots=True)
class MediaMessage:
    telegram_message_id: int
    message_date: str | None
    caption: str | None
    message_url: str | None
    file_name: str | None
    mime_type: str | None
    duration_seconds: int | None
    file_size_bytes: int | None
    media_kind: str
    source_message: object


def build_client(settings: Settings) -> TelegramClient:
    from telethon import TelegramClient
    from telethon.sessions import StringSession
    from telethon.network import ConnectionTcpFull

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")

    session: StringSession | str
    if settings.telegram_session_string:
        session = StringSession(settings.telegram_session_string)
    else:
        session = str(settings.data_dir / f"{settings.telegram_session_name}.session")

    return TelegramClient(
        session,
        settings.telegram_api_id,
        settings.telegram_api_hash,
        use_ipv6=False,
        connection=ConnectionTcpFull,
        connection_retries=5,
        retry_delay=2
    )


def build_client_from_session_string(
    settings: Settings,
    session_string: str,
    api_id: int | None = None,
    api_hash: str | None = None,
) -> Any:
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    final_api_id = api_id or settings.telegram_api_id
    final_api_hash = api_hash or settings.telegram_api_hash

    if not final_api_id or not final_api_hash:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")

    return TelegramClient(
        StringSession(session_string),
        final_api_id,
        final_api_hash,
        use_ipv6=False,
        connection_retries=5,
        retry_delay=2
    )


def _canonical_channel_url(username: str | None, raw_url: str, telegram_id: int) -> str:
    if username:
        return f"https://t.me/{username}"
    parsed = urlparse(raw_url)
    if parsed.scheme and parsed.netloc:
        return raw_url.rstrip("/")
    return f"telegram://channel/{telegram_id}"


async def fetch_channel_info(client: Any, channel_url: str) -> ChannelInfo:
    entity = await client.get_entity(channel_url)
    username = getattr(entity, "username", None)
    return ChannelInfo(
        telegram_id=int(entity.id),
        title=getattr(entity, "title", None),
        username=username,
        canonical_url=_canonical_channel_url(username, channel_url, int(entity.id)),
    )


async def iter_media_messages(
    client: Any,
    *,
    channel_url: str,
    limit: int,
) -> list[MediaMessage]:
    entity = await client.get_entity(channel_url)
    username = getattr(entity, "username", None)
    results: list[MediaMessage] = []

    async for message in client.iter_messages(entity, limit=limit):
        if not message or not message.media:
            continue
        file_name = getattr(message.file, "name", None) if message.file else None
        mime_type = getattr(message.file, "mime_type", None) if message.file else None
        media_kind = detect_media_kind(file_name, mime_type)
        if not media_kind and getattr(message, "video", None):
            media_kind = "video"
        if not media_kind and (
            getattr(message, "audio", None) or getattr(message, "voice", None)
        ):
            media_kind = "audio"
        if not media_kind:
            continue

        message_url = None
        if username:
            message_url = f"https://t.me/{username}/{message.id}"

        duration = getattr(getattr(message, "audio", None), "duration", None)
        if duration is None:
            duration = getattr(getattr(message, "video", None), "duration", None)

        results.append(
            MediaMessage(
                telegram_message_id=int(message.id),
                message_date=message.date.isoformat() if message.date else None,
                caption=message.message,
                message_url=message_url,
                file_name=file_name,
                mime_type=mime_type,
                duration_seconds=int(duration) if duration else None,
                file_size_bytes=getattr(message.file, "size", None) if message.file else None,
                media_kind=media_kind,
                source_message=message,
            )
        )

    return list(reversed(results))


async def download_message_media(
    client: Any,
    *,
    media_message: MediaMessage,
    target_dir: Path,
) -> Path | None:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = await client.download_media(media_message.source_message, file=target_dir)
    return Path(downloaded) if downloaded else None


async def join_chat(client: Any, link: str) -> str:
    from telethon.tl.functions.messages import ImportChatInviteRequest
    from telethon.tl.functions.channels import JoinChannelRequest
    
    # تمیز کردن لینک
    link = link.strip()
    if "t.me/+" in link or "t.me/joinchat/" in link:
        # لینک‌های خصوصی (Invite Link)
        hash_code = link.split("/")[-1].replace("+", "")
        await client(ImportChatInviteRequest(hash_code))
        return "با موفقیت به گروه خصوصی وارد شدم."
    else:
        # لینک‌های عمومی یا یوزرنیم
        entity = link.split("/")[-1]
        await client(JoinChannelRequest(entity))
        return f"به {entity} پیوستم."


async def iter_all_messages(
    client: Any,
    *,
    channel_url: str,
    limit: int,
) -> list[MediaMessage]:
    entity = await client.get_entity(channel_url)
    username = getattr(entity, "username", None)
    results: list[MediaMessage] = []

    async for message in client.iter_messages(entity, limit=limit):
        if not message:
            continue
            
        # اگر فقط متن بود، آن را هم مثل یک آیتم مدیا برای ایندکس شدن در نظر می‌گیریم
        file_name = getattr(message.file, "name", None) if message.file else None
        mime_type = getattr(message.file, "mime_type", None) if message.file else None
        media_kind = detect_media_kind(file_name, mime_type) or "text"
        
        message_url = None
        if username:
            message_url = f"https://t.me/{username}/{message.id}"

        duration = getattr(getattr(message, "audio", None), "duration", None)
        if duration is None:
            duration = getattr(getattr(message, "video", None), "duration", None)

        results.append(
            MediaMessage(
                telegram_message_id=int(message.id),
                message_date=message.date.isoformat() if message.date else None,
                caption=message.message or "",
                message_url=message_url,
                file_name=file_name,
                mime_type=mime_type,
                duration_seconds=int(duration) if duration else None,
                file_size_bytes=getattr(message.file, "size", None) if message.file else None,
                media_kind=media_kind,
                source_message=message,
            )
        )

    return list(reversed(results))


async def request_login_code(
    settings: Settings,
    phone: str,
    api_id: int | None = None,
    api_hash: str | None = None,
) -> dict[str, str]:
    print(f"DEBUG: Initializing client for {phone}...")
    client = build_client_from_session_string(settings, "", api_id=api_id, api_hash=api_hash)
    try:
        print("DEBUG: Connecting to Telegram...")
        await client.connect()
        print("DEBUG: Connected. Requesting code...")
        result = await client.send_code_request(phone)
        print(f"DEBUG: Code sent. Hash: {result.phone_code_hash}")
        return {
            "session_string": client.session.save(),
            "phone_code_hash": result.phone_code_hash,
            "phone": phone,
        }
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        raise
    finally:
        await client.disconnect()


async def sign_in_with_code(
    settings: Settings,
    *,
    phone: str,
    session_string: str,
    code: str,
    phone_code_hash: str,
    api_id: int | None = None,
    api_hash: str | None = None,
) -> dict[str, str]:
    from telethon.errors import SessionPasswordNeededError

    print(f"DEBUG: Signing in with code for {phone}...")
    client = build_client_from_session_string(
        settings, session_string, api_id=api_id, api_hash=api_hash
    )
    try:
        await client.connect()
        print("DEBUG: Connected to Telegram for sign-in.")
        await client.sign_in(
            phone=phone,
            code=code,
            phone_code_hash=phone_code_hash,
        )
        print("DEBUG: Sign-in successful!")
        return {
            "status": "authorized",
            "session_string": client.session.save(),
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }
    except SessionPasswordNeededError:
        print("DEBUG: 2FA Password required.")
        return {
            "status": "password_required",
            "session_string": client.session.save(),
        }
    except Exception as e:
        print(f"DEBUG ERROR during sign-in: {e}")
        raise
    finally:
        await client.disconnect()


async def sign_in_with_password(
    settings: Settings,
    *,
    session_string: str,
    password: str,
    api_id: int | None = None,
    api_hash: str | None = None,
) -> dict[str, str]:
    print("DEBUG: Signing in with 2FA password...")
    client = build_client_from_session_string(
        settings, session_string, api_id=api_id, api_hash=api_hash
    )
    try:
        await client.connect()
        await client.sign_in(password=password)
        print("DEBUG: 2FA Sign-in successful!")
        return {
            "status": "authorized",
            "session_string": client.session.save(),
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        print(f"DEBUG ERROR during password sign-in: {e}")
        raise
    finally:
        await client.disconnect()
