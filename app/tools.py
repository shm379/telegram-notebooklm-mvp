"""MCP tools that act on the caller's own Telegram account.

Every tool resolves the authenticated user from the OAuth access token, borrows
that user's Telethon client from the pool, and returns a JSON envelope
{"ok": true, ...} or {"ok": false, "error": "..."} so the model never sees a
raw stack trace.
"""
import base64
import contextlib
import contextvars
import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field
from telethon import functions, types
from telethon.tl.types import (
    ChannelParticipantsAdmins,
    ChannelParticipantsBanned,
    ChannelParticipantsBots,
    ChannelParticipantsKicked,
    DocumentAttributeFilename,
    InputPeerNotifySettings,
    InputNotifyPeer,
    InputPhoneContact,
    ReactionEmoji,
)

from . import config
from .tg import (
    SESSION_DEAD_ERRORS,
    ClientEntry,
    ClientPool,
    SessionInvalid,
    chat_id_of,
    describe_media,
    display_name,
    fetch_url,
    guarded,
    iso,
    parse_datetime,
    parse_mode_of,
    peer_kind,
    resolve,
    ser_dialog,
    ser_entity,
    ser_message,
)

log = logging.getLogger("telegram_mcp.tools")

_pool: Optional[ClientPool] = None


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)


def _ok(payload: dict) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, indent=2)


def _token_user_id() -> int:
    """The account the OAuth token was minted for — the caller's primary."""
    tok = get_access_token()
    uid = getattr(tok, "user_id", 0) if tok else 0
    if not uid:
        raise SessionInvalid("Not authenticated. Connect your Telegram account first.")
    return int(uid)


def current_owner_key() -> str:
    owner = _pool.db.owner_of(_token_user_id())
    if not owner:
        raise SessionInvalid("This account is no longer connected.")
    return owner


def accounts() -> list[dict]:
    """The caller's connected accounts, first-connected first."""
    return _pool.db.accounts_for_owner(current_owner_key())


def resolve_account(account: Optional[str] = None) -> int:
    """Pick which of the caller's accounts a tool acts on.

    `account` may be a users.id, a Telegram user id, a @username, a phone, or a
    1-based position ("2" = the second account connected). None means the first
    connected account — a fixed rule, so a caller with one account never has to
    say anything and a caller with several gets the same answer every time.

    Only the caller's own accounts are candidates. A valid token for one person
    can never be pointed at another person's phone by naming its id.
    """
    mine = accounts()
    if not mine:
        raise SessionInvalid("No Telegram account is connected.")
    if account is None or str(account).strip() == "":
        return int(mine[0]["id"])
    key = str(account).strip().lstrip("@").lower()
    for i, u in enumerate(mine, start=1):
        if key in {str(u["id"]), str(u["tg_user_id"]), str(i),
                   (u.get("username") or "").lower(), (u.get("phone") or "").replace(" ", "")}:
            return int(u["id"])
    names = ", ".join(f"{i}: {u.get('username') or u.get('phone') or u['tg_user_id']}"
                      for i, u in enumerate(mine, start=1))
    raise ValueError(f"No connected account matches {account!r}. Yours are: {names}")


#: The `account` argument of the tool call in progress. Set by the tool()
#: wrapper, read by entry()/current_user_id(), so thirty-odd tool bodies did not
#: each have to grow a parameter they then forward — and a helper deep in a tool
#: still acts on the account the caller named.
_account_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("tg_account", default=None)


def current_user_id(account: Optional[str] = None) -> int:
    return resolve_account(account if account is not None else _account_ctx.get())


async def entry(account: Optional[str] = None) -> ClientEntry:
    return await _pool.get(current_user_id(account))


def tool(name: str, *, title: str, read_only: bool = False, destructive: bool = False,
         idempotent: bool = False):
    """Register an MCP tool with uniform error handling."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            token = _account_ctx.set(kwargs.pop("account", None))
            try:
                return await fn(*args, **kwargs)
            except SessionInvalid as ex:
                return _err(f"{ex} Reconnect at {config.PUBLIC_BASE_URL}/login")
            except SESSION_DEAD_ERRORS as ex:
                with contextlib.suppress(Exception):
                    uid = current_user_id()
                    _pool.db.mark_session(uid, False)
                    await _pool.drop(uid)
                return _err(f"Telegram session is no longer valid ({type(ex).__name__}). "
                            f"Reconnect at {config.PUBLIC_BASE_URL}/login")
            except (ValueError, RuntimeError, TypeError, KeyError) as ex:
                return _err(f"{type(ex).__name__}: {ex}")
            except Exception as ex:  # Telethon RPC errors etc.
                log.warning("tool %s failed: %s: %s", name, type(ex).__name__, ex)
                return _err(f"{type(ex).__name__}: {ex}")
            finally:
                _account_ctx.reset(token)
        _mcp.tool(
            name=name,
            annotations={"title": title, "readOnlyHint": read_only, "destructiveHint": destructive,
                         "idempotentHint": idempotent, "openWorldHint": True},
        )(wrapper)
        return wrapper
    return deco


_mcp: FastMCP = None  # set by register()

Chat = Annotated[str, Field(description="Chat reference: @username, numeric chat_id (from telegram_list_chats), "
                                        "t.me link, phone number of a contact (+98...), or 'me' for Saved Messages.")]
ParseMode = Annotated[str, Field(description="Text formatting: 'markdown' (default), 'html' or 'none'.")]
Account = Annotated[Optional[str], Field(
    description="Which of YOUR connected Telegram accounts to use: a 1-based position from "
                "telegram_accounts_list ('1' = first connected), a @username, a phone, or an id. "
                "Omit to use the first connected account.")]


def register(mcp: FastMCP, pool: ClientPool) -> None:
    global _mcp, _pool
    _mcp, _pool = mcp, pool

    # ------------------------------------------------------------------ account
    @tool("telegram_accounts_list", title="List connected accounts", read_only=True)
    async def accounts_list() -> str:
        """List every Telegram account you have connected, in the order you connected them.
        Position 1 is what every other tool uses when you do not name an account; pass a
        position, @username, phone or id as `account` to act on a different one."""
        out = []
        for i, u in enumerate(accounts(), start=1):
            out.append({"position": i, "id": u["id"], "tg_user_id": u["tg_user_id"],
                        "username": u.get("username"), "phone": u.get("phone"),
                        "name": " ".join(x for x in (u.get("first_name"), u.get("last_name")) if x),
                        "connected": bool(u.get("session_ok"))})
        return _ok({"accounts": out, "default": out[0]["position"] if out else None})

    @tool("telegram_me", title="Who am I", read_only=True, idempotent=True)
    async def telegram_me(account: Account = None) -> str:
        """Return the connected Telegram account (name, username, phone, id). Call this first
        to confirm which account the tools act on."""
        en = await entry()
        me = await en.client.get_me()
        return _ok({"account": ser_entity(me), "server": config.APP_NAME})

    @tool("telegram_update_profile", title="Update profile", destructive=True)
    async def telegram_update_profile(
        first_name: Annotated[Optional[str], Field(description="New first name.")] = None,
        last_name: Annotated[Optional[str], Field(description="New last name.")] = None,
        about: Annotated[Optional[str], Field(description="New bio (max 70 chars).")] = None,
        username: Annotated[Optional[str], Field(description="New public @username (without @). Empty string removes it.")] = None, account: Account = None) -> str:
        """Change the account's name, bio and/or username."""
        en = await entry()
        c = en.client
        if first_name is None and last_name is None and about is None and username is None:
            return _err("Provide at least one field to change.")
        if any(v is not None for v in (first_name, last_name, about)):
            await guarded(en, lambda: c(functions.account.UpdateProfileRequest(
                first_name=first_name, last_name=last_name, about=about)))
        if username is not None:
            await guarded(en, lambda: c(functions.account.UpdateUsernameRequest(username=username.lstrip("@"))))
        me = await c.get_me()
        return _ok({"account": ser_entity(me)})

    # -------------------------------------------------------------------- chats
    @tool("telegram_list_chats", title="List chats", read_only=True, idempotent=True)
    async def telegram_list_chats(
        limit: Annotated[int, Field(ge=1, le=500, description="Max chats to return.")] = 50,
        kind: Annotated[Optional[str], Field(description="Filter: user, bot, group, supergroup, channel.")] = None,
        query: Annotated[Optional[str], Field(description="Case-insensitive substring to match the chat title/username.")] = None,
        unread_only: Annotated[bool, Field(description="Only chats with unread messages.")] = False,
        archived: Annotated[Optional[bool], Field(description="True = only archived chats, False = only non-archived, omit = both.")] = None, account: Account = None) -> str:
        """List the account's dialogs (private chats, groups, channels) newest-activity first,
        with unread counts and the last message preview. Use the returned chat_id with the
        other tools."""
        en = await entry()
        out = []
        kind_f = (kind or "").strip().lower() or None
        q = (query or "").strip().lower() or None
        async for d in en.client.iter_dialogs(limit=None if (kind_f or q or unread_only) else limit,
                                              archived=archived):
            if kind_f and peer_kind(d.entity) != kind_f:
                continue
            if unread_only and not (d.unread_count or d.unread_mentions_count):
                continue
            if q and q not in (d.name or "").lower() and q not in (getattr(d.entity, "username", "") or "").lower():
                continue
            en.entities[str(d.id)] = d.entity
            out.append(ser_dialog(d))
            if len(out) >= limit:
                break
        return _ok({"count": len(out), "chats": out})

    @tool("telegram_get_chat", title="Chat info", read_only=True, idempotent=True)
    async def telegram_get_chat(chat: Chat, account: Account = None) -> str:
        """Full information about a user, group or channel (bio/about, member count, admin rights...)."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        full = None
        with contextlib.suppress(Exception):
            if isinstance(ent, types.User):
                full = (await c(functions.users.GetFullUserRequest(id=ent))).full_user
            elif isinstance(ent, types.Channel):
                full = (await c(functions.channels.GetFullChannelRequest(channel=ent))).full_chat
            elif isinstance(ent, types.Chat):
                full = (await c(functions.messages.GetFullChatRequest(chat_id=ent.id))).full_chat
        return _ok({"chat": ser_entity(ent, full)})

    @tool("telegram_search_contacts", title="Search people & chats", read_only=True, idempotent=True)
    async def telegram_search_contacts(
        query: Annotated[str, Field(description="Name, @username or phone fragment.")],
        limit: Annotated[int, Field(ge=1, le=50)] = 20, account: Account = None) -> str:
        """Search the account's contacts plus Telegram's global username search."""
        en = await entry()
        c = en.client
        q = query.strip().lstrip("@")
        seen, found = set(), []
        contacts = await c(functions.contacts.GetContactsRequest(hash=0))
        ql = q.lower()
        for u in getattr(contacts, "users", []):
            hay = " ".join(x for x in [u.first_name, u.last_name, u.username, u.phone] if x).lower()
            if ql in hay:
                seen.add(u.id)
                found.append({**ser_entity(u), "source": "contacts"})
        if len(q) >= 3:
            with contextlib.suppress(Exception):
                res = await c(functions.contacts.SearchRequest(q=q, limit=limit))
                for u in list(res.users) + list(res.chats):
                    if u.id in seen:
                        continue
                    seen.add(u.id)
                    found.append({**ser_entity(u), "source": "global"})
        for f in found:
            en.entities[str(f["chat_id"])] = None  # placeholder cleared below
        en.entities = {k: v for k, v in en.entities.items() if v is not None}
        return _ok({"count": min(len(found), limit), "results": found[:limit]})

    @tool("telegram_get_contacts", title="List contacts", read_only=True, idempotent=True)
    async def telegram_get_contacts(
        limit: Annotated[int, Field(ge=1, le=1000)] = 200, account: Account = None) -> str:
        """List the account's saved contacts."""
        en = await entry()
        res = await en.client(functions.contacts.GetContactsRequest(hash=0))
        users = [ser_entity(u) for u in getattr(res, "users", [])][:limit]
        return _ok({"count": len(users), "contacts": users})

    @tool("telegram_add_contact", title="Add contact", idempotent=True)
    async def telegram_add_contact(
        phone: Annotated[str, Field(description="Phone with country code, e.g. +96890000000.")],
        first_name: str,
        last_name: str = "", account: Account = None) -> str:
        """Add (or update) a contact by phone number."""
        en = await entry()
        res = await guarded(en, lambda: en.client(functions.contacts.ImportContactsRequest(
            contacts=[InputPhoneContact(client_id=0, phone=phone, first_name=first_name, last_name=last_name)])))
        users = [ser_entity(u) for u in res.users]
        if not users:
            return _ok({"imported": 0, "note": "No Telegram account found for this phone (or the user hides their phone)."})
        return _ok({"imported": len(users), "contacts": users})

    @tool("telegram_block_user", title="Block / unblock user", destructive=True, idempotent=True)
    async def telegram_block_user(chat: Chat, block: Annotated[bool, Field(description="True to block, False to unblock.")] = True, account: Account = None) -> str:
        """Block or unblock a user."""
        en = await entry()
        ent = await resolve(en, chat)
        req = functions.contacts.BlockRequest if block else functions.contacts.UnblockRequest
        await guarded(en, lambda: en.client(req(id=ent)))
        return _ok({"chat_id": chat_id_of(ent), "blocked": block})

    # ----------------------------------------------------------------- messages
    @tool("telegram_get_messages", title="Read messages", read_only=True, idempotent=True)
    async def telegram_get_messages(
        chat: Chat,
        limit: Annotated[int, Field(ge=1, le=200, description="How many messages (newest first).")] = 30,
        before_id: Annotated[Optional[int], Field(description="Only messages with id < before_id (paging back).")] = None,
        after_id: Annotated[Optional[int], Field(description="Only messages with id > after_id (newer than).")] = None,
        search: Annotated[Optional[str], Field(description="Filter by text inside this chat.")] = None,
        from_user: Annotated[Optional[str], Field(description="Only messages sent by this user (@username/id), groups only.")] = None,
        ids: Annotated[Optional[list[int]], Field(description="Fetch these specific message ids instead.")] = None,
        text_limit: Annotated[Optional[int], Field(ge=50, description="Truncate each text to this many chars.")] = None,
        reply_to: Annotated[Optional[int], Field(description="Only replies to this message id (comments/threads).")] = None, account: Account = None) -> str:
        """Read a chat's history (newest first) or specific messages by id. Returns text, sender,
        media description, reactions, reply/forward info."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        if ids:
            msgs = await c.get_messages(ent, ids=ids[:200])
            out = [ser_message(m, text_limit) for m in msgs if m is not None]
            return _ok({"count": len(out), "messages": out})
        kw: dict[str, Any] = {"limit": limit}
        if before_id:
            kw["max_id"] = before_id
        if after_id:
            kw["min_id"] = after_id
        if search:
            kw["search"] = search
        if from_user:
            kw["from_user"] = await resolve(en, from_user)
        if reply_to:
            kw["reply_to"] = reply_to
        out = []
        async for m in c.iter_messages(ent, **kw):
            if m is not None:
                out.append(ser_message(m, text_limit))
        return _ok({"chat_id": chat_id_of(ent), "count": len(out), "messages": out,
                    "oldest_id_in_page": min((m["id"] for m in out), default=None)})

    @tool("telegram_search_messages", title="Search messages", read_only=True, idempotent=True)
    async def telegram_search_messages(
        query: Annotated[str, Field(description="Text to search for.")],
        chat: Annotated[Optional[str], Field(description="Limit to one chat; omit to search across all chats.")] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        text_limit: Annotated[Optional[int], Field(ge=50)] = 400, account: Account = None) -> str:
        """Search messages by text, in one chat or globally across every chat of the account."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat) if chat else None
        out = []
        async for m in c.iter_messages(ent, search=query, limit=limit):
            if m is None:
                continue
            d = ser_message(m, text_limit)
            if ent is None:
                with contextlib.suppress(Exception):
                    ch = await m.get_chat()
                    d["chat_title"] = display_name(ch)
            out.append(d)
        return _ok({"count": len(out), "messages": out})

    async def _materialize_file(file_url: Optional[str], file_base64: Optional[str], file_name: Optional[str],
                                *, require_image: bool = False) -> tuple[Optional[Path], Optional[Path]]:
        """Return (path, tmpdir) for an upload given a URL or base64 payload."""
        if not (file_url or file_base64):
            return None, None
        if file_url and file_base64:
            raise ValueError("Pass file_url or file_base64, not both.")
        if file_url:
            data, ctype, name = await fetch_url(file_url, max_bytes=config.MAX_UPLOAD_BYTES, require_image=require_image)
        else:
            try:
                data = base64.b64decode(file_base64, validate=True)
            except Exception as ex:
                raise ValueError("file_base64 is not valid base64.") from ex
            if len(data) > config.MAX_UPLOAD_BYTES:
                raise ValueError(f"File too large ({len(data)} bytes > {config.MAX_UPLOAD_BYTES}).")
            name = "upload.jpg" if require_image else "file.bin"
        name = (file_name or name or "file").replace("/", "_").replace("\\", "_")[:120]
        tmpdir = Path(tempfile.mkdtemp(prefix="tgmcp_up_"))
        path = tmpdir / name
        path.write_bytes(data)
        return path, tmpdir

    @tool("telegram_send_message", title="Send message", destructive=True)
    async def telegram_send_message(
        chat: Chat,
        text: Annotated[Optional[str], Field(description="Message text (or caption when a file is attached).")] = None,
        reply_to: Annotated[Optional[int], Field(description="Message id to reply to.")] = None,
        parse_mode: ParseMode = "markdown",
        link_preview: Annotated[bool, Field(description="Show link previews.")] = True,
        silent: Annotated[bool, Field(description="Send without notification sound.")] = False,
        schedule_at: Annotated[Optional[str], Field(description="ISO 8601 datetime to schedule the message, e.g. 2026-09-03T09:00:00+03:30.")] = None,
        file_url: Annotated[Optional[str], Field(description="Public http(s) URL of a file/photo to attach; the server downloads it.")] = None,
        file_base64: Annotated[Optional[str], Field(description="Inline base64 file content (small files only).")] = None,
        file_name: Annotated[Optional[str], Field(description="File name for the attachment (affects how Telegram shows it).")] = None,
        as_document: Annotated[bool, Field(description="Send images/videos as uncompressed files.")] = False, account: Account = None) -> str:
        """Send a text message, optionally with an attached file/photo, as a reply, silently,
        or scheduled for later. Returns the sent message."""
        en = await entry()
        c = en.client
        if not text and not (file_url or file_base64):
            return _err("Provide text and/or a file.")
        ent = await resolve(en, chat)
        pm = parse_mode_of(parse_mode)
        when = parse_datetime(schedule_at)
        path, tmpdir = await _materialize_file(file_url, file_base64, file_name)
        try:
            if path is not None:
                attrs = [DocumentAttributeFilename(file_name=path.name)] if as_document else None
                msg = await guarded(en, lambda: c.send_file(
                    ent, str(path), caption=text or "", parse_mode=pm, reply_to=reply_to, silent=silent,
                    schedule=when, force_document=as_document, attributes=attrs))
            else:
                msg = await guarded(en, lambda: c.send_message(
                    ent, text, parse_mode=pm, reply_to=reply_to, link_preview=link_preview,
                    silent=silent, schedule=when))
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
        if isinstance(msg, list):
            msg = msg[0]
        return _ok({"message": ser_message(msg), "scheduled": bool(when)})

    @tool("telegram_send_poll", title="Send poll", destructive=True)
    async def telegram_send_poll(
        chat: Chat,
        question: str,
        options: Annotated[list[str], Field(min_length=2, max_length=10)],
        multiple_choice: bool = False,
        anonymous: bool = True,
        quiz_correct_option: Annotated[Optional[int], Field(ge=0, description="Index of the correct option to make this a quiz.")] = None, account: Account = None) -> str:
        """Send a poll or quiz to a chat."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        answers = [types.PollAnswer(text=types.TextWithEntities(text=o, entities=[]), option=bytes([i]))
                   for i, o in enumerate(options)]
        poll = types.Poll(id=0, question=types.TextWithEntities(text=question, entities=[]), answers=answers,
                          public_voters=not anonymous, multiple_choice=multiple_choice and quiz_correct_option is None,
                          quiz=quiz_correct_option is not None)
        media = types.InputMediaPoll(
            poll=poll,
            correct_answers=[bytes([quiz_correct_option])] if quiz_correct_option is not None else None)
        msg = await guarded(en, lambda: c.send_message(ent, file=media))
        return _ok({"message": ser_message(msg)})

    @tool("telegram_edit_message", title="Edit message", destructive=True)
    async def telegram_edit_message(
        chat: Chat,
        message_id: int,
        text: Annotated[Optional[str], Field(description="New text/caption.")] = None,
        parse_mode: ParseMode = "markdown",
        image_url: Annotated[Optional[str], Field(description="Replace the photo with this public image URL (only if the message already has media).")] = None,
        image_base64: Annotated[Optional[str], Field(description="Replace the photo with this inline base64 image.")] = None,
        link_preview: bool = True, account: Account = None) -> str:
        """Edit one of your own messages (or a channel post you administer): change the text
        and/or replace its photo. Telegram cannot add media to a text-only message."""
        en = await entry()
        c = en.client
        if text is None and not (image_url or image_base64):
            return _err("Provide text and/or an image.")
        ent = await resolve(en, chat)
        m = await c.get_messages(ent, ids=message_id)
        if m is None:
            return _err(f"Message {message_id} not found.")
        note = None
        replace = bool(image_url or image_base64) and bool(m.media)
        if (image_url or image_base64) and not m.media:
            note = "Message has no media; only the text was edited."
        path, tmpdir = (await _materialize_file(image_url, image_base64, None, require_image=True)) if replace else (None, None)
        pm = parse_mode_of(parse_mode)
        try:
            if path is not None:
                msg = await guarded(en, lambda: c.edit_message(ent, message_id, text=text, file=str(path), parse_mode=pm))
            else:
                msg = await guarded(en, lambda: c.edit_message(ent, message_id, text=text, parse_mode=pm, link_preview=link_preview))
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
        return _ok({"message": ser_message(msg), "replaced_media": path is not None, **({"note": note} if note else {})})

    @tool("telegram_delete_messages", title="Delete messages", destructive=True, idempotent=True)
    async def telegram_delete_messages(
        chat: Chat,
        message_ids: Annotated[list[int], Field(min_length=1, max_length=100)],
        revoke: Annotated[bool, Field(description="Delete for everyone (True) or only for me (False).")] = True, account: Account = None) -> str:
        """Delete messages. IRREVERSIBLE."""
        en = await entry()
        ent = await resolve(en, chat)
        res = await guarded(en, lambda: en.client.delete_messages(ent, message_ids, revoke=revoke))
        deleted = sum(getattr(r, "pts_count", 0) for r in res) if res else 0
        return _ok({"requested": len(message_ids), "deleted": deleted, "revoke": revoke})

    @tool("telegram_forward_messages", title="Forward messages", destructive=True)
    async def telegram_forward_messages(
        from_chat: Chat,
        message_ids: Annotated[list[int], Field(min_length=1, max_length=100)],
        to_chat: Chat,
        drop_author: Annotated[bool, Field(description="Hide the original sender (forward without attribution).")] = False,
        silent: bool = False, account: Account = None) -> str:
        """Forward messages from one chat to another."""
        en = await entry()
        src = await resolve(en, from_chat)
        dst = await resolve(en, to_chat)
        msgs = await guarded(en, lambda: en.client.forward_messages(dst, message_ids, from_peer=src,
                                                                    drop_author=drop_author, silent=silent))
        if not isinstance(msgs, list):
            msgs = [msgs]
        return _ok({"forwarded": len([m for m in msgs if m]), "messages": [ser_message(m, 200) for m in msgs if m]})

    @tool("telegram_react", title="React to message", idempotent=True)
    async def telegram_react(
        chat: Chat,
        message_id: int,
        emoji: Annotated[Optional[str], Field(description="Reaction emoji, e.g. 👍. Omit/empty to remove your reaction.")] = None,
        big: bool = False, account: Account = None) -> str:
        """Add or remove an emoji reaction on a message."""
        en = await entry()
        ent = await resolve(en, chat)
        reaction = [ReactionEmoji(emoticon=emoji)] if emoji else []
        await guarded(en, lambda: en.client(functions.messages.SendReactionRequest(
            peer=ent, msg_id=message_id, reaction=reaction, big=big)))
        return _ok({"message_id": message_id, "reaction": emoji or None})

    @tool("telegram_mark_read", title="Mark as read", idempotent=True)
    async def telegram_mark_read(
        chat: Chat,
        max_id: Annotated[Optional[int], Field(description="Mark read up to this message id (default: everything).")] = None, account: Account = None) -> str:
        """Mark a chat's messages as read (clears the unread counter)."""
        en = await entry()
        ent = await resolve(en, chat)
        await guarded(en, lambda: en.client.send_read_acknowledge(ent, max_id=max_id or 0, clear_mentions=True))
        return _ok({"chat_id": chat_id_of(ent), "read": True})

    @tool("telegram_pin_message", title="Pin / unpin message", idempotent=True)
    async def telegram_pin_message(chat: Chat, message_id: int, unpin: bool = False, notify: bool = False, account: Account = None) -> str:
        """Pin (or unpin) a message in a chat."""
        en = await entry()
        ent = await resolve(en, chat)
        if unpin:
            await guarded(en, lambda: en.client.unpin_message(ent, message_id))
        else:
            await guarded(en, lambda: en.client.pin_message(ent, message_id, notify=notify))
        return _ok({"message_id": message_id, "pinned": not unpin})

    @tool("telegram_download_media", title="Download media", read_only=True, idempotent=True)
    async def telegram_download_media(
        chat: Chat,
        message_id: int,
        max_kb: Annotated[int, Field(ge=16, le=8192, description="Max size to return inline, in KB.")] = 1536, account: Account = None):
        """Fetch a message's photo or file. Photos (and image documents) come back as an image
        the model can look at; text-like documents come back as text; other files return
        metadata only unless small enough (base64)."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        m = await c.get_messages(ent, ids=message_id)
        if m is None or not m.media:
            return _err("Message not found or has no media.")
        cap = min(max_kb * 1024, config.MAX_INLINE_MEDIA_BYTES * 6)
        meta = describe_media(m) or {}
        if m.photo:
            sizes = [s for s in m.photo.sizes if hasattr(s, "size") and getattr(s, "size", 0)]
            sizes.sort(key=lambda s: s.size, reverse=True)
            chosen = next((s for s in sizes if s.size <= cap), sizes[-1] if sizes else None)
            data = await c.download_media(m, bytes, thumb=chosen)
            if not data:
                return _err("Could not download the photo.")
            return [_ok({"message_id": m.id, "media": meta, "bytes": len(data)}), Image(data=data, format="jpeg")]
        doc = m.document
        size = getattr(doc, "size", 0) or 0
        mime = (getattr(doc, "mime_type", "") or "").lower()
        if size > cap:
            return _ok({"message_id": m.id, "media": meta, "inline": False,
                        "note": f"File is {size} bytes, larger than the {cap} byte inline cap. "
                                "Forward it with telegram_forward_messages or raise max_kb."})
        data = await c.download_media(m, bytes)
        if not data:
            return _err("Could not download the file.")
        if mime.startswith("image/") and mime != "image/svg+xml":
            fmt = "png" if "png" in mime else ("gif" if "gif" in mime else ("webp" if "webp" in mime else "jpeg"))
            return [_ok({"message_id": m.id, "media": meta, "bytes": len(data)}), Image(data=data, format=fmt)]
        texty = mime.startswith("text/") or mime in {"application/json", "application/xml", "application/x-yaml",
                                                     "application/csv", "application/javascript"}
        if texty or (meta.get("file_name", "").lower().endswith((".txt", ".md", ".csv", ".json", ".log", ".py", ".html"))):
            text = data.decode("utf-8", errors="replace")
            return _ok({"message_id": m.id, "media": meta, "text": text[:200_000], "truncated": len(text) > 200_000})
        return _ok({"message_id": m.id, "media": meta, "base64": base64.b64encode(data).decode(), "bytes": len(data)})

    # ------------------------------------------------------------ groups/channels
    @tool("telegram_get_participants", title="List members", read_only=True, idempotent=True)
    async def telegram_get_participants(
        chat: Chat,
        limit: Annotated[int, Field(ge=1, le=1000)] = 100,
        search: Annotated[str, Field(description="Filter members by name.")] = "",
        filter: Annotated[str, Field(description="all (default), admins, bots, kicked, banned.")] = "all", account: Account = None) -> str:
        """List members of a group or channel (as far as the account is allowed to see)."""
        en = await entry()
        ent = await resolve(en, chat)
        f = {"admins": ChannelParticipantsAdmins(), "bots": ChannelParticipantsBots(),
             "kicked": ChannelParticipantsKicked(q=search), "banned": ChannelParticipantsBanned(q=search)}.get(filter.lower())
        out = []
        async for u in en.client.iter_participants(ent, limit=limit, search=search if f is None else "", filter=f):
            d = ser_entity(u)
            p = getattr(u, "participant", None)
            if p is not None:
                d["role"] = {"ChannelParticipantCreator": "creator", "ChannelParticipantAdmin": "admin",
                             "ChatParticipantCreator": "creator", "ChatParticipantAdmin": "admin"}.get(type(p).__name__, "member")
            out.append(d)
        return _ok({"chat_id": chat_id_of(ent), "count": len(out), "members": out})

    @tool("telegram_create_chat", title="Create group/channel", destructive=True)
    async def telegram_create_chat(
        title: str,
        kind: Annotated[str, Field(description="group (small private group), supergroup, or channel (broadcast).")] = "group",
        users: Annotated[Optional[list[str]], Field(description="Members to add (@username / id / phone).")] = None,
        about: Annotated[str, Field(description="Description (supergroup/channel).")] = "", account: Account = None) -> str:
        """Create a new group, supergroup or broadcast channel, optionally inviting members."""
        en = await entry()
        c = en.client
        k = kind.lower().strip()
        members = [await resolve(en, u) for u in (users or [])]
        if k == "group":
            if not members:
                return _err("A basic group needs at least one member; use kind='supergroup' for an empty one.")
            res = await guarded(en, lambda: c(functions.messages.CreateChatRequest(users=members, title=title)))
            chats = getattr(getattr(res, "updates", res), "chats", None) or getattr(res, "chats", [])
            ent = chats[0] if chats else None
        elif k in ("supergroup", "channel"):
            res = await guarded(en, lambda: c(functions.channels.CreateChannelRequest(
                title=title, about=about, megagroup=(k == "supergroup"), broadcast=(k == "channel"))))
            ent = res.chats[0] if getattr(res, "chats", None) else None
            if ent is not None and members:
                await guarded(en, lambda: c(functions.channels.InviteToChannelRequest(channel=ent, users=members)))
        else:
            return _err("kind must be group, supergroup or channel.")
        if ent is None:
            return _ok({"created": True, "note": "Created, but Telegram did not return the chat object."})
        en.entities[str(chat_id_of(ent))] = ent
        return _ok({"chat": ser_entity(ent)})

    @tool("telegram_invite_users", title="Invite users", destructive=True)
    async def telegram_invite_users(chat: Chat, users: Annotated[list[str], Field(min_length=1, max_length=50)], account: Account = None) -> str:
        """Add users to a group or channel."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        members = [await resolve(en, u) for u in users]
        if isinstance(ent, types.Channel):
            await guarded(en, lambda: c(functions.channels.InviteToChannelRequest(channel=ent, users=members)))
        else:
            for u in members:
                await guarded(en, lambda u=u: c(functions.messages.AddChatUserRequest(chat_id=ent.id, user_id=u, fwd_limit=50)))
        return _ok({"chat_id": chat_id_of(ent), "invited": len(members)})

    @tool("telegram_join_chat", title="Join chat", idempotent=True)
    async def telegram_join_chat(link: Annotated[str, Field(description="Public @username / t.me link, or a private invite link (t.me/+hash).")], account: Account = None) -> str:
        """Join a public channel/group or accept a private invite link."""
        en = await entry()
        c = en.client
        s = link.strip()
        if "t.me/+" in s or "joinchat/" in s or (s.startswith("+") and not s[1:].isdigit()):
            h = s.rsplit("/", 1)[-1].lstrip("+")
            res = await guarded(en, lambda: c(functions.messages.ImportChatInviteRequest(hash=h)))
            ent = res.chats[0] if getattr(res, "chats", None) else None
        else:
            ent = await resolve(en, s)
            await guarded(en, lambda: c(functions.channels.JoinChannelRequest(channel=ent)))
        return _ok({"joined": True, **({"chat": ser_entity(ent)} if ent is not None else {})})

    @tool("telegram_leave_chat", title="Leave chat", destructive=True, idempotent=True)
    async def telegram_leave_chat(chat: Chat, account: Account = None) -> str:
        """Leave a group/channel or delete a private chat's dialog."""
        en = await entry()
        ent = await resolve(en, chat)
        await guarded(en, lambda: en.client.delete_dialog(ent))
        en.entities.pop(chat.strip().lower(), None)
        return _ok({"chat_id": chat_id_of(ent), "left": True})

    @tool("telegram_archive_chat", title="Archive / unarchive chat", idempotent=True)
    async def telegram_archive_chat(chat: Chat, archive: bool = True, account: Account = None) -> str:
        """Move a chat to (or out of) the Archive folder."""
        en = await entry()
        ent = await resolve(en, chat)
        await guarded(en, lambda: en.client.edit_folder(ent, folder=1 if archive else 0))
        return _ok({"chat_id": chat_id_of(ent), "archived": archive})

    @tool("telegram_mute_chat", title="Mute / unmute chat", idempotent=True)
    async def telegram_mute_chat(
        chat: Chat,
        mute: bool = True,
        hours: Annotated[Optional[float], Field(gt=0, description="Mute duration in hours; omit for forever.")] = None, account: Account = None) -> str:
        """Mute or unmute notifications for a chat."""
        en = await entry()
        ent = await resolve(en, chat)
        until = 0 if not mute else (int(time.time() + hours * 3600) if hours else 2 ** 31 - 1)
        await guarded(en, lambda: en.client(functions.account.UpdateNotifySettingsRequest(
            peer=InputNotifyPeer(peer=ent), settings=InputPeerNotifySettings(mute_until=until))))
        return _ok({"chat_id": chat_id_of(ent), "muted": mute,
                    "until": iso(datetime.fromtimestamp(until, tz=timezone.utc)) if mute and hours else ("forever" if mute else None)})

    # --------------------------------------------------------------- backup/restore
    def _backup_dir(user_id: int, chat_id: int) -> Path:
        return config.user_dir(user_id) / "backup" / str(chat_id)

    @tool("telegram_backup_messages", title="Backup messages", idempotent=True)
    async def telegram_backup_messages(
        chat: Chat,
        limit: Annotated[Optional[int], Field(ge=1, description="Max messages (newest first); omit for all.")] = None,
        download_photos: bool = True, account: Account = None) -> str:
        """Back up a chat's messages (text + original photos) on the server BEFORE bulk edits,
        so telegram_restore_message can roll back. Replaces any previous backup of this chat."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        cid = chat_id_of(ent)
        bdir = _backup_dir(current_user_id(), cid)
        bdir.mkdir(parents=True, exist_ok=True)
        tmp = bdir / "posts.jsonl.tmp"
        n = photos = 0
        try:
            with tmp.open("w", encoding="utf-8") as out:
                async for m in c.iter_messages(ent, limit=limit):
                    if m is None:
                        continue
                    img = ""
                    if download_photos and m.photo:
                        img = str(bdir / f"orig_{m.id}.jpg")
                        await c.download_media(m, file=img)
                        photos += 1
                    out.write(json.dumps({"id": m.id, "date": iso(m.date), "text": m.message or "",
                                          "has_photo": bool(m.photo), "orig_image": img,
                                          "grouped_id": m.grouped_id}, ensure_ascii=False) + "\n")
                    n += 1
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, bdir / "posts.jsonl")
        except Exception:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        return _ok({"chat_id": cid, "backed_up": n, "with_photos": photos})

    @tool("telegram_restore_message", title="Restore message", destructive=True, idempotent=True)
    async def telegram_restore_message(chat: Chat, message_id: int, account: Account = None) -> str:
        """Restore one message's text (and original photo) from the server-side backup."""
        en = await entry()
        c = en.client
        ent = await resolve(en, chat)
        f = _backup_dir(current_user_id(), chat_id_of(ent)) / "posts.jsonl"
        if not f.exists():
            return _err("No backup for this chat. Run telegram_backup_messages first.")
        rec = None
        for line in f.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                r = json.loads(line) if line.strip() else None
                if r and r.get("id") == message_id:
                    rec = r
                    break
        if rec is None:
            return _err(f"Message {message_id} is not in the backup.")
        has_img = rec.get("has_photo") and rec.get("orig_image") and Path(rec["orig_image"]).exists()
        if has_img:
            await guarded(en, lambda: c.edit_message(ent, rec["id"], text=rec.get("text") or "", file=rec["orig_image"]))
        else:
            await guarded(en, lambda: c.edit_message(ent, rec["id"], text=rec.get("text") or ""))
        return _ok({"id": rec["id"], "restored_photo": bool(has_img)})


def content_hash(*parts: str) -> int:
    """Stable positive int from strings (used as synthetic message keys)."""
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:12], 16)
