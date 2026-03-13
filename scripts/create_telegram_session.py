from __future__ import annotations

import asyncio
import os

from telethon import TelegramClient
from telethon.sessions import StringSession


async def main() -> None:
    api_id_raw = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id_raw or not api_hash:
        raise SystemExit("Set TELEGRAM_API_ID and TELEGRAM_API_HASH first.")

    api_id = int(api_id_raw)
    client = TelegramClient(StringSession(), api_id, api_hash)

    async with client:
        await client.start()
        print(client.session.save())


if __name__ == "__main__":
    asyncio.run(main())
