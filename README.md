# Telegram Notebook MVP

یک MVP برای این use case:

1. لینک یک کانال عمومی تلگرام را می‌دهید.
2. پست‌های دارای ویدیو/صوت دانلود می‌شوند.
3. صوت از ویدیو استخراج و به متن تبدیل می‌شود.
4. متن chunk و embed می‌شود.
5. روی transcriptها جست‌وجوی keyword + semantic دارید.
6. از داخل Settings می‌توانید provider و model را بین OpenAI و Gemini عوض کنید.

## چرا Bot API کافی نیست؟

اگر بخواهید محتوای یک کانال عمومی دلخواه را ingest کنید، معمولا باید با API سطح کاربر (`MTProto`) کار کنید. در Bot API، بات فقط پیام‌های کانال‌هایی را می‌گیرد که خودش عضو آن‌ها باشد. برای همین این MVP از `Telethon` استفاده می‌کند.

## معماری

- Python `http.server`: API و UI سبک
- `Telethon`: خواندن پیام‌های کانال عمومی و دانلود media
- `OpenAI` یا `Gemini`: transcription و embedding
- JSON file store: ذخیره‌سازی سبک برای MVP
- Python lexical + cosine search: جست‌وجوی ترکیبی برای MVP

## پیش‌نیازها

- Python 3.11+
- `ffmpeg` روی سیستم برای استخراج صوت از ویدیو و segment کردن فایل‌های بزرگ
- Telegram API credentials:
  - `TELEGRAM_API_ID`
  - `TELEGRAM_API_HASH`
  - ترجیحا `TELEGRAM_SESSION_STRING`
- `TELEGRAM_BOT_TOKEN` اگر می‌خواهید ربات orchestration را هم اجرا کنید
- `OPENAI_API_KEY` یا `GEMINI_API_KEY`

## نصب

```bash
cd /Users/mrchatgpt/Sites/telegram-notebooklm-mvp
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

## گرفتن Telegram credentials

1. از [my.telegram.org](https://my.telegram.org) یک `api_id` و `api_hash` بسازید.
2. برای production بهتر است یک session string بسازید:

```bash
export TELEGRAM_API_ID=...
export TELEGRAM_API_HASH=...
uv run python scripts/create_telegram_session.py
```

خروجی اسکریپت را در `.env` داخل `TELEGRAM_SESSION_STRING` بگذارید.

3. این MVP اگر `TELEGRAM_SESSION_STRING` نداشته باشد، از session file محلی استفاده می‌کند و در اولین اجرا login تعاملی لازم می‌شود.

## اجرا

```bash
.venv/bin/python -m telegram_notebook.main
```

UI روی این آدرس بالا می‌آید:

```text
http://127.0.0.1:8000
```

برای اجرای ربات تلگرام:

```bash
.venv/bin/python -m telegram_notebook.bot
```

## API

### ingest

```bash
curl -X POST http://127.0.0.1:8000/api/channels/ingest \
  -H 'content-type: application/json' \
  -d '{
    "channel_url": "https://t.me/example_channel",
    "limit": 50
  }'
```

### search

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "هوش مصنوعی و مدل زبانی",
    "channel_url": "https://t.me/example_channel",
    "top_k": 5
  }'
```

### settings

- `GET /api/settings`: تنظیمات فعال و وضعیت keyها
- `POST /api/settings`: ذخیره provider/model/keyها در `.env`
- `GET /api/models?provider=gemini&capability=transcription`: گرفتن لیست مدل‌ها از provider

پیش‌فرض فعلی تنظیمات محلی:

- transcription: `gemini-flash-latest`
- embedding: `gemini-embedding-001`

## Bot Commands

- `/start`: معرفی و شروع flow
- `/connect`: شروع اتصال اکانت واقعی تلگرام
- `/status`: وضعیت اتصال session
- `/search <query>`: جست‌وجو در archive
- `/cancel`: لغو flow جاری

Flow اتصال:

1. کاربر در ربات `/connect` می‌زند.
2. شماره را به‌صورت contact یا متن می‌فرستد.
3. backend با `Telethon` کد لاگین را به اکانت واقعی می‌فرستد.
4. کاربر کد را در ربات می‌فرستد.
5. اگر 2FA فعال باشد، ربات پسورد را می‌گیرد.
6. `session string` روی backend ذخیره می‌شود.

## محدودیت‌های MVP

- transcription فقط روی mediaهای صوتی/ویدیویی انجام می‌شود.
- برای فایل‌های خیلی بزرگ، pipeline آن‌ها را به segmentهای کوچک‌تر می‌شکند.
- storage در این نسخه فایل JSON است. برای دیتاست بزرگ بهتر است `Postgres + pgvector` یا یک vector DB اضافه شود.
- برای Gemini در این MVP transcription با `generateContent` روی فایل صوتی انجام می‌شود و مدل‌های قابل انتخاب از API provider خوانده می‌شوند.
- Bot API فقط برای orchestration و مدیریت کاربر استفاده می‌شود؛ خواندن خود کانال‌ها همچنان با `Telethon/MTProto` و اکانت واقعی انجام می‌شود.
- این پروژه فعلا chat-style answer generation مثل NotebookLM ندارد؛ فقط ingest + search را پیاده می‌کند.

## مسیر توسعه بعدی

- افزودن پاسخ‌ساز RAG روی نتایج search
- queue/background jobs با Celery یا Dramatiq
- progress tracking برای ingestهای طولانی
- multi-channel collections
- summary generation per post / per channel
