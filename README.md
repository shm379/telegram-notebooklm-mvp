# Telegram NotebookLM MVP

**Languages / زبان‌ها:** [English](README.md) · [فارسی](README.fa.md) · [العربية](README.ar.md) · [Español](README.es.md) · [简体中文](README.zh.md)

This is an MVP for building a **smart Telegram archive**; a project that can collect the content of channels, chats, files, videos, and forwarded messages, convert them into searchable text, and ultimately answer the user's questions like an **internal NotebookLM for Telegram**.

The project's ultimate goal is to let the user turn their Telegram content into a searchable memory that can be connected to AI tools; from within the Telegram bot, the web dashboard, and in the future via MCP for connecting to tools such as ChatGPT, Claude, Cursor, Codex-like agents, and other AI clients.

---

## Core idea

This project targets three main modes:

### 1. Import Channel / Chat

The user provides the link or ID of a public channel or a chat they have access to, and the system retrieves its messages, captions, and media.

Example:

```text
/ingest https://t.me/example_channel
```

### 2. Forwarded Inbox

The user can forward a message, post, file, photo, video, PDF, or any content to the bot. The system stores, processes, tags, and makes it searchable.

This part is meant to act like a **Smart Telegram Inbox**.

### 3. AI Notebook / RAG

After content is stored and indexed, the user can ask questions of their archive:

```text
/ask از بین پیام‌هایی که درباره Al Mouj ذخیره کردم، کدام‌ها درباره townhouse بودند؟
```

or:

```text
/ask ابزارهای AI که در کانال‌ها درباره ساخت ویدیو معرفی شده‌اند را دسته‌بندی کن
```

The answer should come with the source, the message link, and the related texts.

---

## وضعیت فعلی MVP

در نسخه فعلی، پروژه این قابلیت‌ها را دارد:

- دریافت لینک کانال تلگرام و خواندن پیام‌ها با `Telethon`
- دانلود و پردازش پیام‌های متنی، صوتی و ویدیویی
- استخراج صوت از ویدیو با `ffmpeg`
- تبدیل صوت/ویدیو به متن با OpenAI یا Gemini
- chunk کردن متن‌ها
- ساخت embedding برای جستجوی معنایی
- جستجوی keyword + semantic search
- پاسخ‌سازی اولیه با RAG از روی نتایج جستجو
- داشبورد وب سبک با `Python http.server`
- ربات تلگرام برای orchestration و دستورات اصلی
- اتصال اکانت واقعی تلگرام کاربر با session string از طریق Telethon
- **Forwarded Inbox**: فوروارد هر پیام به ربات، متن/کپشن آن را در inbox شخصی و قابل‌جستجوی کاربر ذخیره می‌کند
- **پردازش مدیای فورواردشده**: فایل صوتی/ویدیو/voice به‌صورت خودکار transcribe و عکس/PDF با OCR (Gemini) به متن قابل‌جستجو تبدیل می‌شوند. فایل‌های DOCX/XLSX هم به‌صورت محلی (بدون کلید API و بدون شبکه) استخراج می‌شوند
- **پردازش مدیا در import کامل کانال**: همان مسیر روی import کانال هم اجرا می‌شود — صوت/ویدیو transcribe، عکس/PDF با OCR، و DOCX/XLSX به‌صورت محلی استخراج می‌شوند (نه فقط متن/کپشن)
- **پردازش مدیای فورواردشده**: فایل صوتی/ویدیو/voice به‌صورت خودکار transcribe و عکس/PDF با OCR (Gemini) به متن قابل‌جستجو تبدیل می‌شوند
- **Rule Engine + Tags**: تعریف قانون keyword→tag، تگ‌گذاری خودکار محتوای واردشده، و فیلتر `/search` و `/ask` با `--tag`
- **Auto-forward به کانال آرشیو**: با `/setarchive`، هر پیامی که فوروارد می‌کنید یا از import کانال می‌آید و با یک قانون tag مطابقت دارد، به‌صورت خودکار به کانال آرشیو شما هم فوروارد می‌شود
- **Import Jobs**: import کامل کانال در background با صف، progress tracking، resume بعد از قطع‌شدن، و امکان لغو
- **خلاصه‌سازی (NotebookLM)**: `/summarize` برای ساخت خلاصه‌ی ساختارمند از کل آرشیو، یک منبع خاص، یا یک تگ
- **Digest**: `/digest [days]` خلاصه‌ی AI از محتوای اخیر (پیش‌فرض ۷ روز) را می‌سازد
- **Audio Overview (Podcast)**: `/podcast` با الهام از NotebookLM، یک پادکست گفتگوی صوتی از کل آرشیو، یک منبع، یک تگ یا یک collection می‌سازد (با [Podcastfy](https://github.com/souzatharsis/podcastfy)؛ به‌صورت extra اختیاری)
- **Topic clustering**: `/topics` محتوای آرشیو را به‌صورت آفلاین (روی embeddingهای موجود) خوشه‌بندی موضوعی می‌کند؛ در صورت وجود کلید Gemini، برچسب هر خوشه با LLM ساخته می‌شود (وگرنه از پرتکرارترین واژه‌ها)
- **Timeline**: `/timeline` آرشیو را بر اساس تاریخ (ماه یا روز) گروه‌بندی می‌کند — مکمل زمانیِ `/topics`
- **Export**: `/export` کل آرشیو، یک منبع یا یک تگ را به‌صورت یک فایل Markdown قابل‌دانلود خروجی می‌گیرد
- **Stats**: `/stats` نمای کلی آرشیو (تعداد آیتم‌ها، منابع، تگ‌ها، نوع مدیا، و بازه‌ی زمانی) را نشان می‌دهد
- **Collections (Notebooks)**: `/collection` چند تگ را زیر یک نام گروه می‌کند و آیتم‌های مجموعه را نشان می‌دهد
- **MCP Server (read-only)**: expose کردن آرشیو به ابزارهای AI با JSON-RPC روی stdio (`python -m telegram_notebook.mcp_server`)
- **Telegram Toolset** (الهام از [uburuntu/Telegram-Toolset](https://github.com/uburuntu/Telegram-Toolset)؛ روی اکانت connect‌شده): `/account` (اطلاعات اکانت)، `/scheduled` (لیست/لغو پیام‌های زمان‌بندی‌شده)، `/llmexport` (خروجی AI-friendly از یک چت)، `/resend` (ارسال به‌جای کاربر)، و `/watchdeleted` + `/deleted` (بازیابی پیام‌های حذف‌شده با watcher همیشه‌روشن opt-in). همه‌ی این‌ها در سایت هم به‌صورت کارت «Telegram Toolset» و endpointهای `/api/account`، `/api/scheduled`، `/api/deleted`، `/api/llmexport`، `/api/resend` در دسترس‌اند.

---

## Why isn't the Bot API enough?

To capture a complete archive of a channel or chat, the Bot API alone is not enough. The Bot API usually only sees new messages that the bot has access to.

To import the history of channels and chats, this project uses `Telethon` and MTProto; that is, the same level of access as a user account, not just a bot token.

---

## Current architecture

```text
Telegram Bot
  |
  | user commands: /connect, /ingest, /search, /ask
  v
Python Backend
  |
  +-- Telethon Client
  |     reading channels and chats
  |
  +-- Ingestion Pipeline
  |     downloading media, extracting text, transcription
  |
  +-- Chunking + Embedding
  |     preparing for semantic search
  |
  +-- Search Service
  |     keyword search + vector search
  |
  +-- RAG Answer Generator
        building the answer from the found sources
```

---

## Technologies

- Python 3.11+
- Telethon
- OpenAI API
- Google Gemini / Google GenAI
- ffmpeg
- SQLite / JSON-compatible local store for the MVP
- Python lexical search + cosine similarity
- Telegram Bot API for the user interface
- A lightweight Web UI with `http.server`

---

## Bot commands

```text
/start
Project introduction and getting started

/connect
Connect the user's real Telegram account

/status
Check the connection status

/ingest <channel_url>
Fast, inline indexing of a channel

/import <channel_url> [limit]
Queue a full, resumable import in the background

/backup
Guide for importing a Telegram backup file; just send the result.json or .zip file to the bot

/jobs
Show the status and progress of import jobs

/canceljob <id>
Cancel a queued or running job

/search <query>
Search the archive

/search <query> --source <channel_url>
Search only within a specific source

/search <query> --tag <tag>
Search only within tagged content

/ask <question>
Ask the archive with AI

/ask <question> --source <channel_url>
Ask only from a specific channel or source

/ask <question> --tag <tag>
Ask only from the content of a specific tag

/summarize [--source <url>] [--tag <tag>]
Summarize the entire archive, a source, or a tag

/digest [days]
AI summary of recent content (default 7 days)

/podcast [--source <url>] [--tag <tag>] [--collection <name>]
ساخت Audio Overview (پادکست) از کل آرشیو، یک منبع، یک تگ یا یک collection

/topics [--source <url>] [--tag <tag>]
Topic clustering of the content

/timeline [--source <url>] [--tag <tag>] [--day]
Temporal view of the archive by month (or day with --day)

/export [--source <url>] [--tag <tag>]
Download a Markdown export of the entire archive, a source, or a tag

/recent [n]
Show the n most recent items (default 10)

/stats
Archive overview (item, source, tag counts, media types, time range)

/sources
Show the indexed sources

/delete <channel_url>
Delete a source's data

/rule add <keyword> -> <tag>
Define a keyword rule for automatic content tagging

/rule add-ai <criterion> -> <tag>
Define an AI rule (matched by an LLM, during /rule apply)

/rule list
Show the rules

/rule remove <id>
Remove a rule

/rule apply
Re-apply the rules to existing content (backfill)

/airules on|off
Automatically run AI rules on every new forward (opt-in; off by default)

/tags
Show the tags and the item count for each tag

/tag rename <old> -> <new>
Rename a tag (or merge it into an existing tag)

/tag delete <tag>
Delete a tag from all items

/collection new|add|list|remove|show <name>
Group multiple tags under a "notebook" and show its items
(you can then summarize/export the whole notebook with /summarize --collection <name> or /export --collection <name>)

/setarchive <@channel | off>
Set the archive channel; tagged forwards are sent to it automatically

/account
نمایش اطلاعات اکانت تلگرامِ connect‌شده

/scheduled <chat> [cancel <id>]
نمایش (یا لغو) پیام‌های زمان‌بندی‌شده‌ی یک چت

/llmexport <chat> [limit]
خروجی گرفتن یک چت به‌صورت transcript مارک‌داون دوست‌دار AI

/resend <target> <text>
ارسال یک پیام از طرف اکانت شما به یک مقصد

/watchdeleted on|off
روشن/خاموش‌کردن watcher بازیابی پیام‌های حذف‌شده (فقط پیام‌های بعد از فعال‌سازی)

/deleted [n]
نمایش پیام‌های حذف‌شده‌ای که اخیراً بازیابی شده‌اند

/cancel
Cancel the current flow
```

---

## Core APIs

### Ingest Channel

```bash
curl -X POST http://127.0.0.1:8000/api/channels/ingest \
  -H 'content-type: application/json' \
  -d '{
    "channel_url": "https://t.me/example_channel",
    "limit": 50
  }'
```

### Search

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "هوش مصنوعی و تولید ویدیو",
    "channel_url": "https://t.me/example_channel",
    "top_k": 5
  }'
```

### Ask AI

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'content-type: application/json' \
  -d '{
    "query": "از این کانال چه ابزارهایی برای ساخت ویدیو معرفی شده؟",
    "channel_url": "https://t.me/example_channel"
  }'
```

### Stats / Recent / Timeline (read-only)

```bash
curl http://127.0.0.1:8000/api/stats
curl 'http://127.0.0.1:8000/api/recent?limit=10'
curl 'http://127.0.0.1:8000/api/timeline?granularity=month'
```

These endpoints, like the rest of the API, are protected with `WEB_API_TOKEN` (or loopback only when no token is set).

---

## Final product direction

The ultimate goal of this project is not just simple search. The product direction is as follows:

```text
Telegram AI Archive
  |
  +-- Full import of channels and chats
  +-- Forwarded Inbox for forwarded messages
  +-- Rule Engine for separating content with keyword or AI
  +-- Tag / Folder / Collection
  +-- Lexical and semantic search
  +-- Internal NotebookLM for Q&A
  +-- MCP Server for connecting to AI tools
```

---

## Rule Engine + Tags

The user can define keyword→tag rules:

```text
/rule add Claude -> AI Tools
/rule add Al Mouj -> Real Estate
/rule add golden visa -> Oman Visa
/rule add قیمت -> Leads
```

Any new content that enters the system (channel ingest, media transcript, or Forwarded Inbox) is checked against its text and caption. If a rule's keyword (as a substring and case-insensitive) is in the text:

- The corresponding tag is attached to that item
- It can later be filtered with `/search ... --tag <tag>` and `/ask ... --tag <tag>`
- `/tags` shows the tags and the item count for each tag
- `/tag rename <old> -> <new>` renames a tag (or merges it into an existing tag), and `/tag delete <tag>` removes it from all items
- `/rule apply` re-applies the current rules to the existing content (backfill)

**AI-based rules:** in addition to keyword rules, you can define a rule with a natural-language criterion whose match an LLM decides on:

```text
/rule add-ai پست‌هایی که درباره‌ی ابزارهای ساخت ویدیو با هوش مصنوعی هستند -> Video AI
/rule add-ai هر چیزی مرتبط با قیمت و فروش ملک -> Leads
```

Because of the LLM cost, AI rules run only during `/rule apply` (one LLM call per item) and require a Gemini key; if no key is set, they are ignored and this is reported in the output. Keyword rules are still applied automatically on every ingest.

**Auto-forward:** with `/setarchive <@channel>` an archive channel is set; from then on, any message you forward to the bot whose text matches a tag rule is, in addition to being stored in the inbox, also forwarded to that channel with its source/tags/link (the bot must be an admin of the channel). To disable: `/setarchive off`.

In addition to `/rule apply`, AI rules also run automatically on every new forward with `/airules on` (opt-in, one LLM call per item; bulk channel imports are never auto-classified).

**اضافه‌شده:** auto-forward و دانلود/پردازش مدیا (OCR/transcription/DOCX-XLSX) حالا روی import کامل کانال هم اجرا می‌شوند — نه فقط مسیر Forwarded Inbox.
**Not yet added (follow-up):** auto-forward for channel imports (for now, only the Forwarded Inbox path) and downloading/processing media within the full channel import.

---

## Telegram backup import (JSON / ZIP)

You can import the full history of a chat or account without needing `/connect`:

1. In **Telegram Desktop**, go to `Settings → Advanced → Export Telegram data` (or, on a chat: `Export chat history`).
2. Set the format to **Machine-readable JSON** (not HTML).
3. The output is a `result.json` file (or a folder containing it along with media); you can zip the folder.

Then:

- **From the bot:** send the `result.json` or `.zip` file directly to the bot. The bot imports it, makes the content searchable, and returns a Markdown copy. (A 20 MB cap because of the Bot API download limit; for a larger file, use the web.)
- **From the web:** in `/app`, in the "Import Telegram Backup" card, upload the file. The content becomes searchable in the web archive and a Markdown download button appears.

Each chat becomes a synthetic source `backup://<id>` and the import is idempotent (re-importing the same file adds nothing).

### API

```bash
curl -X POST 'http://127.0.0.1:8000/api/backup/import' \
  -H 'X-Filename: result.json' \
  -H 'content-type: application/octet-stream' \
  --data-binary @result.json
```

The response includes the number of imported chats/messages and the full Markdown text. Like the rest of the API, it is protected with `WEB_API_TOKEN` (or loopback).

---

## MCP Server

A **Telegram MCP Server** (read-only) has been implemented so that the user's Telegram archive isn't confined to the bot and can be connected to other AI tools (Claude, Cursor, …). It works with JSON-RPC 2.0 over stdio and is written using only the standard library (no new dependencies).

Run:

```bash
MCP_OWNER_ID=0 python -m telegram_notebook.mcp_server
```

`MCP_OWNER_ID` determines which user's archive is exposed (default `0` = the web dashboard archive; for a bot user's archive, provide their `bot_user_id`).

The current MCP tools:

```text
list_sources              Show channels/chats and the forwarded inbox
list_tags                 Show the tags and the item count for each tag
search_telegram_archive   Search (with an optional source/tag filter)
get_message               The full text of an item by media_item_id
ask_telegram_notebook     RAG Q&A over the archive
summarize_source          Summarize the entire archive, a source, or a tag
list_topics               Topic clustering of the content (offline, from embeddings)
timeline                  Count items by time period (month/day)
archive_stats             Archive overview (counts, media types, time range)
list_recent               List of the latest archive items (newest first)
```

All tools are read-only; sensitive tools (import, forward, delete, create_rule) are deliberately not exposed and, if needed, should be added later with permission and confirmation.

---

## Audio Overview (Podcast)

قابلیت امضای NotebookLM یعنی «Audio Overview»: تبدیل آرشیو به یک پادکست گفتگوی صوتی. این کار با کتابخانه‌ی [Podcastfy](https://github.com/souzatharsis/podcastfy) انجام می‌شود و چون وابستگی سنگینی دارد، به‌صورت **extra اختیاری** نصب می‌شود:

```bash
pip install ".[podcast]"
```

سپس در ربات:

```text
/podcast                      # کل آرشیو
/podcast --tag "AI Tools"     # فقط یک تگ
/podcast --source https://t.me/example_channel
/podcast --collection myNotebook
```

- موتور TTS با `PODCAST_TTS_MODEL` انتخاب می‌شود: `edge` (رایگان و بدون کلید، پیش‌فرض)، یا `openai`/`gemini`/`elevenlabs` (با کلید provider مربوطه).
- LLM که متن گفتگو را می‌نویسد به‌صورت پیش‌فرض از کلید Gemini کاربر (یا `GEMINI_API_KEY`) استفاده می‌کند؛ در نبود آن از OpenAI. با `PODCAST_LLM_MODEL` قابل override است.
- اگر extra نصب نشده باشد، `/podcast` پیام راهنمای نصب می‌دهد (به‌جای crash).
- خروجی markdown دستیار (`/ask`، `/summarize`، `/digest`، …) حالا با [telegramify-markdown](https://github.com/sudoskys/telegramify-markdown) به MarkdownV2 امن تبدیل می‌شود تا فرمت درست رندر شود و کاراکترهای خاص پیام را نشکنند.

---

## Telegram Toolset

پنج ماژول [uburuntu/Telegram-Toolset](https://github.com/uburuntu/Telegram-Toolset) به ربات و سایت پورت شده‌اند. این قابلیت‌ها روی **اکانت کاربرِ connect‌شده** کار می‌کنند (در ربات per-user با `/connect`؛ در سایت روی اکانت سرور یعنی `TELEGRAM_SESSION_STRING`).

| ماژول | ربات | سایت |
|---|---|---|
| account-info | `/account` | `GET /api/account` + کارت |
| scheduled | `/scheduled <chat> [cancel <id>]` | `GET /api/scheduled?peer=` + کارت |
| llm-export | `/llmexport <chat> [limit]` | `POST /api/llmexport` (دانلود `.md`) |
| resend | `/resend <target> <text>` | `POST /api/resend` |
| export-deleted | `/watchdeleted on|off` + `/deleted [n]` | `GET /api/deleted` + watcher با `WEB_WATCH_DELETED=1` |

### نکته‌ی مهم درباره‌ی بازیابی پیام حذف‌شده
تلگرام **API‌ای برای گرفتن پیام‌های قبلاً حذف‌شده ندارد**. این قابلیت یک **watcher همیشه‌روشنِ opt-in** است که هر پیام ورودی را cache می‌کند و وقتی تلگرام رویداد حذف می‌فرستد، نسخه‌ی cache‌شده را بازیابی می‌کند. بنابراین:

- فقط پیام‌هایی را بازیابی می‌کند که **بعد از فعال‌سازی** دریافت شده باشند — حذف‌های گذشته قابل بازیابی نیستند.
- نیازمند یک کلاینت Telethon دائمی per-user است (در ربات با `/watchdeleted on`؛ در سایت با `WEB_WATCH_DELETED=1`).
- از نظر حریم خصوصی سنگین است (متن همه‌ی پیام‌های ورودی ذخیره می‌شود)؛ با `/watchdeleted off` متوقف می‌شود.

---

## توسعه و تست

CI runs lint and tests on every push and PR (`.github/workflows/ci.yml`). To run locally:

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## Installation

```bash
git clone https://github.com/shm379/telegram-notebooklm-mvp.git
cd telegram-notebooklm-mvp

uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

On Windows:

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
copy .env.example .env
```

---

## Prerequisites

- Python 3.11+
- ffmpeg
- Telegram API credentials:
  - `TELEGRAM_API_ID`
  - `TELEGRAM_API_HASH`
  - `TELEGRAM_SESSION_STRING` is preferable for production runs
- `TELEGRAM_BOT_TOKEN` to run the bot
- One of these providers:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`

---

## Creating a Telegram Session

```bash
export TELEGRAM_API_ID=...
export TELEGRAM_API_HASH=...
uv run python scripts/create_telegram_session.py
```

Put the output in `.env` under `TELEGRAM_SESSION_STRING`.

If you don't have a `TELEGRAM_SESSION_STRING`, the project uses a local session file and the first run requires an interactive login.

---

## Running the Web UI

```bash
python -m telegram_notebook.main
```

Then open:

```text
http://127.0.0.1:8000        # landing page (introduction)
http://127.0.0.1:8000/app    # dashboard: Ingest / Search / Ask / Library / Backup import
```

---

## Running the Telegram bot

```bash
python -m telegram_notebook.bot
```

---

## Current limitations

- جداسازی داده بین کاربران انجام شده است (هر کاربر فقط دیتای خودش را می‌بیند؛ مالکیت با `owner_id` روی کانال‌ها اعمال می‌شود).
- احراز هویت Web API با `WEB_API_TOKEN` و رمزنگاری secrets در دیتابیس با `SECRETS_KEY` اضافه شده؛ برای production هر دو متغیر را تنظیم کنید.
- storage فعلی برای MVP مناسب است، نه دیتاست بزرگ.
- session string و API keyها باید قبل از production رمزنگاری شوند.
- import کامل کانال با صف، progress و resume از طریق `/import` پشتیبانی می‌شود (یک worker در background)؛ هنوز یک sandbox تست برای کل مسیر Telethon وجود ندارد.
- پردازش مدیا در هر دو مسیر یکسان است: صوت/ویدیو/voice با transcription، عکس/PDF با OCR (Gemini multimodal)، و DOCX/XLSX با استخراج محلی (zipfile + XML، بدون کلید API). این پردازش هم روی Forwarded Inbox و هم روی import کامل کانال انجام می‌شود (routing مشترک در `media.route_media`).
- Rule Engine بر اساس تطبیق keyword (substring) است؛ قوانین AI-based و forward خودکار به کانال آرشیو هنوز اضافه نشده.
- `/topics` خوشه‌بندی موضوعی را روی embeddingهای موجود انجام می‌دهد (greedy cosine، آفلاین)؛ نیازمند آن است که محتوا با کلید embedding ایندکس شده باشد. برچسب خوشه‌ها در صورت وجود کلید Gemini با LLM ساخته می‌شود و در غیر این صورت به پرتکرارترین واژه‌ها برمی‌گردد. `/timeline` نمای زمانی (ماه/روز) را روی تاریخ پیام‌ها می‌سازد.
- برای دیتاست بزرگ بهتر است به PostgreSQL + pgvector یا Qdrant مهاجرت شود.
- Data isolation between users is in place (each user sees only their own data; ownership is enforced via `owner_id` on channels).
- Web API authentication with `WEB_API_TOKEN` and encryption of secrets in the database with `SECRETS_KEY` have been added; set both variables for production.
- The current storage is suitable for the MVP, not a large dataset.
- The session string and API keys should be encrypted before production.
- Full channel import with a queue, progress, and resume is supported via `/import` (a background worker); there is not yet a test sandbox for the entire Telethon path.
- In addition to text/caption, the Forwarded Inbox processes forwarded media: audio/video/voice via transcription, photos/PDFs via OCR (Gemini multimodal), and DOCX/XLSX via local extraction (zipfile + XML, no API key) are converted to text. Downloading media within the full channel import has not been added yet.
- The Rule Engine is based on keyword matching (substring); AI-based rules and automatic forwarding to an archive channel have not been added yet.
- `/topics` performs topic clustering over the existing embeddings (greedy cosine, offline); it requires that the content be indexed with an embedding key. Cluster labels are built with an LLM if a Gemini key is present, and otherwise fall back to the most frequent terms. `/timeline` builds a temporal view (month/day) over the message dates.
- For a large dataset, migrating to PostgreSQL + pgvector or Qdrant is preferable.

---

## Important security notes

- Do not commit any real token, API key, session string, or credential into the repo.
- If a real token was previously committed into `.env.example` or the project history, revoke/regenerate that token immediately.
- For production, user sessions must be encrypted.
- A user_id filter must be applied for every search or ask.
- The user must be able to `disconnect` and `delete my data`.
- MCP tools should initially be read-only.

---

## Suggested Roadmap

### Phase 1 — Stabilize Core

- Cleaning secrets out of the repo
- Fixing the README and env example
- Stabilizing `/connect`, `/ingest`, `/search`, `/ask`
- Fixing error handling and logging

### Phase 2 — Multi-user Data Model

- Adding user_id to sources, messages, media, chunks
- Full isolation of users' data
- Permission and access control

### Phase 3 — Forwarded Inbox

- Processing forwarded messages
- Storing text, caption, media, document
- OCR for photos
- Text extraction from PDF/DOCX/Excel

### Phase 4 — Rules + Tags

- Defining keyword rules
- Tag and collection
- Automatic forwarding to archive channels
- AI-based rules

### Phase 5 — Full Import Jobs

- Full channel import from start to finish
- Resume after interruption
- Progress tracking
- Queue/background worker

### Phase 6 — Internal NotebookLM

- Better answer generation with sources
- Summary per source
- Summary per tag
- Timeline and topic clustering ✅ (`/timeline`, `/topics`)

### Phase 7 — MCP Server

- Read-only MCP tools
- Connecting to AI clients
- The search, ask, list_sources, get_message tools

---

## Summary

Telegram NotebookLM MVP attempts to turn Telegram into a smart memory; a place where the user can store their channels, chats, and forwarded messages, search within them with keyword or semantic search, separate content with rules, and ultimately ask questions of their own archive like NotebookLM.

This project is a foundation for building a larger product:

```text
Telegram Memory for AI Assistants
```
