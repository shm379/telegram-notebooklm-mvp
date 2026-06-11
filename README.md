# Telegram NotebookLM MVP

یک MVP برای ساخت **آرشیو هوشمند تلگرام** است؛ پروژه‌ای که بتواند محتوای کانال‌ها، چت‌ها، فایل‌ها، ویدیوها و پیام‌های فورواردشده را جمع‌آوری کند، آن‌ها را به متن قابل جستجو تبدیل کند و در نهایت مثل یک **NotebookLM داخلی برای تلگرام** به سؤال‌های کاربر پاسخ دهد.

هدف نهایی پروژه این است که کاربر بتواند محتوای تلگرام خود را به یک حافظه قابل جستجو و قابل اتصال به ابزارهای AI تبدیل کند؛ از داخل ربات تلگرام، داشبورد وب، و در آینده از طریق MCP برای اتصال به ابزارهایی مثل ChatGPT، Claude، Cursor، Codex-like agents و سایر AI clients.

---

## ایده اصلی

این پروژه سه حالت اصلی را هدف می‌گیرد:

### 1. Import Channel / Chat

کاربر لینک یا آیدی یک کانال عمومی یا چتی که به آن دسترسی دارد را می‌دهد و سیستم پیام‌ها، کپشن‌ها و مدیاهای آن را دریافت می‌کند.

نمونه:

```text
/ingest https://t.me/example_channel
```

### 2. Forwarded Inbox

کاربر می‌تواند پیام، پست، فایل، عکس، ویدیو، PDF یا هر محتوایی را به ربات فوروارد کند. سیستم آن را ذخیره، پردازش، تگ‌گذاری و قابل جستجو می‌کند.

این بخش قرار است شبیه یک **Smart Telegram Inbox** عمل کند.

### 3. AI Notebook / RAG

بعد از ذخیره و ایندکس شدن محتوا، کاربر می‌تواند از آرشیو خود سؤال بپرسد:

```text
/ask از بین پیام‌هایی که درباره Al Mouj ذخیره کردم، کدام‌ها درباره townhouse بودند؟
```

یا:

```text
/ask ابزارهای AI که در کانال‌ها درباره ساخت ویدیو معرفی شده‌اند را دسته‌بندی کن
```

پاسخ باید همراه با منبع، لینک پیام و متن‌های مرتبط باشد.

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
- **Rule Engine + Tags**: تعریف قانون keyword→tag، تگ‌گذاری خودکار محتوای واردشده، و فیلتر `/search` و `/ask` با `--tag`
- **Import Jobs**: import کامل کانال در background با صف، progress tracking، resume بعد از قطع‌شدن، و امکان لغو
- **خلاصه‌سازی (NotebookLM)**: `/summarize` برای ساخت خلاصه‌ی ساختارمند از کل آرشیو، یک منبع خاص، یا یک تگ

---

## چرا Bot API کافی نیست؟

برای گرفتن آرشیو کامل یک کانال یا چت، Bot API به‌تنهایی کافی نیست. Bot API معمولاً فقط پیام‌های جدیدی را می‌بیند که ربات به آن‌ها دسترسی دارد.

برای import کردن تاریخچه کانال‌ها و چت‌ها، این پروژه از `Telethon` و MTProto استفاده می‌کند؛ یعنی همان سطح دسترسی user account، نه فقط bot token.

---

## معماری فعلی

```text
Telegram Bot
  |
  | دستورات کاربر: /connect, /ingest, /search, /ask
  v
Python Backend
  |
  +-- Telethon Client
  |     خواندن کانال‌ها و چت‌ها
  |
  +-- Ingestion Pipeline
  |     دانلود مدیا، استخراج متن، transcription
  |
  +-- Chunking + Embedding
  |     آماده‌سازی برای semantic search
  |
  +-- Search Service
  |     keyword search + vector search
  |
  +-- RAG Answer Generator
        ساخت پاسخ از روی منابع پیدا شده
```

---

## تکنولوژی‌ها

- Python 3.11+
- Telethon
- OpenAI API
- Google Gemini / Google GenAI
- ffmpeg
- SQLite / JSON-compatible local store برای MVP
- Python lexical search + cosine similarity
- Telegram Bot API برای رابط کاربر
- Web UI سبک با `http.server`

---

## دستورات ربات

```text
/start
معرفی پروژه و شروع کار

/connect
اتصال اکانت واقعی تلگرام کاربر

/status
بررسی وضعیت اتصال

/ingest <channel_url>
ایندکس سریع و inline یک کانال

/import <channel_url> [limit]
صف‌کردن یک import کامل و resumable در background

/jobs
نمایش وضعیت و پیشرفت jobهای import

/canceljob <id>
لغو یک job در صف یا در حال اجرا

/search <query>
جستجو در آرشیو

/search <query> --source <channel_url>
جستجو فقط داخل یک منبع خاص

/search <query> --tag <tag>
جستجو فقط داخل محتوای تگ‌خورده

/ask <question>
پرسش از آرشیو با AI

/ask <question> --source <channel_url>
پرسش فقط از یک کانال یا منبع خاص

/ask <question> --tag <tag>
پرسش فقط از محتوای یک تگ خاص

/summarize [--source <url>] [--tag <tag>]
خلاصه‌سازی کل آرشیو، یک منبع، یا یک تگ

/sources
نمایش منابع ایندکس‌شده

/delete <channel_url>
حذف داده‌های یک منبع

/rule add <keyword> -> <tag>
تعریف قانون برای تگ‌گذاری خودکار محتوا

/rule list
نمایش قوانین

/rule remove <id>
حذف یک قانون

/rule apply
اعمال دوباره قوانین روی محتوای موجود (backfill)

/tags
نمایش تگ‌ها و تعداد آیتم هر تگ

/cancel
لغو flow فعلی
```

---

## APIهای اصلی

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

---

## مسیر محصول نهایی

هدف نهایی این پروژه فقط search ساده نیست. مسیر محصول به این شکل است:

```text
Telegram AI Archive
  |
  +-- Import کامل کانال‌ها و چت‌ها
  +-- Forwarded Inbox برای پیام‌های فورواردشده
  +-- Rule Engine برای جدا کردن محتواها با keyword یا AI
  +-- Tag / Folder / Collection
  +-- Search متنی و معنایی
  +-- NotebookLM داخلی برای پرسش و پاسخ
  +-- MCP Server برای اتصال به AI tools
```

---

## Rule Engine + Tags

کاربر می‌تواند قانون keyword→tag تعریف کند:

```text
/rule add Claude -> AI Tools
/rule add Al Mouj -> Real Estate
/rule add golden visa -> Oman Visa
/rule add قیمت -> Leads
```

هر محتوای جدیدی که وارد سیستم شود (ingest کانال، transcript مدیا، یا Forwarded Inbox) از نظر متن و کپشن بررسی می‌شود. اگر keyword یک rule (به‌صورت substring و case-insensitive) در متن باشد:

- tag مربوطه به آن آیتم وصل می‌شود
- بعداً با `/search ... --tag <tag>` و `/ask ... --tag <tag>` قابل فیلتر است
- `/tags` تگ‌ها و تعداد آیتم هر تگ را نشان می‌دهد
- `/rule apply` قوانین فعلی را روی محتوای موجود دوباره اعمال می‌کند (backfill)

**هنوز اضافه نشده (follow-up):** قوانین AI-based و forward خودکار آیتم‌های match‌شده به یک کانال آرشیو.

---

## MCP Roadmap

یکی از مسیرهای مهم پروژه، ساخت یک **Telegram MCP Server** است تا آرشیو تلگرام کاربر فقط داخل ربات نماند و بتواند به ابزارهای AI دیگر وصل شود.

ابزارهای پیشنهادی MCP:

```text
search_telegram_archive
جستجو در آرشیو تلگرام

ask_telegram_notebook
پرسش و پاسخ از منابع ذخیره‌شده

list_sources
نمایش کانال‌ها، چت‌ها و inboxها

list_tags
نمایش tagها و collectionها

get_message
دریافت متن کامل یک پیام یا آیتم

summarize_source
خلاصه‌سازی یک کانال، چت یا collection
```

در فاز اول، MCP باید read-only باشد. ابزارهای حساس مثل import، forward، delete یا create_rule باید بعداً با permission و confirmation اضافه شوند.

---

## نصب

```bash
git clone https://github.com/shm379/telegram-notebooklm-mvp.git
cd telegram-notebooklm-mvp

uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

روی ویندوز:

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
copy .env.example .env
```

---

## پیش‌نیازها

- Python 3.11+
- ffmpeg
- Telegram API credentials:
  - `TELEGRAM_API_ID`
  - `TELEGRAM_API_HASH`
  - `TELEGRAM_SESSION_STRING` برای اجرای production بهتر است
- `TELEGRAM_BOT_TOKEN` برای اجرای ربات
- یکی از این providerها:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`

---

## ساخت Telegram Session

```bash
export TELEGRAM_API_ID=...
export TELEGRAM_API_HASH=...
uv run python scripts/create_telegram_session.py
```

خروجی را در `.env` داخل `TELEGRAM_SESSION_STRING` بگذارید.

اگر `TELEGRAM_SESSION_STRING` نداشته باشید، پروژه از session file محلی استفاده می‌کند و اولین اجرا نیاز به login تعاملی دارد.

---

## اجرای Web UI

```bash
python -m telegram_notebook.main
```

سپس باز کنید:

```text
http://127.0.0.1:8000
```

---

## اجرای ربات تلگرام

```bash
python -m telegram_notebook.bot
```

---

## محدودیت‌های فعلی

- جداسازی داده بین کاربران انجام شده است (هر کاربر فقط دیتای خودش را می‌بیند؛ مالکیت با `owner_id` روی کانال‌ها اعمال می‌شود).
- احراز هویت Web API با `WEB_API_TOKEN` و رمزنگاری secrets در دیتابیس با `SECRETS_KEY` اضافه شده؛ برای production هر دو متغیر را تنظیم کنید.
- storage فعلی برای MVP مناسب است، نه دیتاست بزرگ.
- session string و API keyها باید قبل از production رمزنگاری شوند.
- import کامل کانال با صف، progress و resume از طریق `/import` پشتیبانی می‌شود (یک worker در background)؛ هنوز یک sandbox تست برای کل مسیر Telethon وجود ندارد.
- Forwarded Inbox فعلاً فقط متن/کپشن پیام‌های فورواردشده را ایندکس می‌کند؛ دانلود و transcription مدیا، OCR عکس و استخراج متن از PDF/DOCX هنوز اضافه نشده.
- Rule Engine بر اساس تطبیق keyword (substring) است؛ قوانین AI-based و forward خودکار به کانال آرشیو هنوز اضافه نشده.
- `/summarize` خلاصه‌ی per-source/per-tag/کل آرشیو می‌سازد؛ topic clustering و timeline خودکار هنوز اضافه نشده.
- برای دیتاست بزرگ بهتر است به PostgreSQL + pgvector یا Qdrant مهاجرت شود.

---

## نکات امنیتی مهم

- هیچ token، API key، session string یا credential واقعی را داخل repo commit نکنید.
- اگر قبلاً token واقعی داخل `.env.example` یا history پروژه commit شده، آن token را فوراً revoke/regenerate کنید.
- برای production، session کاربران باید encrypt شود.
- برای هر search یا ask باید فیلتر user_id اعمال شود.
- کاربر باید امکان `disconnect` و `delete my data` داشته باشد.
- ابزارهای MCP در ابتدا باید read-only باشند.

---

## Roadmap پیشنهادی

### Phase 1 — Stabilize Core

- پاکسازی secrets از repo
- اصلاح README و env example
- پایدارسازی `/connect`, `/ingest`, `/search`, `/ask`
- اصلاح error handling و logging

### Phase 2 — Multi-user Data Model

- اضافه کردن user_id به sources, messages, media, chunks
- جداسازی کامل دیتای کاربران
- permission و access control

### Phase 3 — Forwarded Inbox

- پردازش پیام‌های فورواردشده
- ذخیره text, caption, media, document
- OCR برای عکس‌ها
- استخراج متن از PDF/DOCX/Excel

### Phase 4 — Rules + Tags

- تعریف keyword rule
- tag و collection
- forward اتومات به کانال‌های آرشیو
- ruleهای AI-based

### Phase 5 — Full Import Jobs

- import کامل کانال از اول تا آخر
- resume بعد از قطع شدن
- progress tracking
- queue/background worker

### Phase 6 — NotebookLM داخلی

- پاسخ‌سازی بهتر با منبع
- summary per source
- summary per tag
- timeline و topic clustering

### Phase 7 — MCP Server

- read-only MCP tools
- اتصال به AI clients
- ابزارهای search, ask, list_sources, get_message

---

## خلاصه

Telegram NotebookLM MVP تلاش می‌کند تلگرام را به یک حافظه هوشمند تبدیل کند؛ جایی که کاربر بتواند کانال‌ها، چت‌ها و پیام‌های فورواردی خود را ذخیره کند، با keyword یا semantic search داخل آن‌ها بگردد، محتواها را با rule جدا کند و در نهایت مثل NotebookLM از آرشیو خودش سؤال بپرسد.

این پروژه پایه‌ای برای ساخت یک محصول بزرگ‌تر است:

```text
Telegram Memory for AI Assistants
```