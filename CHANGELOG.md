# Changelog

## Opt-in AI auto-tagging on forwards (2026-06-12)

تکمیل AI rules: اجرای خودکار آن‌ها روی فورواردهای جدید، به‌صورت opt-in.

### Behaviour
- `/airules on|off` (پیش‌فرض خاموش) تعیین می‌کند که قوانین AI روی هر فوروارد جدید به‌صورت خودکار اجرا شوند یا نه. در حالت روشن، به‌ازای هر آیتم یک فراخوانی LLM انجام می‌شود؛ **import انبوه کانال هیچ‌وقت auto-classify نمی‌شود** (هزینه کنترل‌شده).
- نیازمند کلید Gemini؛ در نبود کلید هنگام روشن‌کردن اطلاع داده می‌شود.

### Design
- `IngestionPipeline` پارامتر اختیاری `ai_classifier` گرفت؛ `_apply_rules` در صورت وجود آن، قوانین AI را هم اعمال می‌کند (با بلعیدن خطا). فقط مسیر Forwarded Inbox آن را wire می‌کند (و فقط وقتی کاربر opt-in کرده و کلید دارد).
- ستون `ai_autotag` روی `bot_users` با مهاجرت idempotent؛ `Repository.set_ai_autotag` و helper `_ai_classifier_for_user`.

### Tests
- `tests/test_ai_autotag.py`: اعمال AI rules فقط با classifier، بلعیدن خطای classifier، persist تنظیم، گیتینگ `_ai_classifier_for_user`، و هندلر `/airules`.

## Query collections (2026-06-12)

تکمیل Collections: حالا یک دفترچه قابل خلاصه/خروجی‌گرفتن است.

### Behaviour
- `/summarize --collection <name>` و `/export --collection <name>` همه‌ی آیتم‌های دارای هر یک از تگ‌های مجموعه را (از `items_for_tags`) خلاصه یا به Markdown خروجی می‌گیرند. اگر مجموعه وجود نداشته باشد پیام مناسب می‌دهد.

### Components
- helper خالص `NotebookBot._extract_collection(args)` (جداکردن فلگ `--collection <name>`) و `_collection_items` (resolve مجموعه به items + scope label).

### Tests
- `tests/test_collections.py`: `_extract_collection`، `/summarize --collection` (اجتماع تگ‌ها و مجموعه‌ی ناموجود)، و `/export --collection` (محتوای درست سند).

## Collections / notebooks (2026-06-12)

گروه‌بندی چند تگ زیر یک «دفترچه» (collection).

### Behaviour
- `/collection new <name>` (نام تک‌کلمه)، `/collection add <name> <tag>`، `/collection list`، `/collection remove <name>`، و `/collection show <name>` که آیتم‌های دارای هر کدام از تگ‌های مجموعه را (به‌صورت distinct، جدیدترین اول) نشان می‌دهد. همه scoped به owner.

### Components
- جداول `collections` و `collection_tags` (با index یکتای `(owner_id, name)`).
- متدهای repository: `create_collection`، `add_collection_tag`، `list_collections`، `collection_tags`، `remove_collection`، و `items_for_tags(owner_id, tags, limit)` (اجتماع distinct).

### Tests
- `tests/test_collections.py`: CRUD و افزودن تگ، ایزولاسیون per-user، `items_for_tags` (اجتماع/distinct/scoping)، و مسیرهای کامل هندلر (new/add/list/show/remove + خطاها).

## Dashboard Library panel (2026-06-12)

- داشبورد وب یک کارت «Library» گرفت که با دکمه، `/api/stats` و `/api/recent` را فراخوانی و خلاصه‌ی آرشیو (تعداد آیتم/منبع/تگ و نوع مدیا) و آخرین آیتم‌ها را نمایش می‌دهد.
- تست smoke در `tests/test_web_api.py` که وجود پنل و ارجاع به endpointها را در `INDEX_HTML` بررسی می‌کند.

## Web API: stats / recent / timeline (2026-06-12)

پاریتی داشبورد وب با قابلیت‌های جدید (لایه‌ی JSON API).

### Behaviour
- سه endpoint فقط-خواندنی `GET /api/stats`، `GET /api/recent?limit=N` و `GET /api/timeline?granularity=month|day` که آرشیو داشبورد (owner ثابت `0`) را برمی‌گردانند. مثل بقیه‌ی API با `WEB_API_TOKEN` (یا loopback در نبود توکن) محافظت می‌شوند و از همان متدهای repository و توابع خالص `recent_rows`/`build_timeline`/`archive_stats` استفاده می‌کنند.
- helper `_query_int` برای خواندن امن و clamp‌شده‌ی پارامترهای عددی query.

### Outside scope (follow-up)
- نمایش این داده‌ها در رابط HTML داشبورد (فعلاً فقط JSON API).

### Tests
- `tests/test_web_api.py`: خروجی `/api/stats`، `/api/recent` (سقف limit و ترتیب) و `/api/timeline`، و الزام auth در غیرلوکال.

## Recent items browse (2026-06-12)

مرور سریع آخرین آیتم‌ها — مکمل `/timeline` و `/digest`.

### Behaviour
- `/recent [n]` (پیش‌فرض ۱۰، حداکثر ۵۰) آخرین آیتم‌ها را با منبع، تاریخ، snippet و لینک نشان می‌دهد. ابزار MCP `list_recent` همان فهرست را می‌دهد.

### Components
- ماژول خالص `recent.py` با `recent_rows(items, *, limit, snippet_chars)` (نرمال‌سازی whitespace و کوتاه‌سازی snippet)؛ از `timeline_items` (جدیدترین اول) تغذیه می‌شود.

### Tests
- `tests/test_recent.py`: نرمال‌سازی/سقف `recent_rows`، منبع ناشناخته، هندلر (ترتیب جدید→قدیم و آرشیو خالی)، و ابزار MCP.

## Tag management (2026-06-12)

مدیریت دستی تگ‌ها (rename / merge / delete).

### Behaviour
- `/tag rename <old> -> <new>` نام تگ را عوض می‌کند؛ اگر `<new>` از قبل وجود داشته باشد، دو تگ ادغام می‌شوند (بدون خطای کلید تکراری). `/tag delete <tag>` تگ را از همه‌ی آیتم‌ها برمی‌دارد. هر دو scoped به owner.

### Components
- `Repository.rename_tag` (INSERT OR IGNORE سپس DELETE برای ادغام امن) و `Repository.delete_tag`.

### Tests
- `tests/test_tag_management.py`: rename، merge در تگ موجود، delete، ایزولاسیون per-user، و مسیرهای هندلر (rename/delete/usage/missing).

## Recent digest (2026-06-12)

«بهم بگو چی از دست دادم» — خلاصه‌ی AI از محتوای اخیر.

### Behaviour
- `/digest [days]` (پیش‌فرض ۷، بازه‌ی ۱ تا ۹۰): محتوای ثبت‌شده در N روز اخیر را با همان موتور `summarize` خلاصه می‌کند. در نبود کلید Gemini، به یک خلاصه‌ی ساده (تعداد آیتم + منابع) برمی‌گردد؛ در نبود محتوای اخیر، پیام مناسب می‌دهد.

### Components
- `Repository.recent_items(owner_id, since_date, limit)` — آیتم‌های با `message_date >= since` (جدیدترین اول).

### Tests
- `tests/test_digest.py`: فیلتر تاریخ/owner در `recent_items`، و مسیرهای هندلر (بدون محتوا، fallback بدون کلید، و استفاده از summarize با کلید).

## Archive stats (2026-06-12)

نمای کلی آرشیو با `/stats` و ابزار MCP `archive_stats`.

### Behaviour
- `/stats` تعداد آیتم‌ها، منابع، تگ‌ها، شمارش بر اساس نوع مدیا، و بازه‌ی زمانی (اولین/آخرین تاریخ) را نشان می‌دهد. ابزار MCP `archive_stats` همان خروجی را می‌دهد.

### Components
- `Repository.archive_stats(owner_id)` با aggregate queries (scoped به owner).
- ماژول خالص `stats.py` با `format_stats(stats)`.

### Tests
- `tests/test_stats.py`: قالب‌بندی (خالی/پر)، aggregate و scoping per-user در `archive_stats`، و ابزار MCP.

## Markdown export (2026-06-12)

خروجی‌گرفتن آرشیو به یک فایل Markdown قابل‌دانلود.

### Behaviour
- `/export [--source <url>] [--tag <tag>]` کل آرشیو، یک منبع یا یک تگ را به یک سند Markdown (با عنوان، منبع، لینک و متن هر آیتم) تبدیل و به‌صورت فایل به کاربر می‌فرستد.

### Components
- ماژول خالص `export.py` با `build_markdown_export(scope_label, items)`.
- `TelegramBotApi.send_document` برای آپلود فایل.
- handler `_handle_export` که سند را در یک فایل موقت می‌نویسد، می‌فرستد و پاک‌سازی می‌کند.

### Tests
- `tests/test_export.py`: ساختار Markdown و فیلدهای ناقص، و orchestration هندلر (ارسال سند با محتوای درست، و پیام آرشیو خالی).

## LLM topic labels (2026-06-12)

نام‌گذاری خوشه‌های `/topics` با LLM (در صورت وجود کلید Gemini).

### Behaviour
- `/topics` و ابزار MCP `list_topics` حالا برچسب هر خوشه را با یک فراخوانی LLM (Gemini) از روی نمونه‌متن‌های خوشه می‌سازند؛ در نبود کلید یا خطا/پاسخ خالی، به برچسب مبتنی بر پرتکرارترین واژه‌ها (`top_terms`) برمی‌گردند. خروجی ربات HTML-escape می‌شود.

### Design
- در `clustering.py`: توابع خالص `build_label_prompt` و `parse_topic_label` و `label_cluster(texts, *, generate)` با فراخوانی LLM تزریق‌شده؛ `build_topics` پارامتر اختیاری `namer` گرفت که per-cluster برچسب می‌سازد و با خطا/خالی fallback می‌کند.
- در `bot.py` و `mcp_server.py` فقط وقتی کلید Gemini باشد namer ساخته می‌شود.

### Tests
- `tests/test_clustering.py`: ساخت/پارس prompt برچسب، `label_cluster` با generate تزریق‌شده، و `build_topics` با namer (برچسب‌گذاری، و fallback روی خطا/خالی).

## Forwarded media processing (2026-06-12)

تکمیل Forwarded Inbox: مدیای فورواردشده دانلود و به متن قابل‌جستجو تبدیل می‌شود.

### Behaviour
- فایل‌های صوت/ویدیو/voice/video_note به‌صورت خودکار transcribe می‌شوند (همان `TranscriptionService`)، و عکس‌ها و اسناد PDF/تصویری با OCR (Gemini multimodal) به متن تبدیل می‌شوند. متن استخراج‌شده در inbox ذخیره، تگ‌گذاری، embed و قابل `/search`/`/ask` می‌شود (و در صورت تطبیق tag، auto-forward هم می‌شود).
- اگر کلید Gemini نباشد یا نوع مدیا پشتیبانی نشود، به کاربر اطلاع داده می‌شود و فقط ارجاع/کپشن ذخیره می‌شود.

### Components
- `TelegramBotApi.get_file` + `download_file` (و `file_base_url`) برای دانلود فایل از Bot API.
- `provider_http.gemini_extract_document` (OCR/استخراج متن چندوجهی) و سرویس نازک `ExtractionService` هم‌سطح `TranscriptionService`.
- در `bot.py`: helperهای خالص `_forward_file_ref` (انتخاب فایل، بزرگ‌ترین سایز عکس) و `_media_route` (مسیر transcribe/extract)، و هسته‌ی orchestration `_process_forwarded_media` با تزریق سرویس‌ها و download برای تست آفلاین کامل.

### Outside scope (follow-up)
- استخراج DOCX/Excel و پردازش مدیا در مسیر import کامل کانال.

### Tests
- `tests/test_inbox_media.py`: انتخاب فایل و routing، orchestration برای transcribe/extract، رد در نبود سرویس/route/دانلود، بلعیدن خطای سرویس، و `file_base_url`.

## AI-based rules (2026-06-12)

قوانین tag مبتنی بر LLM، در کنار قوانین keyword موجود.

### Behaviour
- `/rule add-ai <criterion> -> <tag>` یک قانون با معیار زبان طبیعی تعریف می‌کند؛ `/rule list` نوع هر قانون را با آیکن (📝 keyword / 🤖 ai) نشان می‌دهد.
- قوانین AI فقط هنگام `/rule apply` ارزیابی می‌شوند (یک فراخوانی LLM به‌ازای هر آیتم، پوشش همه‌ی قوانین AI). در نبود کلید Gemini نادیده گرفته می‌شوند و در خروجی اطلاع داده می‌شود. قوانین keyword مثل قبل روی هر ingest اعمال می‌شوند.
- `match_tags` حالا قوانین AI را در مسیرهای خودکار رد می‌کند.

### Design
- ماژول `rules.py` با توابع خالص `build_classify_prompt` و `parse_classified_tags` و `classify_ai_tags(text, ai_rules, *, generate)` که فراخوانی LLM را inject می‌کند تا کاملاً offline-testable بماند.
- ستون `kind` روی جدول `rules` با مهاجرت idempotent `_ensure_rule_columns`؛ `add_rule`/`list_rules` با پشتیبانی از `kind`.

### Outside scope (follow-up)
- اعمال خودکار قوانین AI روی هر ingest (فعلاً فقط `/rule apply`).

### Tests
- `tests/test_ai_rules.py`: رد قوانین AI در `match_tags`، ساخت/پارس prompt، `classify_ai_tags` با generate تزریق‌شده و short-circuit، ذخیره‌ی `kind`، و `/rule apply` با ترکیب keyword+AI (LLM جعلی) و رد AI بدون کلید.

## Timeline (2026-06-11)

نمای زمانی آرشیو — مکمل زمانیِ topic clustering.

### Behaviour
- ماژول جدید `timeline.py` (pure-Python، بدون وابستگی): `build_timeline` آیتم‌های دارای تاریخ را در bucketهای تقویمی (ماه `YYYY-MM` یا روز `YYYY-MM-DD`) گروه می‌کند و per-period شمارش/منابع/نمونه می‌دهد؛ چون تاریخ‌ها ISO 8601 هستند، bucket فقط prefix تاریخ است. تاریخ‌های نامعتبر کنار گذاشته می‌شوند.
- `Repository.timeline_items` آیتم‌های دارای `message_date` را (scoped به owner + source/tag، جدیدترین اول) برمی‌گرداند.
- دستور ربات `/timeline [--source <url>] [--tag <tag>] [--day]` (پیش‌فرض ماه) و ابزار MCP `timeline`. فیلدهای کاربر در خروجی HTML با `html.escape` فرار داده می‌شوند.
- `/help`، README و CHANGELOG به‌روز شدند.

### Tests
- `tests/test_timeline.py`: `period_key` (bucket و رد تاریخ بد)، گروه‌بندی ماه/روز و ترتیب نزولی، scoping و ترتیب `timeline_items`، و ابزار MCP `timeline`.

## Fix: HTML-escape archive forwards (2026-06-11)

- چون `send_message` با `parse_mode: HTML` ارسال می‌کند، فیلدهای کاربر-کنترل (label منبع، tagها، متن، لینک) در auto-forward و پیام تأیید inbox حالا با `html.escape` فرار داده می‌شوند. پیش‌تر وجود `<`، `>` یا `&` باعث خطای parser تلگرام و در نتیجه نرسیدن بی‌صدای آیتم به کانال آرشیو می‌شد.
- تست جدید در `tests/test_autoforward.py` که escape شدن این کاراکترها را بررسی می‌کند.

## Auto-forward to an archive channel (2026-06-11)

forward خودکار آیتم‌های tag‌خورده به یک کانال آرشیو (از follow-up های Rule Engine).

### Behaviour
- دستور `/setarchive <@channel | chat id>` کانال آرشیو کاربر را تنظیم می‌کند؛ `/setarchive off` آن را غیرفعال و `/setarchive` بدون آرگومان وضعیت فعلی را نشان می‌دهد.
- در مسیر Forwarded Inbox، بعد از ذخیره‌ی موفق، متن فوروارد با قوانین کاربر (`match_tags`) سنجیده می‌شود؛ اگر حداقل یک tag مطابقت کند و کانال آرشیو تنظیم شده باشد، آیتم با ذکر منبع، tagها، متن و لینک به کانال آرشیو فوروارد می‌شود. خطای ارسال بی‌صدا لاگ می‌شود و جریان اصلی را نمی‌شکند.

### Data
- ستون جدید `archive_chat_id` روی `bot_users` با مهاجرت idempotent `_ensure_bot_user_columns` (ALTER TABLE در صورت نبود ستون). متد `Repository.set_archive_chat`.

### Outside scope (follow-up)
- قوانین AI-based و auto-forward برای import کانال‌ها (فعلاً فقط Forwarded Inbox).

### Tests
- `tests/test_autoforward.py`: تصمیم/قالب‌بندی `_auto_forward` (ارسال هنگام وجود archive+tag، رد در نبود هرکدام، بلعیدن خطای ارسال)، چرخه‌ی `/setarchive` (set/show/clear)، و مهاجرت ستون + scoping per-user.

## Topic clustering (2026-06-11)

خوشه‌بندی موضوعی محتوای آرشیو (از follow-up های NotebookLM).

### Behaviour
- ماژول جدید `clustering.py` (pure-Python، بدون وابستگی): خوشه‌بندی greedy تک‌پاس بر اساس شباهت کسینوسی نسبت به centroidهای متحرک، و `top_terms` برای ساخت برچسب خوشه از پرتکرارترین واژه‌های معنادار (با stopword چندزبانه). چون chunkها embedding ذخیره‌شده دارند، کاملاً آفلاین کار می‌کند.
- `Repository.chunks_with_embeddings` chunkهای دارای embedding را (scoped به owner + source/tag) برمی‌گرداند و BLOB را decode می‌کند.
- دستور ربات `/topics [--source <url>] [--tag <tag>]` و ابزار MCP `list_topics`.
- `/help` و README به‌روزرسانی شدند.

### Outside scope (follow-up)
- نام‌گذاری خوشه‌ها با LLM و timeline خودکار.

### Tests
- `tests/test_clustering.py`: `top_terms`، جداسازی خوشه‌ها، سقف خوشه‌ها، رد آیتم‌های بدون embedding، برچسب/مرتب‌سازی `build_topics`، decode و scoping در `chunks_with_embeddings`، و ابزار MCP `list_topics`.

## CI — pytest + ruff (2026-06-11)

افزودن یک pipeline یکپارچه‌سازی (CI) تا کد خراب به `main` نرود؛ پیش‌تر GitHub Actions فقط deploy می‌کرد.

### CI
- workflow جدید `.github/workflows/ci.yml` روی هر push و pull_request: نصب وابستگی‌ها، سپس `ruff check` و `pytest`.
- کل سوییت (۷۳ تست) در CI اجرا می‌شود؛ `test_telegram_client` هم بدون نیاز به اجرای واقعی Telethon پاس می‌شود (importها lazy هستند).

### Lint
- پیکربندی `ruff` در `pyproject.toml` (rule set `E,F,I,UP,B`؛ `line-length=140`) و افزودن `ruff` به dev dependencies.
- رفع همه‌ی یافته‌های lint: حذف importهای بلااستفاده، مرتب‌سازی importها، `datetime.UTC`، `zip(..., strict=True)` در مسیرهای crypto/cosine، `raise ... from` در except، و annotation امن `TelegramClient` زیر `TYPE_CHECKING`.

### Run locally
```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## Phase 8 — MCP Server (2026-06-09)

فاز آخر Roadmap: یک MCP Server فقط-خواندنی تا آرشیو تلگرام کاربر به ابزارهای AI دیگر وصل شود.

### Behaviour
- ماژول جدید `mcp_server.py`: JSON-RPC 2.0 روی stdio، فقط با کتابخانه‌ی استاندارد (بدون وابستگی جدید). `handle_request` تابع خالص dict→dict است و `serve_stdio` یک حلقه‌ی newline-delimited نازک روی آن.
- متدهای پروتکل: `initialize` (protocolVersion، serverInfo، capabilities.tools)، `notifications/initialized` (بدون پاسخ)، `tools/list`، `tools/call`.
- ابزارها (همه read-only): `list_sources`، `list_tags`، `search_telegram_archive` (با فیلتر source/tag)، `get_message` (متن کامل یک آیتم با `media_item_id`)، `ask_telegram_notebook` (RAG)، `summarize_source`.
- scoped به یک owner از `MCP_OWNER_ID` (پیش‌فرض `0` = آرشیو وب). همه‌ی کوئری‌ها از ایزولاسیون `owner_id` عبور می‌کنند.
- اجرا: `python -m telegram_notebook.mcp_server`.

### Repository
- متد جدید `get_media_item(owner_id, media_item_id)` برای ابزار `get_message`.

### Tests
- `tests/test_mcp_server.py`: initialize/tools-list، رفتار notification، خطای method ناشناخته، list_sources/search/get_message، ابزار ناشناخته (isError)، ایزولاسیون per-owner، و roundtrip کامل `serve_stdio`.

## Phase 7 — Summaries / NotebookLM (2026-06-09)

خلاصه‌سازی آرشیو از Roadmap (summary per source و per tag).

### Behaviour
- `/summarize [--source <url>] [--tag <tag>]` — اگر فیلتری ندهید کل آرشیو، با `--source` یک منبع، و با `--tag` یک تگ خلاصه می‌شود (از همان پارسر `_split_filters`).
- محتوا (یک ردیف به‌ازای هر آیتم، با متن و منبع) از `Repository.summary_items` گرفته می‌شود (scoped به owner + source/tag، با محدودیت پیش‌فرض ۲۰۰ آیتم).
- خلاصه با `SearchService.summarize` ساخته می‌شود؛ prompt در `_build_summary_prompt` (تابع خالص) با ذکر منابع و کوتاه‌سازی متن هر آیتم تولید و به `gemini_generate_content` داده می‌شود.

### Outside scope (follow-up)
- topic clustering و timeline خودکار.

### Tests
- `tests/test_summarize.py`: ساخت prompt (شامل منابع و scope، کوتاه‌سازی متن)، پیام خالی، و scoping متد `summary_items` (کل/تگ/منبع و ایزولاسیون per-user).

## Phase 6 — Full Import Jobs (2026-06-09)

import کامل کانال از Roadmap: صف، background worker، progress tracking، resume بعد از قطع‌شدن، و لغو.

### Data model
- جدول `jobs` (`owner_id`, `channel_url`, `status`, `total`, `processed`, `cursor`, `limit_count`, `error`, `cancel_requested`, زمان‌ها). status یکی از `queued|running|done|failed|cancelled`.
- متدهای repository: `create_job`, `get_job`, `list_jobs`, `claim_next_queued_job` (انتخاب اتمیک قدیمی‌ترین job و انتقال به running)، `update_job_progress`, `finish_job`, `request_job_cancel`, `is_cancel_requested`, و `requeue_running_jobs` (بازگرداندن jobهای running جامانده از worker کرش‌کرده به queued).

### Worker
- ماژول جدید `jobs.py` با `JobWorker` (یک thread دیمن). از Telegram جدا و با یک `runner` تزریق‌شده کار می‌کند تا state machine به‌طور کامل unit-testable باشد.
- روی استارت، jobهای running جامانده را requeue می‌کند (resume بعد از کرش).

### Pipeline
- `ingest_channel` پارامترهای `resume_from` (min_id برای ادامه)، `progress_cb(processed, total, last_msg_id)` و `should_cancel()` گرفت. به‌ازای هر پیام، cancel چک و progress/cursor به‌روزرسانی می‌شود. به‌خاطر idempotent بودن ذخیره‌سازی، resume امن است.
- `iter_all_messages` پارامتر `min_id` گرفت و `limit` حالا اختیاری (`None` = همه‌ی پیام‌ها).

### Bot
- `/import <channel_url> [limit]` (صف‌کردن import کامل/resumable)، `/jobs` (وضعیت و پیشرفت)، `/canceljob <id>`.
- `/ingest` به‌عنوان مسیر سریع inline باقی می‌ماند. worker در `run_forever` استارت می‌شود و در پایان هر job پیام done/failed/cancelled به کاربر می‌فرستد.
- `/help` به‌روزرسانی شد.

### Tests
- `tests/test_jobs.py`: چرخه‌ی حیات job، claim اتمیک و ترتیب، progress/cancel/requeue، و state machine worker با runner جعلی (done/failed/cancelled و انتقال cursor برای resume).

## Phase 5 — Rules + Tags (2026-06-09)

موتور قانون (Rule Engine) و سیستم تگ از Roadmap. کاربر قانون keyword→tag تعریف می‌کند و محتوای واردشده به‌صورت خودکار تگ می‌خورد و قابل فیلتر در جستجو/پرسش می‌شود.

### Data model
- جدول `rules` (`owner_id`, `keyword`, `tag`, `created_at`) با ایندکس یکتای `(owner_id, keyword, tag)`.
- جدول `content_tags` (`owner_id`, `media_item_id`, `tag`) با primary key ترکیبی (تگ‌گذاری idempotent).
- هر دو با `CREATE TABLE IF NOT EXISTS` ساخته می‌شوند؛ برای دیتابیس‌های موجود نیازی به مهاجرت خاص نیست.

### Matching & auto-tagging
- ماژول جدید `rules.py` با تابع خالص `match_tags(text, rules)` (substring، case-insensitive).
- pipeline در هر سه مسیر ingest (متن کانال، transcript مدیا، Forwarded Inbox) بعد از ذخیره‌ی متن، قوانین owner را اعمال و تگ‌ها را ذخیره می‌کند (`_apply_rules`). `owner_id` به helperهای داخلی pipeline اضافه شد.

### Bot commands
- `/rule add <keyword> -> <tag>`، `/rule list`، `/rule remove <id>`، و `/rule apply` (پاک‌کردن و بازمحاسبه‌ی تگ‌ها از روی متن‌های ذخیره‌شده).
- `/tags` — تگ‌ها و تعداد آیتم متمایز هر تگ.
- فیلتر `--tag <tag>` برای `/search` و `/ask`. پارسر `_split_source` با `_split_filters` جایگزین شد که هم `--source` (تک‌توکن) و هم `--tag` (تا انتهای خط، چندکلمه‌ای) را می‌فهمد.
- `/help` به‌روزرسانی شد.

### Search
- `SearchService.search` پارامتر `tag` گرفت. مسیر keyword با join روی `content_tags` فیلتر می‌شود؛ مسیر معنایی (Vertex) با allowlist از `media_ids_for_tag` پس‌فیلتر می‌شود.

### Tests
- `tests/test_rules.py`: تطبیق خالص، پارس `/rule add`، CRUD قوانین و یکتایی، ذخیره/شمارش تگ، تگ‌گذاری خودکار هنگام ingest، جستجوی فیلترشده با تگ، و backfill.
- `tests/test_normalize.py`: تست `_split_filters` (به‌جای `_split_source`).

## Phase 4 — Forwarded Inbox (MVP) (2026-06-09)

پیاده‌سازی فاز بعدی Roadmap: «Smart Telegram Inbox». حالا کاربر می‌تواند هر پیامی را به ربات فوروارد کند و متن/کپشن آن در یک inbox شخصی و قابل‌جستجو ذخیره می‌شود.

### Behaviour
- ربات پیام‌های فورواردشده را تشخیص می‌دهد (هم فرمت جدید `forward_origin` و هم فیلدهای legacy مثل `forward_from`/`forward_from_chat`/`forward_sender_name`) و قبل از منطق auth-flow مسیر می‌دهد، بنابراین با پاسخ‌های متنی فرایند `/connect` تداخل ندارد.
- متن (`text`) یا `caption` فوروارد، همراه با یک تگ نوع مدیا (مثلاً `[Forwarded document: report.pdf]`) و منبع (نام کانال/کاربر مبدأ) ذخیره می‌شود.
- وقتی مبدأ یک کانال عمومی باشد، لینک `https://t.me/<username>/<id>` به‌عنوان منبع ساخته می‌شود.
- محتوای ذخیره‌شده از طریق همان `/search` و `/ask` قابل پرس‌وجوست (chunk + embedding، با fallback به keyword اگر embedding در دسترس نباشد).

### Data model
- inbox به‌صورت یک «کانال» مصنوعی per-user با `channel_url = inbox://forwarded` پیاده شده تا از schema و مسیر جستجوی موجود (و ایزولاسیون `owner_id` فاز ۲) دوباره استفاده شود.
- متد جدید `IngestionPipeline.ingest_forwarded_message` (idempotent بر اساس message_id فوروارد).

### Bot UX
- `/start` و `/help` به‌روزرسانی شدند تا قابلیت فوروارد را توضیح دهند.
- پیام راهنما برای آیتم‌های فقط-مدیا بدون متن (که در این نسخه هنوز ایندکس نمی‌شوند).
- refactor: ساخت Vertex-config مربوط به ایندکس در یک helper مشترک (`_vertex_ingest_config`) جمع شد تا `/ingest` و inbox از آن استفاده کنند.

### خارج از scope (follow-up)
- دانلود و transcription مدیای فورواردشده از طریق Bot API، OCR عکس‌ها، و استخراج متن از PDF/DOCX/Excel.

### Tests
- `tests/test_forwarded.py`: تشخیص فوروارد، استخراج منبع/لینک/تگ مدیا، و ingest end-to-end (ذخیره و جستجوپذیری، idempotency، و per-user بودن inbox).

## Phase 3 — Web API Auth & Secret Encryption (2026-06-09)

دو مورد امنیتی باقی‌مانده از تحلیل: احراز هویت Web API و رمزنگاری secrets در دیتابیس.

### Web API authentication
- متغیر جدید `WEB_API_TOKEN`. وقتی تنظیم شده باشد، همه‌ی endpointهای `/api/*` (به‌جز `/api/health`) به توکن نیاز دارند؛ توکن از طریق `Authorization: Bearer <token>` یا هدر `X-API-Token` ارسال می‌شود (مقایسه‌ی constant-time).
- وقتی توکن تنظیم نشده باشد، API فقط درخواست‌های loopback (localhost) را می‌پذیرد و دسترسی شبکه‌ای بدون احراز هویت با ۴۰۱ رد می‌شود (secure-by-default؛ پیش‌تر کاملاً باز بود).
- `/api/health` برای healthcheck داکر عمومی می‌ماند.
- UI داشبورد: همه‌ی فراخوانی‌ها از `fetchJson` عبور می‌کنند؛ این تابع توکن را از `localStorage` می‌فرستد و در پاسخ ۴۰۱ یک‌بار از کاربر توکن می‌پرسد و ذخیره می‌کند.

### Secret encryption at rest
- ماژول جدید `crypto.py`: رمزنگاری authenticated فقط با کتابخانه‌ی استاندارد (جداسازی کلید با HKDF-SHA256، keystream با HMAC-SHA256 در حالت CTR، و Encrypt-then-MAC با HMAC-SHA256؛ nonce تصادفی ۱۲۸ بیتی برای هر مقدار). بدون هیچ وابستگی جدید.
- ستون‌های حساس قبل از ذخیره در SQLite رمز می‌شوند: در `bot_users` → `api_hash`, `session_string`, `gemini_api_key`؛ در `auth_flows` → `api_hash`, `session_string`, `phone_code_hash`. خواندن (`get_bot_user`/`get_auth_flow`) به‌صورت شفاف رمزگشایی می‌کند.
- کلید از `SECRETS_KEY` خوانده می‌شود. اگر تنظیم نشده باشد، رمزنگاری no-op است (با هشدار) و دیتابیس‌های plaintext قدیمی همچنان کار می‌کنند؛ مقادیر رمزشده با پیشوند `enc::` از plaintext قدیمی تفکیک می‌شوند تا مهاجرت بدون دردسر باشد.

### Tests
- `tests/test_crypto.py`: roundtrip، non-determinism، رد tampering/کلید اشتباه، passthrough برای None/خالی/plaintext قدیمی، و رفتار no-op بدون کلید.
- `tests/test_web_auth.py`: پذیرش bearer/`X-API-Token`، رد توکن غلط/نبود توکن، و محدودیت loopback وقتی توکن تنظیم نشده.
- `tests/test_db.py`: تست‌های جدید برای ذخیره‌ی رمزشده‌ی secrets و رمزگشایی شفاف هنگام خواندن.

### .env.example
- افزوده‌شدن `WEB_API_TOKEN` و `SECRETS_KEY` همراه با دستور تولید مقدار.

## Phase 2 — Per-user Data Isolation (2026-06-09)

تمرکز این فاز روی رفع نشت داده بین کاربران است: تا پیش از این `/search` و `/ask` (و Web API) روی **همه‌ی** کانال‌های دیتابیس کار می‌کردند و کاربران دیتای یکدیگر را می‌دیدند.

### Data model
- ستون `owner_id` به جدول `channels` اضافه شد و مالکیت در همین سطح اعمال می‌شود؛ چون هر `message`/`media_item`/`chunk` از طریق FK به یک کانال وصل است، فیلتر روی `channels.owner_id` در joinها داده را به‌طور کامل ایزوله می‌کند.
- قید سراسری `UNIQUE(channel_url)` با ایندکس ترکیبی `UNIQUE(owner_id, channel_url)` جایگزین شد تا دو کاربر بتوانند مستقل از هم یک کانال یکسان را ingest کنند بدون اینکه ردیف مشترک شوند.
- مهاجرت خودکار (`Repository._ensure_channel_owner`) برای دیتابیس‌های قدیمی: جدول `channels` بازسازی می‌شود، ستون `owner_id` اضافه می‌گردد و ردیف‌های legacy با `owner_id = NULL` می‌مانند؛ یعنی به‌جای نشت بین کاربران، برای کوئری‌های per-user نامرئی می‌شوند (در صورت نیاز باید دوباره ingest شوند).

### Scope enforcement
- متدهای repository که داده برمی‌گردانند یا حذف می‌کنند حالا `owner_id` می‌گیرند: `upsert_channel`, `keyword_candidates`, `embedding_candidates`, `list_channels`, `delete_channel_data`, `get_chunk_by_media_and_index`.
- `SearchService.search` و `IngestionPipeline.ingest_channel` پارامتر `owner_id` می‌گیرند.
- ربات تلگرام `bot_user_id` کاربر را به‌عنوان `owner_id` پاس می‌دهد؛ بنابراین `/search`, `/ask`, `/ingest`, `/sources`, `/delete`, `/status` فقط روی دیتای همان کاربر کار می‌کنند.
- داشبورد وب (که login per-user ندارد) از یک `WEB_OWNER_ID = 0` ثابت استفاده می‌کند تا آرشیو آن از آرشیو کاربران ربات جدا بماند.

### Hardening
- `LIMIT` در `keyword_candidates` به‌جای string-interpolation حالا با پارامتر bind می‌شود.

### Tests
- تست‌های `Repository` برای پاس‌دادن `owner_id` به‌روزرسانی شدند.
- تست جدید `test_data_is_isolated_per_owner`: دو کاربر با یک URL یکسان دیتای هم را نمی‌بینند و حذف یکی روی دیگری اثر ندارد.
- تست جدید `test_migrates_legacy_channels_table_without_owner_id`: مهاجرت دیتابیس قدیمی بدون `owner_id`.

## Phase 1 — Stabilize Core (2026-06-08)

تمرکز این فاز طبق Roadmap داخل README روی پایدارسازی هسته است: امنیت، رفع باگ، دستورهای ربات، logging و تست.

### Security
- توکن واقعی ربات از `.env.example` حذف و خالی شد.
  - ⚠️ این توکن قبلاً در history گیت ثبت شده (commit `5501fda`) و عملاً عمومی است. فقط خالی‌کردن فایل کافی نیست؛ باید همین حالا در **@BotFather** با `/revoke` توکن را باطل و توکن جدید بسازید.
- شناسه‌های مخصوص محیط (`VERTEX_INDEX_ID`، `VERTEX_DEPLOYED_INDEX_ID`) در فایل نمونه خالی شدند.

### Bug fixes
- `/search` و `/ask`: کاربر با `bot_user_id` واقعی خوانده می‌شود، نه `chat_id` (در گروه‌ها این دو فرق دارند).
- Web API: `/api/search` و `/api/ask` حالا `vertex_config` (و `project_id`/`region` برای ask) را پاس می‌دهند؛ پیش‌تر همیشه به keyword search سقوط می‌کردند.
- پاسخ `/ask` در ربات از `<b>` (HTML) استفاده می‌کند تا با `parse_mode=HTML` درست نمایش داده شود (قبلاً `**` خام بود).
- پیش‌فرض `DB_PATH` در `.env.example` با `config.py` یکی شد: `data/store.db`.

### دستورهای جدید ربات
- `/status` — وضعیت اتصال، کلید AI، پیکربندی Vertex و تعداد منابع ایندکس‌شده.
- `/disconnect` — حذف session و credentialهای کاربر («delete my data»).
- `/help` — فهرست دستورها.
- دستورها دیگر با پسوند `@botname` و حروف بزرگ/کوچک مشکل ندارند و دیگر به‌اشتباه وارد flow اتصال نمی‌شوند.
- guard برای ورودی خالی در `/search`، `/ask`، `/ingest`، `/join`، `/delete`.

### Logging و error handling
- ماژول جدید `logging_config.py` با `setup_logging()` (سطح از `LOG_LEVEL`، پیش‌فرض INFO).
- همه‌ی `print()`های دیباگ با `logging` جایگزین شدند؛ مقادیر حساس (شماره تلفن، کد ورود، `phone_code_hash`) دیگر لاگ نمی‌شوند.
- یک update خراب دیگر کل polling ربات را متوقف نمی‌کند (لاگ می‌شود و ادامه می‌دهد).

### Tests
- مجموعه `tests/` با pytest؛ ۲۶ تست بدون نیاز به شبکه: `chunking`، cosine similarity، `normalize_phone`/`normalize_code`، canonical URL، ترکیب متن، sanitize نام کانال، `Repository` روی SQLite موقت، و `upsert_env_values`.
- اجرا: `pip install -e ".[dev]"` سپس `pytest`.

### Follow-ups (برای فازهای بعد)
- `normalize_phone` برای شماره‌های دارای کد کشور هنوز خام است (مثلاً `09123456789` → `+09123456789`).
- `import re` در `bot.py` بعد از حذف regexها بلااستفاده مانده و قابل پاک‌کردن است.
- `main.py` هنوز state سراسری را هنگام import می‌سازد؛ بهتر است lazy شود.
- جداسازی per-user داده‌ها (Phase 2) هنوز انجام نشده: `/search` و `/ask` روی همه‌ی کانال‌ها کار می‌کنند، نه فقط دیتای همان کاربر.
