# Changelog

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
