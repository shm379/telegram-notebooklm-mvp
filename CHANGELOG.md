# Changelog

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
