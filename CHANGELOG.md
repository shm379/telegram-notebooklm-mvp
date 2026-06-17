# Changelog

**Languages / زبان‌ها:** [English](CHANGELOG.md) · [فارسی](CHANGELOG.fa.md) · [العربية](CHANGELOG.ar.md) · [Español](CHANGELOG.es.md) · [简体中文](CHANGELOG.zh.md)

## DOCX / XLSX extraction for forwarded documents (2026-06-17)

Text extraction from forwarded Office documents, performed locally and without an API key.

### Behaviour
- Forwarding a `.docx` or `.xlsx` file (detected by extension or MIME type) now extracts its text and stores it in a searchable inbox. Unlike OCR/PDF, which requires Gemini, this path is **fully local** (no API key or network needed).

### Design
- A pure `office.py` module with `detect_office_kind`, `extract_docx_text`, `extract_xlsx_text`, and the `extract_office_text` dispatcher — using only the standard library (`zipfile` + `xml.etree`). Tag matching is done on the local-name so that both namespace variants (transitional/strict) are supported.
- DOCX: paragraph text (including inside tables) by joining runs. XLSX: resolving sharedStrings + inline strings + numbers, with tab/newline separators and sheet separation.
- `NotebookBot._media_route` returns the new `"office"` route, and `_process_forwarded_media` runs it without needing a service or `enabled`.

### Tests
- `tests/test_office.py`: type detection, DOCX extraction (joining runs/paragraphs), XLSX (shared/inline/number, multiple sheets, missing sharedStrings), and rejection of an unknown format.
- `tests/test_inbox_media.py`: routing of Office documents and full orchestration without any service.

## 0.3.0 — Notebook feature set (2026-06-17)

Release summary: in addition to the MVP core (ingest, transcription, search/ask, Forwarded Inbox, Rule Engine, Import Jobs, MCP), the following was added:

- Infrastructure: CI (ruff + pytest) on every push/PR.
- Organization: topic clustering (`/topics` with LLM labeling), `/timeline`, Collections (`/collection`, plus `/summarize`/`/export --collection`), tag management (`/tag rename|delete`).
- Content: forwarded media processing (transcription + OCR/PDF + local DOCX/XLSX extraction), AI rules (`/rule add-ai`) and opt-in auto-tagging (`/airules`), auto-forward to an archive channel (`/setarchive`).
- Output/review: `/digest`, `/export` (Markdown), `/stats`, `/recent`, and the web endpoints `/api/{stats,recent,timeline}` with a Library panel in the dashboard.
- New MCP tools: `list_topics`, `timeline`, `archive_stats`, `list_recent`.

No new dependencies; full test suite. Details of each item are in the entries below.

## Opt-in AI auto-tagging on forwards (2026-06-12)

Completing AI rules: running them automatically on new forwards, on an opt-in basis.

### Behaviour
- `/airules on|off` (off by default) controls whether AI rules run automatically on every new forward. When on, one LLM call is made per item; **bulk channel imports are never auto-classified** (controlled cost).
- Requires a Gemini key; if no key is present when turning it on, you are notified.

### Design
- `IngestionPipeline` gained an optional `ai_classifier` parameter; `_apply_rules` also applies AI rules when it is present (swallowing errors). Only the Forwarded Inbox path wires it up (and only when the user has opted in and has a key).
- An `ai_autotag` column on `bot_users` with an idempotent migration; `Repository.set_ai_autotag` and the `_ai_classifier_for_user` helper.

### Tests
- `tests/test_ai_autotag.py`: applying AI rules only with a classifier, swallowing classifier errors, persisting the setting, gating in `_ai_classifier_for_user`, and the `/airules` handler.

## Query collections (2026-06-12)

Completing Collections: a notebook can now be summarized or exported.

### Behaviour
- `/summarize --collection <name>` and `/export --collection <name>` summarize or export to Markdown all items that have any of the collection's tags (from `items_for_tags`). If the collection does not exist, an appropriate message is shown.

### Components
- The pure helper `NotebookBot._extract_collection(args)` (parsing the `--collection <name>` flag) and `_collection_items` (resolving a collection to items + a scope label).

### Tests
- `tests/test_collections.py`: `_extract_collection`, `/summarize --collection` (union of tags and a missing collection), and `/export --collection` (correct document content).

## Collections / notebooks (2026-06-12)

Grouping multiple tags under a single "notebook" (collection).

### Behaviour
- `/collection new <name>` (single-word name), `/collection add <name> <tag>`, `/collection list`, `/collection remove <name>`, and `/collection show <name>`, which shows the items that have any of the collection's tags (distinct, newest first). All scoped to the owner.

### Components
- The `collections` and `collection_tags` tables (with a unique index on `(owner_id, name)`).
- Repository methods: `create_collection`, `add_collection_tag`, `list_collections`, `collection_tags`, `remove_collection`, and `items_for_tags(owner_id, tags, limit)` (distinct union).

### Tests
- `tests/test_collections.py`: CRUD and tag addition, per-user isolation, `items_for_tags` (union/distinct/scoping), and the full handler paths (new/add/list/show/remove + errors).

## Dashboard Library panel (2026-06-12)

- The web dashboard gained a "Library" card that, on a button click, calls `/api/stats` and `/api/recent` and displays an archive summary (item/source/tag counts and media types) and the latest items.
- A smoke test in `tests/test_web_api.py` that checks for the presence of the panel and references to the endpoints in `INDEX_HTML`.

## Web API: stats / recent / timeline (2026-06-12)

Web dashboard parity with the new capabilities (a JSON API layer).

### Behaviour
- Three read-only endpoints `GET /api/stats`, `GET /api/recent?limit=N`, and `GET /api/timeline?granularity=month|day` that return the dashboard archive (fixed owner `0`). Like the rest of the API, they are protected with `WEB_API_TOKEN` (or loopback when no token is set) and use the same repository methods and pure functions `recent_rows`/`build_timeline`/`archive_stats`.
- The `_query_int` helper for safe, clamped reading of numeric query parameters.

### Outside scope (follow-up)
- Displaying this data in the dashboard HTML interface (for now, JSON API only).

### Tests
- `tests/test_web_api.py`: output of `/api/stats`, `/api/recent` (limit cap and ordering) and `/api/timeline`, and the auth requirement when non-local.

## Recent items browse (2026-06-12)

Quick browsing of the latest items — a complement to `/timeline` and `/digest`.

### Behaviour
- `/recent [n]` (default 10, max 50) shows the latest items with source, date, snippet, and link. The MCP tool `list_recent` returns the same list.

### Components
- The pure `recent.py` module with `recent_rows(items, *, limit, snippet_chars)` (whitespace normalization and snippet truncation); fed by `timeline_items` (newest first).

### Tests
- `tests/test_recent.py`: normalization/cap of `recent_rows`, unknown source, the handler (new→old ordering and an empty archive), and the MCP tool.

## Tag management (2026-06-12)

Manual tag management (rename / merge / delete).

### Behaviour
- `/tag rename <old> -> <new>` renames a tag; if `<new>` already exists, the two tags are merged (without a duplicate-key error). `/tag delete <tag>` removes the tag from all items. Both scoped to the owner.

### Components
- `Repository.rename_tag` (INSERT OR IGNORE then DELETE for a safe merge) and `Repository.delete_tag`.

### Tests
- `tests/test_tag_management.py`: rename, merge into an existing tag, delete, per-user isolation, and the handler paths (rename/delete/usage/missing).

## Recent digest (2026-06-12)

"Tell me what I missed" — an AI summary of recent content.

### Behaviour
- `/digest [days]` (default 7, range 1 to 90): summarizes content recorded in the last N days using the same `summarize` engine. Without a Gemini key, it falls back to a simple summary (item count + sources); with no recent content, it shows an appropriate message.

### Components
- `Repository.recent_items(owner_id, since_date, limit)` — items with `message_date >= since` (newest first).

### Tests
- `tests/test_digest.py`: date/owner filtering in `recent_items`, and the handler paths (no content, fallback without a key, and using summarize with a key).

## Archive stats (2026-06-12)

An archive overview via `/stats` and the MCP tool `archive_stats`.

### Behaviour
- `/stats` shows the item, source, and tag counts, counts by media type, and the time range (first/last date). The MCP tool `archive_stats` returns the same output.

### Components
- `Repository.archive_stats(owner_id)` with aggregate queries (scoped to the owner).
- The pure `stats.py` module with `format_stats(stats)`.

### Tests
- `tests/test_stats.py`: formatting (empty/populated), aggregation and per-user scoping in `archive_stats`, and the MCP tool.

## Markdown export (2026-06-12)

Exporting the archive to a downloadable Markdown file.

### Behaviour
- `/export [--source <url>] [--tag <tag>]` converts the entire archive, a single source, or a single tag into a Markdown document (with title, source, link, and text for each item) and sends it to the user as a file.

### Components
- The pure `export.py` module with `build_markdown_export(scope_label, items)`.
- `TelegramBotApi.send_document` for uploading the file.
- The `_handle_export` handler, which writes the document to a temporary file, sends it, and cleans up.

### Tests
- `tests/test_export.py`: Markdown structure and missing fields, and the handler orchestration (sending the document with the correct content, and the empty-archive message).

## LLM topic labels (2026-06-12)

Naming `/topics` clusters with an LLM (if a Gemini key is present).

### Behaviour
- `/topics` and the MCP tool `list_topics` now build each cluster's label with one LLM call (Gemini) based on sample texts from the cluster; without a key, or on an error/empty response, they fall back to a label based on the most frequent meaningful terms (`top_terms`). The bot output is HTML-escaped.

### Design
- In `clustering.py`: the pure functions `build_label_prompt` and `parse_topic_label`, and `label_cluster(texts, *, generate)` with an injected LLM call; `build_topics` gained an optional `namer` parameter that builds a per-cluster label and falls back on error/empty.
- In `bot.py` and `mcp_server.py`, the namer is only built when a Gemini key is present.

### Tests
- `tests/test_clustering.py`: building/parsing the label prompt, `label_cluster` with an injected generate, and `build_topics` with a namer (labeling, and fallback on error/empty).

## Forwarded media processing (2026-06-12)

Completing the Forwarded Inbox: forwarded media is downloaded and converted to searchable text.

### Behaviour
- Audio/video/voice/video_note files are transcribed automatically (the same `TranscriptionService`), and photos and PDF/image documents are converted to text via OCR (Gemini multimodal). The extracted text is stored in the inbox, tagged, embedded, and made `/search`/`/ask`-able (and is also auto-forwarded if a tag matches).
- If there is no Gemini key, or the media type is not supported, the user is notified and only the reference/caption is stored.

### Components
- `TelegramBotApi.get_file` + `download_file` (and `file_base_url`) for downloading the file from the Bot API.
- `provider_http.gemini_extract_document` (multimodal OCR/text extraction) and the thin `ExtractionService`, on a par with `TranscriptionService`.
- In `bot.py`: the pure helpers `_forward_file_ref` (file selection, largest photo size) and `_media_route` (transcribe/extract route), and the orchestration core `_process_forwarded_media` with service and download injection for fully offline testing.

### Outside scope (follow-up)
- DOCX/Excel extraction and media processing in the full channel import path.

### Tests
- `tests/test_inbox_media.py`: file selection and routing, orchestration for transcribe/extract, rejection when a service/route/download is missing, swallowing service errors, and `file_base_url`.

## AI-based rules (2026-06-12)

LLM-based tag rules, alongside the existing keyword rules.

### Behaviour
- `/rule add-ai <criterion> -> <tag>` defines a rule with a natural-language criterion; `/rule list` shows each rule's type with an icon (📝 keyword / 🤖 ai).
- AI rules are evaluated only during `/rule apply` (one LLM call per item, covering all AI rules). Without a Gemini key they are ignored, and this is reported in the output. Keyword rules are applied on every ingest as before.
- `match_tags` now skips AI rules in the automatic paths.

### Design
- The `rules.py` module with the pure functions `build_classify_prompt` and `parse_classified_tags`, and `classify_ai_tags(text, ai_rules, *, generate)`, which injects the LLM call so it stays fully offline-testable.
- A `kind` column on the `rules` table with the idempotent migration `_ensure_rule_columns`; `add_rule`/`list_rules` with `kind` support.

### Outside scope (follow-up)
- Automatically applying AI rules on every ingest (for now, only `/rule apply`).

### Tests
- `tests/test_ai_rules.py`: skipping AI rules in `match_tags`, building/parsing the prompt, `classify_ai_tags` with an injected generate and short-circuit, persisting `kind`, and `/rule apply` with a keyword+AI combination (fake LLM) and skipping AI without a key.

## Timeline (2026-06-11)

A temporal view of the archive — the temporal complement to topic clustering.

### Behaviour
- A new `timeline.py` module (pure Python, no dependencies): `build_timeline` groups dated items into calendar buckets (month `YYYY-MM` or day `YYYY-MM-DD`) and provides per-period counts/sources/sample; because dates are ISO 8601, the bucket is simply a date prefix. Invalid dates are discarded.
- `Repository.timeline_items` returns the items that have a `message_date` (scoped to owner + source/tag, newest first).
- The bot command `/timeline [--source <url>] [--tag <tag>] [--day]` (month by default) and the MCP tool `timeline`. User fields in the output are escaped with `html.escape`.
- `/help`, README, and CHANGELOG were updated.

### Tests
- `tests/test_timeline.py`: `period_key` (bucket and rejection of a bad date), month/day grouping and descending ordering, scoping and ordering of `timeline_items`, and the MCP tool `timeline`.

## Fix: HTML-escape archive forwards (2026-06-11)

- Because `send_message` sends with `parse_mode: HTML`, user-controlled fields (source label, tags, text, link) in auto-forwards and the inbox confirmation message are now escaped with `html.escape`. Previously, the presence of `<`, `>`, or `&` caused a Telegram parser error and, as a result, the item would silently fail to reach the archive channel.
- A new test in `tests/test_autoforward.py` that checks that these characters are escaped.

## Auto-forward to an archive channel (2026-06-11)

Automatic forwarding of tagged items to an archive channel (one of the Rule Engine follow-ups).

### Behaviour
- The `/setarchive <@channel | chat id>` command sets the user's archive channel; `/setarchive off` disables it, and `/setarchive` with no argument shows the current status.
- In the Forwarded Inbox path, after a successful save, the forwarded text is checked against the user's rules (`match_tags`); if at least one tag matches and an archive channel is set, the item is forwarded to the archive channel with its source, tags, text, and link. A send error is logged silently and does not break the main flow.

### Data
- A new `archive_chat_id` column on `bot_users` with the idempotent migration `_ensure_bot_user_columns` (ALTER TABLE if the column is missing). The `Repository.set_archive_chat` method.

### Outside scope (follow-up)
- AI-based rules and auto-forward for channel imports (for now, Forwarded Inbox only).

### Tests
- `tests/test_autoforward.py`: the decision/formatting of `_auto_forward` (sending when archive+tag are present, skipping when either is missing, swallowing send errors), the `/setarchive` cycle (set/show/clear), and the column migration + per-user scoping.

## Topic clustering (2026-06-11)

Topic clustering of archive content (one of the NotebookLM follow-ups).

### Behaviour
- A new `clustering.py` module (pure Python, no dependencies): single-pass greedy clustering based on cosine similarity to moving centroids, and `top_terms` for building a cluster label from the most frequent meaningful terms (with a multilingual stopword list). Because chunks have stored embeddings, it works fully offline.
- `Repository.chunks_with_embeddings` returns chunks that have an embedding (scoped to owner + source/tag) and decodes the BLOB.
- The bot command `/topics [--source <url>] [--tag <tag>]` and the MCP tool `list_topics`.
- `/help` and README were updated.

### Outside scope (follow-up)
- Naming clusters with an LLM and an automatic timeline.

### Tests
- `tests/test_clustering.py`: `top_terms`, cluster separation, the cluster cap, rejection of items without an embedding, `build_topics` labeling/ordering, decode and scoping in `chunks_with_embeddings`, and the MCP tool `list_topics`.

## CI — pytest + ruff (2026-06-11)

Adding a continuous integration (CI) pipeline so that broken code doesn't reach `main`; previously, GitHub Actions only handled deploys.

### CI
- A new workflow `.github/workflows/ci.yml` on every push and pull_request: install dependencies, then `ruff check` and `pytest`.
- The full suite (73 tests) runs in CI; `test_telegram_client` also passes without actually running Telethon (imports are lazy).

### Lint
- `ruff` configuration in `pyproject.toml` (rule set `E,F,I,UP,B`; `line-length=140`) and adding `ruff` to dev dependencies.
- Fixing all lint findings: removing unused imports, sorting imports, `datetime.UTC`, `zip(..., strict=True)` in the crypto/cosine paths, `raise ... from` in except blocks, and a safe `TelegramClient` annotation under `TYPE_CHECKING`.

### Run locally
```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## Phase 8 — MCP Server (2026-06-09)

The final phase of the Roadmap: a read-only MCP Server so that the user's Telegram archive can be connected to other AI tools.

### Behaviour
- A new `mcp_server.py` module: JSON-RPC 2.0 over stdio, using only the standard library (no new dependencies). `handle_request` is a pure dict→dict function, and `serve_stdio` is a thin newline-delimited loop on top of it.
- Protocol methods: `initialize` (protocolVersion, serverInfo, capabilities.tools), `notifications/initialized` (no response), `tools/list`, `tools/call`.
- Tools (all read-only): `list_sources`, `list_tags`, `search_telegram_archive` (with a source/tag filter), `get_message` (the full text of an item by `media_item_id`), `ask_telegram_notebook` (RAG), `summarize_source`.
- Scoped to a single owner from `MCP_OWNER_ID` (default `0` = the web archive). All queries pass through `owner_id` isolation.
- Run: `python -m telegram_notebook.mcp_server`.

### Repository
- A new method `get_media_item(owner_id, media_item_id)` for the `get_message` tool.

### Tests
- `tests/test_mcp_server.py`: initialize/tools-list, notification behaviour, unknown-method error, list_sources/search/get_message, an unknown tool (isError), per-owner isolation, and a full `serve_stdio` roundtrip.

## Phase 7 — Summaries / NotebookLM (2026-06-09)

Archive summarization from the Roadmap (summary per source and per tag).

### Behaviour
- `/summarize [--source <url>] [--tag <tag>]` — with no filter, the entire archive is summarized; with `--source`, a single source; and with `--tag`, a single tag (using the same `_split_filters` parser).
- The content (one row per item, with text and source) is fetched from `Repository.summary_items` (scoped to owner + source/tag, with a default limit of 200 items).
- The summary is built with `SearchService.summarize`; the prompt is produced in `_build_summary_prompt` (a pure function) with sources noted and each item's text truncated, and is passed to `gemini_generate_content`.

### Outside scope (follow-up)
- Topic clustering and an automatic timeline.

### Tests
- `tests/test_summarize.py`: building the prompt (including sources and scope, text truncation), the empty message, and the scoping of the `summary_items` method (all/tag/source and per-user isolation).

## Phase 6 — Full Import Jobs (2026-06-09)

Full channel import from the Roadmap: a queue, a background worker, progress tracking, resume after interruption, and cancellation.

### Data model
- The `jobs` table (`owner_id`, `channel_url`, `status`, `total`, `processed`, `cursor`, `limit_count`, `error`, `cancel_requested`, timestamps). status is one of `queued|running|done|failed|cancelled`.
- Repository methods: `create_job`, `get_job`, `list_jobs`, `claim_next_queued_job` (atomically selecting the oldest job and moving it to running), `update_job_progress`, `finish_job`, `request_job_cancel`, `is_cancel_requested`, and `requeue_running_jobs` (returning running jobs orphaned by a crashed worker back to queued).

### Worker
- A new `jobs.py` module with `JobWorker` (a single daemon thread). It is decoupled from Telegram and works with an injected `runner` so that the state machine is fully unit-testable.
- On startup, it requeues orphaned running jobs (resume after a crash).

### Pipeline
- `ingest_channel` gained the parameters `resume_from` (min_id for continuation), `progress_cb(processed, total, last_msg_id)`, and `should_cancel()`. For each message, cancellation is checked and progress/cursor is updated. Because storage is idempotent, resume is safe.
- `iter_all_messages` gained a `min_id` parameter, and `limit` is now optional (`None` = all messages).

### Bot
- `/import <channel_url> [limit]` (queueing a full/resumable import), `/jobs` (status and progress), `/canceljob <id>`.
- `/ingest` remains the fast inline path. The worker is started in `run_forever` and, at the end of each job, sends a done/failed/cancelled message to the user.
- `/help` was updated.

### Tests
- `tests/test_jobs.py`: the job lifecycle, atomic claim and ordering, progress/cancel/requeue, and the worker state machine with a fake runner (done/failed/cancelled and cursor advancement for resume).

## Phase 5 — Rules + Tags (2026-06-09)

The Rule Engine and tag system from the Roadmap. The user defines a keyword→tag rule and incoming content is tagged automatically and can be filtered in search/ask.

### Data model
- The `rules` table (`owner_id`, `keyword`, `tag`, `created_at`) with a unique index on `(owner_id, keyword, tag)`.
- The `content_tags` table (`owner_id`, `media_item_id`, `tag`) with a composite primary key (idempotent tagging).
- Both are created with `CREATE TABLE IF NOT EXISTS`; no special migration is needed for existing databases.

### Matching & auto-tagging
- A new `rules.py` module with the pure function `match_tags(text, rules)` (substring, case-insensitive).
- In all three ingest paths (channel text, media transcript, Forwarded Inbox), after storing the text the pipeline applies the owner's rules and stores the tags (`_apply_rules`). `owner_id` was added to the pipeline's internal helpers.

### Bot commands
- `/rule add <keyword> -> <tag>`, `/rule list`, `/rule remove <id>`, and `/rule apply` (clearing and recomputing tags from the stored texts).
- `/tags` — the tags and the distinct item count for each tag.
- A `--tag <tag>` filter for `/search` and `/ask`. The `_split_source` parser was replaced with `_split_filters`, which understands both `--source` (single token) and `--tag` (to the end of the line, multi-word).
- `/help` was updated.

### Search
- `SearchService.search` gained a `tag` parameter. The keyword path is filtered with a join on `content_tags`; the semantic path (Vertex) is post-filtered with an allowlist from `media_ids_for_tag`.

### Tests
- `tests/test_rules.py`: pure matching, parsing `/rule add`, rule CRUD and uniqueness, tag storage/counting, automatic tagging on ingest, tag-filtered search, and backfill.
- `tests/test_normalize.py`: testing `_split_filters` (instead of `_split_source`).

## Phase 4 — Forwarded Inbox (MVP) (2026-06-09)

Implementing the next Roadmap phase: the "Smart Telegram Inbox". The user can now forward any message to the bot, and its text/caption is stored in a personal, searchable inbox.

### Behaviour
- The bot detects forwarded messages (both the new `forward_origin` format and legacy fields such as `forward_from`/`forward_from_chat`/`forward_sender_name`) and routes them before the auth-flow logic, so it doesn't conflict with text replies in the `/connect` flow.
- The forward's `text` or `caption`, along with a media-type tag (e.g. `[Forwarded document: report.pdf]`) and the source (the origin channel/user name), is stored.
- When the origin is a public channel, the link `https://t.me/<username>/<id>` is built as the source.
- The stored content is queryable through the same `/search` and `/ask` (chunk + embedding, with a keyword fallback if no embedding is available).

### Data model
- The inbox is implemented as a synthetic per-user "channel" with `channel_url = inbox://forwarded`, reusing the existing schema and search path (and the Phase 2 `owner_id` isolation).
- A new method `IngestionPipeline.ingest_forwarded_message` (idempotent based on the forward's message_id).

### Bot UX
- `/start` and `/help` were updated to explain the forwarding capability.
- A guidance message for media-only items without text (which are not yet indexed in this version).
- Refactor: the index-related Vertex config was consolidated into a shared helper (`_vertex_ingest_config`) so that `/ingest` and the inbox both use it.

### Outside scope (follow-up)
- Downloading and transcribing forwarded media via the Bot API, OCR for photos, and text extraction from PDF/DOCX/Excel.

### Tests
- `tests/test_forwarded.py`: forward detection, extraction of the source/link/media tag, and end-to-end ingest (storage and searchability, idempotency, and the per-user nature of the inbox).

## Phase 3 — Web API Auth & Secret Encryption (2026-06-09)

The two remaining security items from the analysis: Web API authentication and encryption of secrets in the database.

### Web API authentication
- A new variable `WEB_API_TOKEN`. When set, all `/api/*` endpoints (except `/api/health`) require the token; the token is sent via `Authorization: Bearer <token>` or the `X-API-Token` header (constant-time comparison).
- When no token is set, the API accepts only loopback (localhost) requests, and unauthenticated network access is rejected with a 401 (secure-by-default; previously it was completely open).
- `/api/health` stays public for the Docker healthcheck.
- Dashboard UI: all calls go through `fetchJson`; this function sends the token from `localStorage` and, on a 401 response, prompts the user once for a token and stores it.

### Secret encryption at rest
- A new `crypto.py` module: authenticated encryption using only the standard library (key separation with HKDF-SHA256, a keystream with HMAC-SHA256 in CTR mode, and Encrypt-then-MAC with HMAC-SHA256; a random 128-bit nonce for each value). With no new dependencies.
- Sensitive columns are encrypted before storage in SQLite: in `bot_users` → `api_hash`, `session_string`, `gemini_api_key`; in `auth_flows` → `api_hash`, `session_string`, `phone_code_hash`. Reading (`get_bot_user`/`get_auth_flow`) decrypts transparently.
- The key is read from `SECRETS_KEY`. If it is not set, encryption is a no-op (with a warning) and old plaintext databases keep working; encrypted values are distinguished from old plaintext by the `enc::` prefix to make migration painless.

### Tests
- `tests/test_crypto.py`: roundtrip, non-determinism, rejection of tampering/wrong key, passthrough for None/empty/old plaintext, and the no-op behaviour without a key.
- `tests/test_web_auth.py`: acceptance of bearer/`X-API-Token`, rejection of a wrong/missing token, and the loopback restriction when no token is set.
- `tests/test_db.py`: new tests for encrypted storage of secrets and transparent decryption on read.

### .env.example
- Adding `WEB_API_TOKEN` and `SECRETS_KEY` along with the command to generate a value.

## Phase 2 — Per-user Data Isolation (2026-06-09)

The focus of this phase is fixing data leakage between users: previously, `/search` and `/ask` (and the Web API) operated on **all** channels in the database, and users could see each other's data.

### Data model
- An `owner_id` column was added to the `channels` table and ownership is enforced at this level; because every `message`/`media_item`/`chunk` is linked to a channel via an FK, filtering on `channels.owner_id` in joins fully isolates the data.
- The global `UNIQUE(channel_url)` constraint was replaced with a composite `UNIQUE(owner_id, channel_url)` index so that two users can independently ingest the same channel without sharing a row.
- An automatic migration (`Repository._ensure_channel_owner`) for old databases: the `channels` table is rebuilt, the `owner_id` column is added, and legacy rows are kept with `owner_id = NULL`; that is, instead of leaking between users, they become invisible to per-user queries (and must be re-ingested if needed).

### Scope enforcement
- Repository methods that return or delete data now take `owner_id`: `upsert_channel`, `keyword_candidates`, `embedding_candidates`, `list_channels`, `delete_channel_data`, `get_chunk_by_media_and_index`.
- `SearchService.search` and `IngestionPipeline.ingest_channel` take an `owner_id` parameter.
- The Telegram bot passes the user's `bot_user_id` as `owner_id`; therefore `/search`, `/ask`, `/ingest`, `/sources`, `/delete`, `/status` operate only on that user's data.
- The web dashboard (which has no per-user login) uses a fixed `WEB_OWNER_ID = 0` so its archive stays separate from the bot users' archives.

### Hardening
- The `LIMIT` in `keyword_candidates` is now bound as a parameter instead of via string interpolation.

### Tests
- The `Repository` tests were updated to pass `owner_id`.
- A new test `test_data_is_isolated_per_owner`: two users with the same URL don't see each other's data, and deleting one's has no effect on the other's.
- A new test `test_migrates_legacy_channels_table_without_owner_id`: migration of an old database without `owner_id`.

## Phase 1 — Stabilize Core (2026-06-08)

Per the Roadmap in the README, this phase focuses on stabilizing the core: security, bug fixes, bot commands, logging, and tests.

### Security
- The real bot token was removed from `.env.example` and emptied.
  - ⚠️ This token was previously committed to git history (commit `5501fda`) and is effectively public. Emptying the file is not enough; you must immediately `/revoke` the token in **@BotFather** and create a new one.
- Environment-specific identifiers (`VERTEX_INDEX_ID`, `VERTEX_DEPLOYED_INDEX_ID`) were emptied in the example file.

### Bug fixes
- `/search` and `/ask`: the user is read by the real `bot_user_id`, not `chat_id` (in groups these two differ).
- Web API: `/api/search` and `/api/ask` now pass `vertex_config` (and `project_id`/`region` for ask); previously they always fell back to keyword search.
- The `/ask` response in the bot uses `<b>` (HTML) so it renders correctly with `parse_mode=HTML` (previously it was raw `**`).
- The `DB_PATH` default in `.env.example` was aligned with `config.py`: `data/store.db`.

### New bot commands
- `/status` — connection status, AI key, Vertex configuration, and the number of indexed sources.
- `/disconnect` — deleting the user's session and credentials ("delete my data").
- `/help` — the command list.
- Commands no longer have issues with the `@botname` suffix and upper/lowercase, and no longer mistakenly enter the connection flow.
- A guard for empty input in `/search`, `/ask`, `/ingest`, `/join`, `/delete`.

### Logging and error handling
- A new `logging_config.py` module with `setup_logging()` (level from `LOG_LEVEL`, default INFO).
- All debug `print()`s were replaced with `logging`; sensitive values (phone number, login code, `phone_code_hash`) are no longer logged.
- A single broken update no longer halts the entire bot polling loop (it is logged and execution continues).

### Tests
- A `tests/` suite with pytest; 26 tests with no network needed: `chunking`, cosine similarity, `normalize_phone`/`normalize_code`, canonical URL, text composition, channel-name sanitization, `Repository` on a temporary SQLite, and `upsert_env_values`.
- Run: `pip install -e ".[dev]"` then `pytest`.

### Follow-ups (for later phases)
- `normalize_phone` is still naive for numbers with a country code (e.g. `09123456789` → `+09123456789`).
- `import re` in `bot.py` is left unused after removing the regexes and can be cleaned up.
- `main.py` still builds global state at import time; it would be better to make it lazy.
- Per-user data isolation (Phase 2) has not been done yet: `/search` and `/ask` operate on all channels, not only that user's data.
