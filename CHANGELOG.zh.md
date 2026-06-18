# Changelog

**Languages / زبان‌ها:** [English](CHANGELOG.md) · [فارسی](CHANGELOG.fa.md) · [العربية](CHANGELOG.ar.md) · [Español](CHANGELOG.es.md) · [简体中文](CHANGELOG.zh.md)

## 转发文档的 DOCX / XLSX 提取（2026-06-17）

从转发的 Office 文档中提取文本，在本地执行且无需 API 密钥。

### 行为
- 转发一个 `.docx` 或 `.xlsx` 文件（通过扩展名或 MIME 类型检测）现在会提取其文本并存入可搜索的收件箱。与需要 Gemini 的 OCR/PDF 不同，这条路径是**完全本地的**（无需 API 密钥或网络）。

### 设计
- 一个纯粹的 `office.py` 模块，包含 `detect_office_kind`、`extract_docx_text`、`extract_xlsx_text` 以及 `extract_office_text` 调度器——仅使用标准库（`zipfile` + `xml.etree`）。标签匹配在本地名（local-name）上进行，以便同时支持两种命名空间变体（过渡型/严格型，transitional/strict）。
- DOCX：通过连接 run 得到段落文本（包括表格内部）。XLSX：解析 sharedStrings + 内联字符串 + 数字，带制表符/换行分隔符以及工作表分隔。
- `NotebookBot._media_route` 返回新的 `"office"` 路由，而 `_process_forwarded_media` 在无需服务或 `enabled` 的情况下运行它。

### 测试
- `tests/test_office.py`：类型检测、DOCX 提取（连接 run/段落）、XLSX（共享/内联/数字、多工作表、缺失 sharedStrings），以及对未知格式的拒绝。
- `tests/test_inbox_media.py`：Office 文档的路由以及在没有任何服务的情况下的完整编排。

## 0.3.0 — Notebook 功能集（2026-06-17）

发布摘要：在 MVP 核心（摄取、转写、搜索/提问、转发收件箱、规则引擎、导入作业、MCP）之外，新增了以下内容：

- 基础设施：在每次 push/PR 时运行 CI（ruff + pytest）。
- 组织能力：主题聚类（`/topics`，带 LLM 标注）、`/timeline`、合集（`/collection`，外加 `/summarize`/`/export --collection`）、标签管理（`/tag rename|delete`）。
- 内容：转发媒体处理（转写 + OCR/PDF + 本地 DOCX/XLSX 提取）、AI 规则（`/rule add-ai`）和选择性启用的自动打标签（`/airules`）、自动转发到归档频道（`/setarchive`）。
- 输出/审阅：`/digest`、`/export`（Markdown）、`/stats`、`/recent`，以及 Web 端点 `/api/{stats,recent,timeline}` 加上仪表板中的资料库（Library）面板。
- 新的 MCP 工具：`list_topics`、`timeline`、`archive_stats`、`list_recent`。

没有新依赖；完整测试套件。每一项的细节在下面的条目中。

## 转发上选择性启用的 AI 自动打标签（2026-06-12）

完善 AI 规则：在选择性启用（opt-in）的基础上，在新转发上自动运行它们。

### 行为
- `/airules on|off`（默认关闭）控制是否在每条新转发上自动运行 AI 规则。开启时，每个条目进行一次 LLM 调用；**批量频道导入永远不会被自动分类**（成本可控）。
- 需要 Gemini 密钥；开启时如果没有密钥，会通知你。

### 设计
- `IngestionPipeline` 新增了一个可选的 `ai_classifier` 参数；当它存在时，`_apply_rules` 也会应用 AI 规则（吞掉错误）。只有转发收件箱路径会接入它（且仅当用户已选择启用并拥有密钥时）。
- `bot_users` 上的一个 `ai_autotag` 列，带幂等迁移；`Repository.set_ai_autotag` 以及 `_ai_classifier_for_user` 辅助函数。

### 测试
- `tests/test_ai_autotag.py`：仅在有分类器时应用 AI 规则、吞掉分类器错误、持久化设置、`_ai_classifier_for_user` 中的门控，以及 `/airules` 处理器。

## 查询合集（2026-06-12）

完善合集：现在可以对一个笔记本进行摘要或导出。

### 行为
- `/summarize --collection <name>` 和 `/export --collection <name>` 会对拥有该合集任意标签的所有条目（来自 `items_for_tags`）进行摘要或导出为 Markdown。如果合集不存在，会显示相应的消息。

### 组件
- 纯辅助函数 `NotebookBot._extract_collection(args)`（解析 `--collection <name>` 标志）和 `_collection_items`（将合集解析为条目 + 一个范围标签）。

### 测试
- `tests/test_collections.py`：`_extract_collection`、`/summarize --collection`（标签的并集以及缺失的合集），以及 `/export --collection`（正确的文档内容）。

## 合集 / 笔记本（2026-06-12）

将多个标签归到一个单一的“笔记本”（合集）之下。

### 行为
- `/collection new <name>`（单词名称）、`/collection add <name> <tag>`、`/collection list`、`/collection remove <name>`，以及 `/collection show <name>`（显示拥有该合集任意标签的条目，去重，从新到旧）。全部限定于所有者范围。

### 组件
- `collections` 和 `collection_tags` 表（在 `(owner_id, name)` 上有唯一索引）。
- Repository 方法：`create_collection`、`add_collection_tag`、`list_collections`、`collection_tags`、`remove_collection`，以及 `items_for_tags(owner_id, tags, limit)`（去重并集）。

### 测试
- `tests/test_collections.py`：CRUD 和标签添加、按用户隔离、`items_for_tags`（并集/去重/范围限定），以及完整的处理器路径（new/add/list/show/remove + 错误）。

## 仪表板资料库面板（2026-06-12）

- Web 仪表板新增了一个“资料库”（Library）卡片，点击按钮后会调用 `/api/stats` 和 `/api/recent`，并显示归档摘要（条目/来源/标签计数和媒体类型）以及最新条目。
- `tests/test_web_api.py` 中的一个冒烟测试，检查面板的存在以及 `INDEX_HTML` 中对各端点的引用。

## Web API：stats / recent / timeline（2026-06-12）

Web 仪表板与新能力的对等性（一个 JSON API 层）。

### 行为
- 三个只读端点 `GET /api/stats`、`GET /api/recent?limit=N` 和 `GET /api/timeline?granularity=month|day`，返回仪表板归档（固定所有者 `0`）。与 API 的其余部分一样，它们受 `WEB_API_TOKEN` 保护（或在未设置令牌时仅限回环地址），并使用相同的 repository 方法和纯函数 `recent_rows`/`build_timeline`/`archive_stats`。
- 用于安全、限幅读取数值查询参数的 `_query_int` 辅助函数。

### 范围之外（后续）
- 在仪表板 HTML 界面中显示这些数据（目前仅有 JSON API）。

### 测试
- `tests/test_web_api.py`：`/api/stats`、`/api/recent`（limit 上限和排序）和 `/api/timeline` 的输出，以及非本地时的鉴权要求。

## 最近条目浏览（2026-06-12）

快速浏览最新条目——是对 `/timeline` 和 `/digest` 的补充。

### 行为
- `/recent [n]`（默认 10，最多 50）显示带来源、日期、片段和链接的最新条目。MCP 工具 `list_recent` 返回相同的列表。

### 组件
- 纯粹的 `recent.py` 模块，包含 `recent_rows(items, *, limit, snippet_chars)`（空白规范化和片段截断）；由 `timeline_items`（从新到旧）提供数据。

### 测试
- `tests/test_recent.py`：`recent_rows` 的规范化/限幅、未知来源、处理器（新→旧排序和空归档），以及 MCP 工具。

## 标签管理（2026-06-12）

手动标签管理（重命名 / 合并 / 删除）。

### 行为
- `/tag rename <old> -> <new>` 重命名一个标签；如果 `<new>` 已存在，两个标签会被合并（不会出现重复键错误）。`/tag delete <tag>` 将该标签从所有条目中移除。两者都限定于所有者范围。

### 组件
- `Repository.rename_tag`（INSERT OR IGNORE 然后 DELETE，用于安全合并）和 `Repository.delete_tag`。

### 测试
- `tests/test_tag_management.py`：重命名、合并到已有标签、删除、按用户隔离，以及处理器路径（rename/delete/usage/missing）。

## 近期简报（2026-06-12）

“告诉我我错过了什么”——近期内容的 AI 摘要。

### 行为
- `/digest [days]`（默认 7，范围 1 到 90）：使用相同的 `summarize` 引擎对最近 N 天记录的内容进行摘要。没有 Gemini 密钥时，回退到简单摘要（条目计数 + 来源）；没有近期内容时，显示相应的消息。

### 组件
- `Repository.recent_items(owner_id, since_date, limit)`——`message_date >= since` 的条目（从新到旧）。

### 测试
- `tests/test_digest.py`：`recent_items` 中的日期/所有者过滤，以及处理器路径（无内容、无密钥的回退，以及在有密钥时使用 summarize）。

## 归档统计（2026-06-12）

通过 `/stats` 和 MCP 工具 `archive_stats` 提供归档概览。

### 行为
- `/stats` 显示条目、来源和标签计数、按媒体类型的计数，以及时间范围（首个/最后日期）。MCP 工具 `archive_stats` 返回相同的输出。

### 组件
- `Repository.archive_stats(owner_id)`，带聚合查询（限定于所有者范围）。
- 纯粹的 `stats.py` 模块，包含 `format_stats(stats)`。

### 测试
- `tests/test_stats.py`：格式化（空/有数据）、`archive_stats` 中的聚合和按用户范围限定，以及 MCP 工具。

## Markdown 导出（2026-06-12）

将归档导出为可下载的 Markdown 文件。

### 行为
- `/export [--source <url>] [--tag <tag>]` 将整个归档、单个来源或单个标签转换为一个 Markdown 文档（每个条目带标题、来源、链接和文本），并作为文件发送给用户。

### 组件
- 纯粹的 `export.py` 模块，包含 `build_markdown_export(scope_label, items)`。
- 用于上传文件的 `TelegramBotApi.send_document`。
- `_handle_export` 处理器，它将文档写入临时文件、发送它，然后清理。

### 测试
- `tests/test_export.py`：Markdown 结构和缺失字段，以及处理器编排（以正确内容发送文档，以及空归档消息）。

## LLM 主题标签（2026-06-12）

用 LLM 为 `/topics` 聚类命名（如果存在 Gemini 密钥）。

### 行为
- `/topics` 和 MCP 工具 `list_topics` 现在会基于聚类中的样本文本，用一次 LLM 调用（Gemini）构建每个聚类的标签；没有密钥时，或在出错/空响应时，回退到基于最高频有意义词项（`top_terms`）的标签。机器人输出经过 HTML 转义。

### 设计
- 在 `clustering.py` 中：纯函数 `build_label_prompt` 和 `parse_topic_label`，以及带注入式 LLM 调用的 `label_cluster(texts, *, generate)`；`build_topics` 新增了一个可选的 `namer` 参数，用于构建每个聚类的标签并在出错/空时回退。
- 在 `bot.py` 和 `mcp_server.py` 中，仅当存在 Gemini 密钥时才构建 namer。

### 测试
- `tests/test_clustering.py`：构建/解析标签提示、带注入式 generate 的 `label_cluster`，以及带 namer 的 `build_topics`（标注，以及出错/空时的回退）。

## 转发媒体处理（2026-06-12）

完善转发收件箱：转发的媒体会被下载并转换为可搜索的文本。

### 行为
- 音频/视频/语音/video_note 文件会被自动转写（相同的 `TranscriptionService`），照片和 PDF/图像文档通过 OCR（Gemini 多模态）转换为文本。提取出的文本会存入收件箱、打标签、嵌入，并可被 `/search`/`/ask` 检索（并且在标签匹配时也会被自动转发）。
- 如果没有 Gemini 密钥，或媒体类型不受支持，会通知用户并仅存储引用/说明文字。

### 组件
- 用于从 Bot API 下载文件的 `TelegramBotApi.get_file` + `download_file`（以及 `file_base_url`）。
- `provider_http.gemini_extract_document`（多模态 OCR/文本提取）以及与 `TranscriptionService` 对等的轻量 `ExtractionService`。
- 在 `bot.py` 中：纯辅助函数 `_forward_file_ref`（文件选择，最大照片尺寸）和 `_media_route`（转写/提取路由），以及编排核心 `_process_forwarded_media`，带服务和下载注入以实现完全离线测试。

### 范围之外（后续）
- 完整频道导入路径中的 DOCX/Excel 提取和媒体处理。

### 测试
- `tests/test_inbox_media.py`：文件选择和路由、转写/提取的编排、缺少服务/路由/下载时的拒绝、吞掉服务错误，以及 `file_base_url`。

## 基于 AI 的规则（2026-06-12）

在已有关键词规则之外，新增基于 LLM 的标签规则。

### 行为
- `/rule add-ai <criterion> -> <tag>` 定义一条带自然语言判据的规则；`/rule list` 用图标显示每条规则的类型（📝 关键词 / 🤖 ai）。
- AI 规则仅在 `/rule apply` 期间被评估（每个条目一次 LLM 调用，覆盖所有 AI 规则）。没有 Gemini 密钥时它们会被忽略，并在输出中报告这一点。关键词规则照旧在每次摄取时应用。
- `match_tags` 现在会在自动路径中跳过 AI 规则。

### 设计
- `rules.py` 模块，包含纯函数 `build_classify_prompt` 和 `parse_classified_tags`，以及 `classify_ai_tags(text, ai_rules, *, generate)`，它注入 LLM 调用以保持完全可离线测试。
- `rules` 表上的一个 `kind` 列，带幂等迁移 `_ensure_rule_columns`；`add_rule`/`list_rules` 支持 `kind`。

### 范围之外（后续）
- 在每次摄取时自动应用 AI 规则（目前仅限 `/rule apply`）。

### 测试
- `tests/test_ai_rules.py`：在 `match_tags` 中跳过 AI 规则、构建/解析提示、带注入式 generate 和短路的 `classify_ai_tags`、持久化 `kind`，以及带关键词+AI 组合（伪 LLM）的 `/rule apply` 和无密钥时跳过 AI。

## 时间线（2026-06-11）

归档的时间视图——是对主题聚类在时间维度上的补充。

### 行为
- 一个新的 `timeline.py` 模块（纯 Python，无依赖）：`build_timeline` 将带日期的条目分组到日历桶中（月 `YYYY-MM` 或日 `YYYY-MM-DD`），并提供每个时段的计数/来源/样本；因为日期是 ISO 8601，桶就是简单的日期前缀。无效日期会被丢弃。
- `Repository.timeline_items` 返回拥有 `message_date` 的条目（限定于所有者 + 来源/标签，从新到旧）。
- 机器人命令 `/timeline [--source <url>] [--tag <tag>] [--day]`（默认按月）和 MCP 工具 `timeline`。输出中的用户字段用 `html.escape` 转义。
- `/help`、README 和 CHANGELOG 已更新。

### 测试
- `tests/test_timeline.py`：`period_key`（桶以及对错误日期的拒绝）、月/日分组和降序排序、`timeline_items` 的范围限定和排序，以及 MCP 工具 `timeline`。

## 修复：对归档转发进行 HTML 转义（2026-06-11）

- 因为 `send_message` 以 `parse_mode: HTML` 发送，自动转发和收件箱确认消息中受用户控制的字段（来源标签、标签、文本、链接）现在用 `html.escape` 转义。此前，`<`、`>` 或 `&` 的存在会导致 Telegram 解析器错误，结果该条目会悄然无法到达归档频道。
- `tests/test_autoforward.py` 中的一个新测试，检查这些字符被转义。

## 自动转发到归档频道（2026-06-11）

将已打标签的条目自动转发到归档频道（规则引擎后续项之一）。

### 行为
- `/setarchive <@channel | chat id>` 命令设置用户的归档频道；`/setarchive off` 禁用它，而不带参数的 `/setarchive` 显示当前状态。
- 在转发收件箱路径中，成功保存后，转发的文本会针对用户的规则进行检查（`match_tags`）；如果至少匹配一个标签且已设置归档频道，该条目会连同其来源、标签、文本和链接被转发到归档频道。发送错误会被静默记录，不会破坏主流程。

### 数据
- `bot_users` 上的一个新 `archive_chat_id` 列，带幂等迁移 `_ensure_bot_user_columns`（若列缺失则 ALTER TABLE）。`Repository.set_archive_chat` 方法。

### 范围之外（后续）
- 基于 AI 的规则以及针对频道导入的自动转发（目前仅限转发收件箱）。

### 测试
- `tests/test_autoforward.py`：`_auto_forward` 的决策/格式化（在有归档+标签时发送、在任一缺失时跳过、吞掉发送错误）、`/setarchive` 周期（设置/显示/清除），以及列迁移 + 按用户范围限定。

## 主题聚类（2026-06-11）

对归档内容进行主题聚类（NotebookLM 后续项之一）。

### 行为
- 一个新的 `clustering.py` 模块（纯 Python，无依赖）：基于与移动质心的余弦相似度的单遍贪心聚类，以及 `top_terms` 用于从最高频有意义词项构建聚类标签（带多语言停用词列表）。因为块（chunk）存有嵌入向量，它能完全离线工作。
- `Repository.chunks_with_embeddings` 返回拥有嵌入向量的块（限定于所有者 + 来源/标签）并解码 BLOB。
- 机器人命令 `/topics [--source <url>] [--tag <tag>]` 和 MCP 工具 `list_topics`。
- `/help` 和 README 已更新。

### 范围之外（后续）
- 用 LLM 为聚类命名以及自动时间线。

### 测试
- `tests/test_clustering.py`：`top_terms`、聚类分离、聚类上限、对无嵌入向量条目的拒绝、`build_topics` 标注/排序、`chunks_with_embeddings` 中的解码和范围限定，以及 MCP 工具 `list_topics`。

## CI — pytest + ruff（2026-06-11）

添加持续集成（CI）流水线，以防有问题的代码进入 `main`；此前，GitHub Actions 只处理部署。

### CI
- 一个新的工作流 `.github/workflows/ci.yml`，在每次 push 和 pull_request 时运行：安装依赖，然后 `ruff check` 和 `pytest`。
- 完整套件（73 个测试）在 CI 中运行；`test_telegram_client` 也能在不实际运行 Telethon 的情况下通过（导入是惰性的）。

### Lint
- `pyproject.toml` 中的 `ruff` 配置（规则集 `E,F,I,UP,B`；`line-length=140`），并将 `ruff` 添加到开发依赖中。
- 修复所有 lint 发现：移除未使用的导入、排序导入、`datetime.UTC`、在 crypto/cosine 路径中的 `zip(..., strict=True)`、except 块中的 `raise ... from`，以及 `TYPE_CHECKING` 下安全的 `TelegramClient` 注解。

### 本地运行
```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## 阶段 8 — MCP 服务器（2026-06-09）

路线图的最后阶段：一个只读 MCP 服务器，以便用户的 Telegram 归档可以连接到其他 AI 工具。

### 行为
- 一个新的 `mcp_server.py` 模块：通过 stdio 的 JSON-RPC 2.0，仅使用标准库（无新依赖）。`handle_request` 是一个纯 dict→dict 函数，而 `serve_stdio` 是其上的一个轻量的、以换行分隔的循环。
- 协议方法：`initialize`（protocolVersion、serverInfo、capabilities.tools）、`notifications/initialized`（无响应）、`tools/list`、`tools/call`。
- 工具（全部只读）：`list_sources`、`list_tags`、`search_telegram_archive`（带来源/标签过滤）、`get_message`（按 `media_item_id` 获取某条目的完整文本）、`ask_telegram_notebook`（RAG）、`summarize_source`。
- 限定于来自 `MCP_OWNER_ID` 的单个所有者（默认 `0` = Web 归档）。所有查询都经过 `owner_id` 隔离。
- 运行：`python -m telegram_notebook.mcp_server`。

### Repository
- 一个新方法 `get_media_item(owner_id, media_item_id)`，用于 `get_message` 工具。

### 测试
- `tests/test_mcp_server.py`：initialize/tools-list、通知行为、未知方法错误、list_sources/search/get_message、未知工具（isError）、按所有者隔离，以及完整的 `serve_stdio` 往返。

## 阶段 7 — 摘要 / NotebookLM（2026-06-09）

路线图中的归档摘要（按来源和按标签摘要）。

### 行为
- `/summarize [--source <url>] [--tag <tag>]`——无过滤时，对整个归档进行摘要；带 `--source` 时，针对单个来源；带 `--tag` 时，针对单个标签（使用相同的 `_split_filters` 解析器）。
- 内容（每个条目一行，带文本和来源）从 `Repository.summary_items` 获取（限定于所有者 + 来源/标签，默认上限 200 个条目）。
- 摘要用 `SearchService.summarize` 构建；提示在 `_build_summary_prompt`（纯函数）中生成，标注来源并截断每个条目的文本，然后传递给 `gemini_generate_content`。

### 范围之外（后续）
- 主题聚类和自动时间线。

### 测试
- `tests/test_summarize.py`：构建提示（包括来源和范围、文本截断）、空消息，以及 `summary_items` 方法的范围限定（全部/标签/来源以及按用户隔离）。

## 阶段 6 — 完整导入作业（2026-06-09）

路线图中的完整频道导入：队列、后台工作线程、进度跟踪、中断后续传，以及取消。

### 数据模型
- `jobs` 表（`owner_id`、`channel_url`、`status`、`total`、`processed`、`cursor`、`limit_count`、`error`、`cancel_requested`、时间戳）。status 是 `queued|running|done|failed|cancelled` 之一。
- Repository 方法：`create_job`、`get_job`、`list_jobs`、`claim_next_queued_job`（原子地选择最旧的作业并将其移至 running）、`update_job_progress`、`finish_job`、`request_job_cancel`、`is_cancel_requested`，以及 `requeue_running_jobs`（将因工作线程崩溃而成为孤儿的 running 作业放回 queued）。

### 工作线程
- 一个新的 `jobs.py` 模块，包含 `JobWorker`（一个单一的守护线程）。它与 Telegram 解耦，并通过注入式 `runner` 工作，使状态机完全可单元测试。
- 启动时，它会重新排队孤儿 running 作业（崩溃后续传）。

### 流水线
- `ingest_channel` 新增了参数 `resume_from`（用于续传的 min_id）、`progress_cb(processed, total, last_msg_id)` 和 `should_cancel()`。对每条消息，会检查取消并更新进度/游标。因为存储是幂等的，续传是安全的。
- `iter_all_messages` 新增了一个 `min_id` 参数，且 `limit` 现在是可选的（`None` = 所有消息）。

### 机器人
- `/import <channel_url> [limit]`（排队一次完整/可续传的导入）、`/jobs`（状态和进度）、`/canceljob <id>`。
- `/ingest` 仍是快速的内联路径。工作线程在 `run_forever` 中启动，并在每个作业结束时向用户发送 done/failed/cancelled 消息。
- `/help` 已更新。

### 测试
- `tests/test_jobs.py`：作业生命周期、原子认领和排序、进度/取消/重新排队，以及带伪 runner 的工作线程状态机（done/failed/cancelled 和用于续传的游标推进）。

## 阶段 5 — 规则 + 标签（2026-06-09）

路线图中的规则引擎和标签系统。用户定义一条关键词→标签规则，传入内容会被自动打标签，并可在搜索/提问中过滤。

### 数据模型
- `rules` 表（`owner_id`、`keyword`、`tag`、`created_at`），在 `(owner_id, keyword, tag)` 上有唯一索引。
- `content_tags` 表（`owner_id`、`media_item_id`、`tag`），带复合主键（幂等打标签）。
- 两者都用 `CREATE TABLE IF NOT EXISTS` 创建；现有数据库无需特殊迁移。

### 匹配与自动打标签
- 一个新的 `rules.py` 模块，包含纯函数 `match_tags(text, rules)`（子串，不区分大小写）。
- 在所有三条摄取路径（频道文本、媒体转写、转发收件箱）中，存储文本后，流水线会应用所有者的规则并存储标签（`_apply_rules`）。`owner_id` 被添加到流水线的内部辅助函数中。

### 机器人命令
- `/rule add <keyword> -> <tag>`、`/rule list`、`/rule remove <id>`，以及 `/rule apply`（清除并从存储的文本重新计算标签）。
- `/tags`——标签以及每个标签的去重条目计数。
- 用于 `/search` 和 `/ask` 的 `--tag <tag>` 过滤。`_split_source` 解析器被替换为 `_split_filters`，它同时理解 `--source`（单个 token）和 `--tag`（到行尾，多词）。
- `/help` 已更新。

### 搜索
- `SearchService.search` 新增了一个 `tag` 参数。关键词路径通过在 `content_tags` 上的 join 进行过滤；语义路径（Vertex）用来自 `media_ids_for_tag` 的允许列表进行后过滤。

### 测试
- `tests/test_rules.py`：纯匹配、解析 `/rule add`、规则 CRUD 和唯一性、标签存储/计数、摄取时的自动打标签、带标签过滤的搜索，以及回填。
- `tests/test_normalize.py`：测试 `_split_filters`（替代 `_split_source`）。

## 阶段 4 — 转发收件箱（MVP）（2026-06-09）

实现路线图的下一阶段：“智能 Telegram 收件箱”。用户现在可以将任何消息转发给机器人，其文本/说明文字会被存入一个个人的、可搜索的收件箱。

### 行为
- 机器人检测转发的消息（既包括新的 `forward_origin` 格式，也包括诸如 `forward_from`/`forward_from_chat`/`forward_sender_name` 等旧字段），并在鉴权流程逻辑之前对它们进行路由，使其不会与 `/connect` 流程中的文本回复冲突。
- 转发的 `text` 或 `caption`，连同一个媒体类型标签（例如 `[Forwarded document: report.pdf]`）和来源（来源频道/用户名称），会被存储。
- 当来源是公开频道时，会构建链接 `https://t.me/<username>/<id>` 作为来源。
- 存储的内容可通过相同的 `/search` 和 `/ask`（块 + 嵌入向量，并在没有可用嵌入向量时回退到关键词）进行查询。

### 数据模型
- 收件箱被实现为一个合成的、每用户的“频道”，其 `channel_url = inbox://forwarded`，复用现有 schema 和搜索路径（以及阶段 2 的 `owner_id` 隔离）。
- 一个新方法 `IngestionPipeline.ingest_forwarded_message`（基于转发的 message_id 幂等）。

### 机器人 UX
- `/start` 和 `/help` 已更新以说明转发能力。
- 针对无文本的纯媒体条目的引导消息（在此版本中尚未被索引）。
- 重构：与索引相关的 Vertex 配置被合并到一个共享辅助函数（`_vertex_ingest_config`）中，使 `/ingest` 和收件箱都使用它。

### 范围之外（后续）
- 通过 Bot API 下载并转写转发的媒体、照片的 OCR，以及从 PDF/DOCX/Excel 提取文本。

### 测试
- `tests/test_forwarded.py`：转发检测、来源/链接/媒体标签的提取，以及端到端摄取（存储和可搜索性、幂等性，以及收件箱的每用户特性）。

## 阶段 3 — Web API 鉴权与机密加密（2026-06-09）

分析中剩下的两个安全项：Web API 鉴权和数据库中机密的加密。

### Web API 鉴权
- 一个新变量 `WEB_API_TOKEN`。设置后，所有 `/api/*` 端点（除 `/api/health` 外）都需要该令牌；令牌通过 `Authorization: Bearer <token>` 或 `X-API-Token` 头发送（常量时间比较）。
- 未设置令牌时，API 只接受回环（localhost）请求，未经鉴权的网络访问会被以 401 拒绝（默认安全；此前它是完全开放的）。
- `/api/health` 为 Docker 健康检查保持公开。
- 仪表板 UI：所有调用都经过 `fetchJson`；该函数发送来自 `localStorage` 的令牌，并在收到 401 响应时提示用户输入一次令牌并存储它。

### 静态机密加密
- 一个新的 `crypto.py` 模块：仅使用标准库的认证加密（用 HKDF-SHA256 进行密钥分离、用 CTR 模式下的 HMAC-SHA256 生成密钥流，以及用 HMAC-SHA256 的 Encrypt-then-MAC；每个值使用一个随机的 128 位 nonce）。无新依赖。
- 敏感列在存入 SQLite 前被加密：在 `bot_users` 中 → `api_hash`、`session_string`、`gemini_api_key`；在 `auth_flows` 中 → `api_hash`、`session_string`、`phone_code_hash`。读取（`get_bot_user`/`get_auth_flow`）会透明解密。
- 密钥从 `SECRETS_KEY` 读取。如果未设置，加密是一个空操作（带警告），旧的明文数据库继续工作；加密值通过 `enc::` 前缀与旧明文区分，以使迁移无痛。

### 测试
- `tests/test_crypto.py`：往返、非确定性、对篡改/错误密钥的拒绝、对 None/空/旧明文的透传，以及无密钥时的空操作行为。
- `tests/test_web_auth.py`：接受 bearer/`X-API-Token`、拒绝错误/缺失令牌，以及未设置令牌时的回环限制。
- `tests/test_db.py`：针对机密加密存储和读取时透明解密的新测试。

### .env.example
- 添加 `WEB_API_TOKEN` 和 `SECRETS_KEY`，以及生成值的命令。

## 阶段 2 — 按用户数据隔离（2026-06-09）

本阶段的重点是修复用户之间的数据泄漏：此前，`/search` 和 `/ask`（以及 Web API）作用于数据库中的**所有**频道，用户可以看到彼此的数据。

### 数据模型
- 向 `channels` 表添加了一个 `owner_id` 列，并在此层级强制所有权；因为每个 `message`/`media_item`/`chunk` 都通过 FK 与一个频道关联，在 join 中按 `channels.owner_id` 过滤可完全隔离数据。
- 全局 `UNIQUE(channel_url)` 约束被替换为复合 `UNIQUE(owner_id, channel_url)` 索引，使两个用户可以各自独立摄取同一个频道而不共享一行。
- 针对旧数据库的自动迁移（`Repository._ensure_channel_owner`）：重建 `channels` 表、添加 `owner_id` 列，并将旧行保留为 `owner_id = NULL`；也就是说，它们不会在用户之间泄漏，而是对按用户查询变得不可见（如有需要必须重新摄取）。

### 范围强制
- 返回或删除数据的 Repository 方法现在接受 `owner_id`：`upsert_channel`、`keyword_candidates`、`embedding_candidates`、`list_channels`、`delete_channel_data`、`get_chunk_by_media_and_index`。
- `SearchService.search` 和 `IngestionPipeline.ingest_channel` 接受一个 `owner_id` 参数。
- Telegram 机器人将用户的 `bot_user_id` 作为 `owner_id` 传递；因此 `/search`、`/ask`、`/ingest`、`/sources`、`/delete`、`/status` 只作用于该用户的数据。
- Web 仪表板（没有按用户登录）使用固定的 `WEB_OWNER_ID = 0`，使其归档与机器人用户的归档保持分离。

### 加固
- `keyword_candidates` 中的 `LIMIT` 现在作为参数绑定，而不是通过字符串插值。

### 测试
- 更新了 `Repository` 测试以传递 `owner_id`。
- 一个新测试 `test_data_is_isolated_per_owner`：两个使用相同 URL 的用户看不到彼此的数据，且删除其中一个的数据不会影响另一个的。
- 一个新测试 `test_migrates_legacy_channels_table_without_owner_id`：迁移一个没有 `owner_id` 的旧数据库。

## 阶段 1 — 稳定核心（2026-06-08）

按照 README 中的路线图，本阶段专注于稳定核心：安全、缺陷修复、机器人命令、日志记录和测试。

### 安全
- 真实的机器人令牌已从 `.env.example` 中移除并清空。
  - ⚠️ 该令牌此前曾被提交到 git 历史中（提交 `5501fda`），实际上已经公开。清空文件还不够；你必须立即在 **@BotFather** 中 `/revoke` 该令牌并创建一个新的。
- 环境特定的标识符（`VERTEX_INDEX_ID`、`VERTEX_DEPLOYED_INDEX_ID`）在示例文件中已清空。

### 缺陷修复
- `/search` 和 `/ask`：用户是通过真实的 `bot_user_id` 读取的，而不是 `chat_id`（在群组中这两者不同）。
- Web API：`/api/search` 和 `/api/ask` 现在传递 `vertex_config`（以及用于 ask 的 `project_id`/`region`）；此前它们总是回退到关键词搜索。
- 机器人中的 `/ask` 响应使用 `<b>`（HTML），以便在 `parse_mode=HTML` 下正确渲染（此前是原始的 `**`）。
- `.env.example` 中的 `DB_PATH` 默认值与 `config.py` 对齐：`data/store.db`。

### 新的机器人命令
- `/status`——连接状态、AI 密钥、Vertex 配置，以及已索引来源的数量。
- `/disconnect`——删除用户的会话和凭据（“删除我的数据”）。
- `/help`——命令列表。
- 命令不再因 `@botname` 后缀和大小写而出现问题，也不再错误地进入连接流程。
- 针对 `/search`、`/ask`、`/ingest`、`/join`、`/delete` 中空输入的防护。

### 日志记录和错误处理
- 一个新的 `logging_config.py` 模块，包含 `setup_logging()`（级别来自 `LOG_LEVEL`，默认 INFO）。
- 所有调试用 `print()` 都被替换为 `logging`；敏感值（电话号码、登录码、`phone_code_hash`）不再被记录。
- 单条有问题的更新不再使整个机器人轮询循环停止（它会被记录并继续执行）。

### 测试
- 一个带 pytest 的 `tests/` 套件；26 个无需网络的测试：`chunking`、余弦相似度、`normalize_phone`/`normalize_code`、规范化 URL、文本组合、频道名称清理、临时 SQLite 上的 `Repository`，以及 `upsert_env_values`。
- 运行：`pip install -e ".[dev]"` 然后 `pytest`。

### 后续项（用于后续阶段）
- 对于带国家代码的号码，`normalize_phone` 仍然很幼稚（例如 `09123456789` → `+09123456789`）。
- `bot.py` 中的 `import re` 在移除正则之后未被使用，可以清理掉。
- `main.py` 仍在导入时构建全局状态；最好将其改为惰性的。
- 按用户数据隔离（阶段 2）尚未完成：`/search` 和 `/ask` 作用于所有频道，而不仅仅是该用户的数据。
