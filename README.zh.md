# Telegram NotebookLM MVP

**Languages / زبان‌ها:** [English](README.md) · [فارسی](README.fa.md) · [العربية](README.ar.md) · [Español](README.es.md) · [简体中文](README.zh.md)

这是一个用于构建**智能 Telegram 归档**的 MVP；这个项目可以收集频道、聊天、文件、视频和转发消息的内容，将它们转换为可搜索的文本，并最终像一个**面向 Telegram 的内部 NotebookLM** 那样回答用户的问题。

该项目的最终目标是让用户能够把自己的 Telegram 内容变成一份可搜索的记忆，并将其连接到 AI 工具；可以在 Telegram 机器人内部、Web 仪表板中，以及未来通过 MCP 连接到 ChatGPT、Claude、Cursor、类 Codex 智能体以及其他 AI 客户端等工具。

---

## 核心理念

该项目针对三种主要模式：

### 1. 导入频道 / 聊天

用户提供一个公开频道或他们有访问权限的聊天的链接或 ID，系统会检索其消息、说明文字（caption）和媒体。

示例：

```text
/ingest https://t.me/example_channel
```

### 2. 转发收件箱（Forwarded Inbox）

用户可以将一条消息、帖子、文件、照片、视频、PDF 或任何内容转发给机器人。系统会对其进行存储、处理、打标签，并使其可搜索。

这部分意在充当一个**智能 Telegram 收件箱**。

### 3. AI 笔记本 / RAG

在内容被存储和索引之后，用户可以对自己的归档提问：

```text
/ask 在我保存的关于 Al Mouj 的消息中，哪些是关于联排别墅（townhouse）的？
```

或者：

```text
/ask 将各频道中介绍的用于制作视频的 AI 工具进行分类
```

答案应当附带来源、消息链接以及相关文本。

---

## 当前 MVP 状态

在当前版本中，该项目具备以下能力：

- 接收 Telegram 频道链接并使用 `Telethon` 读取消息
- 下载并处理文本、音频和视频消息
- 使用 `ffmpeg` 从视频中提取音频
- 使用 OpenAI 或 Gemini 转写音频/视频
- 对文本进行分块（chunking）
- 为语义搜索构建嵌入向量（embeddings）
- 关键词 + 语义搜索
- 基于搜索结果的初步 RAG 答案生成
- 一个使用 `Python http.server` 的轻量级 Web 仪表板
- 一个用于编排和核心命令的 Telegram 机器人
- 通过 Telethon 使用会话字符串（session string）连接用户的真实 Telegram 账户
- **转发收件箱**：将任何消息转发给机器人会把其文本/说明文字存入用户个人的、可搜索的收件箱
- **转发媒体处理**：音频/视频/语音文件会被自动转写，照片/PDF 通过 OCR（Gemini）转换为可搜索文本。DOCX/XLSX 文件也会在本地提取（无需 API 密钥，无需联网）
- **规则引擎 + 标签**：定义关键词→标签规则，对传入内容自动打标签，并使用 `--tag` 过滤 `/search` 和 `/ask`
- **AI 规则**：`/rule add-ai` 使用自然语言判据（由 LLM 匹配），以及 `/airules` 用于在新转发上选择性地（opt-in）自动打标签
- **标签管理和浏览**：`/tag rename|delete`、`/recent` 显示最新条目，以及 Web 端点 `/api/{stats,recent,timeline}` + 仪表板中的资料库（Library）面板
- **自动转发到归档频道**：通过 `/setarchive`，你转发的任何匹配标签规则的消息也会被自动转发到你的归档频道
- **导入作业（Import Jobs）**：在后台进行完整的频道导入，带队列、进度跟踪、中断后续传，以及可取消的能力
- **摘要（NotebookLM）**：`/summarize` 用于为整个归档、特定来源或某个标签构建结构化摘要
- **摘要简报（Digest）**：`/digest [days]` 构建近期内容的 AI 摘要（默认 7 天）
- **主题聚类**：`/topics` 离线地（基于已有的嵌入向量）按主题对归档内容进行聚类；如果存在 Gemini 密钥，每个聚类的标签会用 LLM 构建（否则取自最高频的词项）
- **时间线**：`/timeline` 按日期（月或日）对归档进行分组——是对 `/topics` 在时间维度上的补充
- **导出**：`/export` 将整个归档、单个来源或单个标签导出为可下载的 Markdown 文件
- **统计**：`/stats` 显示归档概览（条目、来源、标签数量、媒体类型和时间范围）
- **合集（笔记本）**：`/collection` 将多个标签归到同一个名称之下，并显示该合集的条目
- **Telegram 备份导入**：接收 Telegram Desktop 的*机器可读 JSON*（Machine-readable JSON）文件（`result.json` 或导出文件夹的 `.zip`）——既支持单聊导出，也支持整个账户导出——使所有消息可搜索，并返回一份 **Markdown** 副本。在机器人中，只需发送文件；在 Web 上，从“导入 Telegram 备份”卡片上传。
- **落地页 + Web 应用**：位于 `/` 的介绍页面，以及位于 `/app` 的完整仪表板（Ingest / Search / Ask / Library / 备份导入）
- **MCP 服务器（只读）**：通过 stdio 上的 JSON-RPC 将归档暴露给 AI 工具（`python -m telegram_notebook.mcp_server`）

---

## 为什么 Bot API 不够用？

要捕获一个频道或聊天的完整归档，仅靠 Bot API 是不够的。Bot API 通常只能看到机器人有权访问的新消息。

为了导入频道和聊天的历史记录，本项目使用 `Telethon` 和 MTProto；也就是说，使用与用户账户相同级别的访问权限，而不仅仅是一个机器人令牌。

---

## 当前架构

```text
Telegram 机器人
  |
  | 用户命令：/connect、/ingest、/search、/ask
  v
Python 后端
  |
  +-- Telethon 客户端
  |     读取频道和聊天
  |
  +-- 摄取流水线（Ingestion Pipeline）
  |     下载媒体、提取文本、转写
  |
  +-- 分块 + 嵌入（Chunking + Embedding）
  |     为语义搜索做准备
  |
  +-- 搜索服务（Search Service）
  |     关键词搜索 + 向量搜索
  |
  +-- RAG 答案生成器
        根据找到的来源构建答案
```

---

## 技术栈

- Python 3.11+
- Telethon
- OpenAI API
- Google Gemini / Google GenAI
- ffmpeg
- 用于 MVP 的 SQLite / 兼容 JSON 的本地存储
- Python 词法搜索 + 余弦相似度
- 用于用户界面的 Telegram Bot API
- 使用 `http.server` 的轻量级 Web UI

---

## 机器人命令

```text
/start
项目介绍和入门

/connect
连接用户的真实 Telegram 账户

/status
检查连接状态

/ingest <channel_url>
对一个频道进行快速的内联索引

/import <channel_url> [limit]
在后台排队进行一次完整的、可续传的导入

/backup
导入 Telegram 备份文件的指南；只需将 result.json 或 .zip 文件发送给机器人

/jobs
显示导入作业的状态和进度

/canceljob <id>
取消一个已排队或正在运行的作业

/search <query>
搜索归档

/search <query> --source <channel_url>
仅在特定来源内搜索

/search <query> --tag <tag>
仅在已打标签的内容内搜索

/ask <question>
用 AI 向归档提问

/ask <question> --source <channel_url>
仅从特定频道或来源提问

/ask <question> --tag <tag>
仅从特定标签的内容提问

/summarize [--source <url>] [--tag <tag>]
对整个归档、某个来源或某个标签进行摘要

/digest [days]
近期内容的 AI 摘要（默认 7 天）

/topics [--source <url>] [--tag <tag>]
对内容进行主题聚类

/timeline [--source <url>] [--tag <tag>] [--day]
按月查看归档的时间视图（或使用 --day 按日）

/export [--source <url>] [--tag <tag>]
下载整个归档、某个来源或某个标签的 Markdown 导出

/recent [n]
显示最近的 n 个条目（默认 10）

/stats
归档概览（条目、来源、标签数量、媒体类型、时间范围）

/sources
显示已索引的来源

/delete <channel_url>
删除某个来源的数据

/rule add <keyword> -> <tag>
定义一条关键词规则用于自动给内容打标签

/rule add-ai <criterion> -> <tag>
定义一条 AI 规则（在 /rule apply 期间由 LLM 匹配）

/rule list
显示规则

/rule remove <id>
移除一条规则

/rule apply
将规则重新应用于已有内容（回填，backfill）

/airules on|off
在每条新转发上自动运行 AI 规则（选择性启用；默认关闭）

/tags
显示标签以及每个标签的条目计数

/tag rename <old> -> <new>
重命名一个标签（或将其合并到已有标签中）

/tag delete <tag>
从所有条目中删除一个标签

/collection new|add|list|remove|show <name>
将多个标签归到一个“笔记本”下并显示其条目
（之后可以用 /summarize --collection <name> 或 /export --collection <name> 对整个笔记本进行摘要/导出）

/setarchive <@channel | off>
设置归档频道；已打标签的转发会被自动发送到该频道

/cancel
取消当前流程
```

---

## 核心 API

### 导入频道（Ingest Channel）

```bash
curl -X POST http://127.0.0.1:8000/api/channels/ingest \
  -H 'content-type: application/json' \
  -d '{
    "channel_url": "https://t.me/example_channel",
    "limit": 50
  }'
```

### 搜索（Search）

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "هوش مصنوعی و تولید ویدیو",
    "channel_url": "https://t.me/example_channel",
    "top_k": 5
  }'
```

### AI 提问（Ask AI）

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'content-type: application/json' \
  -d '{
    "query": "از این کانال چه ابزارهایی برای ساخت ویدیو معرفی شده؟",
    "channel_url": "https://t.me/example_channel"
  }'
```

### 统计 / 最近 / 时间线（只读）

```bash
curl http://127.0.0.1:8000/api/stats
curl 'http://127.0.0.1:8000/api/recent?limit=10'
curl 'http://127.0.0.1:8000/api/timeline?granularity=month'
```

这些端点与 API 的其余部分一样，受 `WEB_API_TOKEN` 保护（或在未设置令牌时仅限回环地址）。

---

## 最终产品方向

该项目的最终目标不仅仅是简单的搜索。产品方向如下：

```text
Telegram AI 归档
  |
  +-- 完整导入频道和聊天
  +-- 用于转发消息的转发收件箱
  +-- 用于通过关键词或 AI 区分内容的规则引擎
  +-- 标签 / 文件夹 / 合集
  +-- 词法和语义搜索
  +-- 用于问答的内部 NotebookLM
  +-- 用于连接 AI 工具的 MCP 服务器
```

---

## 规则引擎 + 标签

用户可以定义关键词→标签规则：

```text
/rule add Claude -> AI Tools
/rule add Al Mouj -> Real Estate
/rule add golden visa -> Oman Visa
/rule add قیمت -> Leads
```

任何进入系统的新内容（频道摄取、媒体转写或转发收件箱）都会针对其文本和说明文字进行检查。如果某条规则的关键词（作为子串且不区分大小写）出现在文本中：

- 对应的标签会被附加到该条目上
- 之后可以用 `/search ... --tag <tag>` 和 `/ask ... --tag <tag>` 进行过滤
- `/tags` 显示标签以及每个标签的条目计数
- `/tag rename <old> -> <new>` 重命名一个标签（或将其合并到已有标签中），`/tag delete <tag>` 将其从所有条目中移除
- `/rule apply` 将当前规则重新应用于已有内容（回填，backfill）

**基于 AI 的规则：** 除关键词规则外，你还可以定义一条使用自然语言判据的规则，其匹配由 LLM 决定：

```text
/rule add-ai 关于使用人工智能制作视频的工具的帖子 -> Video AI
/rule add-ai 任何与房产价格和销售相关的内容 -> Leads
```

由于 LLM 成本，AI 规则仅在 `/rule apply` 期间运行（每个条目一次 LLM 调用）并需要 Gemini 密钥；如果未设置密钥，它们会被忽略，并在输出中报告这一点。关键词规则仍会在每次摄取时自动应用。

**自动转发：** 通过 `/setarchive <@channel>` 设置一个归档频道；从那时起，你转发给机器人的任何文本匹配某条标签规则的消息，除了被存入收件箱之外，还会连同其来源/标签/链接一起被转发到该频道（机器人必须是该频道的管理员）。要禁用：`/setarchive off`。

除 `/rule apply` 外，使用 `/airules on` 时 AI 规则也会在每条新转发上自动运行（选择性启用，每个条目一次 LLM 调用；批量频道导入永远不会被自动分类）。

**尚未添加（后续）：** 针对频道导入的自动转发（目前仅限转发收件箱路径），以及在完整频道导入过程中下载/处理媒体。

---

## Telegram 备份导入（JSON / ZIP）

你无需 `/connect` 即可导入一个聊天或账户的完整历史记录：

1. 在 **Telegram Desktop** 中，前往 `Settings → Advanced → Export Telegram data`（或在某个聊天上：`Export chat history`）。
2. 将格式设置为**机器可读 JSON**（Machine-readable JSON，而不是 HTML）。
3. 输出是一个 `result.json` 文件（或一个包含它以及媒体的文件夹）；你可以将该文件夹打成 zip 包。

然后：

- **从机器人：** 直接将 `result.json` 或 `.zip` 文件发送给机器人。机器人会导入它、使内容可搜索，并返回一份 Markdown 副本。（由于 Bot API 下载限制，有 20 MB 上限；对于更大的文件，请使用 Web。）
- **从 Web：** 在 `/app` 中的“导入 Telegram 备份”卡片，上传文件。内容会在 Web 归档中变得可搜索，并出现一个 Markdown 下载按钮。

每个聊天会成为一个合成来源 `backup://<id>`，且导入是幂等的（重新导入同一文件不会添加任何内容）。

### API

```bash
curl -X POST 'http://127.0.0.1:8000/api/backup/import' \
  -H 'X-Filename: result.json' \
  -H 'content-type: application/octet-stream' \
  --data-binary @result.json
```

响应包含已导入的聊天/消息数量以及完整的 Markdown 文本。与 API 的其余部分一样，它受 `WEB_API_TOKEN` 保护（或回环地址）。

---

## MCP 服务器

已实现一个**Telegram MCP 服务器**（只读），以便用户的 Telegram 归档不局限于机器人，并可连接到其他 AI 工具（Claude、Cursor 等）。它通过 stdio 上的 JSON-RPC 2.0 工作，并且仅使用标准库编写（没有新依赖）。

运行：

```bash
MCP_OWNER_ID=0 python -m telegram_notebook.mcp_server
```

`MCP_OWNER_ID` 决定暴露哪个用户的归档（默认 `0` = Web 仪表板归档；对于某个机器人用户的归档，请提供其 `bot_user_id`）。

当前的 MCP 工具：

```text
list_sources              显示频道/聊天以及转发收件箱
list_tags                 显示标签以及每个标签的条目计数
search_telegram_archive   搜索（带可选的来源/标签过滤）
get_message               按 media_item_id 获取某条目的完整文本
ask_telegram_notebook     对归档进行 RAG 问答
summarize_source          对整个归档、某个来源或某个标签进行摘要
list_topics               对内容进行主题聚类（离线，基于嵌入向量）
timeline                  按时间段（月/日）统计条目数量
archive_stats             归档概览（计数、媒体类型、时间范围）
list_recent               最新归档条目列表（从新到旧）
```

所有工具都是只读的；敏感工具（导入、转发、删除、create_rule）被有意不暴露，如有需要，应在日后带权限和确认机制再添加。

---

## 开发与测试

CI 会在每次 push 和 PR 时运行 lint 和测试（`.github/workflows/ci.yml`）。要在本地运行：

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -q
```

## 安装

```bash
git clone https://github.com/shm379/telegram-notebooklm-mvp.git
cd telegram-notebooklm-mvp

uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

在 Windows 上：

```powershell
uv venv
.venv\Scripts\activate
uv pip install -e .
copy .env.example .env
```

---

## 先决条件

- Python 3.11+
- ffmpeg
- Telegram API 凭据：
  - `TELEGRAM_API_ID`
  - `TELEGRAM_API_HASH`
  - 对于生产环境运行，`TELEGRAM_SESSION_STRING` 更可取
- 用于运行机器人的 `TELEGRAM_BOT_TOKEN`
- 以下提供方之一：
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`

---

## 创建 Telegram 会话

```bash
export TELEGRAM_API_ID=...
export TELEGRAM_API_HASH=...
uv run python scripts/create_telegram_session.py
```

将输出放入 `.env` 中的 `TELEGRAM_SESSION_STRING`。

如果你没有 `TELEGRAM_SESSION_STRING`，项目会使用本地会话文件，并且首次运行需要交互式登录。

---

## 运行 Web UI

```bash
python -m telegram_notebook.main
```

然后打开：

```text
http://127.0.0.1:8000        # 落地页（介绍）
http://127.0.0.1:8000/app    # 仪表板：Ingest / Search / Ask / Library / 备份导入
```

---

## 运行 Telegram 机器人

```bash
python -m telegram_notebook.bot
```

---

## 当前限制

- 用户之间的数据隔离已经到位（每个用户只能看到自己的数据；通过频道上的 `owner_id` 强制所有权）。
- 已添加使用 `WEB_API_TOKEN` 的 Web API 鉴权，以及使用 `SECRETS_KEY` 对数据库中机密的加密；生产环境中请同时设置这两个变量。
- 当前的存储适用于 MVP，不适合大型数据集。
- 会话字符串和 API 密钥应在投入生产前加密。
- 通过 `/import`（一个后台工作线程）支持带队列、进度和续传的完整频道导入；目前尚没有针对整个 Telethon 路径的测试沙箱。
- 除文本/说明文字外，转发收件箱还会处理转发的媒体：音频/视频/语音通过转写、照片/PDF 通过 OCR（Gemini 多模态）、DOCX/XLSX 通过本地提取（zipfile + XML，无需 API 密钥）被转换为文本。在完整频道导入过程中下载媒体尚未添加。
- 规则引擎基于关键词匹配（子串）；基于 AI 的规则和自动转发到归档频道尚未添加。
- `/topics` 在已有的嵌入向量上执行主题聚类（贪心余弦，离线）；它要求内容已用嵌入密钥进行索引。如果存在 Gemini 密钥，聚类标签会用 LLM 构建，否则回退到最高频的词项。`/timeline` 在消息日期上构建时间视图（月/日）。
- 对于大型数据集，迁移到 PostgreSQL + pgvector 或 Qdrant 更可取。

---

## 重要安全提示

- 不要将任何真实的令牌、API 密钥、会话字符串或凭据提交到仓库中。
- 如果之前曾将真实令牌提交到 `.env.example` 或项目历史中，请立即吊销/重新生成该令牌。
- 对于生产环境，用户会话必须加密。
- 每次搜索或提问都必须应用 user_id 过滤。
- 用户必须能够 `disconnect`（断开连接）和 `delete my data`（删除我的数据）。
- MCP 工具最初应当是只读的。

---

## 建议路线图

### 阶段 1 — 稳定核心

- 清理仓库中的机密
- 修复 README 和 env 示例
- 稳定 `/connect`、`/ingest`、`/search`、`/ask`
- 修复错误处理和日志记录

### 阶段 2 — 多用户数据模型

- 向 sources、messages、media、chunks 添加 user_id
- 完整隔离用户数据
- 权限和访问控制

### 阶段 3 — 转发收件箱

- 处理转发的消息
- 存储文本、说明文字、媒体、文档
- 照片的 OCR
- 从 PDF/DOCX/Excel 提取文本

### 阶段 4 — 规则 + 标签

- 定义关键词规则
- 标签和合集
- 自动转发到归档频道
- 基于 AI 的规则

### 阶段 5 — 完整导入作业

- 从头到尾的完整频道导入
- 中断后续传
- 进度跟踪
- 队列/后台工作线程

### 阶段 6 — 内部 NotebookLM

- 更好的带来源答案生成
- 按来源摘要
- 按标签摘要
- 时间线和主题聚类 ✅（`/timeline`、`/topics`）

### 阶段 7 — MCP 服务器

- 只读 MCP 工具
- 连接到 AI 客户端
- search、ask、list_sources、get_message 工具

---

## 总结

Telegram NotebookLM MVP 尝试把 Telegram 变成一份智能记忆；一个用户可以存储自己的频道、聊天和转发消息的地方，可以在其中用关键词或语义搜索进行查找，用规则区分内容，并最终像 NotebookLM 那样对自己的归档提问。

这个项目是构建一个更大产品的基础：

```text
面向 AI 助手的 Telegram 记忆
```
