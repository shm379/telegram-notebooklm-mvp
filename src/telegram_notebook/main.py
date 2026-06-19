from __future__ import annotations

import asyncio
import hmac
import json
import logging
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import get_settings, upsert_env_values
from .db import Repository, connect
from .embeddings import EmbeddingService
from .extraction import ExtractionService
from .logging_config import setup_logging
from .model_catalog import ModelCatalogService
from .pipeline import IngestionPipeline
from .recent import recent_rows
from .search import SearchService
from .telegram_backup import (
    count_messages,
    enrich_with_media,
    make_zip_extractor,
    open_backup_zip,
    parse_export,
    read_export,
    render_markdown,
)
from .timeline import build_timeline
from .transcription import TranscriptionService

logger = logging.getLogger(__name__)

# The web dashboard has no per-user login, so all of its data lives under a
# single fixed owner id. This keeps the dashboard's archive isolated from the
# per-user (bot_user_id) archives created through the Telegram bot.
WEB_OWNER_ID = 0


@dataclass(slots=True)
class RuntimeConfig:
    transcription_provider: str
    transcription_model: str
    embedding_provider: str
    embedding_model: str
    openai_enabled: bool
    gemini_enabled: bool


class AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.catalog = ModelCatalogService()
        self.reload()

    def reload(self) -> None:
        with self.lock:
            get_settings.cache_clear()
            self.settings = get_settings()
            self.repository = Repository(connect(self.settings.db_path))
            self.repository.init()
            self.embeddings = EmbeddingService(
                provider=self.settings.embedding_provider,
                api_key=self._api_key_for(self.settings.embedding_provider),
                model=self.settings.embedding_model,
            )
            self.transcription = TranscriptionService(
                provider=self.settings.transcription_provider,
                api_key=self._api_key_for(self.settings.transcription_provider),
                model=self.settings.transcription_model,
            )
            self.pipeline = IngestionPipeline(
                settings=self.settings,
                repository=self.repository,
                transcription=self.transcription,
                embeddings=self.embeddings,
            )
            self.search_service = SearchService(self.repository, self.embeddings)

    def _api_key_for(self, provider: str) -> str | None:
        if provider == "gemini":
            return self.settings.gemini_api_key
        if provider == "openai":
            return self.settings.openai_api_key
        return None

    def vertex_search_config(self) -> dict[str, object] | None:
        """Vertex AI Search config from settings, or None to use local search."""
        s = self.settings
        if s.vertex_project_id and s.vertex_region and s.vertex_endpoint_id and s.vertex_deployed_index_id:
            return {
                "api_key": s.gemini_api_key,
                "project_id": s.vertex_project_id,
                "region": s.vertex_region,
                "index_endpoint_id": s.vertex_endpoint_id,
                "deployed_index_id": s.vertex_deployed_index_id,
            }
        return None

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            transcription_provider=self.settings.transcription_provider,
            transcription_model=self.settings.transcription_model,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.embedding_model,
            openai_enabled=bool(self.settings.openai_api_key),
            gemini_enabled=bool(self.settings.gemini_api_key),
        )

    def list_models(self, *, provider: str, capability: str | None) -> list[dict[str, object]]:
        provider = provider.lower()
        api_key = self._api_key_for(provider)
        mapped_capability = capability
        if provider == "gemini":
            if capability == "transcription":
                mapped_capability = "generateContent"
            elif capability == "embedding":
                mapped_capability = "embedContent"
        return self.catalog.list_models(
            provider=provider,
            api_key=api_key,
            capability=mapped_capability,
        )


state = AppState()


INDEX_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telegram Notebook — Dashboard</title>
    <style>
      :root {
        --bg: #f6f0e7;
        --card: rgba(255, 252, 247, 0.88);
        --ink: #1d1b19;
        --muted: #6c6257;
        --accent: #0d7c66;
        --accent-2: #b45f06;
        --line: rgba(29, 27, 25, 0.12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(13,124,102,0.14), transparent 28%),
          radial-gradient(circle at bottom right, rgba(180,95,6,0.14), transparent 30%),
          var(--bg);
      }
      .wrap {
        max-width: 1180px;
        margin: 0 auto;
        padding: 28px 16px 64px;
      }
      .hero {
        padding: 28px;
        border-bottom: 1px solid var(--line);
      }
      h1 {
        margin: 0;
        font-size: clamp(2rem, 5vw, 4.6rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
      }
      p {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
        margin-top: 20px;
      }
      .card {
        backdrop-filter: blur(16px);
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 18px;
        box-shadow: 0 18px 50px rgba(29, 27, 25, 0.08);
      }
      .settings-card {
        grid-column: 1 / -1;
      }
      .settings-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }
      label {
        display: block;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: var(--muted);
      }
      input, textarea, button, select {
        width: 100%;
        border-radius: 16px;
        border: 1px solid var(--line);
        padding: 12px 14px;
        font: inherit;
        background: white;
      }
      textarea { min-height: 110px; resize: vertical; }
      button {
        background: linear-gradient(120deg, var(--accent), #0c5d4f);
        color: white;
        border: none;
        cursor: pointer;
      }
      button.secondary {
        background: linear-gradient(120deg, var(--accent-2), #8a4805);
      }
      .results {
        margin-top: 20px;
        display: grid;
        gap: 12px;
      }
      .result {
        padding: 14px;
        border-radius: 18px;
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
      }
      .meta {
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 8px;
      }
      .status {
        min-height: 28px;
        color: var(--accent);
      }
      .tiny {
        font-size: 0.85rem;
        color: var(--muted);
      }
      a { color: var(--accent); }
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <h1>Telegram Notebook</h1>
        <p>
          لینک کانال عمومی را ingest کن، ویدیو و صوت را به متن تبدیل کن، و بعد مثل یک دفتر جست‌وجوی معنایی روی archive خودت داشته باش. حالا می‌توانی بین OpenAI و Gemini جابه‌جا شوی و مدل دلخواه را از خود API بگیری.
        </p>
      </section>

      <section class="grid">
        <div class="card settings-card">
          <h2>Settings</h2>
          <div class="settings-grid">
            <div>
              <label for="transcriptionProvider">Transcription Provider</label>
              <select id="transcriptionProvider">
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
            <div>
              <label for="transcriptionModel">Transcription Model</label>
              <select id="transcriptionModel"></select>
            </div>
            <div>
              <label for="embeddingProvider">Embedding Provider</label>
              <select id="embeddingProvider">
                <option value="openai">OpenAI</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
            <div>
              <label for="embeddingModel">Embedding Model</label>
              <select id="embeddingModel"></select>
            </div>
            <div>
              <label for="geminiApiKey">Gemini API Key</label>
              <input id="geminiApiKey" type="password" placeholder="اختیاری؛ فقط برای بروزرسانی" />
            </div>
            <div>
              <label for="openaiApiKey">OpenAI API Key</label>
              <input id="openaiApiKey" type="password" placeholder="اختیاری؛ فقط برای بروزرسانی" />
            </div>
          </div>
          <div style="margin-top:12px; display:flex; gap:12px; flex-wrap:wrap;">
            <button id="reloadModelsBtn" type="button">Reload Models</button>
            <button id="saveSettingsBtn" class="secondary" type="button">Save Settings</button>
          </div>
          <div class="tiny" id="settingsSummary"></div>
          <div class="status" id="settingsStatus"></div>
        </div>

        <div class="card">
          <h2>Ingest Channel</h2>
          <label for="channelUrl">Channel URL</label>
          <input id="channelUrl" value="https://t.me/example_channel" />
          <label for="limit">Recent posts limit</label>
          <input id="limit" type="number" value="50" min="1" max="500" />
          <button id="ingestBtn">Start Ingest</button>
          <div class="status" id="ingestStatus"></div>
        </div>

        <div class="card">
          <h2>Search & Ask</h2>
          <label for="query">Query / Question</label>
          <textarea id="query">هوش مصنوعی و مدل‌های زبانی</textarea>
          <label for="searchChannel">Optional channel filter</label>
          <input id="searchChannel" placeholder="https://t.me/example_channel" />
          <div style="display:flex; gap:10px;">
            <button class="secondary" id="searchBtn">Search Transcript</button>
            <button id="askBtn">Ask AI Brain</button>
          </div>
          <div class="status" id="searchStatus"></div>
        </div>

        <div class="card">
          <h2>Library</h2>
          <p class="tiny">An overview of your archive and its most recent items.</p>
          <button id="loadLibraryBtn" type="button">Load library</button>
          <div class="status" id="libraryStatus"></div>
          <div class="tiny" id="libraryStats" style="margin-top:10px;"></div>
          <div class="results" id="libraryRecent"></div>
        </div>

        <div class="card">
          <h2>Import Telegram Backup</h2>
          <p class="tiny">
            در Telegram Desktop مسیر «Export chat history» را با فرمت
            <b>Machine-readable JSON</b> بزن، سپس فایل <code>result.json</code> یا
            <code>.zip</code> آن را اینجا آپلود کن تا قابل جستجو شود و یک نسخه‌ی Markdown بگیری.
          </p>
          <label for="backupFile">Backup file (.json / .zip)</label>
          <input id="backupFile" type="file" accept=".json,.zip,application/json,application/zip" />
          <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <button id="importBackupBtn" type="button">Import & Convert</button>
            <button id="convertBackupBtn" class="secondary" type="button">Convert only</button>
          </div>
          <div class="status" id="backupStatus"></div>
          <div class="tiny" id="backupDownload" style="margin-top:10px;"></div>
        </div>

        <div class="card">
          <h2>Sources</h2>
          <p class="tiny">Channels, chats and backups indexed in this workspace, with item counts.</p>
          <button id="loadSourcesBtn" type="button">Load sources</button>
          <div class="status" id="sourcesStatus"></div>
          <div class="results" id="sourcesList" style="margin-top:10px;"></div>
        </div>
      </section>

      <div id="brainAnswer" style="display:none; margin-top:20px;" class="card">
        <h3>AI Brain Response</h3>
        <p id="answerText" style="white-space: pre-wrap; color: var(--ink);"></p>
        <div class="tiny" id="answerMeta"></div>
      </div>

      <section class="results" id="results"></section>
    </div>
    <script>
      const ingestBtn = document.getElementById("ingestBtn");
      const searchBtn = document.getElementById("searchBtn");
      const askBtn = document.getElementById("askBtn");
      const reloadModelsBtn = document.getElementById("reloadModelsBtn");
      const saveSettingsBtn = document.getElementById("saveSettingsBtn");
      const ingestStatus = document.getElementById("ingestStatus");
      const searchStatus = document.getElementById("searchStatus");
      const settingsStatus = document.getElementById("settingsStatus");
      const settingsSummary = document.getElementById("settingsSummary");
      const results = document.getElementById("results");
      const brainAnswer = document.getElementById("brainAnswer");
      const answerText = document.getElementById("answerText");

      function displayResults(dataItems) {
        results.innerHTML = "";
        for (const item of dataItems) {
          const div = document.createElement("article");
          div.className = "result";
          div.innerHTML = `
            <div class="meta">
              <strong>${item.channel_title || item.channel_url}</strong>
              • ${item.media_kind}
              • score=${item.score}
              ${item.message_url ? `• <a href="${item.message_url}" target="_blank" rel="noreferrer">post</a>` : ""}
            </div>
            <div>${item.chunk_text}</div>
          `;
          results.appendChild(div);
        }
      }

      const transcriptionProvider = document.getElementById("transcriptionProvider");
      const transcriptionModel = document.getElementById("transcriptionModel");
      const embeddingProvider = document.getElementById("embeddingProvider");
      const embeddingModel = document.getElementById("embeddingModel");
      const geminiApiKey = document.getElementById("geminiApiKey");
      const openaiApiKey = document.getElementById("openaiApiKey");

      function fillSelect(select, models, selectedId) {
        select.innerHTML = "";
        for (const model of models) {
          const option = document.createElement("option");
          option.value = model.id;
          option.textContent = model.display_name || model.id;
          if (model.id === selectedId) {
            option.selected = true;
          }
          select.appendChild(option);
        }
        if (!models.length && selectedId) {
          const fallback = document.createElement("option");
          fallback.value = selectedId;
          fallback.textContent = selectedId;
          fallback.selected = true;
          select.appendChild(fallback);
        }
      }

      function withToken(options) {
        const opts = { ...(options || {}) };
        opts.headers = { ...(opts.headers || {}) };
        const token = localStorage.getItem("apiToken");
        if (token) {
          opts.headers["X-API-Token"] = token;
        }
        return opts;
      }

      async function fetchJson(url, options = undefined) {
        let response = await fetch(url, withToken(options));
        if (response.status === 401) {
          const token = window.prompt("This API requires a token (WEB_API_TOKEN). Enter it:");
          if (token) {
            localStorage.setItem("apiToken", token.trim());
            response = await fetch(url, withToken(options));
          }
        }
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Request failed");
        }
        return data;
      }

      async function loadSettings() {
        const data = await fetchJson("/api/settings");
        transcriptionProvider.value = data.transcription_provider;
        embeddingProvider.value = data.embedding_provider;
        settingsSummary.textContent =
          `OpenAI key: ${data.openai_enabled ? "set" : "missing"} | Gemini key: ${data.gemini_enabled ? "set" : "missing"}`;
        await reloadModels();
        transcriptionModel.value = data.transcription_model;
        embeddingModel.value = data.embedding_model;
      }

      async function reloadModels() {
        settingsStatus.textContent = "در حال گرفتن لیست مدل‌ها...";
        try {
          const [transcriptionModels, embeddingModels] = await Promise.all([
            fetchJson(`/api/models?provider=${encodeURIComponent(transcriptionProvider.value)}&capability=transcription`),
            fetchJson(`/api/models?provider=${encodeURIComponent(embeddingProvider.value)}&capability=embedding`)
          ]);
          fillSelect(transcriptionModel, transcriptionModels.models, transcriptionModel.value);
          fillSelect(embeddingModel, embeddingModels.models, embeddingModel.value);
          settingsStatus.textContent = "لیست مدل‌ها بروزرسانی شد";
        } catch (error) {
          settingsStatus.textContent = error.message;
        }
      }

      transcriptionProvider.addEventListener("change", reloadModels);
      embeddingProvider.addEventListener("change", reloadModels);
      reloadModelsBtn.addEventListener("click", reloadModels);

      saveSettingsBtn.addEventListener("click", async () => {
        settingsStatus.textContent = "در حال ذخیره تنظیمات...";
        try {
          const data = await fetchJson("/api/settings", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              transcription_provider: transcriptionProvider.value,
              transcription_model: transcriptionModel.value,
              embedding_provider: embeddingProvider.value,
              embedding_model: embeddingModel.value,
              gemini_api_key: geminiApiKey.value || undefined,
              openai_api_key: openaiApiKey.value || undefined
            })
          });
          geminiApiKey.value = "";
          openaiApiKey.value = "";
          settingsSummary.textContent =
            `OpenAI key: ${data.openai_enabled ? "set" : "missing"} | Gemini key: ${data.gemini_enabled ? "set" : "missing"}`;
          settingsStatus.textContent = "تنظیمات ذخیره شد";
        } catch (error) {
          settingsStatus.textContent = error.message;
        }
      });

      ingestBtn.addEventListener("click", async () => {
        ingestStatus.textContent = "در حال ingest...";
        try {
          const data = await fetchJson("/api/channels/ingest", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              channel_url: document.getElementById("channelUrl").value,
              limit: Number(document.getElementById("limit").value || 50)
            })
          });
          ingestStatus.textContent = `کانال ${data.channel_title || data.channel_url} پردازش شد. media موفق: ${data.processed_media}`;
        } catch (error) {
          ingestStatus.textContent = error.message;
        }
      });

      searchBtn.addEventListener("click", async () => {
        searchStatus.textContent = "در حال جست‌وجو...";
        results.innerHTML = "";
        brainAnswer.style.display = "none";
        try {
          const data = await fetchJson("/api/search", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              query: document.getElementById("query").value,
              channel_url: document.getElementById("searchChannel").value || null,
              top_k: 8
            })
          });
          searchStatus.textContent = `${data.results.length} نتیجه پیدا شد`;
          displayResults(data.results);
        } catch (error) {
          searchStatus.textContent = error.message;
        }
      });

      askBtn.addEventListener("click", async () => {
        searchStatus.textContent = "در حال تفکر...";
        results.innerHTML = "";
        brainAnswer.style.display = "none";
        try {
          const data = await fetchJson("/api/ask", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              query: document.getElementById("query").value,
              channel_url: document.getElementById("searchChannel").value || null,
            })
          });
          searchStatus.textContent = "پاسخ آماده شد!";
          brainAnswer.style.display = "block";
          answerText.textContent = data.answer;
          displayResults(data.sources);
        } catch (error) {
          searchStatus.textContent = error.message;
        }
      });

      const loadLibraryBtn = document.getElementById("loadLibraryBtn");
      const libraryStatus = document.getElementById("libraryStatus");
      const libraryStats = document.getElementById("libraryStats");
      const libraryRecent = document.getElementById("libraryRecent");

      loadLibraryBtn.addEventListener("click", async () => {
        libraryStatus.textContent = "در حال بارگذاری...";
        libraryStats.textContent = "";
        libraryRecent.innerHTML = "";
        try {
          const [stats, recent] = await Promise.all([
            fetchJson("/api/stats"),
            fetchJson("/api/recent?limit=10"),
          ]);
          const kinds = Object.entries(stats.by_kind || {}).map(([k, v]) => `${k}: ${v}`).join(", ");
          libraryStats.textContent =
            `Items: ${stats.items} · Sources: ${stats.sources} · Tags: ${stats.tags}` +
            (kinds ? ` · ${kinds}` : "");
          for (const item of (recent.items || [])) {
            const el = document.createElement("div");
            el.className = "result";
            const meta = document.createElement("div");
            meta.className = "meta";
            meta.textContent = `${item.source}${item.date ? " · " + item.date : ""}`;
            const body = document.createElement("div");
            body.textContent = item.snippet || "";
            el.appendChild(meta);
            el.appendChild(body);
            libraryRecent.appendChild(el);
          }
          libraryStatus.textContent = "";
        } catch (error) {
          libraryStatus.textContent = error.message;
        }
      });

      const backupFile = document.getElementById("backupFile");
      const importBackupBtn = document.getElementById("importBackupBtn");
      const convertBackupBtn = document.getElementById("convertBackupBtn");
      const backupStatus = document.getElementById("backupStatus");
      const backupDownload = document.getElementById("backupDownload");

      async function runBackup(endpoint, importing) {
        const file = backupFile.files && backupFile.files[0];
        if (!file) {
          backupStatus.textContent = "اول یک فایل .json یا .zip انتخاب کن.";
          return;
        }
        backupStatus.textContent = importing
          ? "در حال آپلود و ایندکس... (ممکن است کمی طول بکشد)"
          : "در حال تبدیل به Markdown...";
        backupDownload.innerHTML = "";
        try {
          const data = await fetchJson(endpoint, {
            method: "POST",
            headers: { "content-type": "application/octet-stream", "X-Filename": file.name },
            body: file,
          });
          if (importing) {
            backupStatus.textContent =
              `وارد شد: ${data.messages} پیام از ${data.chats} چت` +
              (data.media ? ` (${data.media} مدیا OCR/رونویسی شد)` : "") +
              `. حالا با Search/Ask قابل جستجوست.`;
          } else {
            backupStatus.textContent = `تبدیل شد: ${data.total_messages} پیام از ${data.chats} چت (بدون ایندکس).`;
          }
          const blob = new Blob([data.markdown || ""], { type: "text/markdown" });
          const link = document.createElement("a");
          link.href = URL.createObjectURL(blob);
          link.download = "telegram-backup.md";
          link.textContent = "دانلود فایل Markdown";
          backupDownload.appendChild(link);
        } catch (error) {
          backupStatus.textContent = error.message;
        }
      }

      importBackupBtn.addEventListener("click", () => runBackup("/api/backup/import", true));
      convertBackupBtn.addEventListener("click", () => runBackup("/api/backup/convert", false));

      const loadSourcesBtn = document.getElementById("loadSourcesBtn");
      const sourcesStatus = document.getElementById("sourcesStatus");
      const sourcesList = document.getElementById("sourcesList");

      loadSourcesBtn.addEventListener("click", async () => {
        sourcesStatus.textContent = "در حال بارگذاری...";
        sourcesList.innerHTML = "";
        try {
          const data = await fetchJson("/api/sources");
          const sources = data.sources || [];
          sourcesStatus.textContent = sources.length ? "" : "هنوز منبعی ایندکس نشده.";
          for (const s of sources) {
            const el = document.createElement("div");
            el.className = "result";
            const meta = document.createElement("div");
            meta.className = "meta";
            meta.textContent = `${s.items} item(s)`;
            const body = document.createElement("div");
            body.textContent = s.channel_title || s.channel_url;
            el.appendChild(meta);
            el.appendChild(body);
            sourcesList.appendChild(el);
          }
        } catch (error) {
          sourcesStatus.textContent = error.message;
        }
      });

      loadSettings().catch(error => {
        settingsStatus.textContent = error.message;
      });
    </script>
  </body>
</html>
"""


LANDING_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telegram Notebook — حافظه‌ی هوشمند تلگرام شما</title>
    <meta name="description" content="آرشیو کانال‌ها، چت‌ها و بکاپ‌های تلگرام را به یک حافظه‌ی قابل‌جستجو و قابل‌اتصال به ابزارهای AI تبدیل کن." />
    <style>
      :root {
        --bg: #f6f0e7;
        --card: rgba(255, 252, 247, 0.9);
        --ink: #1d1b19;
        --muted: #6c6257;
        --accent: #0d7c66;
        --accent-2: #b45f06;
        --line: rgba(29, 27, 25, 0.12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Vazirmatn", Tahoma, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(13,124,102,0.16), transparent 30%),
          radial-gradient(circle at bottom right, rgba(180,95,6,0.16), transparent 32%),
          var(--bg);
      }
      a { color: var(--accent); text-decoration: none; }
      .wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 72px; }
      nav {
        display: flex; align-items: center; justify-content: space-between;
        padding: 8px 0 24px; gap: 12px; flex-wrap: wrap;
      }
      .brand { font-weight: 700; font-size: 1.25rem; letter-spacing: -0.02em; }
      .nav-links { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
      .btn {
        display: inline-block; padding: 12px 22px; border-radius: 999px;
        background: linear-gradient(120deg, var(--accent), #0c5d4f);
        color: white; border: none; cursor: pointer; font: inherit; font-weight: 600;
      }
      .btn.secondary { background: linear-gradient(120deg, var(--accent-2), #8a4805); }
      .btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
      .hero { padding: 36px 0 12px; }
      .hero h1 {
        margin: 0 0 10px; font-size: clamp(2.2rem, 6vw, 4.4rem);
        line-height: 1.02; letter-spacing: -0.03em;
      }
      .hero p { color: var(--muted); font-size: 1.15rem; line-height: 1.9; max-width: 56ch; }
      .cta { display: flex; gap: 12px; margin-top: 22px; flex-wrap: wrap; }
      section.block { margin-top: 56px; }
      h2 { font-size: clamp(1.5rem, 3vw, 2.2rem); letter-spacing: -0.02em; margin: 0 0 18px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
      .card {
        backdrop-filter: blur(16px); background: var(--card);
        border: 1px solid var(--line); border-radius: 22px; padding: 20px;
        box-shadow: 0 18px 50px rgba(29, 27, 25, 0.08);
      }
      .card h3 { margin: 0 0 8px; font-size: 1.2rem; }
      .card p { color: var(--muted); line-height: 1.8; margin: 0; }
      .steps { counter-reset: step; display: grid; gap: 14px; }
      .step { display: flex; gap: 14px; align-items: flex-start; }
      .step .num {
        flex: 0 0 auto; width: 38px; height: 38px; border-radius: 50%;
        display: grid; place-items: center; font-weight: 700; color: white;
        background: linear-gradient(120deg, var(--accent), #0c5d4f);
      }
      .step .num.two { background: linear-gradient(120deg, var(--accent-2), #8a4805); }
      .panel {
        background: var(--card); border: 1px solid var(--line); border-radius: 22px;
        padding: 28px; box-shadow: 0 18px 50px rgba(29, 27, 25, 0.08);
      }
      footer { margin-top: 64px; padding-top: 22px; border-top: 1px solid var(--line); color: var(--muted); }
      .muted { color: var(--muted); }
    </style>
  </head>
  <body>
    <div class="wrap">
      <nav>
        <div class="brand">📓 Telegram Notebook</div>
        <div class="nav-links">
          <a href="#features">امکانات</a>
          <a href="#how">چطور کار می‌کند</a>
          <a href="#backup">Import بکاپ</a>
          <a class="btn ghost" href="/app">ورود به اپ</a>
        </div>
      </nav>

      <header class="hero">
        <h1>تلگرام شما، به یک حافظه‌ی هوشمند تبدیل می‌شود.</h1>
        <p>
          کانال‌ها، چت‌ها، پیام‌های فورواردشده و حتی <b>فایل بکاپ تلگرام</b> را وارد کن؛
          همه به متن قابل‌جستجو تبدیل می‌شوند و می‌توانی مثل یک NotebookLM شخصی از آرشیو
          خودت سؤال بپرسی — هم از داخل ربات تلگرام، هم از همین وب‌سایت.
        </p>
        <div class="cta">
          <a class="btn" href="/app">شروع از وب‌اپ</a>
          <a class="btn secondary" href="#backup">Import فایل بکاپ</a>
        </div>
      </header>

      <section id="features" class="block">
        <h2>چه کاری انجام می‌دهد؟</h2>
        <div class="grid">
          <div class="card">
            <h3>🔎 جستجوی معنایی</h3>
            <p>جستجوی keyword و semantic روی کل آرشیو؛ نتایج همراه با منبع و لینک پیام.</p>
          </div>
          <div class="card">
            <h3>🧠 پرسش و پاسخ (RAG)</h3>
            <p>مثل NotebookLM از آرشیو خودت سؤال بپرس و پاسخ مستند بگیر.</p>
          </div>
          <div class="card">
            <h3>🗂️ Import بکاپ تلگرام</h3>
            <p>فایل JSON/ZIP خروجی Telegram Desktop را بده تا قابل‌جستجو شود و Markdown بگیری.</p>
          </div>
          <div class="card">
            <h3>🏷️ تگ و دفترچه</h3>
            <p>قوانین keyword/AI برای تگ‌گذاری خودکار و گروه‌بندی تگ‌ها زیر یک collection.</p>
          </div>
          <div class="card">
            <h3>📨 Forwarded Inbox</h3>
            <p>هر پیام/فایل را به ربات فوروارد کن تا ذخیره، پردازش و قابل‌جستجو شود.</p>
          </div>
          <div class="card">
            <h3>🔌 MCP برای ابزارهای AI</h3>
            <p>آرشیو را به‌صورت read-only به Claude، Cursor و سایر AI clientها وصل کن.</p>
          </div>
        </div>
      </section>

      <section id="how" class="block">
        <h2>چطور کار می‌کند</h2>
        <div class="steps">
          <div class="step">
            <div class="num">۱</div>
            <div><b>وارد کن.</b> یک کانال را ingest کن، پیام‌ها را فوروارد کن، یا فایل بکاپ تلگرام را آپلود کن.</div>
          </div>
          <div class="step">
            <div class="num two">۲</div>
            <div><b>پردازش.</b> محتوا به متن تبدیل، chunk و ایندکس می‌شود و با قوانین تو تگ می‌خورد.</div>
          </div>
          <div class="step">
            <div class="num">۳</div>
            <div><b>بپرس.</b> از وب‌اپ یا ربات تلگرام جستجو کن، سؤال بپرس، خلاصه و timeline بگیر.</div>
          </div>
        </div>
      </section>

      <section id="backup" class="block">
        <h2>Import فایل بکاپ تلگرام</h2>
        <div class="panel">
          <p class="muted" style="line-height:1.9;">
            در <b>Telegram Desktop</b> به <b>Settings → Advanced → Export Telegram data</b>
            (یا روی یک چت: <b>Export chat history</b>) برو و فرمت را روی
            <b>Machine-readable JSON</b> بگذار. سپس فایل <code>result.json</code> یا
            <code>.zip</code> خروجی را در وب‌اپ آپلود کن. تمام پیام‌ها قابل‌جستجو می‌شوند و
            یک نسخه‌ی <b>Markdown</b> تحویل می‌گیری.
          </p>
          <div class="cta">
            <a class="btn" href="/app">باز کردن وب‌اپ و آپلود بکاپ</a>
          </div>
        </div>
      </section>

      <footer>
        <p>
          ساخته‌شده برای تبدیل تلگرام به «حافظه‌ای برای دستیارهای AI».
          <a href="/app">ورود به اپ</a> ·
          <a href="https://github.com/shm379/telegram-notebooklm-mvp">سورس روی GitHub</a>
        </p>
      </footer>
    </div>
  </body>
</html>
"""


def _query_int(query: dict, name: str, *, default: int, lo: int, hi: int) -> int:
    """Read a clamped integer query param, falling back to ``default`` on bad input."""
    try:
        value = int(query.get(name, [str(default)])[0])
    except (ValueError, TypeError):
        value = default
    return max(lo, min(hi, value))


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "TelegramNotebook/0.2"

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str, status: int = 200) -> None:
        raw = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _client_is_loopback(self) -> bool:
        host = self.client_address[0] if self.client_address else ""
        return host in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

    def _presented_token(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer "):].strip()
        token = self.headers.get("X-API-Token")
        return token.strip() if token else None

    def _require_auth(self) -> bool:
        """Guard for /api endpoints. Returns True if the request may proceed.

        When WEB_API_TOKEN is configured, a matching bearer/X-API-Token is required.
        When it is not configured, only loopback clients are allowed so the API is not
        exposed unauthenticated over the network. Sends a 401 and returns False on failure.
        """
        configured = state.settings.web_api_token
        if configured:
            presented = self._presented_token()
            if presented and hmac.compare_digest(presented, configured):
                return True
            self._send_json({"detail": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
            return False
        if self._client_is_loopback():
            return True
        self._send_json(
            {"detail": "Unauthorized: set WEB_API_TOKEN to allow non-local access to the API."},
            status=HTTPStatus.UNAUTHORIZED,
        )
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(LANDING_HTML)
                return
            if parsed.path in ("/app", "/app/"):
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/health":
                config = state.runtime_config()
                self._send_json(
                    {
                        "ok": True,
                        **asdict(config),
                    }
                )
                return
            if parsed.path == "/api/settings":
                if not self._require_auth():
                    return
                self._send_json(asdict(state.runtime_config()))
                return
            if parsed.path == "/api/models":
                if not self._require_auth():
                    return
                query = parse_qs(parsed.query)
                provider = (query.get("provider", ["gemini"])[0] or "gemini").lower()
                capability = query.get("capability", [None])[0]
                models = state.list_models(provider=provider, capability=capability)
                self._send_json(
                    {
                        "provider": provider,
                        "capability": capability,
                        "models": models,
                    }
                )
                return
            if parsed.path == "/api/stats":
                if not self._require_auth():
                    return
                self._send_json(state.repository.archive_stats(owner_id=WEB_OWNER_ID))
                return
            if parsed.path == "/api/recent":
                if not self._require_auth():
                    return
                limit = _query_int(parse_qs(parsed.query), "limit", default=10, lo=1, hi=50)
                items = state.repository.timeline_items(owner_id=WEB_OWNER_ID, limit=limit)
                self._send_json({"items": recent_rows(items, limit=limit)})
                return
            if parsed.path == "/api/timeline":
                if not self._require_auth():
                    return
                query = parse_qs(parsed.query)
                granularity = "day" if query.get("granularity", ["month"])[0] == "day" else "month"
                items = state.repository.timeline_items(owner_id=WEB_OWNER_ID)
                self._send_json({"granularity": granularity, "periods": build_timeline(items, granularity=granularity)})
                return
            if parsed.path == "/api/sources":
                if not self._require_auth():
                    return
                self._send_json({"sources": state.repository.source_counts(owner_id=WEB_OWNER_ID)})
                return
            self._send_json({"detail": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.exception("GET %s failed", parsed.path)
            self._send_json({"detail": str(exc)}, status=400)

    def _handle_backup_upload(self, parsed, *, ingest: bool) -> None:
        """Read an uploaded Telegram backup and return its Markdown.

        The file arrives as the raw request body with its name in ``X-Filename``
        (or ``?filename=``), so there is no multipart parsing to do. With
        ``ingest=True`` the backup is also indexed into the shared web archive
        (``WEB_OWNER_ID``) with media OCR/transcription; with ``ingest=False`` it
        is only converted to Markdown (no indexing, no LLM calls).
        """
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        if not body:
            self._send_json({"detail": "Empty upload"}, status=400)
            return
        filename = self.headers.get("X-Filename") or parse_qs(parsed.query).get("filename", [""])[0]
        try:
            chats = parse_export(read_export(body, filename))
        except Exception as exc:
            self._send_json({"detail": f"Could not read backup: {exc}"}, status=400)
            return

        payload: dict[str, object] = {
            "ok": True,
            "chats": len(chats),
            "total_messages": count_messages(chats),
        }
        if ingest:
            payload["media"] = self._enrich_web_backup_media(chats, body, filename)
            try:
                result = asyncio.run(state.pipeline.ingest_backup(owner_id=WEB_OWNER_ID, chats=chats))
            except Exception as exc:
                logger.exception("Backup import failed")
                self._send_json({"detail": str(exc)}, status=400)
                return
            payload["messages"] = result["messages"]
        payload["markdown"] = render_markdown(chats)
        self._send_json(payload)

    def _enrich_web_backup_media(self, chats, body: bytes, filename: str | None) -> int:
        """OCR/transcribe media bundled in a backup zip for the web import path.

        Uses the web instance's transcription service plus a Gemini extractor.
        No-op for raw JSON or when no Gemini key is configured.
        """
        zf = open_backup_zip(body, filename)
        if zf is None:
            return 0
        try:
            tx = state.transcription
            ex = ExtractionService(
                provider="gemini",
                api_key=state.settings.gemini_api_key,
                model=state.settings.transcription_model,
            )
            tx_enabled = bool(tx and tx.enabled)
            ex_enabled = bool(ex.enabled)
            if not (tx_enabled or ex_enabled):
                return 0

            def transcribe(path):
                return tx.transcribe_media(path, path.parent)

            def ocr(path):
                return ex.extract(path)

            tmpdir = Path(tempfile.mkdtemp(prefix="webbackup_"))
            try:
                extractor = make_zip_extractor(
                    zf, tmpdir,
                    transcribe=transcribe if tx_enabled else None,
                    ocr=ocr if ex_enabled else None,
                )
                return enrich_with_media(chats, extract=extractor)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            logger.exception("Web backup media enrichment failed")
            return 0
        finally:
            zf.close()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._require_auth():
            return

        if parsed.path in ("/api/backup/import", "/api/backup/convert"):
            try:
                self._handle_backup_upload(parsed, ingest=parsed.path.endswith("/import"))
            except Exception as exc:
                logger.exception("Backup upload handler failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json({"detail": "Invalid JSON payload"}, status=400)
            return

        if parsed.path == "/api/settings":
            updates: dict[str, str | None] = {}
            for key in (
                "transcription_provider",
                "transcription_model",
                "embedding_provider",
                "embedding_model",
                "gemini_api_key",
                "openai_api_key",
            ):
                if key in payload:
                    value = str(payload[key]).strip() if payload[key] is not None else ""
                    updates[key.upper()] = value
            if updates:
                upsert_env_values(updates)
                state.reload()
            self._send_json(asdict(state.runtime_config()))
            return

        if parsed.path == "/api/channels/ingest":
            channel_url = str(payload.get("channel_url", "")).strip()
            limit = int(payload.get("limit", 50))
            if not channel_url:
                self._send_json({"detail": "channel_url is required"}, status=400)
                return
            try:
                stats = asyncio.run(
                    state.pipeline.ingest_channel(owner_id=WEB_OWNER_ID, channel_url=channel_url, limit=limit)
                )
                self._send_json(
                    {
                        "channel_url": stats.channel_url,
                        "channel_title": stats.channel_title,
                        "processed_messages": stats.processed_messages,
                        "processed_media": stats.processed_media,
                        "skipped_media": stats.skipped_media,
                    }
                )
            except Exception as exc:
                logger.exception("Ingest request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/search":
            query = str(payload.get("query", "")).strip()
            channel_url = payload.get("channel_url")
            top_k = int(payload.get("top_k", state.settings.default_result_limit))
            if not query:
                self._send_json({"detail": "query is required"}, status=400)
                return
            try:
                results = state.search_service.search(
                    owner_id=WEB_OWNER_ID,
                    query=query,
                    channel_url=str(channel_url).strip() if channel_url else None,
                    top_k=top_k,
                    vertex_config=state.vertex_search_config(),
                )
                self._send_json(
                    {
                        "query": query,
                        "results": [result.to_dict() for result in results],
                    }
                )
            except Exception as exc:
                logger.exception("Search request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/ask":
            query = str(payload.get("query", "")).strip()
            channel_url = payload.get("channel_url")
            if not query:
                self._send_json({"detail": "query is required"}, status=400)
                return
            try:
                vertex_config = state.vertex_search_config()
                # 1. Search for relevant chunks
                results = state.search_service.search(
                    owner_id=WEB_OWNER_ID,
                    query=query,
                    channel_url=str(channel_url).strip() if channel_url else None,
                    top_k=5,
                    vertex_config=vertex_config,
                )
                # 2. Generate answer
                answer = state.search_service.generate_answer(
                    query=query,
                    results=results,
                    api_key=state.settings.gemini_api_key,
                    project_id=state.settings.vertex_project_id,
                    region=state.settings.vertex_region or "us-central1",
                )
                self._send_json(
                    {
                        "query": query,
                        "answer": answer,
                        "sources": [result.to_dict() for result in results],
                    }
                )
            except Exception as exc:
                logger.exception("Ask request failed")
                self._send_json({"detail": str(exc)}, status=400)
            return

        self._send_json({"detail": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    setup_logging()
    server = ThreadingHTTPServer((host, port), RequestHandler)
    logger.info("Serving on http://%s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    run()
