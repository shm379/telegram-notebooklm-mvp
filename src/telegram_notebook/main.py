from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import get_settings, upsert_env_values
from .db import Repository, connect
from .embeddings import EmbeddingService
from .model_catalog import ModelCatalogService
from .pipeline import IngestionPipeline
from .search import SearchService
from .transcription import TranscriptionService


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
<html lang="fa">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telegram Notebook</title>
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

      async function fetchJson(url, options = undefined) {
        const response = await fetch(url, options);
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

      loadSettings().catch(error => {
        settingsStatus.textContent = error.message;
      });
    </script>
  </body>
</html>
"""


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

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
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
                self._send_json(asdict(state.runtime_config()))
                return
            if parsed.path == "/api/models":
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
            self._send_json({"detail": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"detail": str(exc)}, status=400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
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
                    state.pipeline.ingest_channel(channel_url=channel_url, limit=limit)
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
                    query=query,
                    channel_url=str(channel_url).strip() if channel_url else None,
                    top_k=top_k,
                )
                self._send_json(
                    {
                        "query": query,
                        "results": [result.to_dict() for result in results],
                    }
                )
            except Exception as exc:
                self._send_json({"detail": str(exc)}, status=400)
            return

        if parsed.path == "/api/ask":
            query = str(payload.get("query", "")).strip()
            channel_url = payload.get("channel_url")
            if not query:
                self._send_json({"detail": "query is required"}, status=400)
                return
            try:
                # 1. Search for relevant chunks
                results = state.search_service.search(
                    query=query,
                    channel_url=str(channel_url).strip() if channel_url else None,
                    top_k=5,
                )
                # 2. Generate answer
                answer = state.search_service.generate_answer(
                    query=query,
                    results=results,
                    api_key=state.settings.gemini_api_key,
                )
                self._send_json(
                    {
                        "query": query,
                        "answer": answer,
                        "sources": [result.to_dict() for result in results],
                    }
                )
            except Exception as exc:
                self._send_json({"detail": str(exc)}, status=400)
            return

        self._send_json({"detail": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
