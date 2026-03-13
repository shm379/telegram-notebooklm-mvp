from __future__ import annotations

import base64
import json
import ssl
from pathlib import Path
from urllib import parse, request

import certifi

from .media import guess_mime_type


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _json_request(
    *,
    url: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    raw = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=raw, headers=req_headers)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with request.urlopen(req, timeout=90, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def list_gemini_models(*, api_key: str, capability: str | None = None) -> list[dict[str, object]]:
    url = f"{GEMINI_BASE_URL}/models?key={parse.quote(api_key)}"
    data = _json_request(url=url)
    models = []
    for model in data.get("models", []):
        supported = list(model.get("supportedGenerationMethods", []) or [])
        if capability and capability not in supported:
            continue
        name = str(model.get("name", ""))
        models.append(
            {
                "id": name.removeprefix("models/"),
                "name": name,
                "display_name": model.get("displayName"),
                "description": model.get("description"),
                "supported_actions": supported,
            }
        )
    models.sort(key=lambda item: item["id"])
    return models


def list_openai_models(*, api_key: str, capability: str | None = None) -> list[dict[str, object]]:
    data = _json_request(
        url=f"{OPENAI_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models = []
    for model in data.get("data", []):
        model_id = str(model.get("id", ""))
        if capability == "embeddings" and "embedding" not in model_id:
            continue
        if capability == "transcription" and "transcribe" not in model_id:
            continue
        models.append(
            {
                "id": model_id,
                "name": model_id,
                "display_name": model_id,
                "description": None,
                "supported_actions": [],
            }
        )
    models.sort(key=lambda item: item["id"])
    return models


def gemini_embed_text(
    *,
    api_key: str,
    model: str,
    text: str,
    task_type: str | None = None,
) -> list[float] | None:
    payload: dict[str, object] = {
        "content": {"parts": [{"text": text}]},
    }
    if task_type:
        payload["taskType"] = task_type
    data = _json_request(
        url=f"{GEMINI_BASE_URL}/models/{parse.quote(model)}:embedContent?key={parse.quote(api_key)}",
        payload=payload,
    )
    embedding = data.get("embedding", {})
    values = embedding.get("values", [])
    return list(values) if values else None


def gemini_transcribe_audio(
    *,
    api_key: str,
    model: str,
    audio_path: Path,
) -> str:
    mime_type = guess_mime_type(audio_path) or "audio/mpeg"
    encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    data = _json_request(
        url=f"{GEMINI_BASE_URL}/models/{parse.quote(model)}:generateContent?key={parse.quote(api_key)}",
        payload={
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Generate a verbatim transcript of the spoken audio. Return only the transcript text."
                        },
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": encoded_audio,
                            }
                        },
                    ]
                }
            ]
        },
    )
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    texts = [str(part.get("text", "")).strip() for part in parts if part.get("text")]
    return "\n".join(texts).strip()
