from __future__ import annotations

import base64
import json
import ssl
from pathlib import Path
from urllib import parse, request

import certifi

from .media import guess_mime_type


GEMINI_BASE_URL = "https://aiplatform.googleapis.com/v1/publishers/google"


def _json_request(
    *,
    url: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object] | list[dict[str, object]]:
    raw = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        raw = json.dumps(payload).encode("utf-8")
    
    req = request.Request(url, data=raw, headers=req_headers)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    try:
        with request.urlopen(req, timeout=90, context=ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Vertex AI Request Error: {e} to {url}")
        raise


def gemini_embed_text(
    *,
    api_key: str,
    model: str,
    text: str,
    task_type: str | None = None,
) -> list[float] | None:
    # Vertex AI Embedding API
    url = f"{GEMINI_BASE_URL}/models/{model}:predict?key={api_key}"
    payload = {
        "instances": [{"content": text}],
    }
    data = _json_request(url=url, payload=payload)
    if isinstance(data, dict):
        predictions = data.get("predictions", [])
        if predictions:
            return predictions[0].get("embeddings", {}).get("values", [])
    return None


def gemini_transcribe_audio(
    *,
    api_key: str,
    model: str,
    audio_path: Path,
) -> str:
    # استفاده از مدل درخواستی کاربر یا پیش‌فرض ورتکس
    if "flash" not in model:
        model = "gemini-2.0-flash-lite"
        
    mime_type = guess_mime_type(audio_path) or "audio/mpeg"
    encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    
    # دقیقاً مطابق فرمت ارسالی شما
    url = f"{GEMINI_BASE_URL}/models/{model}:streamGenerateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Generate a verbatim transcript of the spoken audio. Return only the transcript text."},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": encoded_audio,
                        }
                    },
                ]
            }
        ]
    }
    
    data = _json_request(url=url, payload=payload)
    
    full_text = []
    if isinstance(data, list):
        for chunk in data:
            parts = chunk.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    full_text.append(p["text"])
    elif isinstance(data, dict):
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                full_text.append(p["text"])
                
    return "".join(full_text).strip()
