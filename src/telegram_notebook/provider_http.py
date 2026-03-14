from __future__ import annotations

import base64
import json
import ssl
import subprocess
from pathlib import Path
from urllib import parse, request

import certifi

from .media import guess_mime_type


GEMINI_BASE_URL = "https://aiplatform.googleapis.com/v1/publishers/google"


def get_gcloud_access_token() -> str | None:
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error getting gcloud access token: {e}")
        return None


def _json_request(
    *,
    url: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    use_gcloud_auth: bool = False,
) -> dict[str, object] | list[dict[str, object]]:
    raw = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    
    if use_gcloud_auth:
        token = get_gcloud_access_token()
        if token:
            req_headers["Authorization"] = f"Bearer {token}"
            
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


def vertex_ai_search(
    *,
    api_key: str | None = None,
    project_id: str,
    region: str,
    index_endpoint_id: str,
    deployed_index_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, object]]:
    # https://cloud.google.com/vertex-ai/docs/vector-search/query-index-public-endpoint
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexEndpoints/{index_endpoint_id}:findNeighbors?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexEndpoints/{index_endpoint_id}:findNeighbors"
        use_gcloud = True
    
    payload = {
        "deployed_index_id": deployed_index_id,
        "queries": [
            {
                "datapoint": {"feature_vector": query_embedding},
                "neighbor_count": top_k
            }
        ]
    }
    
    data = _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)
    results = []
    if isinstance(data, dict):
        # Vertex AI returns a list of nearestNeighbors for each query
        nearest_neighbors_list = data.get("nearestNeighbors", [])
        if nearest_neighbors_list:
            neighbors = nearest_neighbors_list[0].get("neighbors", [])
            for n in neighbors:
                results.append({
                    "id": n.get("datapoint", {}).get("datapointId"),
                    "distance": n.get("distance")
                })
    return results


def vertex_ai_upsert(
    *,
    api_key: str | None = None,
    project_id: str,
    region: str,
    index_id: str,
    datapoints: list[dict[str, object]],
) -> None:
    # https://cloud.google.com/vertex-ai/docs/vector-search/upsert-datapoints
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexes/{index_id}:upsertDatapoints?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/indexes/{index_id}:upsertDatapoints"
        use_gcloud = True
    
    payload = {
        "datapoints": datapoints
    }
    
    _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)


def gemini_embed_text(
    *,
    api_key: str | None = None,
    model: str,
    text: str,
    task_type: str | None = None,
    project_id: str | None = None,
    region: str = "us-central1",
) -> list[float] | None:
    # Vertex AI Embedding API
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/publishers/google/models/{model}:predict?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:predict"
        use_gcloud = True
        
    payload = {
        "instances": [{"content": text}],
    }
    data = _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)
    if isinstance(data, dict):
        predictions = data.get("predictions", [])
        if predictions:
            return predictions[0].get("embeddings", {}).get("values", [])
    return None


def gemini_transcribe_audio(
    *,
    api_key: str | None = None,
    model: str,
    audio_path: Path,
    project_id: str | None = None,
    region: str = "us-central1",
) -> str:
    if "flash" not in model:
        model = "gemini-2.5-flash-lite"
        
    mime_type = guess_mime_type(audio_path) or "audio/mpeg"
    encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:streamGenerateContent"
        use_gcloud = True
    
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
    
    data = _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)
    
    full_text = []
    if isinstance(data, list):
        for chunk in data:
            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        full_text.append(p["text"])
    elif isinstance(data, dict):
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    full_text.append(p["text"])
                
    return "".join(full_text).strip()


def gemini_generate_content(
    *,
    api_key: str | None = None,
    model: str = "gemini-1.5-flash",
    prompt: str,
    project_id: str | None = None,
    region: str = "us-central1",
) -> str:
    if api_key:
        url = f"https://{region}-aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={api_key}"
        use_gcloud = False
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:streamGenerateContent"
        use_gcloud = True
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40,
        }
    }
    
    data = _json_request(url=url, payload=payload, use_gcloud_auth=use_gcloud)
    
    full_text = []
    if isinstance(data, list):
        for chunk in data:
            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        full_text.append(p["text"])
    elif isinstance(data, dict):
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    full_text.append(p["text"])
                
    return "".join(full_text).strip()
