from __future__ import annotations

import math
from typing import Sequence

from .provider_http import gemini_embed_text


class EmbeddingService:
    def __init__(self, *, provider: str, api_key: str | None, model: str) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.client = None

    @property
    def enabled(self) -> bool:
        return self.api_key is not None

    def _get_client(self):
        if not self.api_key:
            return None
        if self.client is None:
            if self.provider != "gemini":
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
        return self.client

    def embed(self, text: str, *, task_type: str | None = None) -> list[float] | None:
        if not self.api_key:
            return None

        if self.provider == "gemini":
            return gemini_embed_text(
                api_key=self.api_key,
                model=self.model,
                text=text,
                task_type=task_type,
            )

        client = self._get_client()
        if not client:
            return None
        response = client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return list(response.data[0].embedding)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
