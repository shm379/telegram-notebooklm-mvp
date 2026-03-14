from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .db import Repository
from .embeddings import EmbeddingService
from .provider_http import vertex_ai_search, gemini_generate_content
from .models import SearchResult


class SearchService:
    def __init__(self, repository: Repository, embeddings: EmbeddingService) -> None:
        self.repository = repository
        self.embeddings = embeddings

    def generate_answer(
        self,
        *,
        query: str,
        results: list[SearchResult],
        api_key: str | None = None,
        project_id: str | None = None,
        region: str = "us-central1",
    ) -> str:
        if not results:
            return "متأسفانه موردی پیدا نشد."

        context_parts = []
        for i, res in enumerate(results, 1):
            source = res.channel_title or res.channel_url or "Unknown Source"
            context_parts.append(f"Source [{i}] ({source}):\n{res.chunk_text}")

        context = "\n\n".join(context_parts)
        prompt = f"""شما یک دستیار هوشمند هستید که به کاربر در تحلیل محتوای کانال‌های تلگرامی کمک می‌کنید.
بر اساس اطلاعات زیر که از جست‌وجوی معنایی استخراج شده است، به سوال کاربر پاسخ دقیق و مستند بدهید.
اگر اطلاعات کافی نیست، بگویید که بر اساس منابع موجود نمی‌توانید پاسخ کاملی بدهید.

اطلاعات استخراج شده:
{context}

سوال کاربر:
{query}

پاسخ (به زبان فارسی):"""

        return gemini_generate_content(
            api_key=api_key,
            prompt=prompt,
            project_id=project_id,
            region=region
        )

    def search(
        self,
        *,
        query: str,
        channel_url: str | None = None,
        top_k: int = 5,
        vertex_config: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not self.embeddings.enabled:
            rows = self.repository.keyword_candidates(query=query, top_k=top_k, channel_url=channel_url)
            return [SearchResult(score=1.0, **r) for r in rows]

        v_proj = vertex_config.get("project_id") if vertex_config else None
        v_reg = vertex_config.get("region", "us-central1") if vertex_config else "us-central1"
        query_vector = self.embeddings.embed(query, task_type="RETRIEVAL_QUERY", project_id=v_proj, region=v_reg)
        if not query_vector:
            return []

        if vertex_config and vertex_config.get("index_endpoint_id"):
            try:
                raw_results = vertex_ai_search(
                    api_key=vertex_config.get("api_key"),
                    project_id=vertex_config["project_id"],
                    region=vertex_config["region"],
                    index_endpoint_id=vertex_config["index_endpoint_id"],
                    deployed_index_id=vertex_config["deployed_index_id"],
                    query_embedding=query_vector,
                    top_k=top_k
                )
                
                final_results = []
                seen_messages = set()
                
                for res in raw_results:
                    # res["id"] is like "c_{media_item_id}_{chunk_index}"
                    id_parts = res["id"].split("_")
                    if len(id_parts) >= 3:
                        media_item_id = int(id_parts[1])
                        chunk_index = int(id_parts[2])
                        chunk = self.repository.get_chunk_by_media_and_index(media_item_id, chunk_index)
                        if chunk:
                            # Avoid duplicates based on message_url
                            if chunk["message_url"] in seen_messages:
                                continue
                            seen_messages.add(chunk["message_url"])
                            
                            final_results.append(SearchResult(
                                score=res.get("distance", 1.0),
                                channel_title=chunk.get("channel_title"),
                                channel_url=chunk.get("channel_url"),
                                message_url=chunk.get("message_url"),
                                media_kind=chunk.get("media_kind", "text"),
                                file_name=chunk.get("file_name"),
                                chunk_text=chunk.get("chunk_text"),
                                caption=chunk.get("caption")
                            ))
                
                if final_results:
                    return final_results
                    
            except Exception as e:
                print(f"Vertex Search Error: {e}")

        # Fallback to keyword search for now
        rows = self.repository.keyword_candidates(query=query, top_k=top_k, channel_url=channel_url)
        return [SearchResult(score=1.0, 
                             channel_title=r.get("channel_title"),
                             channel_url=r.get("channel_url"),
                             message_url=r.get("message_url"),
                             media_kind="text",
                             file_name=None,
                             chunk_text=r.get("chunk_text"),
                             caption=None) for r in rows]
