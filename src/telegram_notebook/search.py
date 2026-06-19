from __future__ import annotations

import logging
from typing import Any

from .db import Repository
from .embeddings import EmbeddingService
from .models import SearchResult
from .provider_http import gemini_generate_content, vertex_ai_search

logger = logging.getLogger(__name__)


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
هر ادعا یا اطلاعاتی که می‌گویید را با ذکر شماره‌ی منبعِ مربوط داخل کروشه مستند کنید، دقیقاً به شکل [1] یا [2]؛ اگر یک جمله از چند منبع آمده، همه را ذکر کنید مثل [1][3].
فقط از همین منابع استفاده کنید و هیچ شماره‌ای خارج از فهرست بالا نسازید.
اگر اطلاعات کافی نیست، صادقانه بگویید که بر اساس منابع موجود نمی‌توانید پاسخ کاملی بدهید.

اطلاعات استخراج شده:
{context}

سوال کاربر:
{query}

پاسخ مستند (به زبان فارسی، با ارجاع‌های [شماره]):"""

        return gemini_generate_content(
            api_key=api_key,
            prompt=prompt,
            project_id=project_id,
            region=region
        )

    @staticmethod
    def _build_summary_prompt(scope_label: str, items: list[dict[str, Any]], per_item_chars: int = 1500) -> str:
        parts = []
        for i, item in enumerate(items, 1):
            source = item.get("channel_title") or item.get("channel_url") or "Unknown Source"
            text = (item.get("text") or "").strip()[:per_item_chars]
            parts.append(f"[{i}] ({source}) {text}")
        context = "\n\n".join(parts)
        return f"""شما یک دستیار هوشمند هستید که آرشیو تلگرام کاربر را خلاصه می‌کنید.
بر اساس آیتم‌های زیر، یک خلاصه‌ی ساختارمند و مفید درباره‌ی «{scope_label}» بنویسید.
خلاصه باید شامل موضوعات اصلی، نکات کلیدی و در صورت امکان دسته‌بندی مطالب باشد.
فقط بر اساس همین آیتم‌ها بنویسید و چیزی از خودتان اضافه نکنید.

آیتم‌ها:
{context}

خلاصه (به زبان فارسی، با تیترها و bullet در صورت نیاز):"""

    def summarize(
        self,
        *,
        scope_label: str,
        items: list[dict[str, Any]],
        api_key: str | None = None,
        project_id: str | None = None,
        region: str = "us-central1",
    ) -> str:
        if not items:
            return "موردی برای خلاصه‌سازی پیدا نشد."
        prompt = self._build_summary_prompt(scope_label, items)
        return gemini_generate_content(
            api_key=api_key,
            prompt=prompt,
            project_id=project_id,
            region=region,
        )

    def search(
        self,
        *,
        owner_id: int,
        query: str,
        channel_url: str | None = None,
        tag: str | None = None,
        top_k: int = 5,
        vertex_config: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not self.embeddings.enabled:
            rows = self.repository.keyword_candidates(owner_id=owner_id, query=query, top_k=top_k, channel_url=channel_url, tag=tag)
            return [SearchResult(score=1.0, **r) for r in rows]

        v_proj = vertex_config.get("project_id") if vertex_config else None
        v_reg = vertex_config.get("region", "us-central1") if vertex_config else "us-central1"
        try:
            query_vector = self.embeddings.embed(
                query,
                task_type="RETRIEVAL_QUERY",
                project_id=v_proj,
                region=v_reg,
            )
        except Exception:
            logger.warning("Embedding query failed; falling back to keyword search", exc_info=True)
            query_vector = None
        if not query_vector:
            rows = self.repository.keyword_candidates(owner_id=owner_id, query=query, top_k=top_k, channel_url=channel_url, tag=tag)
            return [SearchResult(score=1.0, **r) for r in rows]

        allowed_media_ids = self.repository.media_ids_for_tag(owner_id=owner_id, tag=tag) if tag else None

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
                        if allowed_media_ids is not None and media_item_id not in allowed_media_ids:
                            continue
                        chunk = self.repository.get_chunk_by_media_and_index(
                            owner_id=owner_id, media_item_id=media_item_id, chunk_index=chunk_index
                        )
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
                    
            except Exception:
                logger.exception("Vertex AI search failed; falling back to keyword search")

        # Fallback to keyword search for now
        rows = self.repository.keyword_candidates(owner_id=owner_id, query=query, top_k=top_k, channel_url=channel_url, tag=tag)
        return [SearchResult(score=1.0, 
                             channel_title=r.get("channel_title"),
                             channel_url=r.get("channel_url"),
                             message_url=r.get("message_url"),
                             media_kind="text",
                             file_name=None,
                             chunk_text=r.get("chunk_text"),
                             caption=None) for r in rows]
