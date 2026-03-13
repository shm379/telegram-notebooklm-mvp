from __future__ import annotations

import json

from .db import Repository
from .embeddings import EmbeddingService, cosine_similarity
from .models import SearchResult


class SearchService:
    def __init__(self, repository: Repository, embeddings: EmbeddingService) -> None:
        self.repository = repository
        self.embeddings = embeddings

    def search(self, *, query: str, channel_url: str | None, top_k: int) -> list[SearchResult]:
        # ۱. دریافت کاندیداهای متنی (Lexical)
        keyword_rows = self.repository.keyword_candidates(
            query=query,
            top_k=top_k * 3,
            channel_url=channel_url,
        )

        lexical_scores: dict[int, float] = {}
        candidate_rows = {int(row["chunk_id"]): row for row in keyword_rows} if "chunk_id" in (keyword_rows[0] if keyword_rows else {}) else {}
        
        # ۲. جستجوی معنایی در Vertex AI
        semantic_scores: dict[int, float] = {}
        if self.embeddings.enabled:
            query_embedding = self.embeddings.embed(query, task_type="RETRIEVAL_QUERY")
            if query_embedding:
                try:
                    # فراخوانی ورتکس برای پیدا کردن نزدیک‌ترین همسایه‌ها
                    vertex_results = vertex_ai_search(
                        api_key=self.embeddings.api_key,
                        project_id="aesthetic-petal-486120-j2",
                        region="us-central1",
                        index_endpoint_id="7123456789012345678", # این آیدی باید توسط شما جایگزین شود
                        deployed_index_id="my_telegram_index_123", # این آیدی باید توسط شما جایگزین شود
                        query_embedding=query_embedding,
                        top_k=top_k * 5
                    )
                    
                    for v in vertex_results:
                        chunk_id = int(v["id"])
                        similarity = float(v["distance"])
                        semantic_scores[chunk_id] = similarity
                        
                        # اگر در کاندیداهای متنی نبود، اطلاعاتش را از دیتابیس بگیر
                        if chunk_id not in candidate_rows:
                            row = self.repository.get_chunk_with_metadata(chunk_id)
                            if row:
                                candidate_rows[chunk_id] = row
                except Exception as e:
                    print(f"Vertex Search Failed: {e}")

        # ۳. ترکیب نتایج و امتیازدهی نهایی
        scored: list[tuple[float, SearchResult]] = []
        for chunk_id, row in candidate_rows.items():
            lexical = lexical_scores.get(chunk_id, 0.0)
            semantic = semantic_scores.get(chunk_id, 0.0)
            
            # امتیاز نهایی (ترکیب ورتکس و لکسیکال)
            final_score = max(lexical, semantic)
            if final_score <= 0 and not lexical: continue
            
            scored.append(
                (
                    final_score,
                    SearchResult(
                        score=round(final_score, 4),
                        channel_title=row.get("channel_title"),
                        channel_url=row.get("channel_url", ""),
                        message_url=row.get("message_url"),
                        media_kind=row.get("media_kind", "text"),
                        file_name=row.get("file_name"),
                        chunk_text=row.get("chunk_text", ""),
                        caption=row.get("caption"),
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored[:top_k]]
