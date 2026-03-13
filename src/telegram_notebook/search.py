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
        keyword_rows = self.repository.keyword_candidates(
            query=query,
            top_k=top_k * 3,
            channel_url=channel_url,
        )

        lexical_scores: dict[int, float] = {}
        candidate_rows = {int(row["chunk_id"]): row for row in keyword_rows}
        for index, row in enumerate(keyword_rows):
            lexical_scores[int(row["chunk_id"])] = 1.0 - (index / max(len(keyword_rows), 1))

        semantic_scores: dict[int, float] = {}
        if self.embeddings.enabled:
            query_embedding = self.embeddings.embed(
                query,
                task_type="RETRIEVAL_QUERY",
            )
            if query_embedding:
                embedding_rows = self.repository.embedding_candidates(channel_url=channel_url)
                scored_rows: list[tuple[float, object]] = []
                for row in embedding_rows:
                    chunk_id = int(row["chunk_id"])
                    candidate_rows.setdefault(chunk_id, row)
                    similarity = cosine_similarity(
                        query_embedding,
                        json.loads(row["embedding_json"]),
                    )
                    scored_rows.append((similarity, row))
                scored_rows.sort(key=lambda item: item[0], reverse=True)
                for index, (similarity, row) in enumerate(scored_rows[: top_k * 5]):
                    chunk_id = int(row["chunk_id"])
                    position_bonus = 1.0 - (index / max(top_k * 5, 1))
                    semantic_scores[chunk_id] = max(similarity, 0.0) * 0.8 + position_bonus * 0.2

        scored: list[tuple[float, SearchResult]] = []
        for chunk_id, row in candidate_rows.items():
            lexical = lexical_scores.get(chunk_id, 0.0)
            semantic = semantic_scores.get(chunk_id, 0.0)
            final_score = lexical * 0.45 + semantic * 0.55
            if final_score <= 0:
                continue
            scored.append(
                (
                    final_score,
                    SearchResult(
                        score=round(final_score, 4),
                        channel_title=row["channel_title"],
                        channel_url=row["channel_url"],
                        message_url=row["message_url"],
                        media_kind=row["media_kind"],
                        file_name=row["file_name"],
                        chunk_text=row["chunk_text"],
                        caption=row["caption"],
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored[:top_k]]
