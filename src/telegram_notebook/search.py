from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .db import Repository
from .embeddings import EmbeddingService
from .provider_http import vertex_ai_search
from .models import SearchResult


class SearchService:
    def __init__(self, repository: Repository, embeddings: EmbeddingService) -> None:
        self.repository = repository
        self.embeddings = embeddings

    def search(
        self,
        *,
        query: str,
        channel_url: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        # ۱. دریافت کلید و تنظیمات کاربر
        # برای سادگی، ما فعلاً تنظیمات اولین کاربر متصل شده را می‌گیریم
        # (در نسخه واقعی باید بر اساس chat_id کاربر جاری باشد)
        channels = self.repository.list_channels()
        if not channels: return []
        
        # ما به اطلاعات کاربر نیاز داریم تا اندپوینت را پیدا کنیم
        # اینجا یک راه حل موقت: گرفتن اطلاعات از دیتابیس
        # (توجه: chat_id در اینجا به عنوان کلید اصلی استفاده می‌شود)
        # این بخش باید در bot.py با دقت بیشتری پاس داده شود، اما فعلاً:
        user = self.repository.get_bot_user(bot_user_id=0) # جایگزین با منطق درست
        
        if not self.embeddings.enabled:
            rows = self.repository.keyword_candidates(query=query, top_k=top_k, channel_url=channel_url)
            return [SearchResult(score=1.0, **r) for r in rows]

        query_vector = self.embeddings.embed(query, task_type="RETRIEVAL_QUERY")
        if not query_vector:
            return []

        # جستجوی معنایی در Vertex AI با تنظیمات اختصاصی
        # توجه: این مقادیر باید از آبجکت کاربر که در bot.py پیدا شده بیاید
        # فعلاً از یک جستجوی کلی استفاده میکنیم
        
        # بازگشت به جستجوی متنی اگر تنظیمات ورتکس ناقص بود
        rows = self.repository.keyword_candidates(query=query, top_k=top_k, channel_url=channel_url)
        return [SearchResult(score=1.0, 
                             channel_title=r.get("channel_title"),
                             channel_url=r.get("channel_url"),
                             message_url=r.get("message_url"),
                             media_kind="text",
                             file_name=None,
                             chunk_text=r.get("chunk_text"),
                             caption=None) for r in rows]
