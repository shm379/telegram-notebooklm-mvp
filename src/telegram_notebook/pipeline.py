from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import split_text
from .config import Settings
from .db import Repository
from .embeddings import EmbeddingService
from .transcription import TranscriptionService


@dataclass(slots=True)
class IngestStats:
    channel_url: str
    channel_title: str | None
    processed_messages: int = 0
    processed_media: int = 0
    skipped_media: int = 0


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: Repository,
        transcription: TranscriptionService | None,
        embeddings: EmbeddingService,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.transcription = transcription
        self.embeddings = embeddings

    async def ingest_channel(
        self,
        *,
        channel_url: str,
        limit: int,
        api_id: int | None = None,
        api_hash: str | None = None,
        session_string: str | None = None,
        vertex_config: dict[str, str] | None = None,
    ) -> IngestStats:
        from .telegram_client import (
            build_client_from_session_string,
            fetch_channel_info,
            iter_all_messages,
            download_message_media
        )
        
        client = build_client_from_session_string(
            self.settings,
            session_string or "",
            api_id=api_id,
            api_hash=api_hash,
        )
        async with client:
            channel = await fetch_channel_info(client, channel_url)
            channel_id = self.repository.upsert_channel(
                telegram_id=channel.telegram_id,
                channel_url=channel.canonical_url,
                title=channel.title,
                username=channel.username,
            )
            stats = IngestStats(
                channel_url=channel.canonical_url,
                channel_title=channel.title,
            )

            messages = await iter_all_messages(client, channel_url=channel_url, limit=limit)
            stats.processed_messages = len(messages)

            for msg in messages:
                message_id = self.repository.create_or_get_message(
                    channel_id=channel_id,
                    telegram_message_id=msg.telegram_message_id,
                    message_date=msg.message_date,
                    message_url=msg.message_url,
                    caption=msg.caption,
                )

                if msg.media_kind == "text" and msg.caption:
                    await self._process_text_message(message_id, msg.caption, vertex_config)
                    stats.processed_media += 1
                    continue

            return stats

    async def _process_text_message(self, message_id: int, text: str, vertex_config: dict | None = None) -> None:
        media_id = self.repository.create_or_get_media(
            message_id=message_id,
            file_name="text_message",
            file_path="",
            mime_type="text/plain",
            media_kind="text",
            duration_seconds=None,
            file_size_bytes=len(text.encode("utf-8")),
        )
        if self.repository.media_already_transcribed(media_id):
            return
        await self._process_text_data(media_id, text, vertex_config)
        self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=text)

    async def _process_text_data(self, media_item_id: int, text: str, vertex_config: dict | None = None) -> None:
        chunks = split_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        
        datapoints = []
        indexed_chunks = []
        
        from .provider_http import vertex_ai_upsert
        
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed(chunk.text, task_type="RETRIEVAL_DOCUMENT")
            if embedding:
                chunk_id = f"c_{media_item_id}_{i}"
                datapoints.append({
                    "datapoint_id": chunk_id,
                    "feature_vector": embedding
                })
                indexed_chunks.append({
                    "chunk_index": i,
                    "text": chunk.text,
                    "embedding": None, # وکتور در SQLite ذخیره نمیشود
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                })

        # BATCH UPSERT TO VERTEX AI
        if datapoints and vertex_config and vertex_config.get("index_id"):
            try:
                vertex_ai_upsert(
                    api_key=self.embeddings.api_key,
                    project_id=vertex_config["project_id"],
                    region=vertex_config["region"],
                    index_id=vertex_config["index_id"],
                    datapoints=datapoints
                )
            except Exception as e:
                print(f"Batch Upsert Failed: {e}")

        self.repository.replace_chunks(media_item_id=media_item_id, chunks=indexed_chunks)
