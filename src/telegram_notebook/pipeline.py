from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import split_text
from .config import Settings
from .db import Repository
from .embeddings import EmbeddingService
from .telegram_client import (
    build_client,
    download_message_media,
    fetch_channel_info,
    iter_media_messages,
)
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
        transcription: TranscriptionService,
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

            channel_slug = (channel.username or str(channel.telegram_id)).replace("/", "_")
            channel_media_dir = self.settings.media_dir / channel_slug

            for msg in messages:
                message_id = self.repository.create_or_get_message(
                    channel_id=channel_id,
                    telegram_message_id=msg.telegram_message_id,
                    message_date=msg.message_date,
                    message_url=msg.message_url,
                    caption=msg.caption,
                )

                if msg.media_kind == "text" and msg.caption:
                    await self._process_text_message(message_id, msg.caption)
                    stats.processed_media += 1
                    continue

                if msg.media_kind in ("audio", "video", "voice"):
                    downloaded_path = await download_message_media(
                        client,
                        media_message=msg,
                        target_dir=channel_media_dir,
                    )
                    if downloaded_path is None:
                        stats.skipped_media += 1
                        continue

                    media_item_id = self.repository.create_or_get_media(
                        message_id=message_id,
                        file_name=msg.file_name or downloaded_path.name,
                        file_path=str(downloaded_path),
                        mime_type=msg.mime_type,
                        media_kind=msg.media_kind,
                        duration_seconds=msg.duration_seconds,
                        file_size_bytes=msg.file_size_bytes,
                    )

                    if self.repository.media_already_transcribed(media_item_id):
                        stats.skipped_media += 1
                        continue

                    try:
                        transcript = self.transcription.transcribe_media(
                            downloaded_path,
                            work_dir=channel_media_dir / f"work_{msg.telegram_message_id}",
                        )
                        if not transcript:
                            self.repository.mark_media_failed(
                                media_item_id=media_item_id,
                                error="Empty transcript returned by transcription service",
                            )
                            stats.skipped_media += 1
                            continue

                        await self._process_text_data(media_item_id, transcript)
                        stats.processed_media += 1
                    except Exception as exc:
                        self.repository.mark_media_failed(
                            media_item_id=media_item_id,
                            error=str(exc),
                        )
                        stats.skipped_media += 1

            return stats

    async def _process_text_message(self, message_id: int, text: str) -> None:
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
        await self._process_text_data(media_id, text)
        self.repository.mark_media_transcribed(media_item_id=media_id, transcript_text=text)

    async def _process_text_data(self, media_item_id: int, text: str) -> None:
        chunks = split_text(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        indexed_chunks: list[dict[str, object]] = []
        for chunk in chunks:
            indexed_chunks.append(
                {
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "embedding": self.embeddings.embed(
                        chunk.text,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )
        self.repository.replace_chunks(
            media_item_id=media_item_id,
            chunks=indexed_chunks,
        )
