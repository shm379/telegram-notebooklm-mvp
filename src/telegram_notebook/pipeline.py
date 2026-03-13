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

    async def ingest_channel(self, *, channel_url: str, limit: int) -> IngestStats:
        client = build_client(self.settings)
        async with client:
            if not await client.is_user_authorized():
                raise RuntimeError(
                    "Telegram session is not authorized. Provide TELEGRAM_SESSION_STRING or authorize the session file first."
                )

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

            messages = await iter_media_messages(client, channel_url=channel_url, limit=limit)
            stats.processed_messages = len(messages)

            channel_slug = (channel.username or str(channel.telegram_id)).replace("/", "_")
            channel_media_dir = self.settings.media_dir / channel_slug

            for media_message in messages:
                message_id = self.repository.create_or_get_message(
                    channel_id=channel_id,
                    telegram_message_id=media_message.telegram_message_id,
                    message_date=media_message.message_date,
                    message_url=media_message.message_url,
                    caption=media_message.caption,
                )

                downloaded_path = await download_message_media(
                    client,
                    media_message=media_message,
                    target_dir=channel_media_dir,
                )
                if downloaded_path is None:
                    stats.skipped_media += 1
                    continue

                media_item_id = self.repository.create_or_get_media(
                    message_id=message_id,
                    file_name=media_message.file_name or downloaded_path.name,
                    file_path=str(downloaded_path),
                    mime_type=media_message.mime_type,
                    media_kind=media_message.media_kind,
                    duration_seconds=media_message.duration_seconds,
                    file_size_bytes=media_message.file_size_bytes,
                )

                if self.repository.media_already_transcribed(media_item_id):
                    stats.skipped_media += 1
                    continue

                try:
                    transcript = self.transcription.transcribe_media(
                        downloaded_path,
                        work_dir=channel_media_dir / f"work_{media_message.telegram_message_id}",
                    )
                    if not transcript:
                        self.repository.mark_media_failed(
                            media_item_id=media_item_id,
                            error="Empty transcript returned by transcription service",
                        )
                        stats.skipped_media += 1
                        continue

                    chunks = split_text(
                        transcript,
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

                    self.repository.mark_media_transcribed(
                        media_item_id=media_item_id,
                        transcript_text=transcript,
                    )
                    self.repository.replace_chunks(
                        media_item_id=media_item_id,
                        chunks=indexed_chunks,
                    )
                    stats.processed_media += 1
                except Exception as exc:
                    self.repository.mark_media_failed(
                        media_item_id=media_item_id,
                        error=str(exc),
                    )
                    stats.skipped_media += 1

            return stats
