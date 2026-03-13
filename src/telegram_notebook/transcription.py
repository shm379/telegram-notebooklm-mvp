from __future__ import annotations

from pathlib import Path

from .media import normalize_audio_input
from .provider_http import gemini_transcribe_audio


class TranscriptionService:
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

    def transcribe_media(self, source_path: Path, work_dir: Path) -> str:
        if not self.api_key:
            raise RuntimeError(f"API key for provider '{self.provider}' is required for transcription")

        segments = normalize_audio_input(source_path, work_dir)
        texts: list[str] = []
        for segment in segments:
            if self.provider == "gemini":
                text = gemini_transcribe_audio(
                    api_key=self.api_key,
                    model=self.model,
                    audio_path=segment,
                )
                if text:
                    texts.append(text)
            else:
                client = self._get_client()
                if not client:
                    raise RuntimeError("OPENAI_API_KEY is required for transcription")
                with segment.open("rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="text",
                    )
                if hasattr(response, "text"):
                    texts.append(response.text)
                else:
                    texts.append(str(response))
        return "\n".join(part.strip() for part in texts if part.strip()).strip()
