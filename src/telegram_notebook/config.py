from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(".env")


def _int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _str_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


@dataclass(slots=True)
class Settings:
    app_name: str
    data_dir: Path
    db_path: Path
    media_dir: Path
    telegram_api_id: int | None
    telegram_api_hash: str | None
    telegram_session_string: str | None
    telegram_session_name: str
    telegram_bot_token: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    transcription_provider: str
    transcription_model: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    default_result_limit: int

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    load_dotenv(override=True)
    settings = Settings(
        app_name=_str_env("APP_NAME", "Telegram Notebook") or "Telegram Notebook",
        data_dir=Path(_str_env("DATA_DIR", "data") or "data"),
        db_path=Path(_str_env("DB_PATH", "data/store.db") or "data/store.db"),
        media_dir=Path(_str_env("MEDIA_DIR", "data/media") or "data/media"),
        telegram_api_id=_int_env("TELEGRAM_API_ID"),
        telegram_api_hash=_str_env("TELEGRAM_API_HASH"),
        telegram_session_string=_str_env("TELEGRAM_SESSION_STRING"),
        telegram_session_name=_str_env("TELEGRAM_SESSION_NAME", "telegram-notebook")
        or "telegram-notebook",
        telegram_bot_token=_str_env("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_str_env("OPENAI_API_KEY"),
        gemini_api_key=_str_env("GEMINI_API_KEY"),
        transcription_provider=(
            _str_env("TRANSCRIPTION_PROVIDER", "openai") or "openai"
        ).lower(),
        transcription_model=_str_env(
            "TRANSCRIPTION_MODEL",
            "gpt-4o-mini-transcribe",
        )
        or "gpt-4o-mini-transcribe",
        embedding_provider=(_str_env("EMBEDDING_PROVIDER", "openai") or "openai").lower(),
        embedding_model=_str_env("EMBEDDING_MODEL", "text-embedding-3-small")
        or "text-embedding-3-small",
        chunk_size=_int_env("CHUNK_SIZE") or 900,
        chunk_overlap=_int_env("CHUNK_OVERLAP") or 120,
        default_result_limit=_int_env("DEFAULT_RESULT_LIMIT") or 8,
    )
    settings.ensure_directories()
    return settings


def upsert_env_values(updates: dict[str, str | None]) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    pending = dict(updates)
    output: list[str] = []

    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            output.append(line)
            continue

        key, _, _ = line.partition("=")
        if key in pending:
            value = pending.pop(key)
            if value is None:
                output.append(f"{key}=")
            else:
                output.append(f"{key}={value}")
        else:
            output.append(line)

    for key, value in pending.items():
        output.append(f"{key}={'' if value is None else value}")

    ENV_PATH.write_text("\n".join(output) + "\n", encoding="utf-8")
    get_settings.cache_clear()
