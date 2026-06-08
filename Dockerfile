# Telegram NotebookLM MVP — runtime image (bot worker + web UI share this image)
FROM python:3.11-slim

# ffmpeg is required to extract audio from video before transcription.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    DATA_DIR=/app/data \
    DB_PATH=/app/data/store.db \
    MEDIA_DIR=/app/data/media

WORKDIR /app

# Install dependencies first to maximise Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source.
COPY . .

# Persistent data dir (mounted as a named volume by docker-compose).
RUN mkdir -p /app/data/media

# Default process is the Telegram bot worker; the web service overrides this.
CMD ["python", "-m", "telegram_notebook.bot"]
