"""Regression tests for the three defects found in the 2026-09-03 audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from telegram_notebook import config
from telegram_notebook.bot_api import safe_filename
from telegram_notebook.clustering import cluster_embeddings

# --- bot_api: a Telegram-supplied file name must not escape the temp dir ---

@pytest.mark.parametrize(
    "supplied",
    [
        "../../../../etc/passwd",
        "/etc/cron.d/evil",
        "..",
        ".",
        "",
        None,
        "../.env",
        "sub/dir/payload.txt",
    ],
)
def test_safe_filename_yields_one_harmless_segment(supplied):
    name = safe_filename(supplied)
    assert name, "must never return an empty name"
    assert Path(name).name == name, "must be a single path segment"
    assert not name.startswith("."), "must not produce a dotfile or a traversal"
    # The decisive property: joining it can never leave the directory.
    base = Path("/tmp/dl").resolve()
    assert (base / name).resolve().parent == base


def test_safe_filename_keeps_ordinary_names():
    assert safe_filename("archive.zip") == "archive.zip"
    assert safe_filename("my-export_2026.json") == "my-export_2026.json"


# --- config: a newline in a settings value must not define a second variable ---

def test_upsert_env_rejects_newline_injection(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("WEB_API_TOKEN=legit\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)

    with pytest.raises(ValueError):
        config.upsert_env_values({"GEMINI_API_KEY": "x\nWEB_API_TOKEN=attacker"})

    # The file must be untouched — the whole point is that nothing is written.
    assert env.read_text(encoding="utf-8") == "WEB_API_TOKEN=legit\n"


def test_upsert_env_still_writes_ordinary_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)

    config.upsert_env_values({"A": "2", "B": "three"})
    body = env.read_text(encoding="utf-8")
    assert "A=2" in body
    assert "B=three" in body


def test_upsert_env_blanks_none(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)

    config.upsert_env_values({"TOKEN": None})
    assert "TOKEN=\n" in env.read_text(encoding="utf-8")


# --- clustering: mixed embedding dimensions must not abort the run ---

def test_cluster_embeddings_survives_mixed_dimensions():
    """An archive re-embedded with a different model holds two vector sizes."""
    items = [
        {"embedding": [1.0, 0.0, 0.0, 0.0]},
        {"embedding": [0.99, 0.01, 0.0, 0.0]},
        {"embedding": [0.0, 1.0, 0.0, 0.0]},
        {"embedding": [1.0, 0.0]},  # a 2-dim straggler from the old model
    ]
    clusters = cluster_embeddings(items, threshold=0.99, max_clusters=3)
    assert clusters, "must return clusters rather than raising"
    total = sum(len(c) for c in clusters)
    assert total == len(items), "every item must land in exactly one cluster"
