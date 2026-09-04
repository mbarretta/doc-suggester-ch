"""Tests for archive parsing and staleness detection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from doc_suggester_ch.blog_manager import (
    archive_path,
    get_most_recent_blog_date,
    is_archive_stale,
    parse_blog_index,
)

ARCHIVE = """# ClickHouse Blog Archive

*Articles from [clickhouse.com/blog](https://clickhouse.com/blog)*

---

## First Post

*Source: https://clickhouse.com/blog/first | 2026-04-02 | Dale Cooper*

Body of the first post.

With two paragraphs.

---

## Second Post

*Source: https://clickhouse.com/blog/second | 2026-05-10 | Ann Ito, Bo Li*

Second body.

---

## Dateless Post

*Source: https://clickhouse.com/blog/dateless*

Third body.

---

"""


def _write_archive(root: Path, text: str = ARCHIVE) -> Path:
    path = archive_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_checkpoint(root: Path, data: dict) -> None:
    path = root / "output" / "checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_parse_blog_index_reads_all_entries(tmp_path: Path):
    posts = parse_blog_index(_write_archive(tmp_path))
    assert [p.title for p in posts] == ["First Post", "Second Post", "Dateless Post"]


def test_parse_blog_index_extracts_fields(tmp_path: Path):
    posts = parse_blog_index(_write_archive(tmp_path))
    first = posts[0]
    assert first.url == "https://clickhouse.com/blog/first"
    assert first.date == "2026-04-02"
    assert first.authors == ["Dale Cooper"]
    assert "Body of the first post." in first.full_content
    assert "With two paragraphs." in first.full_content
    assert "---" not in first.full_content


def test_parse_blog_index_splits_multiple_authors(tmp_path: Path):
    posts = parse_blog_index(_write_archive(tmp_path))
    assert posts[1].authors == ["Ann Ito", "Bo Li"]


def test_parse_blog_index_handles_missing_date_and_authors(tmp_path: Path):
    posts = parse_blog_index(_write_archive(tmp_path))
    dateless = posts[2]
    assert dateless.date == ""
    assert dateless.authors == []
    assert dateless.url == "https://clickhouse.com/blog/dateless"


def test_parse_blog_index_excerpt_is_truncated(tmp_path: Path):
    long_body = "x" * 900
    text = (
        "## Long\n\n*Source: https://clickhouse.com/blog/long | 2026-01-01*\n\n"
        f"{long_body}\n\n---\n\n"
    )
    posts = parse_blog_index(_write_archive(tmp_path, text))
    assert len(posts[0].excerpt) == 300
    assert len(posts[0].full_content) == 900


def test_parse_blog_index_missing_file(tmp_path: Path):
    assert parse_blog_index(tmp_path / "nope.md") == []


def test_get_most_recent_blog_date(tmp_path: Path):
    _write_checkpoint(tmp_path, {
        "a": {"date": "2026-01-01"},
        "b": {"date": "2026-06-15"},
        "c": {"date": ""},
        "d": {"date": "garbage"},
    })
    assert get_most_recent_blog_date(tmp_path).date().isoformat() == "2026-06-15"


def test_get_most_recent_blog_date_no_checkpoint(tmp_path: Path):
    assert get_most_recent_blog_date(tmp_path) is None


def test_is_archive_stale_when_missing(tmp_path: Path):
    assert is_archive_stale(tmp_path) is True


def test_is_archive_stale_without_checkpoint(tmp_path: Path):
    _write_archive(tmp_path)
    assert is_archive_stale(tmp_path) is True


def test_is_archive_stale_with_fresh_post(tmp_path: Path):
    _write_archive(tmp_path)
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    _write_checkpoint(tmp_path, {"a": {"date": recent}})
    assert is_archive_stale(tmp_path) is False


def test_is_archive_stale_with_old_post(tmp_path: Path):
    _write_archive(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    _write_checkpoint(tmp_path, {"a": {"date": old}})
    assert is_archive_stale(tmp_path) is True
