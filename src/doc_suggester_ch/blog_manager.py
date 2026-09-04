"""Blog archive freshness checks and parsing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from doc_suggester_ch.blog_scraper import (
    ARCHIVE_NAME,
    load_checkpoint,
)

logger = logging.getLogger(__name__)

STALE_DAYS = 7


@dataclass
class BlogPost:
    title: str
    url: str
    date: str
    excerpt: str
    full_content: str
    authors: list[str] = field(default_factory=list)


# Matches: ## Title\n\n*Source: URL[ | date][ | authors]*\n\nbody\n\n---
_ENTRY_RE = re.compile(
    r"^## (.+?)\n\n\*Source: (https?://[^\s|*]+)"
    r"(?:\s*\|\s*([^|*\n]+?))?"
    r"(?:\s*\|\s*([^*\n]+?))?"
    r"\*\n\n([\s\S]*?)(?=\n\n---)",
    re.MULTILINE,
)


def archive_path(project_root: Path) -> Path:
    return project_root / "output" / ARCHIVE_NAME


def _parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def get_most_recent_blog_date(project_root: Path) -> datetime | None:
    """Return the newest publish date recorded in the checkpoint, or None."""
    most_recent: datetime | None = None
    for entry in load_checkpoint(project_root).values():
        parsed = _parse_date(str(entry.get("date", "")))
        if parsed and (most_recent is None or parsed > most_recent):
            most_recent = parsed
    return most_recent


def is_archive_stale(project_root: Path) -> bool:
    """True if the archive is missing, unreadable, or its newest post is old."""
    if not archive_path(project_root).exists():
        return True
    most_recent = get_most_recent_blog_date(project_root)
    if most_recent is None:
        return True
    return (datetime.now(timezone.utc) - most_recent).days > STALE_DAYS


def parse_blog_index(path: Path) -> list[BlogPost]:
    """Parse the markdown archive into BlogPost objects."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    posts: list[BlogPost] = []

    for match in _ENTRY_RE.finditer(text):
        title, url, second, third, body = match.groups()
        body = body.strip()

        # The two optional trailing fields are `date` then `authors`; a post with
        # only one of them is disambiguated by whether it parses as a date.
        date, authors_raw = "", ""
        for value in (second, third):
            if not value:
                continue
            value = value.strip()
            if not date and _parse_date(value):
                date = value
            else:
                authors_raw = value

        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]

        posts.append(BlogPost(
            title=title.strip(),
            url=url.strip(),
            date=date,
            excerpt=body[:300],
            full_content=body,
            authors=authors,
        ))

    return posts
