"""Content-hashing helper shared by cache-invalidation checks across scrapers."""

from __future__ import annotations

import hashlib


def _content_hash(text: str) -> str:
    """Short, stable fingerprint of `text` used to detect content changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
