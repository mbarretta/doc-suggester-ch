"""LLM-generated blog synopses, cached in output/blog-synopses.json.

The blog index sent to the model holds ~1000 posts. Raw excerpts are the first
300 characters of a post, which is often a preamble that says nothing about the
subject. A short retrieval-oriented synopsis per post makes the index far more
selective for the same token budget, and only has to be regenerated when a
post's content changes (tracked via a content hash in the cache).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from doc_suggester_ch.blog_manager import BlogPost
from doc_suggester_ch.blog_scraper import url_to_slug
from doc_suggester_ch.hashing import _content_hash
from doc_suggester_ch.llm import LLMProvider, resolve_provider

logger = logging.getLogger(__name__)

_CONCURRENCY = 10
_SYNOPSES_NAME = "blog-synopses.json"

_PROMPT = """\
Generate an information-retrieval synopsis for this ClickHouse blog post.
Output ONLY the synopsis — no preamble or explanation.
Format: semicolon-separated key topics, technologies, problems addressed, and use cases.
Target: 100-150 characters.

Title: {title}

Content:
{content}
"""


def synopses_path(project_root: Path) -> Path:
    return project_root / "output" / _SYNOPSES_NAME


def load_synopses(project_root: Path) -> dict[str, dict[str, str]]:
    """Read the cached {slug: {"synopsis", "content_hash"}} mapping.

    Returns {} when the file is missing or corrupt. A legacy bare-string
    entry ({slug: "synopsis text"}) is normalized to
    {"synopsis": text, "content_hash": ""} on read - "" never matches a
    real content hash, so the entry regenerates exactly once and then
    upgrades to the current on-disk shape.
    """
    try:
        data = json.loads(synopses_path(project_root).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for slug, entry in data.items():
        if isinstance(entry, str):
            normalized[slug] = {"synopsis": entry, "content_hash": ""}
        elif isinstance(entry, dict):
            normalized[slug] = entry
    return normalized


def _flatten(cache: dict[str, dict[str, str]]) -> dict[str, str]:
    return {slug: entry["synopsis"] for slug, entry in cache.items()}


async def generate_synopses(
    project_root: Path,
    posts: list[BlogPost],
    provider: LLMProvider | str | None = None,
) -> dict[str, str]:
    """Generate and cache synopses for posts that lack one or changed since caching.

    Returns the full mapping of slug -> synopsis text (cached plus newly
    generated); the richer on-disk {synopsis, content_hash} shape never
    leaks into this return value.
    """
    cache = load_synopses(project_root)
    missing = [
        post
        for post in posts
        if cache.get(url_to_slug(post.url), {}).get("content_hash")
        != _content_hash(post.full_content)
    ]

    if not missing:
        return _flatten(cache)

    print(
        f"Generating synopses for {len(missing)} posts "
        "(this may take a minute on first run)...",
        file=sys.stderr,
        flush=True,
    )
    llm = resolve_provider(provider) if provider is None or isinstance(provider, str) else provider
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    failures: list[str] = []

    async def generate_one(post: BlogPost) -> tuple[str, str | None, str]:
        slug = url_to_slug(post.url)
        content_hash = _content_hash(post.full_content)
        prompt = _PROMPT.format(title=post.title, content=post.full_content[:3000])
        async with semaphore:
            try:
                text = await llm.complete(prompt, max_tokens=200)
                return slug, text.strip() or None, content_hash
            except Exception as exc:  # noqa: BLE001 — one post must not kill the run
                logger.debug("failed to generate synopsis for %s: %s", slug, exc)
                failures.append(f"{type(exc).__name__}: {exc}")
                return slug, None, content_hash

    results = await asyncio.gather(*(generate_one(post) for post in missing))

    if len(failures) == len(missing):
        # Usually a missing ANTHROPIC_API_KEY. The blog index falls back to raw
        # excerpts, which is degraded but still usable.
        print(
            f"Synopsis generation failed for all {len(missing)} posts ({failures[0]}); "
            "the blog index will use excerpts instead.",
            file=sys.stderr,
            flush=True,
        )
        return _flatten(cache)

    for slug, synopsis, content_hash in results:
        if synopsis:
            cache[slug] = {"synopsis": synopsis, "content_hash": content_hash}

    path = synopses_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(cache.items())), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return _flatten(cache)
