"""LLM-generated blog synopses, cached in output/blog-synopses.json.

The blog index sent to the model holds ~1000 posts. Raw excerpts are the first
300 characters of a post, which is often a preamble that says nothing about the
subject. A short retrieval-oriented synopsis per post makes the index far more
selective for the same token budget, and only has to be generated once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from doc_suggester_ch.blog_manager import BlogPost
from doc_suggester_ch.blog_scraper import url_to_slug
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


def load_synopses(project_root: Path) -> dict[str, str]:
    """Read cached synopses; returns {} when missing or corrupt."""
    try:
        data = json.loads(synopses_path(project_root).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


async def generate_synopses(
    project_root: Path,
    posts: list[BlogPost],
    provider: LLMProvider | str | None = None,
) -> dict[str, str]:
    """Generate and cache synopses for posts that lack one.

    Returns the full mapping of slug -> synopsis (cached plus newly generated).
    """
    synopses = load_synopses(project_root)
    missing = [post for post in posts if url_to_slug(post.url) not in synopses]

    if not missing:
        return synopses

    print(
        f"Generating synopses for {len(missing)} posts "
        "(this may take a minute on first run)...",
        file=sys.stderr,
        flush=True,
    )
    llm = resolve_provider(provider) if provider is None or isinstance(provider, str) else provider
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    failures: list[str] = []

    async def generate_one(post: BlogPost) -> tuple[str, str | None]:
        slug = url_to_slug(post.url)
        prompt = _PROMPT.format(title=post.title, content=post.full_content[:3000])
        async with semaphore:
            try:
                text = await llm.complete(prompt, max_tokens=200)
                return slug, text.strip() or None
            except Exception as exc:  # noqa: BLE001 — one post must not kill the run
                logger.debug("failed to generate synopsis for %s: %s", slug, exc)
                failures.append(f"{type(exc).__name__}: {exc}")
                return slug, None

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
        return synopses

    for slug, synopsis in results:
        if synopsis:
            synopses[slug] = synopsis

    path = synopses_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(synopses.items())), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return synopses
