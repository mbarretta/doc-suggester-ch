"""Scrapes clickhouse.com/blog into a markdown archive.

Post discovery comes from the site sitemap rather than the paginated blog
listing — the sitemap enumerates every post in one request and carries a
`lastmod` timestamp per URL.

Writes two files under `<project_root>/output/`:
  - clickhouse-blog-archive.md — one `## Title` section per post
  - checkpoint.json            — {slug: {title, url, date, scraped_at}}

The checkpoint lets subsequent runs scrape only posts they haven't seen.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from doc_suggester_ch.fetcher import (
    DEFAULT_CONCURRENCY,
    fetch_text,
    make_client,
    parse_sitemap,
)

logger = logging.getLogger(__name__)

SITEMAP_URL = "https://clickhouse.com/sitemap.xml"
BLOG_PREFIX = "https://clickhouse.com/blog/"

ARCHIVE_NAME = "clickhouse-blog-archive.md"
CHECKPOINT_NAME = "checkpoint.json"

_ARCHIVE_HEADER = (
    "# ClickHouse Blog Archive\n\n"
    "*Articles from [clickhouse.com/blog](https://clickhouse.com/blog)*\n\n"
    "---\n\n"
)

# JSON-LD carries the authoritative publish date; sitemap lastmod is a fallback.
_DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
_AUTHOR_RE = re.compile(r'"author"\s*:\s*\{[^}]*?"name"\s*:\s*"([^"]+)"')
_EXCESS_BLANKS_RE = re.compile(r"\n{3,}")

# Trailing marketing blocks that survive <article> scoping on some posts.
_BOILERPLATE_RES = [
    re.compile(r"(?is)\n#+\s*(?:get started|ready to get started).*$"),
    re.compile(r"(?is)\nShare this (?:post|article).*$"),
    re.compile(r"(?is)\n#+\s*Related (?:posts|articles|content)\s*\n.*$"),
]

_STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "iframe"]

_ARTICLE_SELECTORS = ["article", "main", '[role="main"]']


@dataclass
class ScrapedPost:
    slug: str
    title: str
    url: str
    date: str  # ISO "YYYY-MM-DD", or "" when unknown
    authors: list[str]
    markdown: str


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def url_to_slug(url: str) -> str:
    """Return the trailing path segment of a blog URL."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def _iso_date(raw: str) -> str:
    """Normalize an ISO-ish timestamp to a YYYY-MM-DD date string."""
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return raw[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", raw) else ""


async def discover_posts(client: httpx.AsyncClient) -> list[tuple[str, str]]:
    """Return (url, lastmod) for every English blog post in the sitemap.

    Requiring the `clickhouse.com/blog/` prefix (rather than just containing
    `/blog/`) deliberately skips the locale-prefixed translations —
    `/ja/blog/...`, `/ko/blog/...` — which are the same posts in another
    language and would only add near-duplicates to the index.
    """
    xml = await fetch_text(client, SITEMAP_URL)
    posts = [
        (url, lastmod)
        for url, lastmod in parse_sitemap(xml)
        if url.startswith(BLOG_PREFIX) and url != BLOG_PREFIX
    ]
    # Sitemap order is alphabetical by slug; newest-first reads better in the archive.
    posts.sort(key=lambda pair: pair[1], reverse=True)
    return posts


def _clean(markdown: str) -> str:
    text = markdown
    for pattern in _BOILERPLATE_RES:
        text = pattern.sub("", text)
    text = _EXCESS_BLANKS_RE.sub("\n\n", text)
    return text.strip()


def parse_post_html(html: str, url: str, fallback_date: str = "") -> ScrapedPost:
    """Extract title, date, authors, and markdown body from a blog post page."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].split(" | ")[0].strip()
    if not title:
        title = url_to_slug(url)

    date = _iso_date(match.group(1)) if (match := _DATE_PUBLISHED_RE.search(html)) else ""
    if not date:
        date = _iso_date(fallback_date)

    authors = list(dict.fromkeys(_AUTHOR_RE.findall(html)))

    body = None
    for selector in _ARTICLE_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate is not None:
            body = candidate
            break
    if body is None:
        body = soup.body or soup

    for tag in body.find_all(_STRIP_TAGS):
        tag.decompose()

    markdown = _clean(markdownify(str(body), heading_style="ATX"))

    return ScrapedPost(
        slug=url_to_slug(url),
        title=title,
        url=url,
        date=date,
        authors=authors,
        markdown=markdown,
    )


def format_post(post: ScrapedPost) -> str:
    """Render one post as an archive section.

    The `*Source: <url> | <date>*` line is the contract `blog_manager`
    parses back out, so keep the shape stable.
    """
    meta = f"*Source: {post.url}"
    if post.date:
        meta += f" | {post.date}"
    if post.authors:
        meta += f" | {', '.join(post.authors)}"
    meta += "*"
    return f"## {post.title}\n\n{meta}\n\n{post.markdown}\n\n---\n\n"


def load_checkpoint(project_root: Path) -> dict[str, dict]:
    path = project_root / "output" / CHECKPOINT_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_checkpoint(project_root: Path, checkpoint: dict[str, dict]) -> None:
    path = project_root / "output" / CHECKPOINT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(checkpoint.items())), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def refresh_blogs(
    project_root: Path,
    force: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Scrape new blog posts into the archive. Returns the number added.

    With `force`, re-scrapes everything and rebuilds the archive from scratch.
    Otherwise scrapes only slugs missing from the checkpoint and appends them.
    """
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME

    checkpoint = {} if force else load_checkpoint(project_root)

    async with make_client() as client:
        discovered = await discover_posts(client)
        if not discovered:
            _status("Warning: no blog posts found in sitemap — leaving archive as-is.")
            return 0

        todo = [(url, lastmod) for url, lastmod in discovered if url_to_slug(url) not in checkpoint]
        if not todo:
            _status(f"Blog archive up to date ({len(discovered)} posts).")
            return 0

        _status(f"Scraping {len(todo)} blog posts ({len(discovered) - len(todo)} already cached)...")

        semaphore = asyncio.Semaphore(concurrency)
        done = 0

        async def scrape_one(url: str, lastmod: str) -> ScrapedPost | None:
            nonlocal done
            async with semaphore:
                try:
                    html = await fetch_text(client, url)
                    post = parse_post_html(html, url, fallback_date=lastmod)
                except Exception as exc:  # noqa: BLE001 — one bad post must not kill the run
                    logger.warning("failed to scrape %s: %s", url, exc)
                    post = None
                done += 1
                if done % 25 == 0 or done == len(todo):
                    _status(f"  [{done}/{len(todo)}] scraped")
                return post

        results = await asyncio.gather(*(scrape_one(u, m) for u, m in todo))

    scraped = {post.slug: post for post in results if post is not None and post.markdown}

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for slug, post in scraped.items():
        checkpoint[slug] = {
            "title": post.title,
            "url": post.url,
            "date": post.date,
            "scraped_at": now,
        }
    save_checkpoint(project_root, checkpoint)

    # Rebuild on force (or when there is no archive yet); otherwise append.
    rebuild = force or not archive_path.exists()
    ordered = [
        scraped[url_to_slug(url)]
        for url, _ in discovered
        if url_to_slug(url) in scraped
    ]
    sections = "".join(format_post(post) for post in ordered)

    if rebuild:
        archive_path.write_text(_ARCHIVE_HEADER + sections, encoding="utf-8")
        _status(f"Archive rebuilt with {len(ordered)} posts: {archive_path}")
    else:
        with archive_path.open("a", encoding="utf-8") as handle:
            handle.write(sections)
        _status(f"Appended {len(ordered)} new posts to {archive_path}")

    return len(ordered)
