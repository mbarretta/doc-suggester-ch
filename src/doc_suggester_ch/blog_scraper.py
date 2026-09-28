"""Scrapes clickhouse.com/blog into a markdown archive.

Post discovery comes from the site sitemap rather than the paginated blog
listing — the sitemap enumerates every post in one request and carries a
`lastmod` timestamp per URL.

Writes two files under `<project_root>/output/`:
  - clickhouse-blog-archive.md — one `## Title` section per post
  - checkpoint.json            — {slug: {title, url, date, scraped_at, lastmod}}

The checkpoint records each slug's sitemap `lastmod` at scrape time, so
subsequent runs re-scrape a slug whenever the site's `lastmod` advances (an
edit), not just when the slug is brand new. The archive is always rebuilt
from a merge of freshly scraped posts and the previously parsed archive,
ordered by the current sitemap — so a post that disappears from the sitemap
is pruned, and the checkpoint is pruned to match, letting a later reappearance
(even with its original `lastmod`) be re-scraped rather than skipped forever.
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
_BARE_RULE_RE = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")

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
    # `---` on its own line is the archive's post separator, and markdownify
    # renders <hr> exactly that way — so a post containing a horizontal rule
    # would be silently truncated when the archive is parsed back. Rewrite to
    # the equivalent markdown rule to keep the separator unambiguous.
    text = _BARE_RULE_RE.sub("***", text)
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


def _needs_scrape(slug: str, lastmod: str, checkpoint: dict[str, dict]) -> bool:
    """True when `slug` is new to the checkpoint or its lastmod has advanced."""
    entry = checkpoint.get(slug)
    return entry is None or entry.get("lastmod", "") != lastmod


def select_todo(
    discovered: list[tuple[str, str]], checkpoint: dict[str, dict]
) -> list[tuple[str, str]]:
    """Return the (url, lastmod) pairs from `discovered` that need (re)scraping."""
    return [
        (url, lastmod)
        for url, lastmod in discovered
        if _needs_scrape(url_to_slug(url), lastmod, checkpoint)
    ]


def _blogpost_to_scraped(post: "BlogPost") -> ScrapedPost:
    """Convert an archive-parsed BlogPost back into a ScrapedPost for merging."""
    return ScrapedPost(
        slug=url_to_slug(post.url),
        title=post.title,
        url=post.url,
        date=post.date,
        authors=post.authors,
        markdown=post.full_content,
    )


def merge_posts(
    discovered: list[tuple[str, str]],
    scraped: dict[str, ScrapedPost],
    existing: dict[str, ScrapedPost],
) -> list[ScrapedPost]:
    """Merge freshly scraped posts over the existing archive, ordered by `discovered`.

    `scraped` wins on slug collision (fresher content replaces stale), and any
    slug no longer present in `discovered` is dropped (prune-on-removal).
    """
    merged = {**existing, **scraped}
    return [merged[url_to_slug(url)] for url, _ in discovered if url_to_slug(url) in merged]


async def refresh_blogs(
    project_root: Path,
    force: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Scrape new/changed blog posts and merge-rebuild the archive.

    Returns the count of posts (re)scraped this run (new + updated), not the
    full archive size. With `force`, re-scrapes everything. Otherwise scrapes
    only slugs that are new or whose sitemap `lastmod` has advanced since the
    last scrape. The archive is always rewritten as a full merge-rebuild —
    freshly scraped posts merged over the previously parsed archive, ordered
    by the current sitemap — so removed posts (and their checkpoint entries)
    are pruned and a re-scraped post never creates a duplicate section.
    """
    # Function-local import: blog_manager imports from blog_scraper at module
    # scope, so importing the other way at module scope would cycle.
    from doc_suggester_ch.blog_manager import archive_path as _archive_path
    from doc_suggester_ch.blog_manager import parse_blog_index

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_file = output_dir / ARCHIVE_NAME

    checkpoint = {} if force else load_checkpoint(project_root)

    async with make_client() as client:
        discovered = await discover_posts(client)
        if not discovered:
            _status("Warning: no blog posts found in sitemap — leaving archive as-is.")
            return 0

        todo = select_todo(discovered, checkpoint)
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
    lastmods_by_slug = {url_to_slug(url): lastmod for url, lastmod in todo}

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for slug, post in scraped.items():
        checkpoint[slug] = {
            "title": post.title,
            "url": post.url,
            "date": post.date,
            "scraped_at": now,
            "lastmod": lastmods_by_slug.get(slug, ""),
        }

    existing: dict[str, ScrapedPost] = {}
    if not force and archive_file.exists():
        existing_posts = (_blogpost_to_scraped(bp) for bp in parse_blog_index(_archive_path(project_root)))
        existing = {post.slug: post for post in existing_posts}

    merged = merge_posts(discovered, scraped, existing)
    sections = "".join(format_post(post) for post in merged)
    archive_file.write_text(_ARCHIVE_HEADER + sections, encoding="utf-8")
    _status(f"Archive rebuilt with {len(merged)} posts: {archive_file}")

    # Prune checkpoint entries for slugs the sitemap no longer carries, so a
    # post that later reappears (even with its original lastmod) is eligible
    # for re-scraping instead of being permanently skipped.
    discovered_slugs = {url_to_slug(url) for url, _ in discovered}
    checkpoint = {slug: entry for slug, entry in checkpoint.items() if slug in discovered_slugs}
    save_checkpoint(project_root, checkpoint)

    return len(scraped)
