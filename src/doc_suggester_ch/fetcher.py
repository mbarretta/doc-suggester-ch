"""Shared async HTTP helpers for the ClickHouse blog and Academy scrapers."""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; doc-suggester-ch/0.1; +https://github.com/mbarretta/doc-suggester-ch)"

DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3

_RETRY_STATUS = {429, 500, 502, 503, 504}


def make_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Build an AsyncClient with the scraper's User-Agent and redirect policy."""
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )


async def fetch_text(
    client: httpx.AsyncClient,
    url: str,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """GET a URL and return its body, retrying transient failures with backoff.

    Raises the final httpx error if every attempt fails.
    """
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.get(url)
            if response.status_code in _RETRY_STATUS and attempt < retries - 1:
                logger.debug("retryable %s for %s", response.status_code, url)
            else:
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            logger.debug("fetch error for %s (attempt %d): %s", url, attempt + 1, exc)
        await asyncio.sleep(delay)
        delay *= 2
    if last_exc is not None:
        raise last_exc
    raise httpx.HTTPError(f"failed to fetch {url}")


_SITEMAP_ENTRY_RE = re.compile(
    r"<url>\s*<loc>([^<]+)</loc>\s*(?:<lastmod>([^<]+)</lastmod>)?",
    re.IGNORECASE,
)


def parse_sitemap(xml: str) -> list[tuple[str, str]]:
    """Extract (loc, lastmod) pairs from a sitemap. lastmod is "" when absent."""
    return [
        (match.group(1).strip(), (match.group(2) or "").strip())
        for match in _SITEMAP_ENTRY_RE.finditer(xml)
    ]
