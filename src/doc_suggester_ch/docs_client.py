"""Async MCP client for the official ClickHouse documentation MCP server.

The server is hosted over HTTP at https://clickhouse.com/docs/mcp — there is
nothing to install and no Docker involved. It exposes two useful tools:

  search_click_house_documentation      full-text search, returns title/link/excerpt
  query_docs_filesystem_...             read-only shell over the docs tree,
                                        e.g. `head -200 /path/to/page.mdx`

Tool names are discovered at connect time rather than hardcoded: the
filesystem tool's name is suffixed with the site slug and could change.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

MCP_URL = "https://clickhouse.com/docs/mcp"

_SEARCH_HINT = "search"
_FILESYSTEM_HINT = "query_docs_filesystem"

# Docs pages are MDX, so results carry JSX component definitions and imports —
# hundreds of tokens of boilerplate per page that say nothing about the subject.
# Not anchored to line start: in search results these appear inline, after the
# result's "Content: " label.
_MDX_EXPORT_RE = re.compile(r"(?ms)export\s+const\s+\w+\s*=[\s\S]*?^\};[ \t]*$\n?")
_MDX_IMPORT_RE = re.compile(r"(?m)^import\s+.*?;[ \t]*$\n?")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# A single search can return ~70KB across all hits. Search exists to locate a
# page; get_doc_page reads it in full. So keep many hits with short excerpts
# rather than a few with entire pages.
_MAX_SEARCH_HITS = 12
_MAX_HIT_CHARS = 700

_HIT_SPLIT_RE = re.compile(r"(?m)^(?=Title:\s)")


def _strip_mdx_boilerplate(text: str) -> str:
    cleaned = _MDX_EXPORT_RE.sub("", text)
    cleaned = _MDX_IMPORT_RE.sub("", cleaned)
    return _BLANK_RUN_RE.sub("\n\n", cleaned).strip()


def _condense_search_results(text: str) -> str:
    """Cap a search response to a readable number of usefully-sized hits."""
    hits = [hit.strip() for hit in _HIT_SPLIT_RE.split(text) if hit.strip()]
    # Decide by shape, not by count: one well-formed hit still needs truncating.
    if not any(hit.startswith("Title:") for hit in hits):
        # Not the expected "Title:/Link:/Page:/Content:" shape — cap length only.
        budget = _MAX_SEARCH_HITS * _MAX_HIT_CHARS
        return text if len(text) <= budget else text[:budget] + "\n[truncated]"

    kept = []
    for hit in hits[:_MAX_SEARCH_HITS]:
        kept.append(hit if len(hit) <= _MAX_HIT_CHARS else hit[:_MAX_HIT_CHARS] + " […]")
    if len(hits) > _MAX_SEARCH_HITS:
        kept.append(f"[{len(hits) - _MAX_SEARCH_HITS} further results omitted]")
    return "\n\n".join(kept)


class DocsClient:
    """Async context manager wrapping one MCP session against the docs server."""

    def __init__(self, url: str = MCP_URL) -> None:
        self._url = url
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._cache: dict[str, str] = {}
        self._search_tool: str | None = None
        self._fs_tool: str | None = None

    async def __aenter__(self) -> "DocsClient":
        async with AsyncExitStack() as stack:
            logger.debug("connecting to docs MCP at %s", self._url)
            streams = await stack.enter_async_context(streamable_http_client(self._url))
            # The client yields (read, write) and may append a session-id getter.
            read, write = streams[0], streams[1]
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            listed = await session.list_tools()
            names = [tool.name for tool in listed.tools]
            self._search_tool = next((n for n in names if _SEARCH_HINT in n), None)
            self._fs_tool = next((n for n in names if _FILESYSTEM_HINT in n), None)
            logger.debug("docs MCP tools: %s", names)

            self._session = session
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    @staticmethod
    def _extract_text(result: Any) -> str:
        parts: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else str(result)

    def _require_session(self) -> None:
        """Fail loudly on unopened use.

        Checked before the per-tool availability checks below, so forgetting the
        context manager raises instead of returning a plausible-looking
        "unavailable" string that the model would treat as a real answer.
        """
        if self._session is None:
            raise RuntimeError("DocsClient must be used as an async context manager")

    async def _call(self, name: str, arguments: dict[str, Any]) -> str:
        self._require_session()
        key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        if key not in self._cache:
            result = await self._session.call_tool(name, arguments=arguments)
            self._cache[key] = _strip_mdx_boilerplate(self._extract_text(result))
        return self._cache[key]

    async def search(self, query: str) -> str:
        """Full-text search over the ClickHouse docs."""
        self._require_session()
        if not self._search_tool:
            return "Docs search is unavailable (tool not offered by the server)."
        return _condense_search_results(
            await self._call(self._search_tool, {"query": query})
        )

    async def get_doc_page(self, path: str, max_lines: int = 200) -> str:
        """Read a docs page by path, e.g. `/concepts/why-clickhouse-is-so-fast`.

        `path` is normalized to the `.mdx` file the filesystem tool expects.
        """
        self._require_session()
        if not self._fs_tool:
            return "Docs page reads are unavailable (tool not offered by the server)."
        clean = path.strip()
        if clean.startswith("http"):
            clean = clean.split("clickhouse.com/docs", 1)[-1] or "/"
        clean = "/" + clean.lstrip("/")
        clean = clean.split("#", 1)[0].split("?", 1)[0].rstrip("/") or "/index"
        if not clean.endswith(".mdx"):
            clean += ".mdx"
        command = f"head -{max(1, max_lines)} {shlex.quote(clean)}"
        return await self._call(self._fs_tool, {"command": command})
