"""Tests for the docs MCP client's path normalization and tool discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from doc_suggester_ch.docs_client import DocsClient


def _client_with_session(tool_names: list[str]) -> DocsClient:
    """Build a DocsClient with a stubbed session, bypassing the network."""
    client = DocsClient()
    session = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text="tool output")]
    session.call_tool = AsyncMock(return_value=result)
    client._session = session
    client._search_tool = next((n for n in tool_names if "search" in n), None)
    client._fs_tool = next((n for n in tool_names if "query_docs_filesystem" in n), None)
    return client


REAL_TOOLS = [
    "search_click_house_documentation",
    "query_docs_filesystem_click_house_documentation",
    "submit_feedback",
]


async def test_search_calls_discovered_tool():
    client = _client_with_session(REAL_TOOLS)
    result = await client.search("primary key")

    client._session.call_tool.assert_awaited_once_with(
        "search_click_house_documentation", arguments={"query": "primary key"}
    )
    assert result == "tool output"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/concepts/why-fast", "/concepts/why-fast.mdx"),
        ("concepts/why-fast", "/concepts/why-fast.mdx"),
        ("/concepts/why-fast/", "/concepts/why-fast.mdx"),
        ("/concepts/why-fast.mdx", "/concepts/why-fast.mdx"),
        ("https://clickhouse.com/docs/concepts/why-fast", "/concepts/why-fast.mdx"),
        ("https://clickhouse.com/docs/concepts/why-fast#section", "/concepts/why-fast.mdx"),
        ("https://clickhouse.com/docs/concepts/why-fast?x=1", "/concepts/why-fast.mdx"),
    ],
)
async def test_get_doc_page_normalizes_paths(path, expected):
    client = _client_with_session(REAL_TOOLS)
    await client.get_doc_page(path)

    command = client._session.call_tool.await_args.kwargs["arguments"]["command"]
    assert command == f"head -200 {expected}"


async def test_get_doc_page_honors_max_lines():
    client = _client_with_session(REAL_TOOLS)
    await client.get_doc_page("/a", max_lines=25)
    command = client._session.call_tool.await_args.kwargs["arguments"]["command"]
    assert command.startswith("head -25 ")


async def test_get_doc_page_clamps_nonpositive_max_lines():
    client = _client_with_session(REAL_TOOLS)
    await client.get_doc_page("/a", max_lines=0)
    command = client._session.call_tool.await_args.kwargs["arguments"]["command"]
    assert command.startswith("head -1 ")


async def test_get_doc_page_quotes_paths_with_spaces():
    client = _client_with_session(REAL_TOOLS)
    await client.get_doc_page("/a b/c")
    command = client._session.call_tool.await_args.kwargs["arguments"]["command"]
    assert "'/a b/c.mdx'" in command


async def test_results_are_cached_per_arguments():
    client = _client_with_session(REAL_TOOLS)
    await client.search("q")
    await client.search("q")
    await client.search("other")
    assert client._session.call_tool.await_count == 2


async def test_missing_tools_degrade_gracefully():
    client = _client_with_session([])
    assert "unavailable" in await client.search("q")
    assert "unavailable" in await client.get_doc_page("/a")
    client._session.call_tool.assert_not_awaited()


async def test_call_without_context_manager_raises():
    with pytest.raises(RuntimeError, match="async context manager"):
        await DocsClient().search("q")


def test_extract_text_joins_blocks_and_handles_dicts():
    result = MagicMock()
    result.content = [MagicMock(text="a"), {"text": "b"}]
    assert DocsClient._extract_text(result) == "a\nb"


def test_strip_mdx_boilerplate_removes_component_definitions():
    from doc_suggester_ch.docs_client import _strip_mdx_boilerplate

    raw = '''# Choosing a primary key

> Page describing how to choose a primary key

import { Foo } from '/snippets/foo.md';

export const Image = ({img, alt, size = "lg"}) => {
  const normalizedSize = ["sm", "md"].includes(size) ? size : "lg";
  return <div className={`ch-image-${normalizedSize}`}>
      <img src={img} alt={alt} />
    </div>;
};

The ordering key determines how rows are sorted on disk.'''

    cleaned = _strip_mdx_boilerplate(raw)
    assert "export const Image" not in cleaned
    assert "normalizedSize" not in cleaned
    assert "import {" not in cleaned
    assert "# Choosing a primary key" in cleaned
    assert "The ordering key determines how rows are sorted on disk." in cleaned
    assert "\n\n\n" not in cleaned


def test_strip_mdx_boilerplate_leaves_plain_prose_untouched():
    from doc_suggester_ch.docs_client import _strip_mdx_boilerplate

    prose = "# Title\n\nSome prose about MergeTree.\n\n## Section\n\nMore prose."
    assert _strip_mdx_boilerplate(prose) == prose


async def test_call_strips_boilerplate_from_results():
    client = _client_with_session(REAL_TOOLS)
    result = MagicMock()
    result.content = [MagicMock(text='export const X = () => {\n  return 1;\n};\nreal prose')]
    client._session.call_tool = AsyncMock(return_value=result)

    assert await client.search("q") == "real prose"


def _hit(n: int, body_chars: int = 50) -> str:
    return (
        f"Title: Result {n}\n"
        f"Link: https://clickhouse.com/docs/page-{n}\n"
        f"Page: page-{n}\n"
        f"Content: {'c' * body_chars}"
    )


def test_condense_search_results_truncates_long_hits():
    from doc_suggester_ch.docs_client import _MAX_HIT_CHARS, _condense_search_results

    out = _condense_search_results(_hit(1, body_chars=5000))
    assert len(out) <= _MAX_HIT_CHARS + 10
    assert out.endswith("[…]")
    assert "Title: Result 1" in out
    assert "Link: https://clickhouse.com/docs/page-1" in out


def test_condense_search_results_caps_hit_count():
    from doc_suggester_ch.docs_client import _MAX_SEARCH_HITS, _condense_search_results

    raw = "\n".join(_hit(n) for n in range(20))
    out = _condense_search_results(raw)
    assert out.count("Title: Result") == _MAX_SEARCH_HITS
    assert f"[{20 - _MAX_SEARCH_HITS} further results omitted]" in out


def test_condense_search_results_keeps_short_responses_intact():
    from doc_suggester_ch.docs_client import _condense_search_results

    raw = "\n".join(_hit(n) for n in range(3))
    out = _condense_search_results(raw)
    assert out.count("Title: Result") == 3
    assert "omitted" not in out
    assert "[…]" not in out


def test_condense_search_results_handles_unexpected_shape():
    from doc_suggester_ch.docs_client import _condense_search_results

    assert _condense_search_results("no results found") == "no results found"
    assert _condense_search_results("x" * 100_000).endswith("[truncated]")


def test_strip_mdx_boilerplate_handles_inline_exports():
    from doc_suggester_ch.docs_client import _strip_mdx_boilerplate

    # In search results the export follows a "Content: " label on the same line
    raw = 'Content: export const Image = ({img}) => {\n  return <img src={img} />;\n};\nreal prose'
    cleaned = _strip_mdx_boilerplate(raw)
    assert "export const" not in cleaned
    assert "real prose" in cleaned


async def test_search_condenses_and_strips():
    client = _client_with_session(REAL_TOOLS)
    raw = "\n".join(
        f"Title: R{n}\nLink: l{n}\nPage: p{n}\n"
        f"Content: export const X = () => {{\n  return 1;\n}};\nprose {n}"
        for n in range(3)
    )
    result = MagicMock()
    result.content = [MagicMock(text=raw)]
    client._session.call_tool = AsyncMock(return_value=result)

    out = await client.search("q")
    assert "export const" not in out
    assert out.count("Title: R") == 3
    assert "prose 0" in out
