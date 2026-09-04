"""Tests for the recommendation loop, tool dispatch, and index building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import FakeProvider

from doc_suggester_ch.blog_manager import BlogPost
from doc_suggester_ch.llm import MAX_TURNS, ProviderError
from doc_suggester_ch.suggester import (
    _build_blog_index_text,
    _build_system_prompt,
    _dispatch_tool,
    _format_tool_status,
    suggest,
)
from doc_suggester_ch.training_manager import TrainingCourse


def _block(type_, **kwargs):
    block = MagicMock()
    block.type = type_
    for key, value in kwargs.items():
        setattr(block, key, value)
    return block


@pytest.fixture
def post():
    return BlogPost(
        title="Wide events",
        url="https://clickhouse.com/blog/wide-events",
        date="2026-04-02",
        excerpt="excerpt text",
        full_content="full body of the post",
        authors=["Dale Cooper"],
    )


@pytest.fixture
def course():
    return TrainingCourse(
        id="1883620",
        title="Observability with ClickStack: Level 1",
        url="https://learn.clickhouse.com/visitor_catalog_class/show/1883620",
        difficulty="beginner",
    )


@pytest.fixture
def docs():
    client = AsyncMock()
    client.search = AsyncMock(return_value="search results")
    client.get_doc_page = AsyncMock(return_value="page body")
    return client


def test_build_blog_index_text_prefers_synopsis(post):
    text = _build_blog_index_text([post], {"wide-events": "observability; wide events"})
    assert "**Wide events** | 2026-04-02" in text
    assert "URL: https://clickhouse.com/blog/wide-events" in text
    assert "Synopsis: observability; wide events" in text


def test_build_blog_index_text_falls_back_to_excerpt(post):
    text = _build_blog_index_text([post], {})
    assert "Synopsis: excerpt text" in text


def test_build_blog_index_text_omits_absent_date(post):
    post.date = ""
    assert "**Wide events**\n" in _build_blog_index_text([post], {})


def test_system_prompt_md_vs_email():
    md = _build_system_prompt("md")
    email = _build_system_prompt("email")
    assert "ClickHouse sales engineers" in md
    assert "Content Conflicts" in md
    assert "follow-up email" in email
    assert "Content Conflicts" not in email


def test_system_prompt_warns_against_inventing_urls():
    assert "Never invent a URL" in _build_system_prompt("md")


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("get_blog_post", {"url": "u"}, "reading blog post: u"),
        ("search_docs", {"query": "q"}, "searching docs: q"),
        ("get_doc_page", {"path": "/p"}, "reading doc page: /p"),
        ("get_training_course", {"course_id": "7"}, "reading course: 7"),
        ("mystery", {}, "tool: mystery"),
    ],
)
def test_format_tool_status(name, args, expected):
    assert expected in _format_tool_status(name, args)


async def test_dispatch_get_blog_post(post, docs, course):
    result = await _dispatch_tool(
        "get_blog_post", {"url": post.url}, {post.url: post}, docs, {course.id: course}
    )
    assert result == "full body of the post"


async def test_dispatch_get_blog_post_tolerates_trailing_slash(post, docs):
    result = await _dispatch_tool(
        "get_blog_post", {"url": post.url + "/"}, {post.url: post}, docs, {}
    )
    assert result == "full body of the post"


async def test_dispatch_get_blog_post_unknown_url(post, docs):
    result = await _dispatch_tool("get_blog_post", {"url": "https://x/y"}, {}, docs, {})
    assert "not found" in result


async def test_dispatch_search_docs(docs):
    result = await _dispatch_tool("search_docs", {"query": "primary key"}, {}, docs, {})
    docs.search.assert_awaited_once_with("primary key")
    assert result == "search results"


async def test_dispatch_get_doc_page_passes_max_lines(docs):
    await _dispatch_tool("get_doc_page", {"path": "/a", "max_lines": 50}, {}, docs, {})
    docs.get_doc_page.assert_awaited_once_with("/a", max_lines=50)


async def test_dispatch_get_doc_page_defaults_max_lines(docs):
    await _dispatch_tool("get_doc_page", {"path": "/a"}, {}, docs, {})
    docs.get_doc_page.assert_awaited_once_with("/a", max_lines=200)


async def test_dispatch_get_training_course(docs, course):
    result = await _dispatch_tool(
        "get_training_course", {"course_id": "1883620"}, {}, docs, {course.id: course}
    )
    assert "Observability with ClickStack: Level 1" in result


async def test_dispatch_get_training_course_unknown(docs):
    result = await _dispatch_tool("get_training_course", {"course_id": "0"}, {}, docs, {})
    assert "Course not found: 0" in result


async def test_dispatch_unknown_tool(docs):
    assert "Unknown tool" in await _dispatch_tool("nope", {}, {}, docs, {})


@pytest.fixture
def docs_ctx(docs):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=docs)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _patch_pipeline(post, course, docs_ctx):
    """Patch suggest()'s data sources so only the provider drives behaviour."""
    return patch.multiple(
        "doc_suggester_ch.suggester",
        is_archive_stale=MagicMock(return_value=False),
        is_training_stale=MagicMock(return_value=False),
        refresh_blogs=AsyncMock(),
        refresh_training=AsyncMock(),
        parse_blog_index=MagicMock(return_value=[post]),
        generate_synopses=AsyncMock(return_value={}),
        load_training=MagicMock(return_value=[course]),
        DocsClient=MagicMock(return_value=docs_ctx),
    )


async def test_suggest_returns_final_text(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider(tool_script=["## Recommendations"])
    with _patch_pipeline(post, course, docs_ctx):
        result = await suggest("prospect wants observability", tmp_path, provider=provider)

    assert result == "## Recommendations"
    assert len(provider.tool_loop_calls) == 1


async def test_suggest_passes_tools_and_system_prompt(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider()
    with _patch_pipeline(post, course, docs_ctx):
        await suggest("notes", tmp_path, provider=provider)

    call = provider.tool_loop_calls[0]
    assert [t.name for t in call["tools"]] == [
        "get_blog_post", "search_docs", "get_doc_page", "get_training_course",
    ]
    assert "ClickHouse sales engineers" in call["system"]
    assert call["max_turns"] == MAX_TURNS


async def test_suggest_includes_both_indexes_in_prompt(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider()
    with _patch_pipeline(post, course, docs_ctx):
        await suggest("prospect notes here", tmp_path, provider=provider)

    content = provider.tool_loop_calls[0]["user_content"]
    assert "prospect notes here" in content
    assert "## Blog Index" in content
    assert "## ClickHouse Academy Index" in content


async def test_suggest_wires_dispatch_to_real_tools(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider(tool_script=[
        [("search_docs", {"query": "primary key"}),
         ("get_blog_post", {"url": post.url})],
        "done",
    ])
    with _patch_pipeline(post, course, docs_ctx):
        result = await suggest("notes", tmp_path, provider=provider)

    assert result == "done"
    results = provider.tool_loop_calls[0]["results"]
    assert results[0] == "search results"
    assert results[1] == "full body of the post"


async def test_suggest_dispatch_reports_unknown_ids(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider(tool_script=[
        [("get_training_course", {"course_id": "nope"})],
        "done",
    ])
    with _patch_pipeline(post, course, docs_ctx):
        await suggest("notes", tmp_path, provider=provider)

    assert "Course not found: nope" in provider.tool_loop_calls[0]["results"][0]


async def test_suggest_refreshes_when_stale(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider()
    refresh_blogs = AsyncMock()
    refresh_training = AsyncMock()

    with patch.multiple(
        "doc_suggester_ch.suggester",
        is_archive_stale=MagicMock(return_value=True),
        is_training_stale=MagicMock(return_value=True),
        refresh_blogs=refresh_blogs,
        refresh_training=refresh_training,
        parse_blog_index=MagicMock(return_value=[post]),
        generate_synopses=AsyncMock(return_value={}),
        load_training=MagicMock(return_value=[course]),
        DocsClient=MagicMock(return_value=docs_ctx),
    ):
        await suggest("notes", tmp_path, provider=provider)

    refresh_blogs.assert_awaited_once()
    refresh_training.assert_awaited_once()
    # The resolved provider is handed to enrichment so it doesn't re-resolve
    assert refresh_training.await_args.kwargs["provider"] is provider


async def test_suggest_resolves_provider_by_name(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider(tool_script=["out"])
    with _patch_pipeline(post, course, docs_ctx), \
         patch("doc_suggester_ch.suggester.resolve_provider", return_value=provider) as resolve:
        result = await suggest("notes", tmp_path, provider="openai")

    resolve.assert_called_once_with("openai")
    assert result == "out"


async def test_suggest_resolves_provider_when_omitted(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider(tool_script=["out"])
    with _patch_pipeline(post, course, docs_ctx), \
         patch("doc_suggester_ch.suggester.resolve_provider", return_value=provider) as resolve:
        await suggest("notes", tmp_path)

    resolve.assert_called_once_with(None)


async def test_suggest_resolves_before_scraping(tmp_path: Path, post, course, docs_ctx):
    """A bad credential must fail fast, not after a multi-minute crawl."""
    refresh_blogs = AsyncMock()
    with patch.multiple(
        "doc_suggester_ch.suggester",
        is_archive_stale=MagicMock(return_value=True),
        is_training_stale=MagicMock(return_value=True),
        refresh_blogs=refresh_blogs,
        refresh_training=AsyncMock(),
        parse_blog_index=MagicMock(return_value=[post]),
        generate_synopses=AsyncMock(return_value={}),
        load_training=MagicMock(return_value=[course]),
        DocsClient=MagicMock(return_value=docs_ctx),
    ), patch(
        "doc_suggester_ch.suggester.resolve_provider",
        side_effect=ProviderError("no key"),
    ):
        with pytest.raises(ProviderError):
            await suggest("notes", tmp_path)

    refresh_blogs.assert_not_awaited()


async def test_suggest_handles_empty_model_output(tmp_path: Path, post, course, docs_ctx):
    provider = FakeProvider(tool_script=["   "])
    with _patch_pipeline(post, course, docs_ctx):
        result = await suggest("notes", tmp_path, provider=provider)

    assert result == "No recommendations generated."
