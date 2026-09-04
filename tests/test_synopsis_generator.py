"""Tests for synopsis generation and caching."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from doc_suggester_ch.blog_manager import BlogPost
from doc_suggester_ch.synopsis_generator import (
    generate_synopses,
    load_synopses,
    synopses_path,
)


def _post(slug: str) -> BlogPost:
    return BlogPost(
        title=f"Post {slug}",
        url=f"https://clickhouse.com/blog/{slug}",
        date="2026-01-01",
        excerpt="e",
        full_content="body " * 50,
    )


def _mock_client(text: str = "topics; technologies; use cases"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock(content=[block])
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


def test_load_synopses_missing(tmp_path: Path):
    assert load_synopses(tmp_path) == {}


def test_load_synopses_corrupt(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert load_synopses(tmp_path) == {}


async def test_generate_synopses_writes_cache(tmp_path: Path):
    client = _mock_client()
    with patch("doc_suggester_ch.synopsis_generator.anthropic.AsyncAnthropic", return_value=client):
        result = await generate_synopses(tmp_path, [_post("alpha"), _post("beta")])

    assert result == {
        "alpha": "topics; technologies; use cases",
        "beta": "topics; technologies; use cases",
    }
    on_disk = json.loads(synopses_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk == result


async def test_generate_synopses_uses_haiku(tmp_path: Path):
    client = _mock_client()
    with patch("doc_suggester_ch.synopsis_generator.anthropic.AsyncAnthropic", return_value=client):
        await generate_synopses(tmp_path, [_post("alpha")])

    assert client.messages.create.await_args.kwargs["model"] == "claude-haiku-4-5"


async def test_generate_synopses_skips_cached_posts(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"alpha": "cached"}), encoding="utf-8")

    client = _mock_client("fresh")
    with patch("doc_suggester_ch.synopsis_generator.anthropic.AsyncAnthropic", return_value=client):
        result = await generate_synopses(tmp_path, [_post("alpha"), _post("beta")])

    assert client.messages.create.await_count == 1
    assert result == {"alpha": "cached", "beta": "fresh"}


async def test_generate_synopses_no_api_call_when_all_cached(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"alpha": "cached"}), encoding="utf-8")

    with patch("doc_suggester_ch.synopsis_generator.anthropic.AsyncAnthropic") as cls:
        result = await generate_synopses(tmp_path, [_post("alpha")])

    cls.assert_not_called()
    assert result == {"alpha": "cached"}


async def test_generate_synopses_survives_api_error(tmp_path: Path):
    import anthropic

    client = AsyncMock()
    client.messages.create = AsyncMock(
        side_effect=anthropic.APIConnectionError(request=MagicMock())
    )
    with patch("doc_suggester_ch.synopsis_generator.anthropic.AsyncAnthropic", return_value=client):
        result = await generate_synopses(tmp_path, [_post("alpha")])

    assert result == {}


async def test_generate_synopses_sorts_cache_keys(tmp_path: Path):
    client = _mock_client()
    with patch("doc_suggester_ch.synopsis_generator.anthropic.AsyncAnthropic", return_value=client):
        await generate_synopses(tmp_path, [_post("zulu"), _post("alpha")])

    raw = synopses_path(tmp_path).read_text(encoding="utf-8")
    assert list(json.loads(raw)) == ["alpha", "zulu"]
