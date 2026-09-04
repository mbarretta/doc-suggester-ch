"""Tests for synopsis generation and caching."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeProvider

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


def test_load_synopses_missing(tmp_path: Path):
    assert load_synopses(tmp_path) == {}


def test_load_synopses_corrupt(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert load_synopses(tmp_path) == {}


async def test_generate_synopses_writes_cache(tmp_path: Path):
    provider = FakeProvider(completions=["topics; technologies; use cases"])
    result = await generate_synopses(
        tmp_path, [_post("alpha"), _post("beta")], provider=provider
    )

    assert result == {
        "alpha": "topics; technologies; use cases",
        "beta": "topics; technologies; use cases",
    }
    assert json.loads(synopses_path(tmp_path).read_text(encoding="utf-8")) == result


async def test_generate_synopses_passes_title_and_body_to_model(tmp_path: Path):
    provider = FakeProvider()
    await generate_synopses(tmp_path, [_post("alpha")], provider=provider)

    prompt, max_tokens = provider.complete_calls[0]
    assert "Post alpha" in prompt
    assert "body" in prompt
    assert max_tokens == 200


async def test_generate_synopses_skips_cached_posts(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"alpha": "cached"}), encoding="utf-8")

    provider = FakeProvider(completions=["fresh"])
    result = await generate_synopses(
        tmp_path, [_post("alpha"), _post("beta")], provider=provider
    )

    assert len(provider.complete_calls) == 1
    assert result == {"alpha": "cached", "beta": "fresh"}


async def test_generate_synopses_no_call_when_all_cached(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"alpha": "cached"}), encoding="utf-8")

    provider = FakeProvider()
    result = await generate_synopses(tmp_path, [_post("alpha")], provider=provider)

    assert provider.complete_calls == []
    assert result == {"alpha": "cached"}


async def test_generate_synopses_survives_provider_error(tmp_path: Path):
    provider = FakeProvider(complete_error=RuntimeError("no credits"))
    result = await generate_synopses(tmp_path, [_post("alpha")], provider=provider)

    # Degrades to no synopses rather than raising; the index falls back to excerpts
    assert result == {}


async def test_generate_synopses_keeps_partial_results(tmp_path: Path):
    """One post failing must not discard the rest."""
    class FlakyProvider(FakeProvider):
        async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
            if "alpha" in prompt:
                raise RuntimeError("boom")
            return "good synopsis"

    result = await generate_synopses(
        tmp_path, [_post("alpha"), _post("beta")], provider=FlakyProvider()
    )
    assert result == {"beta": "good synopsis"}


async def test_generate_synopses_sorts_cache_keys(tmp_path: Path):
    provider = FakeProvider()
    await generate_synopses(tmp_path, [_post("zulu"), _post("alpha")], provider=provider)

    raw = synopses_path(tmp_path).read_text(encoding="utf-8")
    assert list(json.loads(raw)) == ["alpha", "zulu"]
