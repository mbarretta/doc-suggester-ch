"""Tests for synopsis generation and caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conftest import FakeProvider

from doc_suggester_ch.blog_manager import BlogPost
from doc_suggester_ch.synopsis_generator import (
    generate_synopses,
    load_synopses,
    synopses_path,
)


def _post(slug: str, content: str | None = None) -> BlogPost:
    return BlogPost(
        title=f"Post {slug}",
        url=f"https://clickhouse.com/blog/{slug}",
        date="2026-01-01",
        excerpt="e",
        full_content=content if content is not None else "body " * 50,
    )


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _write_cache(tmp_path: Path, cache: dict) -> None:
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def test_load_synopses_missing(tmp_path: Path):
    assert load_synopses(tmp_path) == {}


def test_load_synopses_corrupt(tmp_path: Path):
    path = synopses_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert load_synopses(tmp_path) == {}


def test_load_synopses_upgrades_legacy_flat_string_entries(tmp_path: Path):
    _write_cache(tmp_path, {"alpha": "a cached synopsis"})

    assert load_synopses(tmp_path) == {
        "alpha": {"synopsis": "a cached synopsis", "content_hash": ""}
    }


async def test_generate_synopses_writes_cache(tmp_path: Path):
    provider = FakeProvider(completions=["topics; technologies; use cases"])
    posts = [_post("alpha"), _post("beta")]
    result = await generate_synopses(tmp_path, posts, provider=provider)

    assert result == {
        "alpha": "topics; technologies; use cases",
        "beta": "topics; technologies; use cases",
    }
    on_disk = json.loads(synopses_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk == {
        "alpha": {
            "synopsis": "topics; technologies; use cases",
            "content_hash": _hash(posts[0].full_content),
        },
        "beta": {
            "synopsis": "topics; technologies; use cases",
            "content_hash": _hash(posts[1].full_content),
        },
    }


async def test_generate_synopses_passes_title_and_body_to_model(tmp_path: Path):
    provider = FakeProvider()
    await generate_synopses(tmp_path, [_post("alpha")], provider=provider)

    prompt, max_tokens = provider.complete_calls[0]
    assert "Post alpha" in prompt
    assert "body" in prompt
    assert max_tokens == 200


async def test_generate_synopses_skips_cached_posts(tmp_path: Path):
    alpha = _post("alpha")
    _write_cache(
        tmp_path,
        {"alpha": {"synopsis": "cached", "content_hash": _hash(alpha.full_content)}},
    )

    provider = FakeProvider(completions=["fresh"])
    result = await generate_synopses(
        tmp_path, [alpha, _post("beta")], provider=provider
    )

    assert len(provider.complete_calls) == 1
    assert result == {"alpha": "cached", "beta": "fresh"}


async def test_generate_synopses_no_call_when_all_cached(tmp_path: Path):
    alpha = _post("alpha")
    _write_cache(
        tmp_path,
        {"alpha": {"synopsis": "cached", "content_hash": _hash(alpha.full_content)}},
    )

    provider = FakeProvider()
    result = await generate_synopses(tmp_path, [alpha], provider=provider)

    assert provider.complete_calls == []
    assert result == {"alpha": "cached"}


async def test_generate_synopses_skips_unchanged_content_hash(tmp_path: Path):
    """A slug whose cached content_hash still matches is never re-sent to the model."""
    alpha = _post("alpha", content="original content")
    _write_cache(
        tmp_path,
        {
            "alpha": {
                "synopsis": "original synopsis",
                "content_hash": _hash("original content"),
            }
        },
    )

    provider = FakeProvider()
    result = await generate_synopses(tmp_path, [alpha], provider=provider)

    assert provider.complete_calls == []
    assert result == {"alpha": "original synopsis"}


async def test_generate_synopses_regenerates_on_changed_content(tmp_path: Path):
    """A stored content_hash that no longer matches the post's content forces regeneration."""
    changed = _post("alpha", content="new content")
    _write_cache(
        tmp_path,
        {
            "alpha": {
                "synopsis": "stale synopsis",
                "content_hash": _hash("old content"),
            }
        },
    )

    provider = FakeProvider(completions=["updated synopsis"])
    result = await generate_synopses(tmp_path, [changed], provider=provider)

    assert len(provider.complete_calls) == 1
    assert result == {"alpha": "updated synopsis"}
    on_disk = json.loads(synopses_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk["alpha"] == {
        "synopsis": "updated synopsis",
        "content_hash": _hash("new content"),
    }


async def test_generate_synopses_returns_flat_string_mapping(tmp_path: Path):
    """The return value stays dict[str, str] even though the on-disk shape is richer."""
    cached = _post("alpha")
    _write_cache(
        tmp_path,
        {
            "alpha": {
                "synopsis": "cached synopsis",
                "content_hash": _hash(cached.full_content),
            }
        },
    )

    provider = FakeProvider(completions=["fresh synopsis"])
    result = await generate_synopses(
        tmp_path, [cached, _post("beta")], provider=provider
    )

    assert result == {"alpha": "cached synopsis", "beta": "fresh synopsis"}
    assert all(isinstance(value, str) for value in result.values())


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
