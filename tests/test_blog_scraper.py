"""Tests for sitemap discovery, post parsing, and archive writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_suggester_ch.blog_scraper import (
    ScrapedPost,
    _blogpost_to_scraped,
    _iso_date,
    _needs_scrape,
    format_post,
    load_checkpoint,
    merge_posts,
    parse_post_html,
    save_checkpoint,
    select_todo,
    url_to_slug,
)
from doc_suggester_ch.fetcher import parse_sitemap

SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://clickhouse.com/blog/alpha-post</loc><lastmod>2026-01-05T10:00:00.000Z</lastmod></url>
<url><loc>https://clickhouse.com/blog/beta-post</loc><lastmod>2026-03-11T08:30:00.000Z</lastmod></url>
<url><loc>https://clickhouse.com/ja/blog/beta-post-jp</loc><lastmod>2026-03-12T08:30:00.000Z</lastmod></url>
<url><loc>https://clickhouse.com/docs/intro</loc><lastmod>2026-02-01T00:00:00.000Z</lastmod></url>
<url><loc>https://clickhouse.com/company/events/x</loc></url>
</urlset>
"""

POST_HTML = """<!doctype html>
<html><head>
<title>Wide events | ClickHouse</title>
<meta property="og:title" content="Wide events | ClickHouse">
<meta name="description" content="Why wide events beat metrics">
<script type="application/ld+json">
{"@type":"BlogPosting","datePublished":"2026-04-02T09:15:00.000Z",
 "author":{"@type":"Person","name":"Dale Cooper"}}
</script>
</head><body>
<nav>site nav</nav>
<article>
  <h1>Wide events, not metrics</h1>
  <p>Observability tools force you to pre-aggregate.</p>
  <h2>The fix</h2>
  <p>Store the raw event.</p>
  <script>tracking()</script>
  <footer>Share this post on X</footer>
</article>
<footer>global footer</footer>
</body></html>
"""


def test_parse_sitemap_extracts_loc_and_lastmod():
    entries = parse_sitemap(SITEMAP)
    assert ("https://clickhouse.com/blog/alpha-post", "2026-01-05T10:00:00.000Z") in entries
    # An entry without <lastmod> still parses, with an empty date
    assert ("https://clickhouse.com/company/events/x", "") in entries


def test_blog_prefix_filter_excludes_localized_and_non_blog():
    prefix = "https://clickhouse.com/blog/"
    kept = [u for u, _ in parse_sitemap(SITEMAP) if u.startswith(prefix) and u != prefix]
    assert kept == [
        "https://clickhouse.com/blog/alpha-post",
        "https://clickhouse.com/blog/beta-post",
    ]


def test_url_to_slug():
    assert url_to_slug("https://clickhouse.com/blog/wide-events") == "wide-events"
    assert url_to_slug("https://clickhouse.com/blog/wide-events/") == "wide-events"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-04-02T09:15:00.000Z", "2026-04-02"),
        ("2026-04-02", "2026-04-02"),
        ("", ""),
        ("not a date", ""),
    ],
)
def test_iso_date(raw, expected):
    assert _iso_date(raw) == expected


def test_parse_post_html_extracts_metadata_and_body():
    post = parse_post_html(POST_HTML, "https://clickhouse.com/blog/wide-events")

    assert post.title == "Wide events, not metrics"
    assert post.slug == "wide-events"
    assert post.date == "2026-04-02"
    assert post.authors == ["Dale Cooper"]
    assert "pre-aggregate" in post.markdown
    assert "The fix" in post.markdown
    # nav/footer/script are stripped
    assert "site nav" not in post.markdown
    assert "global footer" not in post.markdown
    assert "tracking()" not in post.markdown


def test_parse_post_html_strips_trailing_share_boilerplate():
    post = parse_post_html(POST_HTML, "https://clickhouse.com/blog/wide-events")
    assert "Share this post" not in post.markdown


def test_parse_post_html_falls_back_to_sitemap_date():
    html = POST_HTML.replace('"datePublished":"2026-04-02T09:15:00.000Z",', "")
    post = parse_post_html(html, "https://clickhouse.com/blog/x", fallback_date="2026-05-06T00:00:00Z")
    assert post.date == "2026-05-06"


def test_parse_post_html_falls_back_to_slug_title():
    html = "<html><body><article><p>body text here</p></article></body></html>"
    post = parse_post_html(html, "https://clickhouse.com/blog/no-heading")
    assert post.title == "no-heading"


def test_format_post_shape_is_parseable_by_blog_manager():
    post = ScrapedPost(
        slug="s", title="T", url="https://clickhouse.com/blog/s",
        date="2026-04-02", authors=["A B"], markdown="body",
    )
    section = format_post(post)
    assert section.startswith("## T\n\n")
    assert "*Source: https://clickhouse.com/blog/s | 2026-04-02 | A B*" in section
    assert section.endswith("\n\n---\n\n")


def test_format_post_omits_missing_date_and_authors():
    post = ScrapedPost(slug="s", title="T", url="https://u/s", date="", authors=[], markdown="b")
    assert "*Source: https://u/s*" in format_post(post)


def test_checkpoint_round_trip(tmp_path: Path):
    data = {"slug-b": {"title": "B", "url": "u", "date": "2026-01-01", "scraped_at": "t"}}
    save_checkpoint(tmp_path, data)
    assert load_checkpoint(tmp_path) == data


def test_load_checkpoint_missing_and_corrupt(tmp_path: Path):
    assert load_checkpoint(tmp_path) == {}
    path = tmp_path / "output" / "checkpoint.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_checkpoint(tmp_path) == {}


def test_save_checkpoint_sorts_keys(tmp_path: Path):
    save_checkpoint(tmp_path, {"z": {"date": ""}, "a": {"date": ""}})
    raw = (tmp_path / "output" / "checkpoint.json").read_text(encoding="utf-8")
    assert list(json.loads(raw)) == ["a", "z"]


RULES_HTML = """<html><body><article>
  <h1>Post with rules</h1>
  <p>Before the rule.</p>
  <hr>
  <p>After the rule.</p>
  <hr/>
  <p>Final paragraph.</p>
</article></body></html>"""


def test_horizontal_rules_do_not_break_the_separator_contract():
    """markdownify renders <hr> as `---`, which is also the archive separator."""
    post = parse_post_html(RULES_HTML, "https://clickhouse.com/blog/rules")

    assert "\n---\n" not in f"\n{post.markdown}\n"
    assert "***" in post.markdown
    assert "Final paragraph." in post.markdown


def test_archive_round_trip_survives_horizontal_rules(tmp_path: Path):
    from doc_suggester_ch.blog_manager import archive_path, parse_blog_index

    post = parse_post_html(RULES_HTML, "https://clickhouse.com/blog/rules")
    path = archive_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# H\n\n---\n\n" + format_post(post), encoding="utf-8")

    parsed = parse_blog_index(path)
    assert len(parsed) == 1
    # Nothing lost: the last paragraph still made it through
    assert parsed[0].full_content == post.markdown
    assert "Final paragraph." in parsed[0].full_content


def test_needs_scrape_flags_new_and_changed_lastmod():
    checkpoint = {"alpha-post": {"lastmod": "2026-01-05T10:00:00.000Z"}}

    # Brand-new slug: no checkpoint entry at all.
    assert _needs_scrape("beta-post", "2026-03-11T08:30:00.000Z", checkpoint) is True
    # Existing slug whose sitemap lastmod has advanced (edited on the site).
    assert _needs_scrape("alpha-post", "2026-02-01T00:00:00.000Z", checkpoint) is True
    # Existing slug with an unchanged lastmod: no re-scrape needed.
    assert _needs_scrape("alpha-post", "2026-01-05T10:00:00.000Z", checkpoint) is False


def test_select_todo_skips_unchanged():
    checkpoint = {"alpha-post": {"lastmod": "2026-01-05T10:00:00.000Z"}}
    discovered = [
        ("https://clickhouse.com/blog/alpha-post", "2026-01-05T10:00:00.000Z"),
        ("https://clickhouse.com/blog/beta-post", "2026-03-11T08:30:00.000Z"),
    ]

    assert select_todo(discovered, checkpoint) == [
        ("https://clickhouse.com/blog/beta-post", "2026-03-11T08:30:00.000Z"),
    ]


def test_merge_posts_prefers_scraped_over_existing():
    discovered = [("https://clickhouse.com/blog/alpha-post", "2026-02-01T00:00:00.000Z")]
    existing = {
        "alpha-post": ScrapedPost(
            slug="alpha-post", title="Old title", url="https://clickhouse.com/blog/alpha-post",
            date="2026-01-01", authors=[], markdown="stale body",
        ),
    }
    scraped = {
        "alpha-post": ScrapedPost(
            slug="alpha-post", title="New title", url="https://clickhouse.com/blog/alpha-post",
            date="2026-02-01", authors=[], markdown="fresh body",
        ),
    }

    merged = merge_posts(discovered, scraped, existing)

    assert len(merged) == 1
    assert merged[0].markdown == "fresh body"
    assert merged[0].title == "New title"


def test_merge_posts_drops_slugs_no_longer_in_discovered():
    discovered = [("https://clickhouse.com/blog/alpha-post", "2026-02-01T00:00:00.000Z")]
    existing = {
        "alpha-post": ScrapedPost(
            slug="alpha-post", title="A", url="https://clickhouse.com/blog/alpha-post",
            date="2026-01-01", authors=[], markdown="a",
        ),
        "removed-post": ScrapedPost(
            slug="removed-post", title="Gone", url="https://clickhouse.com/blog/removed-post",
            date="2025-12-01", authors=[], markdown="gone",
        ),
    }

    merged = merge_posts(discovered, {}, existing)

    assert [post.slug for post in merged] == ["alpha-post"]


def test_blogpost_to_scraped_round_trips_through_format_post(tmp_path: Path):
    from doc_suggester_ch.blog_manager import archive_path, parse_blog_index

    post = ScrapedPost(
        slug="wide-events", title="Wide events, not metrics",
        url="https://clickhouse.com/blog/wide-events", date="2026-04-02",
        authors=["Dale Cooper"], markdown="Store the raw event.",
    )
    path = archive_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# H\n\n---\n\n" + format_post(post), encoding="utf-8")

    parsed = parse_blog_index(path)
    assert len(parsed) == 1

    round_tripped = _blogpost_to_scraped(parsed[0])
    assert round_tripped.slug == post.slug
    assert round_tripped.title == post.title
    assert round_tripped.url == post.url
    assert round_tripped.date == post.date
    assert round_tripped.authors == post.authors
    assert round_tripped.markdown == post.markdown


def test_checkpoint_and_archive_symmetry_allows_revival_after_removal():
    """A post pruned from both the checkpoint and archive on removal must be
    re-selected for scraping if it later reappears — even with its original,
    unchanged lastmod. Without symmetric checkpoint pruning, the stale
    checkpoint entry would suppress it forever."""
    url = "https://clickhouse.com/blog/revived-post"
    original_lastmod = "2026-01-05T10:00:00.000Z"
    checkpoint = {"revived-post": {"lastmod": original_lastmod}}

    # Round 2: the post is removed from the sitemap. `refresh_blogs` prunes
    # any checkpoint entry whose slug is no longer in `discovered`.
    discovered_after_removal: list[tuple[str, str]] = []
    discovered_slugs = {url_to_slug(u) for u, _ in discovered_after_removal}
    checkpoint = {slug: entry for slug, entry in checkpoint.items() if slug in discovered_slugs}
    assert checkpoint == {}

    # Round 3: the post reappears with its ORIGINAL, unchanged lastmod. A
    # naive "have I ever seen this lastmod" check would wrongly skip it, but
    # since the checkpoint entry was pruned, select_todo must still pick it up.
    discovered_after_revival = [(url, original_lastmod)]
    todo = select_todo(discovered_after_revival, checkpoint)

    assert todo == [(url, original_lastmod)]
