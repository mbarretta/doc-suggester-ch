"""Tests for sitemap discovery, post parsing, and archive writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_suggester_ch.blog_scraper import (
    ScrapedPost,
    _iso_date,
    format_post,
    load_checkpoint,
    parse_post_html,
    save_checkpoint,
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
