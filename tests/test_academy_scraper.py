"""Tests for the Academy catalog crawl, parsing, and enrichment caching."""

from __future__ import annotations

from conftest import FakeProvider

from doc_suggester_ch.academy_scraper import (
    Course,
    enrich_courses,
    parse_categories,
    parse_class_ids,
    parse_course_html,
)

CATALOG_HTML = """
<div class="catalog">
  <a href="/visitor_class_catalog/category/115904"><span>Learning Path: Real-time Analytics</span></a>
  <a href="/visitor_class_catalog/category/141040">Learning Path: Observability with ClickStack</a>
  <a href="/visitor_class_catalog/category/143582">ClickHouse&#x306B;&#x3088;&#x308B;&#x5206;&#x6790;</a>
  <a href="/visitor_class_catalog">All</a>
</div>
"""

CATEGORY_HTML = """
<a href="/visitor_catalog_class/show/1883620">Level 1</a>
<a href="/visitor_catalog_class/show/2259908">Level 2</a>
<a href="/visitor_catalog_class/show/1883620">Level 1 again</a>
"""

COURSE_HTML = """<!doctype html>
<html><head><title>ClickHouse Academy - Observability with ClickStack: Level 1</title>
<meta name="description" content="What You'll Learn: a guided introduction to ClickStack.">
</head><body>
<div class="leftColumn">
  <h2>About</h2>
  <p><strong>What You'll Learn</strong>: a guided introduction to <em>ClickStack</em>.</p>
  <ul>
    <li><strong>Module 1</strong>: Introduction to ClickStack</li>
    <li><strong>Module 2</strong>: Ingesting Data</li>
  </ul>
  <img src="/files/badge.png">
  <script>x()</script>
</div>
<div class="block">
  <h2>Info</h2>
  <div>Time zone:</div><div>Eastern Time (US &amp; Canada)</div>
  <div>Style:</div><div>Self paced</div>
  <div>Modules:</div><div>3</div>
  <div>Category:</div><div>Learning Path: Observability with ClickStack</div>
</div>
</body></html>
"""

GENERIC_COURSE_HTML = """<!doctype html>
<html><head><title>ClickHouse Academy - Mystery Course</title>
<meta name="description" content="Learn ClickHouse with the ClickHouse Academy. Become an expert.">
</head><body><div class="block"><h2>Info</h2><div>Style:</div><div>Micro course</div></div></body></html>
"""


def test_parse_categories():
    categories = parse_categories(CATALOG_HTML)
    assert categories["115904"] == "Learning Path: Real-time Analytics"
    assert categories["141040"] == "Learning Path: Observability with ClickStack"
    # HTML entities are unescaped
    assert categories["143582"] == "ClickHouseによる分析"
    # The catalog root link itself is not a category
    assert len(categories) == 3


def test_parse_class_ids_dedupes_and_sorts():
    assert parse_class_ids(CATEGORY_HTML) == ["1883620", "2259908"]


def test_parse_course_html_extracts_facts():
    course = parse_course_html("1883620", COURSE_HTML)

    assert course.title == "Observability with ClickStack: Level 1"
    assert course.id == "1883620"
    assert course.url.endswith("/visitor_catalog_class/show/1883620")
    assert course.member_url.endswith("/user_catalog_class/show/1883620")
    assert course.style == "Self paced"
    assert course.module_count == "3"
    assert course.language == "en"


def test_parse_course_html_keeps_inline_markup_sentences_intact():
    course = parse_course_html("1883620", COURSE_HTML)
    # Inline <strong>/<em> must not fragment the sentence
    assert "a guided introduction to *ClickStack*" in course.description


def test_parse_course_html_extracts_module_outline():
    course = parse_course_html("1883620", COURSE_HTML)
    assert course.modules == [
        "Module 1: Introduction to ClickStack",
        "Module 2: Ingesting Data",
    ]


def test_parse_course_html_drops_badge_images_and_scripts():
    course = parse_course_html("1883620", COURSE_HTML)
    assert "badge.png" not in course.description
    assert "x()" not in course.description


def test_parse_course_html_ignores_generic_site_description():
    course = parse_course_html("999", GENERIC_COURSE_HTML)
    assert course.description == ""
    assert course.style == "Micro course"


def test_parse_course_html_tags_non_english_title():
    html = COURSE_HTML.replace(
        "Observability with ClickStack: Level 1", "ClickHouseによるリアルタイム分析: Level 1"
    )
    assert parse_course_html("1", html).language == "non-en"


def test_content_hash_tracks_description():
    a = parse_course_html("1", COURSE_HTML)
    b = parse_course_html("1", COURSE_HTML.replace("Ingesting Data", "Ingesting Logs"))
    assert a.content_hash != b.content_hash


async def test_enrich_courses_reuses_cache_on_matching_hash():
    course = Course(id="1", title="T", url="u", member_url="m", description="desc")
    course.content_hash = "abc123"
    cached = {"1": {
        "content_hash": "abc123",
        "difficulty": "intermediate",
        "summary": "cached summary",
        "technologies": ["ClickStack"],
        "personas": ["SRE"],
        "problems_addressed": ["cost"],
        "intent_signals": ["too expensive"],
    }}

    provider = FakeProvider()
    await enrich_courses([course], cached, provider=provider)

    assert provider.complete_calls == []
    assert course.difficulty == "intermediate"
    assert course.summary == "cached summary"
    assert course.technologies == ["ClickStack"]


async def test_enrich_courses_calls_model_on_hash_change():
    course = Course(id="1", title="T", url="u", member_url="m", description="desc")
    course.content_hash = "new-hash"
    cached = {"1": {"content_hash": "old-hash", "difficulty": "beginner"}}

    provider = FakeProvider(completions=["""
    {"difficulty":"advanced","summary":"s","technologies":["Kafka"],
     "personas":["data engineer"],"problems_addressed":["p"],"intent_signals":["i"]}
    """])
    await enrich_courses([course], cached, provider=provider)

    assert len(provider.complete_calls) == 1
    assert course.difficulty == "advanced"
    assert course.technologies == ["Kafka"]
    assert course.personas == ["data engineer"]


async def test_enrich_courses_accepts_fenced_json():
    course = Course(id="1", title="T", url="u", member_url="m", description="desc")
    provider = FakeProvider(completions=[
        'Here you go:\n```json\n{"difficulty":"beginner","summary":"s"}\n```'
    ])
    await enrich_courses([course], {}, provider=provider)

    assert course.difficulty == "beginner"


async def test_enrich_courses_prompt_carries_scraped_facts():
    course = Course(
        id="1", title="Observability L1", url="u", member_url="m",
        description="about text", style="Self paced",
        learning_paths=["Learning Path: Observability"], modules=["Module 1: Intro"],
    )
    provider = FakeProvider(completions=['{"difficulty":"beginner"}'])
    await enrich_courses([course], {}, provider=provider)

    prompt = provider.complete_calls[0][0]
    assert "Observability L1" in prompt
    assert "Learning Path: Observability" in prompt
    assert "Self paced" in prompt
    assert "Module 1: Intro" in prompt
    assert "about text" in prompt


async def test_enrich_courses_skips_courses_without_description():
    course = Course(id="1", title="T", url="u", member_url="m", description="")
    provider = FakeProvider()
    await enrich_courses([course], {}, provider=provider)
    assert provider.complete_calls == []


async def test_enrich_courses_survives_bad_model_output():
    course = Course(id="1", title="T", url="u", member_url="m", description="desc")
    provider = FakeProvider(completions=["not json at all"])
    await enrich_courses([course], {}, provider=provider)

    # Failure is logged, not raised; the scraped facts survive
    assert course.difficulty == ""
    assert course.description == "desc"


async def test_enrich_courses_survives_provider_error():
    course = Course(id="1", title="T", url="u", member_url="m", description="desc")
    provider = FakeProvider(complete_error=RuntimeError("no credits"))
    await enrich_courses([course], {}, provider=provider)

    assert course.difficulty == ""
    assert course.description == "desc"
