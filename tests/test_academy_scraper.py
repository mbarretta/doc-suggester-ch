"""Tests for the Academy catalog crawl, parsing, and enrichment caching."""

from __future__ import annotations

from conftest import FakeProvider

from doc_suggester_ch.academy_scraper import (
    Course,
    _assemble_enrichment_text,
    _content_hash,
    _fetch_lesson_transcripts,
    discover_course_slugs,
    enrich_courses,
    parse_course_page,
    parse_lesson_list,
    parse_lesson_transcript,
    parse_skilljar_course_block,
)
from doc_suggester_ch.fetcher import parse_sitemap

SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://learn.clickhouse.com/observability-with-clickstack</loc><lastmod>2026-01-01</lastmod></url>
  <url><loc>https://learn.clickhouse.com/real-time-analytics</loc><lastmod>2026-01-02</lastmod></url>
  <url><loc>https://learn.clickhouse.com/workshops-and-tutorials</loc><lastmod>2026-01-03</lastmod></url>
  <url><loc>https://learn.clickhouse.com/observability-with-clickstack/intro-lesson</loc><lastmod>2026-01-01</lastmod></url>
  <url><loc>https://learn.clickhouse.com/observability-with-clickstack/ingest-lesson</loc><lastmod>2026-01-01</lastmod></url>
  <url><loc>https://learn.clickhouse.com/page/2</loc><lastmod>2026-01-01</lastmod></url>
  <url><loc>https://learn.clickhouse.com/path/observability</loc><lastmod>2026-01-01</lastmod></url>
</urlset>
"""

SKILLJAR_COURSE_HTML = """<!doctype html>
<html><head><title>Observability with ClickStack</title></head>
<body>
<script>
var otherVar = {foo: 'bar'};
var skilljarCourse = {
  id: '998877',
  title: 'Observability with ClickStack: Level 1',
  short_description: 'A guided introduction to ClickStack.',
  long_description_html: 'Learn how to use ClickStack.\\u000D\\u000ABuilt for you\\u0027s team.',
  tags: ["Observability", "ClickStack"]
};
</script>
<a class="lesson-modular" href="/observability-with-clickstack/intro-lesson">Introduction to ClickStack</a>
<a class="lesson-modular" href="/observability-with-clickstack/ingest-lesson">Ingesting Data</a>
<a class="lesson-modular" href="/observability-with-clickstack/ingest-lesson">Ingesting Data</a>
<a class="lesson-modular" href="/observability-with-clickstack/quiz-lesson">Take the Quiz</a>
<a href="/other-course/some-lesson">Not this course</a>
</body></html>
"""

SEPARATOR_HTML = """<!doctype html>
<html><head><title>[separator] Workshops and Tutorials</title></head>
<body>
<script>
var skilljarCourse = {
  id: '1',
  title: '[separator] Workshops and Tutorials',
  short_description: '',
  long_description_html: '',
  tags: []
};
</script>
</body></html>
"""

LESSON_TRANSCRIPT_HTML = """<!doctype html>
<html><body>
<article class="ch-lesson">
  <h2>Introduction</h2>
  <span class="ch-ts">00:00:01</span>
  <p>Welcome to the course.</p>
  <div class="ch-slide"><img src="/slides/1.png"></div>
  <p>Let's get started.</p>
</article>
</body></html>
"""

QUIZ_LESSON_HTML = """<!doctype html>
<html><body>
<div class="quiz-container">
  <h2>Take the Quiz</h2>
  <p>Question 1</p>
</div>
</body></html>
"""


def test_discover_course_slugs_filters_page_and_path():
    entries = parse_sitemap(SITEMAP_XML)
    slugs = discover_course_slugs(entries)
    assert "observability-with-clickstack" in slugs
    assert "real-time-analytics" in slugs
    # The separator decoy passes URL-shape filtering — it can only be
    # excluded once its page title is known (see parse_course_page).
    assert "workshops-and-tutorials" in slugs
    # /page/... and /path/... are two-segment URLs, excluded by shape.
    assert not any("page" in slug for slug in slugs)
    assert not any("path" in slug for slug in slugs)
    # Nested lesson URLs are not course roots.
    assert "observability-with-clickstack/intro-lesson" not in slugs


def test_parse_skilljar_course_block_extracts_and_unescapes_fields():
    block = parse_skilljar_course_block(SKILLJAR_COURSE_HTML)
    assert block is not None
    assert block["id"] == "998877"
    assert block["title"] == "Observability with ClickStack: Level 1"
    assert block["short_description"] == "A guided introduction to ClickStack."
    # \uXXXX sequences and the escaped apostrophe are both unescaped.
    assert "\r\n" in block["long_description_html"]
    assert "Built for you's team." in block["long_description_html"]
    assert block["tags"] == ["Observability", "ClickStack"]


def test_parse_skilljar_course_block_returns_none_without_script():
    assert parse_skilljar_course_block("<html><body>no script here</body></html>") is None


def test_parse_course_page_builds_course():
    result = parse_course_page("observability-with-clickstack", SKILLJAR_COURSE_HTML)
    assert result is not None
    course, lessons = result

    assert course.id == "observability-with-clickstack"
    assert course.title == "Observability with ClickStack: Level 1"
    assert course.url == "https://learn.clickhouse.com/observability-with-clickstack"
    assert course.learning_paths == ["Observability", "ClickStack"]
    assert course.content_hash == ""
    assert "## Lessons" not in course.description
    assert "A guided introduction to ClickStack." in course.description

    # Quiz entry is included in the outline...
    assert course.modules == [
        "Introduction to ClickStack",
        "Ingesting Data",
        "Take the Quiz",
    ]
    # ...but excluded from module_count.
    assert course.module_count == "2"

    # Lessons are ordered, de-duplicated, and scoped to this course's slug.
    assert lessons == [
        ("/observability-with-clickstack/intro-lesson", "Introduction to ClickStack"),
        ("/observability-with-clickstack/ingest-lesson", "Ingesting Data"),
        ("/observability-with-clickstack/quiz-lesson", "Take the Quiz"),
    ]


def test_parse_course_page_skips_separator_decoy():
    assert parse_course_page("workshops-and-tutorials", SEPARATOR_HTML) is None


def test_parse_lesson_list_dedupes_and_scopes_to_slug():
    lessons = parse_lesson_list("observability-with-clickstack", SKILLJAR_COURSE_HTML)
    assert lessons == [
        ("/observability-with-clickstack/intro-lesson", "Introduction to ClickStack"),
        ("/observability-with-clickstack/ingest-lesson", "Ingesting Data"),
        ("/observability-with-clickstack/quiz-lesson", "Take the Quiz"),
    ]


def test_parse_lesson_transcript_strips_timestamps_and_slides():
    transcript = parse_lesson_transcript(LESSON_TRANSCRIPT_HTML)
    assert "00:00:01" not in transcript
    assert "slides/1.png" not in transcript
    assert "Welcome to the course." in transcript
    assert "Let's get started." in transcript


def test_parse_lesson_transcript_returns_empty_for_quiz_pages():
    assert parse_lesson_transcript(QUIZ_LESSON_HTML) == ""


def test_content_hash_changes_with_text():
    a = _content_hash("hello world")
    b = _content_hash("hello world!")
    assert a != b
    assert a == _content_hash("hello world")


def test_assemble_enrichment_text_appends_sections_and_skips_empty():
    lessons = [("/c/l1", "Intro"), ("/c/l2", "Quiz")]
    transcripts = ["Some transcript text.", ""]
    text = _assemble_enrichment_text("A course description.", lessons, transcripts)
    assert text.startswith("A course description.")
    assert "## Lessons" in text
    assert "### Intro" in text
    assert "Some transcript text." in text
    assert "### Quiz" not in text


def test_assemble_enrichment_text_returns_bare_description_when_no_transcripts():
    lessons = [("/c/l1", "Intro"), ("/c/l2", "Quiz")]
    transcripts = ["", ""]
    text = _assemble_enrichment_text("A course description.", lessons, transcripts)
    assert text == "A course description."


async def test_fetch_lesson_transcripts_isolates_lesson_failures(monkeypatch):
    async def fake_fetch_text(client, url):
        if "fail" in url:
            raise RuntimeError("boom")
        return LESSON_TRANSCRIPT_HTML

    monkeypatch.setattr("doc_suggester_ch.academy_scraper.fetch_text", fake_fetch_text)

    lesson_map = {
        "course-a": [
            ("/course-a/l1", "L1"),
            ("/course-a/fail-lesson", "L2"),
            ("/course-a/l3", "L3"),
        ],
    }
    result = await _fetch_lesson_transcripts(client=None, lesson_map=lesson_map)

    # The failed lesson contributes "" at its original position; order and
    # the rest of the course's transcripts are unaffected.
    assert len(result["course-a"]) == 3
    assert result["course-a"][1] == ""
    assert "Welcome to the course." in result["course-a"][0]
    assert "Welcome to the course." in result["course-a"][2]


async def test_enrich_courses_reuses_cache_on_matching_hash():
    course = Course(id="1", title="T", url="u", description="desc")
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
    course = Course(id="1", title="T", url="u", description="desc")
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
    course = Course(id="1", title="T", url="u", description="desc")
    provider = FakeProvider(completions=[
        'Here you go:\n```json\n{"difficulty":"beginner","summary":"s"}\n```'
    ])
    await enrich_courses([course], {}, provider=provider)

    assert course.difficulty == "beginner"


async def test_enrich_courses_prompt_carries_scraped_facts():
    course = Course(
        id="1", title="Observability L1", url="u",
        description="about text",
        learning_paths=["Learning Path: Observability"], modules=["Module 1: Intro"],
    )
    provider = FakeProvider(completions=['{"difficulty":"beginner"}'])
    await enrich_courses([course], {}, provider=provider)

    prompt = provider.complete_calls[0][0]
    assert "Observability L1" in prompt
    assert "Learning Path: Observability" in prompt
    assert "Module 1: Intro" in prompt
    assert "about text" in prompt
    assert "Style:" not in prompt


async def test_enrich_courses_uses_source_text_when_provided():
    course = Course(id="1", title="T", url="u", description="stale description")
    provider = FakeProvider(completions=['{"difficulty":"beginner"}'])
    source_text = {"1": "richer description plus transcripts"}

    await enrich_courses([course], {}, provider=provider, source_text=source_text)

    prompt = provider.complete_calls[0][0]
    assert "richer description plus transcripts" in prompt
    assert "stale description" not in prompt


async def test_enrich_courses_skips_courses_without_description():
    course = Course(id="1", title="T", url="u", description="")
    provider = FakeProvider()
    await enrich_courses([course], {}, provider=provider)
    assert provider.complete_calls == []


async def test_enrich_courses_survives_bad_model_output():
    course = Course(id="1", title="T", url="u", description="desc")
    provider = FakeProvider(completions=["not json at all"])
    await enrich_courses([course], {}, provider=provider)

    # Failure is logged, not raised; the scraped facts survive
    assert course.difficulty == ""
    assert course.description == "desc"


async def test_enrich_courses_survives_provider_error():
    course = Course(id="1", title="T", url="u", description="desc")
    provider = FakeProvider(complete_error=RuntimeError("no credits"))
    await enrich_courses([course], {}, provider=provider)

    assert course.difficulty == ""
    assert course.description == "desc"
