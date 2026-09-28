"""Scrapes the ClickHouse Academy catalog into output/training-catalog.json.

The Academy runs on Skilljar. Course discovery uses the public sitemap.xml
(same shape the blog scraper already consumes), and every course-root page
embeds a structured `skilljarCourse` JS object (title, short/long
description, a flat `tags` list) plus plain anchor tags listing its lessons
in curriculum order. Every lesson page renders a full, publicly accessible,
timestamped transcript with no login required.

Lesson transcripts are fetched and used only as enrichment input for the
one-time LLM call (difficulty/summary/technologies/personas/
problems_addressed/intent_signals) — they are never persisted into
Course.description or training-catalog.json, which stays a small
course-level blurb.

Scraped facts (title, learning paths, module outline, description +
transcripts) are then enriched by one Claude call per course to derive the
fields the LMS does not publish: difficulty, personas, problems addressed,
and intent signals.
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from doc_suggester_ch.fetcher import fetch_text, make_client, parse_sitemap
from doc_suggester_ch.llm import LLMProvider, extract_json, resolve_provider

logger = logging.getLogger(__name__)

BASE_URL = "https://learn.clickhouse.com"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

CATALOG_NAME = "training-catalog.json"

_ENRICH_CONCURRENCY = 5
_LESSON_CONCURRENCY = 10  # mirrors fetcher.DEFAULT_CONCURRENCY

_TAG_RE = re.compile(r"<[^>]+>")

# Course-root URLs are exactly one path segment: https://learn.clickhouse.com/{slug}.
# /page/... and /path/... pages are two segments and already excluded by the
# regex itself; _NON_COURSE_SLUGS is a defensive backstop for a bare single
# -segment match of either name.
_COURSE_ROOT_RE = re.compile(r"^https://learn\.clickhouse\.com/([a-z0-9-]+)$")
_NON_COURSE_SLUGS = {"page", "path"}

# Matched against the whole <script> tag's text, not a pre-isolated {...}
# block — the field names are specific enough not to collide with the
# script's other `var` declarations.
_SCALAR_FIELD_RE = re.compile(r"(\w+):\s*'((?:[^'\\]|\\.)*)'")
_TAGS_ARRAY_RE = re.compile(r"tags:\s*\[(.*?)\]", re.DOTALL)
_TAG_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")

_SEPARATOR_TITLE_RE = re.compile(r"^\s*\[separator\]", re.IGNORECASE)
_QUIZ_TITLE_RE = re.compile(r"take the quiz", re.IGNORECASE)


@dataclass
class Course:
    id: str                                               # course slug, e.g. "data-warehousing-with-clickhouse"
    title: str
    url: str                                              # https://learn.clickhouse.com/{slug} — the only URL now
    learning_paths: list[str] = field(default_factory=list)  # = Skilljar's raw `tags` array, verbatim
    language: str = "en"                                  # unchanged ASCII-title heuristic
    module_count: str = ""                                # count of non-quiz lesson entries
    modules: list[str] = field(default_factory=list)       # ordered lesson titles (quiz entries kept)
    description: str = ""                                 # short_description + markdown(long_description_html)
    difficulty: str = ""
    summary: str = ""
    technologies: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    problems_addressed: list[str] = field(default_factory=list)
    intent_signals: list[str] = field(default_factory=list)
    content_hash: str = ""                                # sha256[:16] of the enrichment text, not of description alone


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _strip_tags(fragment: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", fragment)).strip()


def discover_course_slugs(entries: list[tuple[str, str]]) -> list[str]:
    """Filter fetcher.parse_sitemap() output to candidate course-root slugs.

    Cannot filter out the "[separator] ..." decoy by URL alone — that needs
    the fetched page's title (see parse_course_page).
    """
    slugs: list[str] = []
    for url, _lastmod in entries:
        match = _COURSE_ROOT_RE.match(url)
        if not match:
            continue
        slug = match.group(1)
        if slug in _NON_COURSE_SLUGS:
            continue
        slugs.append(slug)
    return slugs


def _find_skilljar_script(soup: BeautifulSoup) -> str | None:
    for script in soup.find_all("script"):
        text = script.string or script.get_text()
        if text and "skilljarCourse" in text:
            return text
    return None


def _unescape_js_string(raw: str) -> str:
    """Undo the JS single-quoted string escapes we actually see: \\uXXXX and \\'.

    Deliberately manual regex substitution rather than Python's
    unicode_escape codec, which chokes on stray/unrelated backslashes.
    """
    text = _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), raw)
    return text.replace("\\'", "'")


def parse_skilljar_course_block(html: str) -> dict | None:
    """Extract the embedded `skilljarCourse` object via regex only (no JS eval).

    Returns {id, title, short_description, long_description_html, tags} (tags
    may be []), or None if no such script tag is present on the page.
    """
    soup = BeautifulSoup(html, "html.parser")
    script_text = _find_skilljar_script(soup)
    if script_text is None:
        return None

    fields: dict = {
        key: _unescape_js_string(value)
        for key, value in _SCALAR_FIELD_RE.findall(script_text)
    }

    tags_match = _TAGS_ARRAY_RE.search(script_text)
    fields["tags"] = (
        [_unescape_js_string(item) for item in _TAG_ITEM_RE.findall(tags_match.group(1))]
        if tags_match
        else []
    )
    return fields


def _html_to_markdown(html_fragment: str) -> str:
    """Convert a course description fragment to markdown, stripping badge cruft."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "form", "svg"]):
        tag.decompose()
    markdown = markdownify(str(soup), heading_style="ATX")
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)  # LMS badge images
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def parse_lesson_list(slug: str, html: str) -> list[tuple[str, str]]:
    """Return ordered, de-duplicated (lesson_path, lesson_title) tuples for this course."""
    soup = BeautifulSoup(html, "html.parser")
    prefix = f"/{slug}/"
    seen: set[str] = set()
    lessons: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        path = href[len(BASE_URL):] if href.startswith(BASE_URL) else href
        if not path.startswith(prefix):
            continue
        title = anchor.get_text(strip=True)
        if not title or path in seen:
            continue
        seen.add(path)
        lessons.append((path, title))
    return lessons


def parse_course_page(slug: str, html: str) -> tuple[Course, list[tuple[str, str]]] | None:
    """Build a Course + its ordered lesson list from a course-root page.

    Returns None if there's no skilljarCourse block, or if the title matches
    the "[separator] ..." catalog-divider decoy.
    """
    block = parse_skilljar_course_block(html)
    if block is None:
        return None

    title = block.get("title", "")
    if _SEPARATOR_TITLE_RE.match(title):
        return None

    lessons = parse_lesson_list(slug, html)
    modules = [lesson_title for _path, lesson_title in lessons]
    module_count = sum(1 for module in modules if not _QUIZ_TITLE_RE.search(module))

    description = "\n\n".join([
        block.get("short_description", ""),
        _html_to_markdown(block.get("long_description_html", "")),
    ])

    course = Course(
        id=slug,
        title=title,
        url=f"{BASE_URL}/{slug}",
        learning_paths=list(block.get("tags", [])),
        language="en" if title.isascii() else "non-en",
        module_count=str(module_count),
        modules=modules,
        description=description,
        content_hash="",
    )
    return course, lessons


def parse_lesson_transcript(html: str) -> str:
    """Return the lesson transcript as markdown, or "" for quiz pages.

    Strips <span class="ch-ts"> timestamps and <div class="ch-slide">
    images before converting to markdown.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="ch-lesson")
    if article is None:
        return ""
    for tag in article.find_all("span", class_="ch-ts"):
        tag.decompose()
    for tag in article.find_all("div", class_="ch-slide"):
        tag.decompose()
    markdown = markdownify(str(article), heading_style="ATX")
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _assemble_enrichment_text(
    description: str,
    lessons: list[tuple[str, str]],
    transcripts: list[str],
) -> str:
    """Pure. Append a "## Lessons" section for every non-empty transcript, in order.

    Returns the bare description unchanged when no transcript is non-empty.
    """
    sections = [
        f"### {title}\n\n{transcript}"
        for (_path, title), transcript in zip(lessons, transcripts)
        if transcript.strip()
    ]
    if not sections:
        return description
    return description + "\n\n## Lessons\n\n" + "\n\n".join(sections)


async def _fetch_lesson_transcripts(
    client: httpx.AsyncClient,
    lesson_map: dict[str, list[tuple[str, str]]],
) -> dict[str, list[str]]:
    """Fetch every course's lesson transcripts under one shared semaphore.

    A lesson fetch/parse failure yields "" at that lesson's original
    position — it never aborts the course or the wider catalog refresh.
    """
    semaphore = asyncio.Semaphore(_LESSON_CONCURRENCY)

    async def fetch_one(path: str) -> str:
        async with semaphore:
            try:
                html = await fetch_text(client, f"{BASE_URL}{path}")
                return parse_lesson_transcript(html)
            except Exception as exc:  # noqa: BLE001 — one bad lesson must not kill the course
                logger.warning("failed to fetch lesson %s: %s", path, exc)
                return ""

    course_ids = list(lesson_map.keys())
    flat_paths = [path for course_id in course_ids for path, _title in lesson_map[course_id]]
    results = await asyncio.gather(*(fetch_one(path) for path in flat_paths))

    transcripts: dict[str, list[str]] = {}
    offset = 0
    for course_id in course_ids:
        count = len(lesson_map[course_id])
        transcripts[course_id] = results[offset:offset + count]
        offset += count
    return transcripts


async def scrape_catalog(client: httpx.AsyncClient) -> tuple[list[Course], dict[str, str]]:
    """Crawl the sitemap + course/lesson pages and return (courses, enrichment_text).

    enrichment_text maps course.id -> the description-plus-transcripts text
    that fed content_hash and the LLM enrichment call.
    """
    sitemap_xml = await fetch_text(client, SITEMAP_URL)
    entries = parse_sitemap(sitemap_xml)
    slugs = discover_course_slugs(entries)
    if not slugs:
        _status("Warning: no course slugs found in the Academy sitemap.")
        return [], {}
    _status(f"Found {len(slugs)} candidate course slugs.")

    pages = await asyncio.gather(
        *(fetch_text(client, f"{BASE_URL}/{slug}") for slug in slugs),
        return_exceptions=True,
    )

    courses: list[Course] = []
    lesson_map: dict[str, list[tuple[str, str]]] = {}
    for slug, page in zip(slugs, pages):
        if isinstance(page, BaseException):
            logger.warning("failed to fetch course page %s: %s", slug, page)
            continue
        parsed = parse_course_page(slug, page)
        if parsed is None:
            continue
        course, lessons = parsed
        courses.append(course)
        lesson_map[course.id] = lessons

    if not courses:
        _status("Warning: Academy sitemap contained no parseable courses.")
        return [], {}
    _status(f"Found {len(courses)} courses; fetching lesson transcripts...")

    transcripts_by_course = await _fetch_lesson_transcripts(client, lesson_map)

    enrichment_text: dict[str, str] = {}
    for course in courses:
        lessons = lesson_map.get(course.id, [])
        transcripts = transcripts_by_course.get(course.id, [])
        text = _assemble_enrichment_text(course.description, lessons, transcripts)
        enrichment_text[course.id] = text
        course.content_hash = _content_hash(text)

    # English courses first — the localized copies are duplicates for most readers.
    courses.sort(key=lambda c: (c.language != "en", c.learning_paths[:1], c.title))
    _status(f"Scraped {len(courses)} courses across {sum(len(v) for v in lesson_map.values())} lessons.")
    return courses, enrichment_text


_ENRICH_PROMPT = """\
You are building a search index over ClickHouse Academy training courses so a \
sales engineer can match courses to a prospect's stated concerns.

Return ONLY a JSON object with these keys:
  "difficulty": one of "beginner", "intermediate", "advanced"
  "summary": one sentence, under 200 characters, describing what the course covers
  "technologies": array of specific technologies/products named (e.g. "ClickStack", "Kafka", "OpenTelemetry", "BigQuery")
  "personas": array of job roles this suits (e.g. "data engineer", "SRE", "analytics engineer")
  "problems_addressed": array of concrete problems a prospect would recognise
  "intent_signals": array of short phrases a prospect might say that make this course relevant

Course title: {title}
Learning paths: {paths}
Modules: {modules}

Course description:
{description}
"""

_ENRICHED_FIELDS = (
    "difficulty",
    "summary",
    "technologies",
    "personas",
    "problems_addressed",
    "intent_signals",
)


async def enrich_courses(
    courses: list[Course],
    cached: dict[str, dict],
    provider: LLMProvider | str | None = None,
    source_text: dict[str, str] | None = None,
) -> None:
    """Fill LLM-derived fields in place, reusing cached values where valid.

    A cached entry is reused when its content_hash still matches, so
    re-running only pays for courses whose enrichment source actually
    changed. `source_text`, when provided, supplies the richer
    description-plus-transcripts text used to build each prompt (keyed by
    course.id); omitting it falls back to `course.description`, exactly as
    before.
    """
    todo = []
    for course in courses:
        prior = cached.get(course.id)
        if prior and prior.get("content_hash") == course.content_hash:
            for key in _ENRICHED_FIELDS:
                if key in prior:
                    setattr(course, key, prior[key])
            continue
        if course.description:
            todo.append(course)

    if not todo:
        return

    llm = resolve_provider(provider) if provider is None or isinstance(provider, str) else provider
    _status(f"Enriching {len(todo)} courses via {llm.name} ({llm.bulk_model})...")
    semaphore = asyncio.Semaphore(_ENRICH_CONCURRENCY)
    failures: list[str] = []

    async def enrich_one(course: Course) -> None:
        description_source = (source_text or {}).get(course.id, course.description)
        prompt = _ENRICH_PROMPT.format(
            title=course.title,
            paths=", ".join(course.learning_paths) or "(none)",
            modules="; ".join(course.modules) or "(not listed)",
            description=description_source[:6000],
        )
        async with semaphore:
            try:
                text = await llm.complete(prompt, max_tokens=1024)
                data = extract_json(text)
            except Exception as exc:  # noqa: BLE001 — never lose a completed crawl
                logger.debug("enrichment failed for course %s: %s", course.id, exc)
                failures.append(f"{type(exc).__name__}: {exc}")
                return
            course.difficulty = str(data.get("difficulty", "") or "")
            course.summary = str(data.get("summary", "") or "")
            for key in ("technologies", "personas", "problems_addressed", "intent_signals"):
                value = data.get(key) or []
                if isinstance(value, list):
                    setattr(course, key, [str(v) for v in value])

    await asyncio.gather(*(enrich_one(c) for c in todo))

    if len(failures) == len(todo):
        # Usually a missing ANTHROPIC_API_KEY. Say so once, and keep the
        # scraped facts — the recommendation step will fail loudly on its own.
        _status(
            f"Enrichment failed for all {len(todo)} courses ({failures[0]}). "
            "Scraped course facts were still saved."
        )
    elif failures:
        _status(f"Enrichment failed for {len(failures)}/{len(todo)} courses; see --verbose.")


def catalog_path(project_root: Path) -> Path:
    return project_root / "output" / CATALOG_NAME


def _load_cached_courses(project_root: Path) -> dict[str, dict]:
    path = catalog_path(project_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return {str(c.get("id")): c for c in data.get("courses", []) if c.get("id")}


async def refresh_training(
    project_root: Path,
    force: bool = False,
    provider: LLMProvider | str | None = None,
) -> int:
    """Scrape the Academy catalog and write training-catalog.json.

    Returns the number of courses written. With `force`, discards cached
    enrichment and re-derives every course's LLM fields.
    """
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    async with make_client() as client:
        courses, enrichment_text = await scrape_catalog(client)

    if not courses:
        _status("Warning: Academy catalog scrape returned nothing — keeping existing catalog.")
        return 0

    cached = {} if force else _load_cached_courses(project_root)
    await enrich_courses(courses, cached, provider=provider, source_text=enrichment_text)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SITEMAP_URL,
        "courses": [asdict(course) for course in courses],
    }
    catalog_path(project_root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _status(f"Wrote {len(courses)} courses to {catalog_path(project_root)}")
    return len(courses)
