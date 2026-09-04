"""Scrapes the ClickHouse Academy catalog into output/training-catalog.json.

The Academy runs on a Thought Industries LMS. `/main_catalog` and
`/user_catalog_class/...` require a login, but the `visitor_` equivalents of the
same pages are public and carry the same content, so this scraper uses those:

  1. /visitor_class_catalog                 -> learning paths (categories)
  2. /visitor_class_catalog/category/{id}   -> course ids per path
  3. /visitor_catalog_class/show/{id}       -> course detail

Course ids are shared between the visitor and logged-in views, so a course
scraped here resolves under /user_catalog_class/show/{id} for a logged-in
reader. Both URLs are recorded per course.

Scraped facts (title, learning paths, style, module count, About prose) are
then enriched by one Claude call per course to derive the fields the LMS does
not publish: difficulty, personas, problems addressed, and intent signals.
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

import anthropic
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from doc_suggester_ch.fetcher import fetch_text, make_client

logger = logging.getLogger(__name__)

BASE_URL = "https://learn.clickhouse.com"
CATALOG_URL = f"{BASE_URL}/visitor_class_catalog"
VISITOR_CLASS_URL = f"{BASE_URL}/visitor_catalog_class/show"
MEMBER_CLASS_URL = f"{BASE_URL}/user_catalog_class/show"

CATALOG_NAME = "training-catalog.json"

_ENRICH_MODEL = "claude-haiku-4-5"
_ENRICH_CONCURRENCY = 5

_CATEGORY_LINK_RE = re.compile(
    r'href="/visitor_class_catalog/category/(\d+)"[^>]*>(.*?)</a>', re.DOTALL
)
_CLASS_ID_RE = re.compile(r"/visitor_catalog_class/show/(\d+)")
_TAG_RE = re.compile(r"<[^>]+>")
_MODULE_RE = re.compile(r"\*\*Module\s+(\d+)\*\*\s*:?\s*([^\n*]{3,90})|Module\s+(\d+)\s*:\s*([^\n*]{3,90})")

# Fields the LMS renders in the "Info" side panel.
_INFO_KEYS = ("Time zone", "Style", "Modules", "Category", "Duration", "Level")

# Site-wide fallback text used when a course has no description of its own.
_GENERIC_DESC = "Learn ClickHouse with the ClickHouse Academy"


@dataclass
class Course:
    id: str
    title: str
    url: str                  # public (visitor) URL
    member_url: str           # logged-in URL for the same course
    learning_paths: list[str] = field(default_factory=list)
    language: str = "en"      # "en", or "non-en" for the localized paths
    style: str = ""           # e.g. "Self paced", "Micro course"
    module_count: str = ""
    modules: list[str] = field(default_factory=list)
    description: str = ""     # About prose, as markdown
    # LLM-derived (see enrich_courses)
    difficulty: str = ""
    summary: str = ""
    technologies: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    problems_addressed: list[str] = field(default_factory=list)
    intent_signals: list[str] = field(default_factory=list)
    content_hash: str = ""


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _strip_tags(fragment: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", fragment)).strip()


def parse_categories(html: str) -> dict[str, str]:
    """Map category id -> learning path name from the catalog root page."""
    return {
        cid: name
        for cid, raw in _CATEGORY_LINK_RE.findall(html)
        if (name := _strip_tags(raw))
    }


def parse_class_ids(html: str) -> list[str]:
    """Return the course ids linked from a category page."""
    return sorted(set(_CLASS_ID_RE.findall(html)))


def _extract_info(soup: BeautifulSoup) -> dict[str, str]:
    """Read the Info side panel into a {lowercased key: value} dict."""
    lines = [line.strip() for line in soup.get_text("\n").split("\n") if line.strip()]
    if "Info" not in lines:
        return {}
    tail = lines[lines.index("Info") + 1:]
    info: dict[str, str] = {}
    for index, line in enumerate(tail[:-1]):
        key = line.rstrip(":")
        if key in _INFO_KEYS:
            info.setdefault(key.lower().replace(" ", "_"), tail[index + 1])
    return info


def _extract_about(soup: BeautifulSoup) -> str:
    """Convert the About column to markdown.

    Taking the container's HTML (rather than page text) keeps sentences intact —
    the prose is peppered with inline <strong>/<em> that shred a text dump.
    """
    column = soup.find("div", class_="leftColumn")
    if column is None:
        return ""
    for tag in column.find_all(["script", "style", "noscript", "form", "svg"]):
        tag.decompose()
    markdown = markdownify(str(column), heading_style="ATX")
    markdown = re.sub(r"^\s*About\s*\n+", "", markdown)
    markdown = re.sub(r"^\s*##\s*About\s*\n+", "", markdown)
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)  # LMS badge images
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def parse_course_html(class_id: str, html: str) -> Course:
    """Extract a Course from a visitor course-detail page."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        title = re.sub(r"^ClickHouse Academy\s*-\s*", "", title).strip()

    about = _extract_about(soup)
    info = _extract_info(soup)

    if not about:
        meta = soup.find("meta", attrs={"name": "description"})
        candidate = (meta.get("content") or "").strip() if meta else ""
        about = "" if _GENERIC_DESC in candidate else candidate

    modules: list[str] = []
    for groups in _MODULE_RE.findall(about):
        number = groups[0] or groups[2]
        name = (groups[1] or groups[3]).strip(" :-—")
        if number and name:
            entry = f"Module {number}: {name}"
            if entry not in modules:
                modules.append(entry)

    # The catalog carries localized copies of some paths (currently Japanese).
    # Tag rather than drop them, so an APJ prospect can still be matched.
    language = "en" if title.isascii() else "non-en"

    return Course(
        id=class_id,
        title=title or f"Course {class_id}",
        url=f"{VISITOR_CLASS_URL}/{class_id}",
        member_url=f"{MEMBER_CLASS_URL}/{class_id}",
        language=language,
        style=info.get("style", ""),
        module_count=info.get("modules", ""),
        modules=modules,
        description=about,
        content_hash=hashlib.sha256(about.encode("utf-8")).hexdigest()[:16],
    )


async def scrape_catalog(client: httpx.AsyncClient) -> list[Course]:
    """Crawl all three catalog levels and return the courses found."""
    root = await fetch_text(client, CATALOG_URL)
    categories = parse_categories(root)
    if not categories:
        _status("Warning: no learning paths found in the Academy catalog.")
        return []
    _status(f"Found {len(categories)} learning paths.")

    paths_by_class: dict[str, set[str]] = {}
    category_pages = await asyncio.gather(
        *(fetch_text(client, f"{CATALOG_URL}/category/{cid}") for cid in categories),
        return_exceptions=True,
    )
    for (cid, name), page in zip(categories.items(), category_pages):
        if isinstance(page, BaseException):
            logger.warning("failed to fetch category %s: %s", cid, page)
            continue
        for class_id in parse_class_ids(page):
            paths_by_class.setdefault(class_id, set()).add(name)

    if not paths_by_class:
        _status("Warning: learning paths contained no courses.")
        return []
    _status(f"Found {len(paths_by_class)} courses; fetching details...")

    detail_pages = await asyncio.gather(
        *(fetch_text(client, f"{VISITOR_CLASS_URL}/{cid}") for cid in paths_by_class),
        return_exceptions=True,
    )

    courses: list[Course] = []
    for class_id, page in zip(paths_by_class, detail_pages):
        if isinstance(page, BaseException):
            logger.warning("failed to fetch course %s: %s", class_id, page)
            continue
        course = parse_course_html(class_id, page)
        course.learning_paths = sorted(paths_by_class[class_id])
        courses.append(course)

    # English paths first — the localized copies are duplicates for most readers.
    courses.sort(key=lambda c: (c.language != "en", c.learning_paths[:1], c.title))
    return courses


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
Style: {style}
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


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("no JSON object in response")
    return json.loads(match.group(0))


async def enrich_courses(courses: list[Course], cached: dict[str, dict]) -> None:
    """Fill LLM-derived fields in place, reusing cached values where valid.

    A cached entry is reused when its content_hash still matches, so re-running
    only pays for courses whose description actually changed.
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

    _status(f"Enriching {len(todo)} courses with Claude...")
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(_ENRICH_CONCURRENCY)
    failures: list[str] = []

    async def enrich_one(course: Course) -> None:
        prompt = _ENRICH_PROMPT.format(
            title=course.title,
            paths=", ".join(course.learning_paths) or "(none)",
            style=course.style or "(unknown)",
            modules="; ".join(course.modules) or "(not listed)",
            description=course.description[:6000],
        )
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=_ENRICH_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = next((b.text for b in response.content if b.type == "text"), "")
                data = _extract_json(text)
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


async def refresh_training(project_root: Path, force: bool = False) -> int:
    """Scrape the Academy catalog and write training-catalog.json.

    Returns the number of courses written. With `force`, discards cached
    enrichment and re-derives every course's LLM fields.
    """
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    async with make_client() as client:
        courses = await scrape_catalog(client)

    if not courses:
        _status("Warning: Academy catalog scrape returned nothing — keeping existing catalog.")
        return 0

    cached = {} if force else _load_cached_courses(project_root)
    await enrich_courses(courses, cached)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": CATALOG_URL,
        "courses": [asdict(course) for course in courses],
    }
    catalog_path(project_root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _status(f"Wrote {len(courses)} courses to {catalog_path(project_root)}")
    return len(courses)
