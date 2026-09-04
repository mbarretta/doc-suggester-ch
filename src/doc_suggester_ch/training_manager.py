"""Academy catalog freshness checks, loading, and prompt formatting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from doc_suggester_ch.academy_scraper import catalog_path

logger = logging.getLogger(__name__)

STALE_DAYS = 30


@dataclass
class TrainingCourse:
    id: str
    title: str
    url: str
    member_url: str = ""
    learning_paths: list[str] = field(default_factory=list)
    language: str = "en"
    style: str = ""
    module_count: str = ""
    modules: list[str] = field(default_factory=list)
    description: str = ""
    difficulty: str = ""
    summary: str = ""
    technologies: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    problems_addressed: list[str] = field(default_factory=list)
    intent_signals: list[str] = field(default_factory=list)


def is_training_stale(project_root: Path) -> bool:
    """True if the catalog is missing or older than STALE_DAYS."""
    path = catalog_path(project_root)
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).days > STALE_DAYS


def load_training(project_root: Path) -> list[TrainingCourse]:
    """Parse training-catalog.json into TrainingCourse objects."""
    path = catalog_path(project_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    courses: list[TrainingCourse] = []
    for entry in data.get("courses", []):
        if not entry.get("id") or not entry.get("url"):
            continue
        courses.append(TrainingCourse(
            id=str(entry["id"]),
            title=entry.get("title", ""),
            url=entry["url"],
            member_url=entry.get("member_url", ""),
            learning_paths=entry.get("learning_paths", []) or [],
            language=entry.get("language", "en") or "en",
            style=entry.get("style", "") or "",
            module_count=str(entry.get("module_count", "") or ""),
            modules=entry.get("modules", []) or [],
            description=entry.get("description", "") or "",
            difficulty=entry.get("difficulty", "") or "",
            summary=entry.get("summary", "") or "",
            technologies=entry.get("technologies", []) or [],
            personas=entry.get("personas", []) or [],
            problems_addressed=entry.get("problems_addressed", []) or [],
            intent_signals=entry.get("intent_signals", []) or [],
        ))
    return courses


def build_training_index_text(courses: list[TrainingCourse]) -> str:
    """Build the compact catalog index injected into the prompt."""
    if not courses:
        return ""
    lines = ["## ClickHouse Academy Index\n"]
    for course in courses:
        header = f"- **{course.title}**"
        if course.difficulty:
            header += f" [{course.difficulty}]"
        if course.style:
            header += f" ({course.style})"
        if course.language != "en":
            header += f" [{course.language}]"
        lines.append(header)
        lines.append(f"  ID: {course.id} | URL: {course.url}")
        if course.learning_paths:
            lines.append(f"  Learning path: {', '.join(course.learning_paths)}")
        if course.technologies:
            lines.append(f"  Technologies: {', '.join(course.technologies)}")
        if course.intent_signals:
            lines.append(f"  Signals: {', '.join(course.intent_signals[:6])}")
        blurb = course.summary or course.description[:200]
        if blurb:
            lines.append(f"  Summary: {blurb}")
        lines.append("")
    return "\n".join(lines)


def format_training_detail(course: TrainingCourse) -> str:
    """Format one course's full details as a tool result."""
    lines = [
        f"# {course.title}",
        f"**ID**: {course.id}",
        f"**URL**: {course.url}",
    ]
    if course.member_url:
        lines.append(f"**URL (signed in)**: {course.member_url}")
    if course.learning_paths:
        lines.append(f"**Learning path**: {', '.join(course.learning_paths)}")
    if course.difficulty:
        lines.append(f"**Difficulty**: {course.difficulty}")
    if course.style:
        lines.append(f"**Style**: {course.style}")
    if course.language != "en":
        lines.append(f"**Language**: {course.language} (localized course)")
    if course.module_count:
        lines.append(f"**Modules**: {course.module_count}")
    if course.technologies:
        lines.append(f"**Technologies**: {', '.join(course.technologies)}")
    if course.personas:
        lines.append(f"**Personas**: {', '.join(course.personas)}")
    if course.summary:
        lines.append(f"\n**Summary**: {course.summary}")
    if course.modules:
        lines.append("\n**Module outline**:")
        lines.extend(f"- {module}" for module in course.modules)
    if course.problems_addressed:
        lines.append("\n**Problems addressed**:")
        lines.extend(f"- {problem}" for problem in course.problems_addressed)
    if course.intent_signals:
        lines.append(f"\n**Intent signals**: {', '.join(course.intent_signals)}")
    if course.description:
        lines.append(f"\n**Full description**:\n{course.description}")
    return "\n".join(lines)
