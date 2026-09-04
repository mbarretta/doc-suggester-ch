"""Tests for training catalog loading and prompt formatting."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from doc_suggester_ch.academy_scraper import catalog_path
from doc_suggester_ch.training_manager import (
    TrainingCourse,
    build_training_index_text,
    format_training_detail,
    is_training_stale,
    load_training,
)

CATALOG = {
    "generated_at": "2026-09-01T00:00:00+00:00",
    "source": "https://learn.clickhouse.com/visitor_class_catalog",
    "courses": [
        {
            "id": "1883620",
            "title": "Observability with ClickStack: Level 1",
            "url": "https://learn.clickhouse.com/visitor_catalog_class/show/1883620",
            "member_url": "https://learn.clickhouse.com/user_catalog_class/show/1883620",
            "learning_paths": ["Learning Path: Observability with ClickStack"],
            "language": "en",
            "style": "Self paced",
            "module_count": "3",
            "modules": ["Module 1: Introduction to ClickStack"],
            "description": "Full about text.",
            "difficulty": "beginner",
            "summary": "Intro to ClickStack observability.",
            "technologies": ["ClickStack", "OpenTelemetry"],
            "personas": ["SRE"],
            "problems_addressed": ["observability cost"],
            "intent_signals": ["Datadog bill too high"],
        },
        {
            "id": "2330344",
            "title": "ClickHouseによるリアルタイム分析: Level 1",
            "url": "https://learn.clickhouse.com/visitor_catalog_class/show/2330344",
            "language": "non-en",
            "style": "Self paced",
        },
        {"title": "No id — skipped", "url": "https://x"},
        {"id": "999", "title": "No url — skipped"},
    ],
}


def _write_catalog(root: Path, payload: dict = CATALOG) -> Path:
    path = catalog_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_training_parses_full_entry(tmp_path: Path):
    _write_catalog(tmp_path)
    courses = load_training(tmp_path)
    first = courses[0]
    assert first.id == "1883620"
    assert first.difficulty == "beginner"
    assert first.technologies == ["ClickStack", "OpenTelemetry"]
    assert first.member_url.endswith("/user_catalog_class/show/1883620")


def test_load_training_defaults_missing_fields(tmp_path: Path):
    _write_catalog(tmp_path)
    second = load_training(tmp_path)[1]
    assert second.language == "non-en"
    assert second.difficulty == ""
    assert second.technologies == []
    assert second.modules == []


def test_load_training_skips_entries_without_id_or_url(tmp_path: Path):
    _write_catalog(tmp_path)
    assert [c.id for c in load_training(tmp_path)] == ["1883620", "2330344"]


def test_load_training_missing_and_corrupt(tmp_path: Path):
    assert load_training(tmp_path) == []
    path = catalog_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{oops", encoding="utf-8")
    assert load_training(tmp_path) == []


def test_is_training_stale_missing(tmp_path: Path):
    assert is_training_stale(tmp_path) is True


def test_is_training_stale_fresh(tmp_path: Path):
    _write_catalog(tmp_path)
    assert is_training_stale(tmp_path) is False


def test_is_training_stale_old(tmp_path: Path):
    path = _write_catalog(tmp_path)
    old = time.time() - 45 * 86400
    os.utime(path, (old, old))
    assert is_training_stale(tmp_path) is True


def test_build_training_index_text(tmp_path: Path):
    _write_catalog(tmp_path)
    text = build_training_index_text(load_training(tmp_path))
    assert "## ClickHouse Academy Index" in text
    assert "**Observability with ClickStack: Level 1** [beginner] (Self paced)" in text
    assert "ID: 1883620" in text
    assert "Technologies: ClickStack, OpenTelemetry" in text
    assert "Signals: Datadog bill too high" in text


def test_build_training_index_text_flags_non_english(tmp_path: Path):
    _write_catalog(tmp_path)
    text = build_training_index_text(load_training(tmp_path))
    assert "[non-en]" in text


def test_build_training_index_text_empty():
    assert build_training_index_text([]) == ""


def test_build_training_index_falls_back_to_description():
    course = TrainingCourse(id="1", title="T", url="u", description="D" * 400)
    text = build_training_index_text([course])
    assert "Summary: " + "D" * 200 in text


def test_format_training_detail(tmp_path: Path):
    _write_catalog(tmp_path)
    detail = format_training_detail(load_training(tmp_path)[0])
    assert detail.startswith("# Observability with ClickStack: Level 1")
    assert "**ID**: 1883620" in detail
    assert "**Difficulty**: beginner" in detail
    assert "**Modules**: 3" in detail
    assert "- Module 1: Introduction to ClickStack" in detail
    assert "- observability cost" in detail
    assert "**Full description**:\nFull about text." in detail


def test_format_training_detail_notes_localized_course(tmp_path: Path):
    _write_catalog(tmp_path)
    detail = format_training_detail(load_training(tmp_path)[1])
    assert "**Language**: non-en (localized course)" in detail


def test_format_training_detail_omits_absent_fields():
    detail = format_training_detail(TrainingCourse(id="1", title="T", url="u"))
    assert "**Difficulty**" not in detail
    assert "**Technologies**" not in detail
    assert "**Language**" not in detail
