"""Tests for argument parsing, notes resolution, and project-root discovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import FakeProvider

from doc_suggester_ch.cli import _parse_args, _resolve_project_root, _run_init, main
from doc_suggester_ch.llm import ProviderError


def test_parse_args_defaults():
    args = _parse_args(["some notes"])
    assert args.notes == ["some notes"]
    assert args.format == "md"
    assert args.refresh is False
    assert args.notes_file is None
    assert args.project_root is None


def test_parse_args_flags():
    args = _parse_args(["--format", "email", "--refresh", "-v", "a", "b"])
    assert args.format == "email"
    assert args.refresh is True
    assert args.verbose is True
    assert args.notes == ["a", "b"]


def test_parse_args_provider_defaults_to_none():
    assert _parse_args(["notes"]).provider is None


def test_parse_args_accepts_provider():
    assert _parse_args(["--provider", "openai", "notes"]).provider == "openai"


def test_parse_args_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        _parse_args(["--provider", "cohere", "notes"])


def test_parse_args_rejects_unknown_format():
    with pytest.raises(SystemExit):
        _parse_args(["--format", "pdf", "notes"])


def _run_main(argv, suggest_mock):
    with patch("doc_suggester_ch.suggester.suggest", suggest_mock), \
         patch("doc_suggester_ch.cli._resolve_project_root", return_value=Path("/tmp/root")):
        main(argv)


def test_main_joins_positional_notes(capsys):
    suggest = AsyncMock(return_value="recommendations")
    _run_main(["Java", "CVEs", "in", "prod"], suggest)

    assert suggest.await_args.kwargs["se_notes"] == "Java CVEs in prod"
    assert capsys.readouterr().out.strip() == "recommendations"


def test_main_passes_format_and_refresh():
    suggest = AsyncMock(return_value="out")
    _run_main(["--format", "email", "--refresh", "notes"], suggest)

    kwargs = suggest.await_args.kwargs
    assert kwargs["output_format"] == "email"
    assert kwargs["force_refresh"] is True


def test_main_passes_provider():
    suggest = AsyncMock(return_value="out")
    _run_main(["--provider", "openai", "notes"], suggest)
    assert suggest.await_args.kwargs["provider"] == "openai"


def test_main_provider_defaults_to_none():
    suggest = AsyncMock(return_value="out")
    _run_main(["notes"], suggest)
    assert suggest.await_args.kwargs["provider"] is None


def test_main_exits_2_on_provider_error(capsys):
    suggest = AsyncMock(side_effect=ProviderError("No API key found. Set one of: ..."))
    with pytest.raises(SystemExit) as exc:
        _run_main(["notes"], suggest)

    assert exc.value.code == 2
    assert "No API key found" in capsys.readouterr().err


def test_main_reads_notes_file(tmp_path: Path):
    notes_file = tmp_path / "notes.txt"
    notes_file.write_text("  observability prospect  \n", encoding="utf-8")
    suggest = AsyncMock(return_value="out")
    _run_main(["--notes-file", str(notes_file)], suggest)

    assert suggest.await_args.kwargs["se_notes"] == "observability prospect"


def test_main_reads_stdin_when_no_args(monkeypatch):
    stdin = MagicMock()
    stdin.isatty.return_value = False
    stdin.read.return_value = "  piped notes \n"
    monkeypatch.setattr("sys.stdin", stdin)

    suggest = AsyncMock(return_value="out")
    _run_main([], suggest)

    assert suggest.await_args.kwargs["se_notes"] == "piped notes"


def test_main_errors_without_notes(monkeypatch, capsys):
    stdin = MagicMock()
    stdin.isatty.return_value = True
    monkeypatch.setattr("sys.stdin", stdin)

    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    assert "provide SE notes" in capsys.readouterr().err


def test_main_errors_on_empty_notes(tmp_path: Path, capsys):
    notes_file = tmp_path / "empty.txt"
    notes_file.write_text("   \n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["--notes-file", str(notes_file)])

    assert exc.value.code == 1
    assert "empty" in capsys.readouterr().err


def test_main_init_dispatches_to_run_init():
    with patch("doc_suggester_ch.cli.asyncio.run", side_effect=lambda coro: coro.close()) as run, \
         patch("doc_suggester_ch.cli._resolve_project_root", return_value=Path("/tmp/root")):
        main(["init"])
    run.assert_called_once()


def test_main_init_accepts_project_root():
    with patch("doc_suggester_ch.cli.asyncio.run", side_effect=lambda coro: coro.close()), \
         patch("doc_suggester_ch.cli._resolve_project_root", return_value=Path("/tmp/x")) as resolve:
        main(["init", "--project-root", "/tmp/x"])
    resolve.assert_called_once_with("/tmp/x")


async def test_run_init_refreshes_then_generates_synopses(tmp_path: Path):
    post = MagicMock()
    provider = FakeProvider()
    with patch("doc_suggester_ch.blog_scraper.refresh_blogs", new=AsyncMock()) as blogs, \
         patch("doc_suggester_ch.academy_scraper.refresh_training", new=AsyncMock()) as training, \
         patch("doc_suggester_ch.blog_manager.parse_blog_index", return_value=[post]), \
         patch("doc_suggester_ch.llm.resolve_provider", return_value=provider), \
         patch("doc_suggester_ch.synopsis_generator.generate_synopses", new=AsyncMock()) as syn:
        await _run_init(tmp_path)

    blogs.assert_awaited_once()
    training.assert_awaited_once()
    syn.assert_awaited_once()
    # The resolved provider is reused, not re-resolved per call site
    assert training.await_args.kwargs["provider"] is provider
    assert syn.await_args.kwargs["provider"] is provider


async def test_run_init_skips_synopses_when_no_posts(tmp_path: Path):
    with patch("doc_suggester_ch.blog_scraper.refresh_blogs", new=AsyncMock()), \
         patch("doc_suggester_ch.academy_scraper.refresh_training", new=AsyncMock()), \
         patch("doc_suggester_ch.blog_manager.parse_blog_index", return_value=[]), \
         patch("doc_suggester_ch.llm.resolve_provider", return_value=FakeProvider()), \
         patch("doc_suggester_ch.synopsis_generator.generate_synopses", new=AsyncMock()) as syn:
        await _run_init(tmp_path)

    syn.assert_not_awaited()


async def test_run_init_fails_fast_on_missing_credentials(tmp_path: Path):
    with patch("doc_suggester_ch.blog_scraper.refresh_blogs", new=AsyncMock()) as blogs, \
         patch("doc_suggester_ch.llm.resolve_provider", side_effect=ProviderError("no key")):
        with pytest.raises(ProviderError):
            await _run_init(tmp_path)

    blogs.assert_not_awaited()


def test_resolve_project_root_explicit_creates_dir(tmp_path: Path):
    target = tmp_path / "data"
    assert _resolve_project_root(str(target)) == target.resolve()
    assert target.is_dir()


def test_resolve_project_root_finds_repo_checkout():
    # This test file lives in the repo, so discovery walks up to the repo root
    root = _resolve_project_root(None)
    assert (root / "src" / "doc_suggester_ch").is_dir()
