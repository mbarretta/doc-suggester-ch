"""CLI entry point for doc-suggester-ch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_DATA_DIR_NAME = "doc-suggester-ch"


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default=None,
        help=(
            "Which LLM provider to use. Default: whichever key is set, "
            "preferring ANTHROPIC_API_KEY over OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--project-root",
        metavar="DIR",
        default=None,
        help=f"Path to the data directory (default: ~/.local/share/{_DATA_DIR_NAME}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="doc-suggester-ch",
        description="Recommend relevant ClickHouse blogs, docs, and Academy courses given SE notes about a prospect.",
    )
    parser.add_argument(
        "notes",
        nargs="*",
        metavar="NOTES",
        help="SE notes text (reads from stdin if omitted and --notes-file not given).",
    )
    parser.add_argument(
        "--format",
        choices=["md", "email"],
        default="md",
        help="Output format: 'md' for ranked markdown (default), 'email' for a follow-up email draft.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a blog archive and Academy catalog refresh regardless of staleness.",
    )
    parser.add_argument(
        "--notes-file",
        metavar="FILE",
        help="Read SE notes from a file instead of positional args or stdin.",
    )
    _add_common_args(parser)
    return parser.parse_args(argv)


async def _run_init(project_root: Path, provider: str | None = None) -> None:
    """Pre-fetch and process all data sources for first-run readiness."""
    from doc_suggester_ch.academy_scraper import refresh_training
    from doc_suggester_ch.blog_manager import archive_path, parse_blog_index
    from doc_suggester_ch.blog_scraper import refresh_blogs
    from doc_suggester_ch.llm import resolve_provider
    from doc_suggester_ch.synopsis_generator import generate_synopses

    # Resolve up front so a credential problem surfaces before the crawl.
    llm = resolve_provider(provider)
    _status(f"Using {llm.name} ({llm.main_model} / {llm.bulk_model}).")

    _status("Refreshing blog archive and ClickHouse Academy catalog...")
    await asyncio.gather(
        refresh_blogs(project_root, force=True),
        refresh_training(project_root, force=True, provider=llm),
    )

    posts = parse_blog_index(archive_path(project_root))
    if posts:
        _status(f"Generating blog synopses for {len(posts)} posts (this may take a few minutes)...")
        await generate_synopses(project_root, posts, provider=llm)
    else:
        _status("Warning: no blog posts found after refresh — skipping synopsis generation.")

    _status("Init complete. Run 'doc-suggester-ch' normally to get recommendations.")


def _resolve_project_root(explicit: str | None) -> Path:
    # 1. Explicit --project-root flag
    if explicit:
        root = Path(explicit).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    # 2. Walk up from this file — works for `uv run` and editable installs
    candidate = Path(__file__).resolve().parent
    while candidate != candidate.parent:
        if (candidate / "src" / "doc_suggester_ch").is_dir():
            return candidate
        candidate = candidate.parent
    # 3. Standalone (uv tool install): use a per-user data directory
    data_dir = Path.home() / ".local" / "share" / _DATA_DIR_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def main(argv: list[str] | None = None) -> None:
    # Handle the 'init' subcommand before full argparse so it doesn't collide
    # with the positional notes argument.
    raw = argv if argv is not None else sys.argv[1:]
    if raw and raw[0] == "init":
        init_parser = argparse.ArgumentParser(prog="doc-suggester-ch init", add_help=True)
        _add_common_args(init_parser)
        init_args = init_parser.parse_args(raw[1:])
        _setup_logging(init_args.verbose)
        asyncio.run(_run_init(
            _resolve_project_root(init_args.project_root),
            provider=init_args.provider,
        ))
        return

    args = _parse_args(argv)
    _setup_logging(args.verbose)

    if args.notes_file:
        notes = Path(args.notes_file).read_text(encoding="utf-8").strip()
    elif args.notes:
        notes = " ".join(args.notes)
    elif not sys.stdin.isatty():
        notes = sys.stdin.read().strip()
    else:
        print(
            "Error: provide SE notes as arguments, via --notes-file, or via stdin.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not notes:
        print("Error: SE notes are empty.", file=sys.stderr)
        sys.exit(1)

    from doc_suggester_ch.llm import ProviderError
    from doc_suggester_ch.suggester import suggest

    try:
        result = asyncio.run(suggest(
            se_notes=notes,
            project_root=_resolve_project_root(args.project_root),
            force_refresh=args.refresh,
            output_format=args.format,
            provider=args.provider,
        ))
    except ProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(result)


if __name__ == "__main__":
    main()
