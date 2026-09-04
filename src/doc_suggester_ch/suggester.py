"""Core LLM orchestration: turns SE notes into ranked ClickHouse recommendations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from doc_suggester_ch.academy_scraper import refresh_training
from doc_suggester_ch.blog_manager import (
    BlogPost,
    archive_path,
    is_archive_stale,
    parse_blog_index,
)
from doc_suggester_ch.blog_scraper import refresh_blogs, url_to_slug
from doc_suggester_ch.docs_client import DocsClient
from doc_suggester_ch.llm import (
    MAX_TURNS,
    LLMProvider,
    ToolSpec,
    resolve_provider,
)
from doc_suggester_ch.synopsis_generator import generate_synopses
from doc_suggester_ch.training_manager import (
    TrainingCourse,
    build_training_index_text,
    format_training_detail,
    is_training_stale,
    load_training,
)

_SYSTEM_PROMPT_BASE = """\
You are a technical content advisor for ClickHouse sales engineers. Given notes about a \
prospect, you identify the most relevant ClickHouse blog posts, documentation pages, and \
ClickHouse Academy courses.

You have access to:
1. A blog index below (title, URL, date, synopsis) — use get_blog_post to read a full post
2. Tools to search and read ClickHouse documentation
3. A ClickHouse Academy index (self-paced courses and workshops) — use get_training_course \
for full course details

Workflow:
- Scan the blog index for relevant posts based on title and synopsis
- Fetch full content for the most promising posts using get_blog_post
- Use search_docs to find relevant documentation, then get_doc_page to read a page in full
- Fetch full course details with get_training_course before recommending a course
- Select the 5-10 most relevant resources before writing your final output

Guidance specific to ClickHouse prospects:
- Distinguish ClickHouse Cloud from self-managed/OSS deployments, and say which a resource assumes
- Treat ClickStack (the observability stack) and chDB (embedded ClickHouse) as distinct products
- Prefer resources matching the prospect's actual workload — real-time analytics, observability, \
data warehousing, or ML/GenAI — over generic introductions
- Academy courses tagged [non-en] are localized copies of an English course; recommend the \
English one unless the notes indicate the prospect prefers that language
- Never invent a URL. Only recommend resources that appear in the indexes below or that you \
retrieved with a tool.

"""

_OUTPUT_FORMAT_MD = """\
Output format for each recommendation:
### N. [Type] Title
**URL**: <url>
**Date**: <date> (for blog posts)
**Learning path**: <path> (for Academy courses)
**Difficulty**: <level> (for Academy courses)
**Why relevant**: 1-2 sentence explanation tied to the prospect's specific concerns

When a blog post and a documentation page conflict, prefer the more recently dated source. \
If conflicts exist, add a "## Content Conflicts" section at the end noting them.

If no conflicts: end with `*No content conflicts detected.*`
"""

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _build_system_prompt(output_format: str) -> str:
    if output_format == "email":
        fmt = (_PROMPTS_DIR / "email_format.txt").read_text(encoding="utf-8")
    else:
        fmt = _OUTPUT_FORMAT_MD
    return _SYSTEM_PROMPT_BASE + fmt


_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_blog_post",
        description="Fetch the full content of a ClickHouse blog post by its URL.",
        schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The blog post URL, as shown in the index."}
            },
            "required": ["url"],
        },
    ),
    ToolSpec(
        name="search_docs",
        description=(
            "Search the ClickHouse documentation. Returns titles, links, and excerpts. "
            "Use this first to locate a page, then get_doc_page to read it."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."}
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_doc_page",
        description=(
            "Read a ClickHouse documentation page in full. Pass the docs path from a "
            "search result (e.g. '/concepts/best-practices/choosing-a-primary-key') "
            "or a full clickhouse.com/docs URL."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Docs path or full URL."},
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read (default 200).",
                    "default": 200,
                },
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="get_training_course",
        description=(
            "Get full details for a ClickHouse Academy course by its ID, as shown in "
            "the Academy index. Use before recommending a course."
        ),
        schema={
            "type": "object",
            "properties": {
                "course_id": {"type": "string", "description": "Course ID, e.g. '1896608'."}
            },
            "required": ["course_id"],
        },
    ),
]


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _format_tool_status(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "get_blog_post":
        return f"  -> reading blog post: {tool_input.get('url', '')}"
    if tool_name == "search_docs":
        return f"  -> searching docs: {tool_input.get('query', '')}"
    if tool_name == "get_doc_page":
        return f"  -> reading doc page: {tool_input.get('path', '')}"
    if tool_name == "get_training_course":
        return f"  -> reading course: {tool_input.get('course_id', '')}"
    return f"  -> tool: {tool_name}"


def _build_blog_index_text(posts: list[BlogPost], synopses: dict[str, str] | None = None) -> str:
    synopses = synopses or {}
    lines = ["## Blog Index\n"]
    for post in posts:
        date_part = f" | {post.date}" if post.date else ""
        lines.append(f"- **{post.title}**{date_part}")
        lines.append(f"  URL: {post.url}")
        blurb = synopses.get(url_to_slug(post.url)) or post.excerpt[:200]
        lines.append(f"  Synopsis: {blurb}")
        lines.append("")
    return "\n".join(lines)


async def _dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    post_by_url: dict[str, BlogPost],
    docs: DocsClient,
    course_by_id: dict[str, TrainingCourse],
) -> str:
    if tool_name == "get_blog_post":
        url = tool_input.get("url", "")
        post = post_by_url.get(url) or post_by_url.get(url.rstrip("/"))
        return post.full_content if post else f"Blog post not found in archive: {url}"
    if tool_name == "search_docs":
        return await docs.search(tool_input["query"])
    if tool_name == "get_doc_page":
        return await docs.get_doc_page(
            tool_input["path"], max_lines=tool_input.get("max_lines", 200)
        )
    if tool_name == "get_training_course":
        course_id = str(tool_input.get("course_id", ""))
        course = course_by_id.get(course_id)
        return format_training_detail(course) if course else f"Course not found: {course_id}"
    return f"Unknown tool: {tool_name}"


async def suggest(
    se_notes: str,
    project_root: Path,
    force_refresh: bool = False,
    output_format: str = "md",
    provider: LLMProvider | str | None = None,
) -> str:
    """Generate content recommendations for SE notes.

    Args:
        se_notes: Free-form text describing the prospect's interests/concerns.
        project_root: Directory holding the `output/` data files.
        force_refresh: Re-scrape the blog archive and Academy catalog regardless of age.
        output_format: "md" for a ranked markdown list, "email" for a follow-up email.
        provider: An LLMProvider, a provider name ("anthropic"/"openai"), or None
            to pick one from the available API keys.

    Returns:
        Formatted recommendations in the requested format.
    """
    # Resolve before any scraping, so a missing key fails in the first second
    # rather than after a multi-minute crawl.
    llm = resolve_provider(provider) if provider is None or isinstance(provider, str) else provider

    if force_refresh or is_archive_stale(project_root):
        _status("Refreshing blog archive...")
        await refresh_blogs(project_root, force=force_refresh)

    if force_refresh or is_training_stale(project_root):
        _status("Refreshing ClickHouse Academy catalog...")
        await refresh_training(project_root, force=force_refresh, provider=llm)

    posts = parse_blog_index(archive_path(project_root))
    synopses = await generate_synopses(project_root, posts, provider=llm)
    blog_index_text = _build_blog_index_text(posts, synopses)
    post_by_url = {post.url: post for post in posts}

    courses = load_training(project_root)
    course_by_id = {course.id: course for course in courses}
    training_index_text = build_training_index_text(courses)

    user_content = f"SE notes about prospect:\n\n{se_notes}\n\n{blog_index_text}"
    if training_index_text:
        user_content += f"\n\n{training_index_text}"

    _status(f"Asking {llm.name} ({llm.main_model}) for recommendations...")

    async with DocsClient() as docs:
        async def dispatch(name: str, tool_input: dict[str, Any]) -> str:
            return await _dispatch_tool(name, tool_input, post_by_url, docs, course_by_id)

        def on_tool(name: str, tool_input: dict[str, Any]) -> None:
            _status(_format_tool_status(name, tool_input))

        result = await llm.run_tool_loop(
            system=_build_system_prompt(output_format),
            user_content=user_content,
            tools=_TOOLS,
            dispatch=dispatch,
            on_tool=on_tool,
            max_turns=MAX_TURNS,
        )

    return result.strip() or "No recommendations generated."
