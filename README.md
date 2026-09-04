# doc-suggester-ch

Given SE notes about a prospect, recommends relevant ClickHouse blog posts, documentation pages, and ClickHouse Academy courses using Claude.

The ClickHouse counterpart to [doc-suggester-cgr](https://github.com/mbarretta/doc-suggester-cgr) (same idea, Chainguard content).

## How it works

1. Checks if the blog archive is fresh (< 7 days old); scrapes clickhouse.com/blog if not
2. Parses the archive into a lightweight index of ~870 posts (with LLM-generated synopses for faster, more precise filtering)
3. Loads the ClickHouse Academy catalog — self-paced courses and workshops with module outlines, difficulty, and intent signals
4. Connects to the official ClickHouse docs MCP server over HTTP
5. Calls Claude (`claude-opus-5`) with the blog index, Academy index, and doc tools — Claude fetches full content on demand and returns a ranked markdown list

### Content sources

| Source | How it's collected |
| --- | --- |
| **Blog** — [clickhouse.com/blog](https://clickhouse.com/blog) | Posts are enumerated from `clickhouse.com/sitemap.xml` rather than the paginated listing: one request yields every post plus a `lastmod`. Each page is scraped to markdown; publish dates and authors come from the page's JSON-LD. Localized (`/ja/`, `/ko/`) translations are skipped. |
| **Docs** — [clickhouse.com/docs](https://clickhouse.com/docs) | The official ClickHouse documentation MCP server at `https://clickhouse.com/docs/mcp`, over HTTP — nothing to install, no Docker. Provides full-text search plus a read-only filesystem over the docs tree. |
| **Training** — [ClickHouse Academy](https://learn.clickhouse.com) | `/main_catalog` requires a login, but the public `visitor_` views of the same pages do not, so the scraper walks those: catalog root → learning path → course. Course IDs are shared between the two views, so each course records both its public URL and the signed-in `/user_catalog_class/show/{id}` URL. |

Because the Academy pages don't publish difficulty, personas, or intent signals, one Claude Haiku call per course derives them from the scraped description. Results are cached and keyed by a hash of the description, so re-running only pays for courses that actually changed.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- `ANTHROPIC_API_KEY` set in your environment

No Go and no Docker — unlike `doc-suggester-cgr`, the scrapers are pure Python and the docs MCP server is hosted.

## Install

### Standalone (recommended)

Installs `doc-suggester-ch` as a global command — no virtualenv activation or `uv run` prefix needed.

```bash
git clone https://github.com/mbarretta/doc-suggester-ch
uv tool install ./doc-suggester-ch
```

Data is stored in `~/.local/share/doc-suggester-ch/` and refreshed automatically when it goes stale (blogs: 7 days, Academy catalog: 30 days).

### Development

```bash
git clone https://github.com/mbarretta/doc-suggester-ch
cd doc-suggester-ch
uv sync
```

In a repo checkout, data is written to `./output/` instead of the per-user directory.

## Usage

### Basic

```bash
# Standalone
doc-suggester-ch "prospect wants to cut their Datadog bill, high log volume"

# Development (from repo root)
uv run doc-suggester-ch "prospect wants to cut their Datadog bill, high log volume"
```

### Read notes from a file

```bash
doc-suggester-ch --notes-file notes.txt
```

### Pipe from stdin

```bash
echo "fintech, real-time dashboards on Postgres are too slow, evaluating Snowflake" | doc-suggester-ch
```

### Output as a follow-up email

```bash
doc-suggester-ch --format email "prospect wants sub-second dashboards"
```

Produces a ready-to-send follow-up email — warm opener, resources as bullets with inline URLs, and a closing offer to follow up. The default (`--format md`) returns a ranked markdown list with titles, URLs, dates, and relevance explanations.

### Initialize data on first use

Run `init` before your first real query to pre-fetch everything up front. This avoids a long pause on first use:

```bash
doc-suggester-ch init
```

This scrapes the blog archive (~870 posts), generates synopses for all of them, and builds the Academy catalog. Subsequent runs reuse the cached data.

### Force a refresh

```bash
doc-suggester-ch --refresh "prospect interested in ClickStack"
```

`--refresh` re-scrapes the blog archive and Academy catalog regardless of age, and re-derives the Academy LLM fields.

### Other flags

| Flag | Purpose |
| --- | --- |
| `--project-root DIR` | Use `DIR` for data instead of `~/.local/share/doc-suggester-ch` |
| `-v`, `--verbose` | DEBUG logging, including per-item scrape and enrichment failures |

## Generated files

All under `<project-root>/output/`, and all safely deletable — everything is refetchable.

| File | Contents |
| --- | --- |
| `clickhouse-blog-archive.md` | One `## Title` section per post, with a `*Source: <url> \| <date> \| <authors>*` line |
| `checkpoint.json` | `{slug: {title, url, date, scraped_at}}` — drives incremental scraping |
| `blog-synopses.json` | `{slug: synopsis}` — LLM-generated, generated once per post |
| `training-catalog.json` | Academy courses: scraped facts plus LLM-derived metadata |

## Development

```bash
# Run tests
uv run pytest tests/

# Run without installing
uv run python -m doc_suggester_ch "some SE notes"
```

### Module layout

| Module | Responsibility |
| --- | --- |
| `cli.py` | Argument parsing, notes resolution, project-root discovery, `init` |
| `suggester.py` | The Claude tool-use loop and output formatting |
| `fetcher.py` | Shared async HTTP: User-Agent, retries with backoff, sitemap parsing |
| `blog_scraper.py` | Sitemap discovery, post scraping, archive and checkpoint writing |
| `blog_manager.py` | Archive staleness checks and parsing back into `BlogPost`s |
| `academy_scraper.py` | Three-level Academy crawl and LLM enrichment |
| `training_manager.py` | Catalog staleness, loading, and prompt formatting |
| `docs_client.py` | MCP client for the hosted ClickHouse docs server |
| `synopsis_generator.py` | Cached LLM synopses for the blog index |
