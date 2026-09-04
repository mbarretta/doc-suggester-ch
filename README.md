# doc-suggester-ch

Given SE notes about a prospect, recommends relevant ClickHouse blog posts, documentation pages, and ClickHouse Academy courses. Works with either Claude or OpenAI.

The ClickHouse counterpart to [doc-suggester-cgr](https://github.com/mbarretta/doc-suggester-cgr) (same idea, Chainguard content).

## How it works

1. Checks if the blog archive is fresh (< 7 days old); scrapes clickhouse.com/blog if not
2. Parses the archive into a lightweight index of ~870 posts (with LLM-generated synopses for faster, more precise filtering)
3. Loads the ClickHouse Academy catalog — self-paced courses and workshops with module outlines, difficulty, and intent signals
4. Connects to the official ClickHouse docs MCP server over HTTP
5. Calls the model with the blog index, Academy index, and doc tools — it fetches full content on demand and returns a ranked markdown list

### Content sources

| Source | How it's collected |
| --- | --- |
| **Blog** — [clickhouse.com/blog](https://clickhouse.com/blog) | Posts are enumerated from `clickhouse.com/sitemap.xml` rather than the paginated listing: one request yields every post plus a `lastmod`. Each page is scraped to markdown; publish dates and authors come from the page's JSON-LD. Localized (`/ja/`, `/ko/`) translations are skipped. |
| **Docs** — [clickhouse.com/docs](https://clickhouse.com/docs) | The official ClickHouse documentation MCP server at `https://clickhouse.com/docs/mcp`, over HTTP — nothing to install, no Docker. Provides full-text search plus a read-only filesystem over the docs tree. |
| **Training** — [ClickHouse Academy](https://learn.clickhouse.com) | `/main_catalog` requires a login, but the public `visitor_` views of the same pages do not, so the scraper walks those: catalog root → learning path → course. Course IDs are shared between the two views, so each course records both its public URL and the signed-in `/user_catalog_class/show/{id}` URL. |

Because the Academy pages don't publish difficulty, personas, or intent signals, one cheap-model call per course derives them from the scraped description. Results are cached and keyed by a hash of the description, so re-running only pays for courses that actually changed.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- **Either** `ANTHROPIC_API_KEY` **or** `OPENAI_API_KEY` set in your environment

No Go and no Docker — unlike `doc-suggester-cgr`, the scrapers are pure Python and the docs MCP server is hosted.

## Choosing a provider

Provider selection is by available credentials: if `ANTHROPIC_API_KEY` is set it wins, otherwise `OPENAI_API_KEY` is used. Force one with `--provider`:

```bash
doc-suggester-ch --provider openai "prospect wants sub-second dashboards"
```

Each provider uses two models — a capable one for the recommendation loop, and a cheaper one for the high-volume single-shot work (blog synopses, Academy enrichment):

| Provider | Main model | Bulk model |
| --- | --- | --- |
| `anthropic` | `claude-opus-5` | `claude-haiku-4-5` |
| `openai` | `gpt-5.5` | `gpt-5.4-mini` |

Model names churn faster than this code will, so both are overridable without a code change:

```bash
DOC_SUGGESTER_CH_MODEL=gpt-5.6-sol \
DOC_SUGGESTER_CH_BULK_MODEL=gpt-5.4-nano \
  doc-suggester-ch "prospect notes"
```

| Environment variable | Effect |
| --- | --- |
| `DOC_SUGGESTER_CH_PROVIDER` | Default provider, same values as `--provider` |
| `DOC_SUGGESTER_CH_MODEL` | Override the main model |
| `DOC_SUGGESTER_CH_BULK_MODEL` | Override the bulk model |
| `DOC_SUGGESTER_CH_OPENAI_REASONING_EFFORT` | Send `reasoning_effort` on OpenAI calls (`low`/`medium`/`high`). Omitted by default, since models that don't accept it reject the request |

The credential is resolved before any scraping, so a missing or wrong key fails in the first second rather than after a multi-minute crawl.

### Provider differences

Claude runs the loop with adaptive thinking enabled. OpenAI runs via Chat Completions, where reasoning depth is instead controlled by the opt-in `reasoning_effort` above. Recommendation quality will differ between the two; the retrieval and the indexes they see are identical.

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
| `--provider {anthropic,openai}` | Force a provider instead of picking by available key |
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
| `llm.py` | Provider abstraction: model choice, and each provider's tool loop |
| `suggester.py` | Index building, tool definitions and dispatch, output formatting |
| `fetcher.py` | Shared async HTTP: User-Agent, retries with backoff, sitemap parsing |
| `blog_scraper.py` | Sitemap discovery, post scraping, archive and checkpoint writing |
| `blog_manager.py` | Archive staleness checks and parsing back into `BlogPost`s |
| `academy_scraper.py` | Three-level Academy crawl and LLM enrichment |
| `training_manager.py` | Catalog staleness, loading, and prompt formatting |
| `docs_client.py` | MCP client for the hosted ClickHouse docs server |
| `synopsis_generator.py` | Cached LLM synopses for the blog index |

`llm.py` is where the two providers' protocols diverge — Anthropic returns `tool_use` content blocks and takes results back as `tool_result` blocks in one user message, while OpenAI returns `tool_calls` and takes each result back as a separate `tool` message. Rather than adapt one into the other, each provider owns its whole loop behind a shared two-method interface, and callers describe tools neutrally as `ToolSpec`. `tests/test_wire_protocol.py` runs both real SDKs against local servers speaking each protocol, so a serialization mistake fails in CI rather than against a live API.
