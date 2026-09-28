# Part 1: Rewrite `academy_scraper.py` for the Skilljar-based ClickHouse Academy

## Context

ClickHouse Academy migrated its LMS from Thought Industries to **Skilljar**. The current scraper's entire URL scheme is dead — `/visitor_class_catalog` now 308-redirects to `/`, and `/main_catalog` / `/user_catalog_class/show/{id}` return 404/403. This was originally investigated as "the Academy content is now ungated, is there anything net-new to capture," but the honest finding is bigger: the old scraper produces zero courses today and must be rewritten regardless.

Live verification (curled with the repo's own `USER_AGENT`) found the new site is dramatically richer than what the old scraper ever captured:
- A public `sitemap.xml` (allowed by `robots.txt`), same shape the blog scraper already consumes.
- Every course page embeds a structured `skilljarCourse` JS object (title, short/long description, a flat `tags` list) — replacing regex-guessed prose scraping.
- Every course page lists its lessons as plain anchor tags with real lesson titles, in curriculum order — zero extra requests.
- Every lesson page renders a full, publicly accessible, timestamped video transcript (or lab instructions) with no login required.
- One path was investigated and ruled out: the 5 `/path/{category}` taxonomy pages looked like they'd map courses to categories the way the old TI category pages did, but they're client-side rendered (Vue app pulling from an internal API) — no course links exist in the static HTML. Confirmed by live curl; do not build a crawler for these.

Course count also drops from 29 (old catalog) to **19** real courses (20 sitemap slugs minus one `[separator]` catalog-divider decoy) — a real consolidation, not a scraping bug, so don't be alarmed if a first run returns roughly 19.

**Decision on transcript usage** (confirmed with the user): full lesson transcripts are fetched and used **only as enrichment input** — they make the one-time LLM enrichment call (`difficulty`/`summary`/`technologies`/`personas`/`problems_addressed`/`intent_signals`) far more grounded, since day-to-day prospect-matching runs through those enriched fields, not raw description text. They are **not** persisted into `Course.description` or `training-catalog.json` — that field stays a small, course-level blurb, exactly like today, so the live `get_training_course` tool call and the JSON file don't grow unboundedly.

## Downstream contract (must be preserved, verified against `training_manager.py`/`suggester.py`/`cli.py`)

- `training_manager.load_training` only cares about the JSON envelope `{generated_at, source, courses: [...]}` and reads each course entry via `.get(key, default)`. Entries need non-empty `id`/`url`; every other field is optional.
- `member_url` and `style` are safe to stop populating — both are always read via `.get(..., "")` and only shown behind `if course.member_url:` / `if course.style:` guards.
- `content_hash` is **never** read downstream — it's purely `academy_scraper`'s own enrichment-cache key.
- `suggester.py` and `cli.py` only depend on the call signature `refresh_training(project_root, force, provider)` returning an `int` — nothing else about the module's internals matters to them. **No changes needed in `training_manager.py`, `suggester.py`, or `cli.py`.**

## Final `Course` dataclass

```python
@dataclass
class Course:
    id: str                                              # course slug, e.g. "data-warehousing-with-clickhouse"
    title: str
    url: str                                             # https://learn.clickhouse.com/{slug} — the only URL now
    learning_paths: list[str] = field(default_factory=list)  # = Skilljar's raw `tags` array, verbatim
    language: str = "en"                                 # unchanged ASCII-title heuristic
    module_count: str = ""                               # count of non-quiz lesson entries
    modules: list[str] = field(default_factory=list)      # ordered lesson titles (quiz entries kept, for outline completeness)
    description: str = ""                                # short_description + markdown(long_description_html) — course-level only, no transcripts
    difficulty: str = ""
    summary: str = ""
    technologies: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    problems_addressed: list[str] = field(default_factory=list)
    intent_signals: list[str] = field(default_factory=list)
    content_hash: str = ""                                # sha256[:16] of the ENRICHMENT source text (description + transcripts), not of description alone
```

Dropped entirely: `member_url` (no signed-in vs. visitor URL split anymore), `style` (no structured info panel on Skilljar pages). No new `tags` field — the raw tag list is folded straight into the existing `learning_paths`, reusing the existing prompt slot and existing `training_manager` display code with zero schema change.

## New/changed constants and functions in `academy_scraper.py`

Remove: `VISITOR_CLASS_URL`, `MEMBER_CLASS_URL`, `_CATEGORY_LINK_RE`, `_CLASS_ID_RE`, `_MODULE_RE`, `_INFO_KEYS`, `_GENERIC_DESC`, `parse_categories`, `parse_class_ids`, `_extract_info`, `_extract_about`, `parse_course_html`. Rename `CATALOG_URL` → `SITEMAP_URL = f"{BASE_URL}/sitemap.xml"` (used both to fetch the sitemap and as the JSON payload's `"source"` value). Keep `BASE_URL`, `CATALOG_NAME`, `_ENRICH_CONCURRENCY`, `_strip_tags`/`_TAG_RE`, `catalog_path`, `_load_cached_courses` as-is.

**Discovery / filtering**
```python
_COURSE_ROOT_RE = re.compile(r"^https://learn\.clickhouse\.com/([a-z0-9-]+)$")
_NON_COURSE_SLUGS = {"page", "path"}  # defensive backstop; /page/x and /path/x are 2-segment URLs already excluded by the regex

def discover_course_slugs(entries: list[tuple[str, str]]) -> list[str]:
    """Filter fetcher.parse_sitemap() output to candidate course-root slugs.
    Cannot filter out the "[separator] ..." decoy by URL alone — that needs the fetched page's title (see parse_course_page)."""
```

**`skilljarCourse` JS-object extraction (regex-only, no JS eval)**
```python
_SCALAR_FIELD_RE = re.compile(r"(\w+):\s*'((?:[^'\\]|\\.)*)'")
_TAGS_ARRAY_RE = re.compile(r"tags:\s*\[(.*?)\]", re.DOTALL)
_TAG_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")

def _find_skilljar_script(soup: BeautifulSoup) -> str | None: ...
def _unescape_js_string(raw: str) -> str: ...   # \uXXXX via manual regex substitution only — NOT Python's unicode_escape codec
def parse_skilljar_course_block(html: str) -> dict | None: ...  # {id, title, short_description, long_description_html, tags}
def _html_to_markdown(html_fragment: str) -> str: ...  # strips script/style/noscript/form/svg, markdownify, strip badge images — replaces _extract_about
```
Run `_SCALAR_FIELD_RE`/`_TAGS_ARRAY_RE` directly against the whole `<script>` tag's text (already scoped to "the one containing `skilljarCourse`") — no separate block-boundary regex needed; simpler than isolating `{...}` first, and just as safe since the field names are specific enough not to collide with the script's other `var` declarations.

**Lesson list + course assembly**
```python
def parse_lesson_list(slug: str, html: str) -> list[tuple[str, str]]: ...  # ordered, de-duped (lesson_path, lesson_title), scoped to this course's slug
def parse_course_page(slug: str, html: str) -> tuple[Course, list[tuple[str, str]]] | None: ...
    # Returns None if no skilljarCourse block, or if title matches _SEPARATOR_TITLE_RE.
    # Builds Course with learning_paths = tags, modules = lesson titles, module_count = count excluding _QUIZ_TITLE_RE matches,
    # description = "\n\n".join([short_description, markdown(long_description_html)]).
    # content_hash left at "" here — finalized later, once transcripts are known.

_SEPARATOR_TITLE_RE = re.compile(r"^\s*\[separator\]", re.IGNORECASE)
_QUIZ_TITLE_RE = re.compile(r"take the quiz", re.IGNORECASE)
```

**Lesson transcript extraction**
```python
def parse_lesson_transcript(html: str) -> str: ...
    # BeautifulSoup find <article class="ch-lesson">; "" if absent (quiz pages have none — must not error).
    # Strip <span class="ch-ts"> timestamps and <div class="ch-slide"> images before markdownify.
```

**Enrichment-text assembly (transcripts feed enrichment only, never persisted description)**
```python
_LESSON_CONCURRENCY = 10  # mirrors fetcher.DEFAULT_CONCURRENCY

def _content_hash(text: str) -> str: ...  # sha256(text.encode())[:16] — same formula as today, just given a name

def _assemble_enrichment_text(description: str, lessons: list[tuple[str, str]], transcripts: list[str]) -> str: ...
    # Pure. Appends "## Lessons" + "### {title}\n\n{text}" per non-empty transcript, in lesson order.
    # Returns bare `description` unchanged if there are no non-empty transcripts.

async def _fetch_lesson_transcripts(client: httpx.AsyncClient, lesson_map: dict[str, list[tuple[str, str]]]) -> dict[str, list[str]]: ...
    # One shared asyncio.Semaphore(_LESSON_CONCURRENCY) across every course's lessons (~265 total requests).
    # A lesson fetch/parse failure yields "" for that lesson, never aborts the course (mirrors blog_scraper.refresh_blogs's per-item exception handling).
```

**Orchestration**
```python
async def scrape_catalog(client: httpx.AsyncClient) -> tuple[list[Course], dict[str, str]]:
    # 1. fetch SITEMAP_URL -> fetcher.parse_sitemap -> discover_course_slugs
    # 2. asyncio.gather fetch each /{slug} course-root page (unbounded, ~20 requests)
    # 3. parse_course_page per slug -> (Course, lessons); drop None (separator/unparseable)
    # 4. _fetch_lesson_transcripts(client, lesson_map) for all remaining courses
    # 5. for each course: enrichment_text[course.id] = _assemble_enrichment_text(...); course.content_hash = _content_hash(enrichment_text[course.id])
    # 6. sort courses (language, learning_paths[:1], title) — same ordering rule as today
    # returns (courses, enrichment_text)
```

`enrich_courses` gains one new optional parameter, backward-compatible with existing tests that don't pass it:
```python
async def enrich_courses(courses, cached, provider=None, source_text: dict[str, str] | None = None) -> None:
    # inside enrich_one: text = (source_text or {}).get(course.id, course.description); prompt description slot = text[:6000]
    # cache-hit check unchanged: prior.get("content_hash") == course.content_hash
```

`_ENRICH_PROMPT`: remove the `Style: {style}` line entirely (no replacement needed — `Learning paths: {paths}` already carries the tag signal now). `enrich_one` drops the `style=...` format kwarg.

`refresh_training` (unchanged signature/body shape, just updated call sites):
```python
courses, enrichment_text = await scrape_catalog(client)
...
cached = {} if force else _load_cached_courses(project_root)
await enrich_courses(courses, cached, provider=provider, source_text=enrichment_text)
payload = {"generated_at": ..., "source": SITEMAP_URL, "courses": [asdict(c) for c in courses]}
```
`asdict(course)` naturally omits `member_url`/`style` since those fields no longer exist — no extra code needed.

## Test plan — `tests/test_academy_scraper.py`

Replace the 4 Thought-Industries-shaped fixtures (`CATALOG_HTML`, `CATEGORY_HTML`, `COURSE_HTML`, `GENERIC_COURSE_HTML`) with:

1. **`SITEMAP_XML`** — 2 course-root URLs, 3-4 nested lesson URLs, one `/page/...`, one `/path/...`, and the separator-shaped decoy URL. → `test_discover_course_slugs_filters_page_and_path` (separator slug passes URL-shape filtering; page/path don't).
2. **`SKILLJAR_COURSE_HTML`** — `<script>var skilljarCourse = {...}</script>` with a `'`-escaped apostrophe and a `\u000D\u000A`-escaped newline in `long_description_html`, a `tags: [...]` array, and 3 lesson anchors under the matching slug (one "Take the quiz"). → `test_parse_skilljar_course_block_extracts_and_unescapes_fields`, `test_parse_course_page_builds_course` (asserts `learning_paths == tags`, `modules` includes the quiz entry, `module_count` excludes it, `description` has no "## Lessons" section, `content_hash == ""`).
3. **`SEPARATOR_HTML`** — same shape, `title: '[separator] Workshops and Tutorials'`. → `test_parse_course_page_skips_separator_decoy`.
4. **`LESSON_TRANSCRIPT_HTML`** — `<article class="ch-lesson">` with an `<h2>`, a `<span class="ch-ts">` timestamp, a `<div class="ch-slide">` image, prose `<p>`. → `test_parse_lesson_transcript_strips_timestamps_and_slides`.
5. **`QUIZ_LESSON_HTML`** — no `<article class="ch-lesson">` at all. → `test_parse_lesson_transcript_returns_empty_for_quiz_pages`.
6. **Pure helper tests** (no HTML fixtures): `test_content_hash_changes_with_text`; `test_assemble_enrichment_text_appends_sections_and_skips_empty`; `test_assemble_enrichment_text_returns_bare_description_when_no_transcripts`.
7. **`enrich_courses` tests** — keep all existing tests, drop `member_url=`/`style=` kwargs from every `Course(...)` construction, update any test asserting on the literal `"Style:"` prompt text. Add `test_enrich_courses_uses_source_text_when_provided` (prompt built from `source_text[course.id]`, not `course.description`, when the param is passed) and confirm the existing tests still pass unmodified when it's omitted (default `None` falls back to `course.description`).

`tests/test_training_manager.py`, `tests/test_cli.py`, `tests/test_suggester.py` — **no changes**, per the downstream-contract section above (all three already mock/fixture at the JSON-envelope or call-signature level, never at the HTML-parsing level).

## Open risks to verify during implementation (not blocking, spot-check as you go)

1. Escaped-apostrophe title/description on a real course (the sampled example had none) — confirm `_SCALAR_FIELD_RE`'s `\'`-aware capture handles it.
2. Lesson-anchor markup (`class="lesson-modular..."`) was only sampled on one course — spot-check a workshop/tutorial-styled course too, not just a learning-path course.
3. First live run should log final course/lesson counts so a silent 0-course or 1-course result (e.g. a sitemap format change) is obvious rather than passing silently.

---

# Part 2: Fix blog edit-staleness (new work — raised by "how does this stay up to date")

## Problem

`blog_scraper.refresh_blogs`'s checkpoint (`{slug: {title, url, date, scraped_at}}`) only records "have I ever scraped this slug" — once checkpointed, a slug is skipped forever, so an edited existing post's new content is never re-fetched. `synopsis_generator.generate_synopses`'s cache (`{slug: synopsis}`) has the identical bug: any slug with a cached synopsis is never regenerated. Both caches are presence-based, not content-based — this is a pre-existing gap, not something the Academy rewrite introduces, but it's the direct answer to "how does new/changed blog content get picked up."

Verified via grep before designing the fix: **nothing reads `refresh_blogs`'s return value** — `cli.py`'s `_run_init` calls it inside `asyncio.gather` and ignores the result (it re-parses the archive from disk afterward instead), `suggester.py` awaits it and ignores the result, and every test only asserts `awaited`/`not_awaited`. So its return-value semantics are free to change with zero compatibility risk.

## Fix: `blog_scraper.py`

- Checkpoint entries gain a `lastmod` field, storing the sitemap's `lastmod` for that URL at scrape time: `{slug: {title, url, date, scraped_at, lastmod}}`.
- New pure helpers (unit-testable without a mock HTTP layer, matching this repo's existing convention of testing pure parsing/diffing functions directly rather than the async orchestration):
  ```python
  def _needs_scrape(slug: str, lastmod: str, checkpoint: dict[str, dict]) -> bool:
      entry = checkpoint.get(slug)
      return entry is None or entry.get("lastmod", "") != lastmod

  def select_todo(discovered: list[tuple[str, str]], checkpoint: dict[str, dict]) -> list[tuple[str, str]]:
      return [(url, lastmod) for url, lastmod in discovered if _needs_scrape(url_to_slug(url), lastmod, checkpoint)]

  def _blogpost_to_scraped(post: "BlogPost") -> ScrapedPost:
      return ScrapedPost(slug=url_to_slug(post.url), title=post.title, url=post.url, date=post.date, authors=post.authors, markdown=post.full_content)

  def merge_posts(discovered: list[tuple[str, str]], scraped: dict[str, ScrapedPost], existing: dict[str, ScrapedPost]) -> list[ScrapedPost]:
      merged = {**existing, **scraped}  # scraped wins on slug collision — fresher content replaces stale
      return [merged[url_to_slug(url)] for url, _ in discovered if url_to_slug(url) in merged]
  ```
- `refresh_blogs` changes:
  - `todo = select_todo(discovered, checkpoint)` replaces the old `url_to_slug(url) not in checkpoint` filter — now catches both brand-new slugs and slugs whose `lastmod` advanced since last scrape.
  - After scraping, each checkpoint entry also stores `"lastmod"` (the discovered value for that URL).
  - The archive write becomes **always a merge-rebuild**, replacing the old "rebuild on force/missing, else append" branch — this is required, not optional: appending a re-scraped post as a second `## Title` section would create a duplicate entry, since `blog_manager.parse_blog_index` has no dedup logic (it's a pure regex scan of the whole file). Steps: parse the *existing* archive (if present and not `force`) via `blog_manager.parse_blog_index`, using a **function-local import** inside `refresh_blogs` (`blog_manager` already imports from `blog_scraper` at module level, so importing the other way at module scope would cycle; this repo already uses function-local imports for exactly this kind of cross-module call — see `cli.py`'s `_run_init`); convert each parsed `BlogPost` to a `ScrapedPost` via `_blogpost_to_scraped`; call `merge_posts(discovered, scraped, existing)`; write the full file in `discovered` order every time.
  - **Behavior change worth flagging explicitly, not just a bug fix**: because the rebuilt archive is now driven by `discovered` (the current sitemap) rather than accumulated appends, a post that disappears from the sitemap (unpublished/removed) is pruned from the archive on the next refresh instead of lingering forever. This is the correct behavior for a recommendation index, but it's a real change from today — flagging so it's a conscious choice, not a surprise.
  - Return value becomes "count of posts (re)scraped this run" (new + updated, not the full archive size).

## Fix: `synopsis_generator.py`

- On-disk schema changes from `{slug: synopsis_string}` to `{slug: {"synopsis": str, "content_hash": str}}`, hashed the same way as Academy's enrichment cache (`sha256(post.full_content.encode())[:16]`).
- `load_synopses` normalizes any legacy bare-string entry to `{"synopsis": value, "content_hash": ""}` on read — `""` never matches a real hash, so an old entry regenerates exactly once and then upgrades to the new shape. This is ordinary defensive parsing of a pre-existing data file, not a compatibility shim in the logic.
- The `missing` filter becomes: posts whose slug is absent from the cache, **or** whose cached `content_hash` doesn't match the post's current content hash. This is the piece that actually closes the loop — without it, a post that `blog_scraper.py` now correctly re-scrapes would still keep serving its stale synopsis forever, silently undoing the fix above.
- `generate_synopses`'s **return value to callers stays `dict[str, str]`** (slug → synopsis text only) — the richer on-disk shape is a private implementation detail. `suggester.py`'s `_build_blog_index_text(posts, synopses)` needs zero changes.

## Test plan additions

- `tests/test_blog_scraper.py`: `test_needs_scrape_flags_new_and_changed_lastmod`, `test_select_todo_skips_unchanged`, `test_merge_posts_prefers_scraped_over_existing`, `test_merge_posts_drops_slugs_no_longer_in_discovered` (covers the prune-on-removal behavior), `test_blogpost_to_scraped_round_trips_through_format_post` (write via `format_post`, read back via `blog_manager.parse_blog_index`, confirm fidelity — mirrors the existing `test_archive_round_trip_survives_horizontal_rules` pattern).
- `tests/test_synopsis_generator.py` (existing file): `test_load_synopses_upgrades_legacy_flat_string_entries`, `test_generate_synopses_skips_unchanged_content_hash`, `test_generate_synopses_regenerates_on_changed_content`, `test_generate_synopses_returns_flat_string_mapping` (confirms the richer on-disk shape doesn't leak into the return value).

---

## Critical files

- `src/doc_suggester_ch/academy_scraper.py` — Part 1 rewrite.
- `src/doc_suggester_ch/blog_scraper.py` — Part 2 fix (checkpoint `lastmod`, merge-rebuild).
- `src/doc_suggester_ch/synopsis_generator.py` — Part 2 fix (content-hash cache).
- `src/doc_suggester_ch/fetcher.py` — reused as-is (`make_client`, `fetch_text`, `parse_sitemap`, `DEFAULT_CONCURRENCY`); no changes.
- `tests/test_academy_scraper.py`, `tests/test_blog_scraper.py`, `tests/test_synopsis_generator.py` — fixture/test additions per above.
- `README.md` — update "How it works" / "Content sources" / module-layout tables to describe the Skilljar sitemap + lesson-transcript-for-enrichment mechanics (currently says "29 Academy courses" / signed-in URLs), and add a line on the blog archive's edit/removal-aware refresh.

## Verification

1. `uv run pytest tests/` — full suite green, including all rewritten/added tests across both parts.
2. `uv run doc-suggester-ch init --refresh` against the live site — confirm `output/training-catalog.json` has ~19 courses, non-empty `modules`/`description`/enriched fields per course, and that `description` contains no `## Lessons` section (transcripts must feed enrichment only, never leak into the persisted field).
3. Spot-check one course's enriched fields (`summary`, `problems_addressed`, `intent_signals`) look meaningfully grounded in real lesson content, not generic — the whole point of fetching transcripts.
4. `uv run doc-suggester-ch "prospect wants real-time observability with ClickStack"` — confirm a plausible Academy course recommendation appears end-to-end.
5. Diff course count against the last real `training-catalog.json` this repo produced (29 courses) to confirm the drop to ~19 is the expected consolidation, not a scraping regression.
6. Blog edit-detection sanity check: after a normal refresh, manually edit one entry's stored `lastmod` in `checkpoint.json` to an older value (simulating a site-side edit), re-run `--refresh`, and confirm only that one post is re-scraped and re-synopsized — not the full ~870-post catalog — and the archive's total post count is unchanged (no duplicate section).
