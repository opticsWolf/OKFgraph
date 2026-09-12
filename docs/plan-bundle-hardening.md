# Plan: bundle hardening (google-okf findings 1, 4, 2)

Source: coderadar review of `google-okf` (producer-side OKF library) vs
OKFgraph 0.2.0 (consumer/index side). Items ordered by dependency:
reserved-name skip → secret hygiene → pre-import gate. Items 3 and 5 are
deferred at the bottom, written out in full for later.

Invariants (same as retrieval roundup): no new MCP tools, CLI-only
maintenance, stdlib + ladybug only, deterministic outputs, tested +
documented per item, skills updated.

## 1. Reserved filenames (`index.md` skip)

**Problem.** Export generates one `index.md` per directory, but
`ImportManager._import_bundle_inner` ingests every `*.md` under the bundle
root (`root.rglob("*")` + suffix filter). Re-importing an exported bundle
creates `…/index` concepts: noise nodes that self-link, dilute PPR, and show
up in search. google-okf's `RESERVED_FILENAMES = {"index.md", "log.md"}`
is the prior art.

**Design.**
- Module-level constant in `import_.py` (next to `SUPPORTED_SOURCE_EXTS`):
  `RESERVED_FILENAMES = frozenset({"index.md"})`. Only `index.md` for now —
  `log.md` has no producer yet (see deferred §5); reserving the name without
  a writer invites confusion.
- Apply at the single choke point: the `source_files` comprehension in
  `_import_bundle_inner` (`fp.name.lower() not in RESERVED_FILENAMES`).
  Single-file `import_from_okf` keeps accepting an explicit `index.md`
  path (explicit beats implicit; matches `okf import <file>` semantics).
- Case-insensitive (`Index.md` on macOS/Windows checkouts).
- Diff's `state_of_dir` shares the walk → route its file enumeration
  through the same predicate so `diff` and `import` agree on what a
  "concept file" is. (Drift mode compares graph vs dir; if import skips
  index.md but diff counts it, every exported bundle reads as drifted.)

**Acceptance.**
- Round-trip test: export bundle → re-import → concept count unchanged,
  no `*/index` ids, `diff_db_dir` reports identical.
- Existing `import --all` on bundles *without* index files: byte-identical
  behaviour (defaults unchanged).

**Risk.** A user hand-writing a concept literally named `index.md` loses
it on bulk import. Mitigations: explicit single-file import still works;
warning log per skipped file; document in README + ingest skill.

## 4. `resource` URI sanitization

**Problem.** `resource:` frontmatter flows verbatim from parse
(`parse_source_file`) into Ladybug and back out through export. Any future
DB/API producer (or a careless hand-written file) embedding
`mysql://user:pass@host` lands secrets in the graph, exports, vaults, and
logs. google-okf strips credentials (`***@`) at production time; we do it
at parse time so *every* path (bundle import, single file, ingest kinds,
future producers) is covered by one rule.

**Design.**
- Pure helper (lives in `import_.py` next to `parse_source_file`, or
  `links.py` if we want all string hygiene in one place — decide at
  implementation; `import_.py` keeps the diff smaller):
  `sanitize_resource(uri)`: rewrite URI userinfo `scheme://user:pass@`
  → `scheme://***@`. Only when an `@` appears after `://` and before the
  first `/`; `mailto:`, bare paths, `okf-asset://` (UUID, no userinfo)
  untouched. Non-string / missing → passthrough.
- Call it once inside module-level `parse_source_file` on
  `fm.get("resource")`, so batch + single + diff-parse paths inherit it.
- Deterministic, no network, no config flag (hygiene is not optional;
  a flag would default-off and defeat the purpose).

**Acceptance.**
- Unit tests: `mysql://u:p@host/db` → `mysql://***@host/db`,
  `mongodb+srv://u:p@h/` → `mongodb+srv://***@h/`, plain paths /
  `https://` without userinfo / `None` unchanged, `okf-asset://…`
  unchanged.
- End-to-end: import file with credentialed `resource:` → stored +
  re-exported value sanitized.
- Docs note: graphs imported *before* this change keep old values until
  re-import (delta hash sees changed file only if the file changes —
  sanitization happens post-hash, so a `reindex` won't rewrite them;
  call it out in README/ingest-skill one-liner).

**Risk.** Over-matching exotic-but-legit `@` in paths
(`docs/meeting@noon.md` as resource). Guard: only rewrite when the `@`
sits in the authority section (between `://` and next `/`), which paths
without a scheme never satisfy.

## 2. `okf lint` (pre-import bundle gate)

**Problem.** Today a malformed bundle dies mid-import (parse warning +
skip) or, worse, imports half-broken: typeless files silently become
`type: "note"`, dangling relative links become `BrokenLink` rows. Doctor
sees the aftermath graph-side, but nothing validates the bundle *before*
the ~30s cold-boot + embed cycle. google-okf's `lint` is the model:
frontmatter + link resolution with CI exit codes — but file-side, no DB,
no model (same class as `diff` snapshot mode).

**Design.** New CLI-only command `okf lint [DIR]` (default: bundle root).
Three check groups, file-side only:

1. **Parse**: every concept file parses (`parse_source_file`); failures
   are errors (path + exception, sorted).
2. **Frontmatter**: missing `type:` / missing `title:` are *warnings*,
   not errors — import synthesizes both (`note` / stem), so erroring
   would contradict import behaviour. (Deliberate deviation from
   google-okf, which errors on missing `type` because its pipeline has
   no synthesis step.)
3. **Links**: reuse `extract_md_links` / `normalize_path_link` /
   `is_external` from `links.py`; resolve each against the directory
   tree (same rule as import: `](path.md)` by path, `[[wiki]]` by name
   index built from the walk via `build_name_index`). Unresolvable →
   error (would become a BrokenLink row). Anchor-only (`#section`) and
   external skipped, same as import.

Output: human-readable grouped list (sorted for determinism) + `--json`
machine form `{errors: [...], warnings: [...], files: N}`. Exit codes:
`0` clean (warnings ok), `1` any error, `2` usage/dir-missing — same
convention as `diff`/`doctor --strict`.

**Not in scope.** No auto-fix (`--fix` stays doctor's job: lint has no
graph to repair against, and rewriting user files from a linter is a
different trust level). No MCP tool (maintenance stays CLI-only). No
removal of anything from doctor/import — see decision record below.

**Acceptance.**
- Fixture bundle with one of each failure: unparseable frontmatter,
  dangling path link, dangling wikilink, ambiguous wikilink (resolves
  nowhere → error, never guessed), plus a typeless file (warning only).
- Golden `--json` output test.
- Consistency test (locks the doctor/lint relationship): lint-clean
  fixture → `import --all` → `diagnose()` reports zero `broken_link`
  findings. If the two link-checkers ever diverge on what "resolves"
  means, this test fails.
- `okf lint` on a missing dir → exit 2, no traceback.

**Risk.** Two link-resolution implementations (dir-side lint, DB-side
import) drifting apart. Mitigations: both call the same `links.py`
pure functions; the consistency test above pins agreement; any future
resolution-rule change must update the fixture + both callers.

### Decision record: why doctor keeps its link checks

Raised during review ("why factor the link-check out of doctor?").
Answer: nothing leaves doctor. Lint checks files pre-import; doctor
checks rows post-import (drift after edits, purge orphans, unlinted
ingest paths like MCP/thoughts). Import keeps recording links
standalone — it must work on bundles that never saw lint. Shared pure
parsing, separate checks, consistency test to pin agreement.

## Deferred: 3. SQLite producer + `SourceProducer` seam

**Idea.** google-okf's `BaseProducer.produce() -> Dict[str, Concept]` plus
`output_prefix` namespacing, with the highest-value instance implemented
in stdlib: a SQLite producer that maps `PRAGMA table_info` /
`foreign_key_list` to one concept per table (schema table in body,
columns in frontmatter extras, FKs as relative `[t](t.md)` links +
Relationships section) — i.e. their `MySQLProducer` with zero new
dependencies. Output is markdown written under a prefix
(`database/tables/…`); normal `import --all` picks it up, so the bundle
stays source of truth and emitted links become `LINKS_TO` edges through
existing import. Generalize to a `SourceProducer` protocol mirroring the
`DocumentConverter` philosophy (provider owns options, missing optional
dep → clear error, never silent fallback). DOCX-via-extra belongs here
too. **Why deferred:** new surface (CLI `produce` verb? `ingest --kind
db`?), new fixture DBs, and it should ride on top of `lint` (generated
bundles get pre-flighted). Roughly a half-day with fixtures.

## Deferred: 5. Observation notes in generated concepts

**Idea.** Their Mongo merger writes `⚠️ Schema variance: multiple types
detected` inline in the generated body — producers record *observations*,
not just transcriptions. Any future producer (SQLite first) should emit
null-rate / variance / row-count / staleness notes into the concept body
so FTS/PPR can answer "which tables look unhealthy?". This is also the
natural writer of a `log.md` bundle changelog (the second half of
`RESERVED_FILENAMES`), appended by import/produce runs. **Why deferred:**
no producer exists yet to carry it; defining the note schema now would
be speculative. Revisit with §3.
