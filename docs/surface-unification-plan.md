# Surface unification plan: CLI, MCP, Python API

**Status:** implemented in okfgraph 0.10.0 (2026-10-04); the error-code table in §4 is the shipped set. Plan history: draft v3 (2026-10-04). v1 was the user's draft; v2 checks it against `dev` @ `7e88eab` (okfgraph 0.9.0), corrects it, and expands it. v3 records the §11 decisions (Q1–Q5 answered) and adds the Q6 impact appendix (§11.1); Q6 itself is still open.
**Goal:** one vocabulary and one implementation per operation, reached the same way from all three surfaces. One release cut.

## Ground rules

**Backward compatibility is irrelevant. Rename freely; consistency is key.**

- No hidden aliases, no deprecation windows, no additive-then-remove.
- Every rename lands together with its callers, tests, skills and docs in the same change. The whole set ships as one breaking release.
- **Enforced by construction, not by review.** Names are defined once (op signatures + one settings table, §3.1). Conformance tests (§8) fail when a surface drifts from them. A `grep` sweep for old spellings is the final backstop, not the main check.

Release: cut this as **okfgraph 0.10.0** (recorded: Q1 → B). The rename-freely ground rule stays in force after 0.10.0 — 0.x minor releases may still break, so nothing freezes here; a future surface freeze would be its own explicit decision.

## 0. Decisions at a glance

| # | Decision | Why |
|---|---|---|
| D1 | **Ops are the public `OKFRouter` methods** (`router.search`, `router.read`, …), implemented in `okfgraph/ops/` and mixed into the facade. There's no separate `operations.py` layer beside the facade. | v1 had op + facade wrapper + MCP + CLI, so four copies of every signature. With D1 there's one Python signature, and CLI/MCP are checked against it (§8.1). |
| D2 | **Ops return JSON-serialisable data** (dicts/lists, pydantic dumped via `public_dict()`/`model_dump(mode="json")`). They never return models. | Python `data` == MCP `data` == CLI `--json` `data`, byte for byte. Native models stay available on the component layer (`router.search_engine.*`, documented as advanced and not frozen). |
| D3 | **Router-free ops stay router-free**: `lint`, `produce`, `diff` (snapshot mode), `model_info` are `@staticmethod`s on `OKFRouter` (as `model_info` already is). | Keeps the property that `okf lint` / `okf produce` never pay the DB open or the ~30 s model cold boot. |
| D4 | **One settings table** drives CLI flags, MCP boot flags, TOML keys, env vars and the `OKFRouter` constructor (§3.1). | Today five hand-maintained key lists exist (`_parse_toml`, `_load_env`, `_apply_cli`, `_merge`, `cli._router`), plus MCP's own argparse. They have already drifted (MCP reads no TOML/env at all). |
| D5 | **`--json` is explicit; no TTY sniffing.** | "Human on TTY, JSON when piped" changes output under `capsys`, CI and `okf … \| grep` depending on the environment. Same input should give the same bytes. |
| D6 | **Results go to stdout, logs to stderr, always.** | Today `import`, `broken-links`, `repair-links` and `reindex` report results through `logger.info`, so `-q` hides the result itself. |
| D7 | **MCP errors set `isError`** (the tool raises, so the server returns an error result) and carry the error envelope as text. | A JSON string with `"error": {...}` in a *successful* tool result is read as success by clients and approval UIs. |

## 1. What exists today (verified on `dev` @ `7e88eab`)

### 1.1 Inventory

- **CLI** (`okfgraph/cli.py`, 1,560 lines, 20 subcommands):
  - Commands: `init`, `model-info`, `import`, `search`, `read`, `traverse`, `ingest`, `export`, `diff`, `doctor`, `lint`, `produce`, `shell`, `broken-links`, `repair-links`, `reindex`, `deleted-list`, `deleted-recover`, `deleted-purge`, `detach`.
  - Global flags (`_add_global`, hidden from per-command help): `--db --bundle --model --primary --bundle-root(ALIAS=PATH, append) --dim --max-length --cache-dir --device --precision --image-model --image-precision --cpu-arena --chunk-size --chunk-overlap --no-chunking --wal-mode --allow-remote-images --allowed-image-domains`.
  - Logging flags: `-v -q --log-file --profile`.
  - `--json` exists on `diff`, `doctor` and `lint` only.
  - Handlers call component managers directly, including private methods: `purge_mgr._recover_concept`, `search_engine._get_ancestry/_get_siblings`.
  - `shell` builds `argparse.Namespace` objects by hand with the old attribute names (`db=`, `bundle=`, `dim=`, `purge=`), so it is a hidden second copy of every flag.
- **MCP** (`okfgraph/mcp_server.py`, 536 lines, 5 tools): `search`, `read`, `traverse`, `ingest`, `export_bundle`.
  - Returns `json.dumps(...)` strings.
  - Failures come in two spellings: `"error: …"` and `"Concept not found: …"`.
  - Boot flags: `--db-path --bundle-root --root --device --model --precision --cpu-arena --embedding-dim --max-length --no-chunking --log-level`.
  - **It does not load `okfgraph.toml` or `OKFGRAPH_*` env at all.**
  - Its default `bundle_root` is the DB's parent directory; the CLI's is `.`/TOML.
- **Python** (`OKFRouter`, `okfgraph/router.py`, 733 lines):
  - Facade proxies, mostly `*args/**kwargs`: `get_by_id`, `list_directory`, `search_hybrid`, `traverse`, `import_from_okf`, `export_to_okf`, `list_broken_links`, `repair_links`, `diagnose`, `doctor_fix`, `diff_dirs`, `diff_db_dir`.
  - Static: `model_info`, `default_cache_dir`.
  - Everything else is reached through component attributes: `search_engine`, `import_mgr`, `export_mgr`, `doctor_mgr`, `diff_mgr`, `image_mgr`, `ingest_mgr`, `embed_engine`, `schema_mgr`, `purge_mgr`, `delta_mgr`, `encoder`, `vision_encoder`. There is no `roots_mgr`.
- **Legacy** (`okfgraph/tools.py`, 461 lines): 16 OpenAI-style definitions (v1 said 17) with zero runtime consumers. Only `tests/test_router.py` (`TestTools`) and `tests/test_chunking.py` import `TOOLS`.
- **Config** (`okfgraph/config.py`, 541 lines):
  - Precedence CLI > env > TOML > defaults.
  - TOML: `[database] path dim wal_mode`, `[embedding] …`, `[import] …`, top-level `bundle`, `[[roots]]`.
  - Env `OKFGRAPH_`: `DB DIM WAL_MODE MODEL DEVICE PRECISION CPU_ARENA CACHE_DIR MAX_LENGTH IMAGE_MODEL IMAGE_PRECISION MODE BATCH_SIZE CHUNK_SIZE CHUNK_OVERLAP ALLOW_REMOTE_IMAGES ALLOWED_IMAGE_DOMAINS NO_CHUNKING BUNDLE`, plus the ignored-with-warning `OMNI_MODEL_ID`.
- **Examples** (`examples/`): call facade methods that **don't exist**: `router.reindex`, `router.list_images`, `router.import_bundle`, `router.search_images_with_text`, `router.get_image_data` (`hasattr(OKFRouter, …)` is False for all five). They break with `AttributeError` since the `__getattr__` bridge was removed.

### 1.2 Corrections to v1

| v1 said | Actually |
|---|---|
| `model_id`: no env var today, don't add one | `OKFGRAPH_MODEL` exists → rename to `OKFGRAPH_MODEL_ID` |
| ADD `OKFGRAPH_MAX_LENGTH` (missing) | exists |
| `image_model_id`: no CLI flag | `--image-model` exists → `--image-model-id`; `--image-precision` exists |
| audit item: `--cpu-arena` parser presence | present in CLI and MCP |
| audit item: `--strict` semantics | exit 1 when `report["findings"]` is non-empty, after the optional `--fix` (`cli.py:814`) |
| audit item: `recover` entry point | `purge_mgr._recover_concept` (private, `purge.py:292`); siblings `list_deleted_concepts`, `purge_deleted_concepts(older_than)` |
| audit item: `lint_bundle` signature | `lint_bundle(bundle_dir: str \| Path) -> dict` (`lint.py:47`) |
| `produce` wraps `producers.produce` | `producer_for(name).produce(source, out, prefix=, overwrite=)` (`producers.py:499`), then `lint_bundle(out)` |
| `IMPORT_SCOPE_CLASH` (`--bundle` + `--primary`, verify) | confirmed (`cli.py:289`, exit 2); D9 below removes the clash instead of naming it |
| MCP boot flags stay as they are | MCP lacks TOML/env loading and lacks `--cache-dir --image-model --image-precision --chunk-size --chunk-overlap --wal-mode` and the remote-image flags |

### 1.3 Drift and defects found (each one is closed by this plan)

| # | Defect | Where | Closed by |
|---|---|---|---|
| X1 | MCP `read(include="chunks")` returns **pydantic repr strings** (`"id='a' parent_doc_id='p' …"`), not objects: `json.dumps(list[ChunkModel], default=str)` | `mcp_server.py:273` | D2 |
| X2 | Error paths exit **0**: `search --target chunks --rank hub`, `ingest` with missing `--md-file`/`--thoughts`/file not found, `read` of an unknown id, `deleted-recover` failure | `cli.py:352,845,864,881,500,555,949` | §4 |
| X3 | Filters silently ignored: `concept_type/tags/parent_id` are dropped on the `hub_rerank` and `expand` chunk paths and for `target=images` (CLI and MCP alike) | `cli.py:354-375`, `mcp_server.py:215-222` | §3.2 rule |
| X4 | Same op, different defaults: PDF ingest `auto_import` is **False** on CLI and hard-coded **True** on MCP | `cli.py:884`, `mcp_server.py:371` | §3.5 |
| X5 | Config precedence is truthiness-based: `--chunk-overlap 0` is ignored; a higher layer can't set a value back to its default (TOML `dim=1024` + env `OKFGRAPH_DIM=512` → 1024); `--cpu-arena` can't be turned off over TOML `true` | `config.py:432-540` | §3.1 settings table |
| X6 | Invalid TOML is skipped silently; validation errors only `warning` | `config.py:287, 243-247` | `CONFIG_INVALID` |
| X7 | MCP ignores `okfgraph.toml`/env; the default `bundle_root` differs from the CLI's | `mcp_server.py:60, 412-529` | §3.1 |
| X8 | `okf init` without `--dim` logs `"%d" % None` (logging error traceback on stderr) | `cli.py:249` | §3.7 |
| X9 | `model-info --model-id` has its own hard-coded default and ignores `--model`/TOML/env | `cli.py:1255` | §3.1 |
| X10 | `hub_weight=0` becomes 0.3 (`or 0.3`) on CLI concept search | `cli.py:419` | §3.2 |
| X11 | Results printed via `logger.info` disappear under `-q` | `_import`, `_broken_links`, `_repair_links`, `_reindex` | D6 |
| X12 | Examples call non-existent facade methods | `examples/*.py` | §3.7 + §8.4 |
| X13 | `purge_deleted` means two things: an `import_bundle` bool (remove concepts whose files vanished) **and** the soft-delete purge op | `import_.py`, `purge.py` | §3.7 rename |

## 2. Target shape

```
            okfgraph/settings.py — ONE field table (§3.1)
            name, type, default, choices, section, help, scope
                 │ derives
     ┌───────────┼────────────────┬───────────────────┐
     ▼           ▼                ▼                   ▼
  CLI flags   MCP boot flags   TOML keys / env   OKFRouter(**settings)

            okfgraph/ops/*.py — mixed into OKFRouter (D1)
            router.search / read / traverse / ingest / export_* / admin
            canonical snake_case params → JSON-able data
            raises OKFError(code, op, fields, remedy)
                 │
     ┌───────────┼──────────────────────┐
     ▼           ▼                      ▼
  CLI adapter   MCP adapter            Python callers
  parse → op    params 1:1 with op,    call ops directly;
  → render      Field() text, raise    components = advanced,
  (human|json)  → isError + envelope   not frozen
```

Components (`components/*.py`) stay the implementation layer. They change only where a rename reaches them (§3) or where a private method becomes public because an op needs it (`_recover_concept`, `_get_ancestry`, `_get_siblings`).

### Adapter rules (no exceptions)

1. **Canonical names are snake_case, defined once**: in the op signature (op params) or the settings table (connection settings). CLI flags derive mechanically: `max_tokens` → `--max-tokens`, and positional args keep the canonical name as metavar/dest. MCP params are identical to op params.
2. **Booleans defaulting True** get a `--no-x` CLI flag, which the adapter inverts (`extract_images=True` ↔ `--no-extract-images`). Booleans defaulting False get `--x`. Never `--x/--no-x` pairs.
3. **Canonical lists are `list[str]`.** The CLI accepts CSV and splits it at the boundary (`--tags a,b`). No CSV past the adapter. Env lists are CSV as well.
4. **Adapters own transport only.** `ctx: Context`/`_get_router(ctx)` stays in `mcp_server.py`; argparse, renderers and `on_page` progress stay in `cli.py`. `okfgraph/ops/` imports neither.
5. **`--json` is a global flag** (with `-v/-q`), so every command has it. Human rendering is the default (D5).
6. **No adapter-side defaults.** Defaults live in the op signature or settings table. The CLI uses `default=argparse.SUPPRESS`/`None` and passes only what the user set (this kills X4, X9 and X10).
7. **Presentation-only flags are allowed but named canonically and documented as CLI-only**: `read --output-path` (write the document to a file), `--json`, logging flags.

## 3. Canonical naming (the renames)

### 3.1 Connection settings: one table

Canonical name = the `OKFRouter` constructor kwarg. `okfgraph/settings.py` holds a list of `Setting(name, type, default, choices, section, help, cli=True, mcp=True, env=True)`. Everything else is generated from it:

- CLI flag `--<kebab>`
- MCP boot flag (same spelling: the CLI and MCP flag sets become identical)
- TOML key `[<section>] <name>`
- env `OKFGRAPH_<UPPER>`
- the `OKFRouter` kwarg

Precedence stays CLI > env > TOML > defaults. It is computed **per key over the layers that set it**: a layer's value counts when present, whatever it equals (this kills X5). MCP boot goes through the same `Settings.load(cli_args, bundle_root)` (kills X7).

| Canonical | CLI / MCP flag | TOML | Env | Change from today |
|---|---|---|---|---|
| `db_path` | `--db-path` | `[database] db_path` | `OKFGRAPH_DB_PATH` | CLI `--db`, TOML `path`, env `DB` renamed |
| `bundle_root` | `--bundle-root` | top-level `bundle_root` | `OKFGRAPH_BUNDLE_ROOT` | takes today's `--primary` semantics (primary root, **no pin**); see D9 |
| `roots` | `--root ALIAS=PATH` (repeatable) | `[[roots]] alias, path` | — (map shape) | CLI `--bundle-root A=P` → `--root` |
| `embedding_dim` | `--embedding-dim` | `[embedding] embedding_dim` | `OKFGRAPH_EMBEDDING_DIM` | moves from `[database] dim`; `DatabaseConfig.dim` → settings field |
| `model_id` | `--model-id` | `[embedding] model_id` | `OKFGRAPH_MODEL_ID` | CLI/MCP `--model`, env `MODEL`; `model-info` drops its private `--model-id` and reads the setting (X9) |
| `max_length` | `--max-length` | `[embedding] max_length` | `OKFGRAPH_MAX_LENGTH` | — |
| `device`, `precision`, `cpu_arena`, `cache_dir` | same | `[embedding]` | same | MCP gains `--cache-dir` |
| `image_model_id` | `--image-model-id` | `[embedding] image_model_id` | `OKFGRAPH_IMAGE_MODEL_ID` | CLI `--image-model`, env `IMAGE_MODEL`; MCP gains it |
| `image_precision` | `--image-precision` | `[embedding] image_precision` | `OKFGRAPH_IMAGE_PRECISION` | MCP gains it |
| `enable_chunking` | `--no-chunking` (rule 2) | `[import] enable_chunking` | `OKFGRAPH_ENABLE_CHUNKING` | the field is named after the constructor kwarg; TOML/env `no_chunking` renamed |
| `chunk_size`, `chunk_overlap` | same | `[import]` | same | MCP gains them |
| `wal_mode` | `--wal-mode` | `[database] wal_mode` | same | help text says "SQLite": fix to Ladybug |
| `allow_remote_images`, `allowed_image_domains` | same | `[import]` | same | MCP gains them |
| `mode`, `batch_size` | op params, not settings | `[import]` (defaults for the ops) | same | stay TOML/env-configurable as op defaults; the table marks them `cli=False` so they don't appear as global flags |
| logging | `-v/--verbose`, `-q/--quiet`, `--log-file`, `--profile` | — | `OKFGRAPH_LOG_LEVEL` | MCP `--log-level` → the same four flags |
| — | — | — | — | **delete** `omni_model_id` field, its TOML/env warnings and `OKFGRAPH_OMNI_MODEL_ID` (compat irrelevant) |

**D9: `--bundle` / `--primary` / `--bundle-root` collapse.**
- **Today:** `--bundle P` sets the primary root *and* pins `import --all` to that one tree. `--primary P` sets the primary without pinning. `--bundle-root A=P` adds a named root.
- **Target:**
  - `--bundle-root P` is the primary root and never pins.
  - Pinning becomes an op parameter, `import_bundle(bundle_path=…)`, CLI `import --all --bundle-path P`. `detach` and `diff` already take `bundle_path`/`old,new` as op params.
  - The clash disappears, so `IMPORT_SCOPE_CLASH` is never minted.

**Collision hazard (must be handled explicitly).** `--bundle-root` exists today with a *different meaning* (named extra root, `ALIAS=PATH`). The zero-grep sweep can't catch it, because the new spelling is identical. Two guards:
1. **Order:** both renames happen in one commit.
2. **Refusal:** the settings loader rejects a `bundle_root` value that contains `=` and isn't an existing path, with a usage error pointing at `--root`. `test_config.py` asserts the refusal.

### 3.2 `search(query, target, limit, concept_type, tags, parent_id, include_chunks, max_chunks_per_doc, expand, context_hops, hub_rerank, hub_weight, rank)`

- CLI: `--type` → `--concept-type`, `--parent` → `--parent-id`, `--chunks` → `--include-chunks`. Drop `or 0.3` (X10, rule 6).
- MCP: `type_filter` → `concept_type`. **ADD `include_chunks` and `max_chunks_per_doc`** (missing today).
- Python: `router.search(...)` is the op. `search_hybrid` leaves the facade (it remains `search_engine.search_hybrid`, advanced). `exclude_reserved` stays a component-only knob.
- The op owns the target dispatch (images / chunks{hub_rerank > expand > plain} / concepts). One implementation; the two adapter copies are deleted.
- **Rule for params a path doesn't use** (X3): passing a param that the chosen target/path ignores raises `BAD_VALUE` naming the param and the path. Examples: `rank` with `target=chunks` (today's message, now with exit code 2 instead of 0); `concept_type` with `hub_rerank`; any filter with `target=images`; `include_chunks` with `target≠concepts`. A future change may make a path honour a filter instead; silent dropping is never allowed.
  - Default-valued params count as "not passed". The op tells them apart with a sentinel, which is why rule 6 matters.

### 3.3 `read(concept_id, include, max_tokens)`

- Names are already unified. Behaviour isn't:
  - **Unknown id:** MCP returns the bare string `"Concept not found: …"` and the CLI prints and exits 0. Target: `UNKNOWN_CONCEPT` everywhere, including `include=chunks/document/context` (today these return empty results for an unknown id, with no error).
  - **`include=chunks`:** returns `list[dict]` (X1 fix via D2).
  - **`include=context`:** the op owns the assembler (incoming/outgoing links, ancestry, siblings, cap 10). The two copies in `cli.py:530` and `mcp_server.py:276` are deleted. `_get_ancestry`/`_get_siblings` become public `search_engine.get_context(concept_id, cap=10)`.
  - **`include=document`:** returns `{"concept_id", "markdown"}`, not a bare string, so the envelope stays an object.
- CLI-only presentation: `read --output` → `--output-path` (rule 7). `--- BODY ---` moves into the human renderer.

### 3.4 `traverse(start_id, relationship, direction, depth, node_type, target, max_path_length)`

- CLI: `--type` → `--node-type`.
- MCP: **ADD `node_type`**.
- Python: `router.traverse` gets an explicit signature. `search_engine.find_path(start_id, end_id, max_length)` → `find_path(start_id, target, max_path_length)`.
- The op owns three modes:
  - empty `start_id` → root listing (then `relationship/direction/depth/node_type` are `BAD_VALUE` if non-default)
  - `target` set → path
  - otherwise → walk
- An unknown `start_id` is `UNKNOWN_CONCEPT`. Today it returns an empty list.

### 3.5 `ingest(kind, md_path, pdf_path, thoughts, topic, concept_id, title, description, tags, mode, routing_mode, extract_images, auto_import, output_dir, batch_size, prune_missing, force)`

- CLI renames:
  - `--md-file` → `--md-path`
  - `--pdf-file` → `--pdf-path`
  - `--output` → `--output-dir`
  - `--purge` → `--prune-missing` (X13)
- **`auto_import` default (X4):** one default in the op, **True** (MCP behaviour; ingest means "into the graph"). CLI gets `--no-auto-import` (rule 2) with `--output-dir` for the convert-only flow. `on_page` and `converter` stay Python-only advanced params; the CLI passes its progress printer.
- Python: `IngestManager.ingest_md/ingest_pdf/ingest_thoughts` merge into `IngestManager.ingest(kind, …)`. `router.ingest` is the op. That takes five dispatches down to one.
- Missing required params per kind (`md_path` for md, …) raise `MISSING_PARAM` (today: three spellings, exit 0 on the CLI).
- MCP: unchanged names. It does **not** gain `force`, `prune_missing`, `output_dir` or `auto_import=False` (policy, §6).

### 3.6 `export_bundle(output_dir, directory_id, concept_type, tags, flavor)` + `export_concept(concept_id, output_dir, flavor)`

- `export_to_okf(concept_id, output_path, …)` → op `export_concept(concept_id, output_dir, flavor)`. It writes `<output_dir>/<concept_id>.md`, as the CLI already does, and returns `{"concept_id", "path"}`. The component keeps an `output_path`-based internal helper.
  - This removes v1's "`--output` maps to `output_dir`/`output_path` by branch" exception: both ops take `output_dir`, so the flag derives mechanically.
- CLI `export`: `--output` → `--output-dir`, `--type` → `--concept-type`, `--parent` → `--directory-id`. `--all` vs `--concept-id` stays adapter routing between the two ops.
- MCP: `export_bundle` unchanged. **ADD `export_concept`** (6th tool; single-item write, inside policy).
- Return shapes: `export_bundle` returns `{"output_dir", "concept_ids", "flavor"}` (today a bare list).

### 3.7 Admin ops (CLI + Python; MCP excluded by policy, §6)

| Op (canonical) | CLI | Implementation today → change |
|---|---|---|
| `init()` | `init` | constructs the router from settings and closes it. Returns `{"db_path", "embedding_dim", "model_id"}`. Fixes X8 (logs the *adopted* dim). |
| `model_info()` *(static)* | `model-info` | `EmbeddingEngine.model_info`; reads `model_id`/`cache_dir` from settings (X9) |
| `import_bundle(bundle_path, batch_size, mode, prune_missing, force)` | `import --all [--bundle-path P]` | `import_mgr.import_bundle`. `purge_deleted` → `prune_missing`. Returns `{"concept_ids", "images": {id: n}}` so the CLI renders without extra `list_images` calls. `alias` stays Python-only (PDF work-dir namespace). |
| `import_file(file_path, mode, force)` | `import FILE…` (per file; missing files are `FILE_NOT_FOUND`, today a warning) | `import_from_okf` → `import_file`. `rebuild_indexes` stays Python-only. |
| `lint(bundle_dir)` *(static)* | `lint [bundle_dir]` | wraps `lint_bundle` |
| `produce(source_type, source_path, output_dir, prefix, overwrite)` *(static)* | `produce`: `--from` → `--source-type`, `--source` → `--source-path`, `--output` → `--output-dir` | wraps `producer_for(source_type).produce(...)` + `lint_bundle`. Returns `{"producer", "files", "root", "prefix", "lint": {...}}`. |
| `doctor(stale_days, fix, strict)` | `doctor` | composes `doctor_mgr.fix()` + `diagnose()`. Returns `{"report", "fixed"?}`. `strict` turns findings into exit 1 via `DOCTOR_FINDINGS`, with the report still in `data`. `diagnose`/`doctor_fix` leave the facade (components). |
| `diff(old, new)` | `diff [old] [new]` | the op picks snapshot (two dirs, router-free) vs drift (graph vs dir). `diff_dirs`/`diff_db_dir` leave the facade. Exit 1 = different via `data.identical`. |
| `list_broken_links()` / `repair_links()` | `broken-links` / `repair-links` | `import_mgr`; `repair_links` returns `{"repaired": n}` |
| `reindex(if_dirty)` | `reindex [--if-dirty]` | `schema_mgr.reindex(force=not if_dirty)` → `{"rebuilt": bool}` |
| `list_deleted()` / `recover_deleted(concept_id)` / `purge_deleted(older_than)` | `deleted-list` / `deleted-recover CONCEPT_ID` / `deleted-purge [--older-than]` | `purge_mgr`: `list_deleted_concepts` → `list_deleted`, `_recover_concept` → public `recover_deleted` (a failure is `NOT_RECOVERABLE`, today exit 0), `purge_deleted_concepts` → `purge_deleted` |
| `detach(bundle_path, verify, force)` | `detach [--bundle-path P] [--no-verify] [--force]` | `import_mgr.detach` |
| `list_images(concept_id)` / `get_image(asset_id)` | `images [CONCEPT_ID]` / `image ASSET_ID [--output-path P]` | `image_mgr` (`get_image_data` → `get_image`). Full ops on all three surfaces (recorded: Q5 → B; the facade need from X12 is satisfied by the same catalog entries). MCP `get_image` returns metadata + base64 `data`; CLI writes the bytes to `--output-path` (rule 7) and prints the metadata. |
| — | `shell` | CLI-only. **Rewrite to dispatch each line through `build_parser()`** instead of hand-built Namespaces, so it can't drift. |

## 4. Result envelope, errors, exit codes

One envelope on the wire for CLI `--json` and MCP:

```json
{ "ok": true,  "op": "read", "data": { ... }, "warnings": [] }
{ "ok": false, "op": "read", "data": null,    "warnings": [],
  "error": { "code": "UNKNOWN_CONCEPT", "message": "no concept 'x'",
             "fields": { "concept_id": "x" }, "remedy": "search first to find IDs" } }
```

- **Python:** ops return `data`. Warnings go through `warnings.warn(OKFWarning)` and are also attached to the envelope by the adapters. Errors raise `OKFError(code, message, op, fields, remedy)`. `OKFError` subclasses `ValueError` for usage codes and `RuntimeError` for state codes, so existing `except` sites keep working while being converted.
- **MCP:** the success path returns the envelope string. The error path raises `ToolError(envelope_json)`, so the result carries `isError: true` (D7). Exceptions from the components that aren't `OKFError` are wrapped as `INTERNAL` with the type name; there are no bare tracebacks.
- **CLI:**
  - Human renderer by default, envelope with `--json` (D5).
  - **stdout = result, stderr = logs + human error lines** (D6).
  - **Exit codes:**
    - `0` ok
    - `1` domain outcome: `diff` different, `doctor --strict` findings, `lint` errors, any state-class `OKFError`
    - `2` usage: argparse errors, usage-class `OKFError`
    - `130` interrupted
  - All error paths in X2 move to non-zero.
- **Code set** (each has a class; usage → exit 2, state → exit 1):

| Code | Class | Raised by |
|---|---|---|
| `BAD_VALUE` | usage | an ignored param for the chosen path (§3.2), bad choice, out-of-range |
| `MISSING_PARAM` | usage | ingest per-kind requirements |
| `FILE_NOT_FOUND` | usage | ingest/import/lint/produce paths |
| `CONFIG_INVALID` | usage | TOML parse error (X6), validation failure, `bundle_root` containing `=` |
| `UNKNOWN_CONCEPT` | state | read / traverse / export_concept / recover |
| `UNKNOWN_ASSET` | state | get_image / list_images (unknown asset or concept id) |
| `NOT_RECOVERABLE` | state | recover past its window |
| `DETACHED` | state | imports/ingest on a detached graph without `force` |
| `DETACH_REFUSED` | state | `detach` with mismatches / untracked / source-only files and no `force`, or an absent root to verify against (without `no_verify`) |
| `PURGE_REFUSED_ABSENT_ROOT` | state | `prune_missing` with an unmounted root |
| `MODEL_PIN_MISMATCH` / `PRECISION_PIN_MISMATCH` / `DIM_MISMATCH` | state | embedding pins |
| `IMAGE_PIN_MISMATCH` / `VISION_INCOMPATIBLE` | state | vision pins / text-small graph |
| `NO_ORT_RUNTIME` | state | the router's pre-open check (missing vs installed-but-unimportable runtime get different hints) |
| `WRITE_LOCK_TIMEOUT` | state | `_write_lock_ctx` |
| `DB_LOCKED` | state | router open: the database is held by another process (okf-mcp, a parallel `okf`) |
| `SEARCH_UNAVAILABLE` | state | index missing / dirty when the search engine refuses |
| `DOCTOR_FINDINGS` / `LINT_ERRORS` / `DIFF_DIFFERENT` | outcome | exit 1 with the full report in `data`, not `error` (Python: `err.data`) |
| `INTERNAL` | state | anything unconverted (each one found is a bug to convert) |

Finalise the list in Phase 2 by converting every `raise RuntimeError/ValueError/KeyError` reachable from an op (`grep -n "raise " okfgraph/components okfgraph/router.py`) and every `[ERROR]` in `cli.py`. **A code ships only with: code + failing op + offending value(s) in `fields` + a remedy sentence.** `tests/test_error_paths.py` asserts codes, not message text.

## 5. File-by-file work

| File | Work |
|---|---|
| `okfgraph/settings.py` (new) | The settings table (§3.1); `Settings.load(cli_args, bundle_root)` with per-key layer presence; generators `add_cli_flags(parser, scope)`, `toml_schema()`, `env_names()`. Replaces the bodies of `_parse_toml/_load_env/_apply_cli/_merge`. |
| `okfgraph/config.py` | Reduced to TOML discovery (cwd → bundle root → `~/.config/okfgraph/config.toml`) + thin `OKFConfig` = `Settings`, or deleted if nothing else needs the name. Docstring regenerated from the table. |
| `okfgraph/errors.py` (new) | `OKFError`, `OKFWarning`, the code registry (code → class → exit code), `envelope(op, data=None, error=None, warnings=())`. |
| `okfgraph/ops/` (new: `query.py`, `ingest.py`, `export.py`, `admin.py`) | The op methods (§3.2–3.7) as a mixin `OpsMixin` that `OKFRouter` inherits; static ops for router-free ones. Imports nothing from `cli`/`mcp_server`. |
| `okfgraph/router.py` | Inherits `OpsMixin`; delete all `*args/**kwargs` proxies and the empty section-comment scaffolding (`router.py:629-732`); constructor takes the settings fields 1:1 (+ `OKFRouter.from_settings`). |
| `okfgraph/cli.py` | Global flags from `settings.add_cli_flags`; per-command flags derived from op signatures (§3.2–3.7); handlers = parse → op → render; `render_human` per op + one `render_json`; `--json` global; `shell` through `build_parser()`; `[ERROR]` prints → `OKFError`. Expect roughly −40% lines. |
| `okfgraph/mcp_server.py` | Boot via `Settings.load` + `settings.add_cli_flags(parser, scope="mcp")`; tools = `envelope(op, data=router.<op>(**params))`; error path `ToolError`; +`include_chunks`, `max_chunks_per_doc` on search; +`node_type` on traverse; +`export_concept`, `list_images`, `get_image` tools; tool docstrings keep the routing guidance. |
| `okfgraph/components/ingest.py` | Merge the 3 kind-methods into `ingest(kind, …)`. |
| `okfgraph/components/export.py` | `export_to_okf` → `export_concept(concept_id, output_dir, flavor)`. |
| `okfgraph/components/import_.py` | `import_from_okf` → `import_file`; `purge_deleted` param → `prune_missing`. |
| `okfgraph/components/search.py` | `find_path(start_id, target, max_path_length)`; public `get_context`. |
| `okfgraph/components/purge.py` | `list_deleted`, public `recover_deleted`, `purge_deleted`. |
| `okfgraph/tools.py` | DELETE. Delete `TestTools` in `test_router.py` (the conformance tests replace it); repoint `test_chunking.py`. |
| `examples/*.py` | Move to ops (`router.search`, `router.import_bundle`, `router.reindex`, `router.list_images`, `router.get_image`); covered by §8.4. |
| `tests/test_surface_conformance.py` (new) | §8.1–8.3. |
| `tests/test_cli.py`, `test_roundup_cli.py`, `test_error_paths.py`, `test_config.py`, `test_mcp_server.py`, `test_mcp_stdio.py`, `test_ingest_tool.py`, `test_okf_ingest_tool.py`, every test touching a renamed verb | Updated in the same commit as each rename. |
| `docs/harness-integration.md` | Tool surface 5 → 8; renamed params; envelope + `isError`; boot flags = CLI global flags; MCP now reads `okfgraph.toml`/env. |
| `skills/okfgraph-cli`, `skills/okfgraph-mcp`, `skills/okfgraph-ingest` | Need-tables regenerated from the op catalog (identical across the three by construction); Setup sections from the settings table; frontmatter routing lines stay. |
| `docs/architecture.md` | Op catalog + settings table sections, **generated** by `scripts/gen_surface_docs.py` (checked in CI for staleness). |
| `CHANGELOG.md` | One 0.10.0 entry: every rename (generated from the old→new map in the conversion script, not from memory), behaviour changes (X1–X13), envelope/exit-code contract. |

## 6. What stays split (policy, not debt)

- **MCP exposes query, single-item write and image reads:** `search`, `read`, `traverse`, `ingest`, `export_bundle`, `export_concept`, `list_images`, `get_image`.
  - Not exposed: bulk import, `prune_missing`, detach, `doctor --fix`, reindex, the deleted lifecycle, `force`, PDF convert-only (`auto_import=False`/`output_dir`).
  - Reason: bulk mutation, refusal overrides and artifact lifecycle aren't agent-tool operations, and the write-tool approval UI assumes a small blast radius.
  - The conformance test (§8.1) has an explicit **MCP exclusion list** per op, so an omission is a decision, never an accident.
- **Different things, not different names:** `shell` is CLI-only, server boot/transport is MCP-only, `--json`/logging/`--output-path` are CLI presentation.
- **Operational reality stays documented, not merged:** the CLI cold-boots per call (~30 s model load → batching rule), MCP holds a warm router, Python holds a handle. `search(rank="ppr")` and the router-free ops (D3) are the cold-path answers. The skills keep these rules; they stop duplicating names.

## 7. Sequencing (one breaking release; each step lands green)

```
0. Conformance + parity tests, fixture graph. Mark the known gaps xfail
   (they document today's drift: X1–X13, missing MCP params).
1. settings.py + errors.py (no behaviour change yet; config tests ported).
2. Ops extraction, op by op: search → read → traverse → ingest → export
   → admin. Each op lands with its renames, callers (CLI/MCP/shell/
   examples) and tests; its xfails flip to pass.
3. Envelope + OKFError + exit codes + stdout/stderr split (D6, D7);
   test_error_paths.py asserts codes.
4. Settings renames (§3.1, incl. D9 + the --bundle-root collision guard)
   + MCP boot through Settings.load.
5. Delete tools.py, the facade proxies, the omni shims; generated docs.
6. Skills/docs/CHANGELOG; zero-hit sweep; tag 0.10.0.
```

- **Dependencies:** step 2 is the wall. Step 1 can run in parallel with step 0. Step 4 touches disjoint files from step 2 (settings vs ops) and can run alongside it after step 1. Step 3 needs step 2's ops.
- **Effort (rough):** 0: 1d · 1: 1d · 2: 3–4d · 3: 1.5d · 4: 1d · 5: 0.5d · 6: 1d. **Total ≈ 9–10 days.**
- No step leaves two spellings alive: each rename is atomic within its step, and nothing is released until step 6.

## 8. Conformance tests (what replaces "grep and hope")

### 8.1 Signature conformance (`test_surface_conformance.py`)
For every op in the catalog:
- MCP tool input-schema property names == op params − the MCP exclusion list (§6).
- MCP defaults == op defaults.
- MCP `Literal` choices == op `Literal` choices.
- CLI subparser dests (normalised, rule-2 inversions applied) == op params − the CLI exclusions (`on_page`, `converter`, `alias`, `rebuild_indexes`).
- No CLI argument has a non-`None`/`SUPPRESS` default (rule 6).
- Settings: generated CLI flags == generated MCP boot flags (minus `scope`-marked ones); every TOML key and env var round-trips through `Settings.load`.

### 8.2 Behavioural parity
On one fixture graph (built once per session), for each shared op and a matrix of params:
- `router.<op>(**p)` == `json.loads(mcp tool)["data"]` == `json.loads(okf <op> --json …)["data"]`
- Error cases compare `error.code`.

### 8.3 Single-dispatch check
The target/kind/mode dispatch keywords (`target == "images"`, `kind == "pdf"`, `include == "context"`, `find_path(`) appear only under `okfgraph/ops/`. This is an AST/grep test, not a manual `grep -c`.

### 8.4 Examples smoke
Every `router.<name>` attribute referenced in `examples/*.py` and in fenced code in `skills/**/SKILL.md` and `docs/*.md` exists on `OKFRouter` (static scan). Every `okf <cmd> --flag` in those files parses with `build_parser()`. This would have caught X12.

### 8.5 Old-spelling sweep (backstop)
Runs over `okfgraph/ tests/ examples/ skills/ docs/`, excluding `docs/archive/` and `CHANGELOG.md`:

```bash
grep -rnE "type_filter|--md-file|--pdf-file|export_to_okf|import_from_okf|okfgraph\.tools|OKFGRAPH_(DB|DIM|BUNDLE|MODEL|IMAGE_MODEL|NO_CHUNKING)\b|--primary\b|--image-model\b|[^-]--model\b|--dim\b|--db\b|--bundle\b|--purge\b|_recover_concept|list_deleted_concepts|purge_deleted_concepts|search_hybrid\(|omni_model_id"
```

Plus the explicit test that `--bundle-root a=b` is refused (§3.1). Grep can't see a reused spelling.

## 9. Acceptance

- §8.1–8.4 green, with no xfails left from step 0.
- §8.5 returns zero hits.
- `tools.py` gone. No `*args/**kwargs` on `OKFRouter`. No private component method is called from `cli.py`/`mcp_server.py`.
- X1–X13 each have a regression test.
- The MCP server started with only `--db-path` honours `okfgraph.toml` and `OKFGRAPH_*` exactly as the CLI does (test with a temp TOML).
- Skills' need-tables, `docs/architecture.md` op catalog and settings table are generated and up to date (CI staleness check). CHANGELOG 0.10.0 lists every rename from the generated map.
- Full suite green; the hermetic no-torch test still green (ops add no imports).

## 10. Risks

| Risk | Mitigation |
|---|---|
| External harnesses break on renamed MCP params/boot flags (`type_filter`, `--model`) | Accepted by the ground rule. The CHANGELOG has a "harness migration" block with the exact `mcp.json` diff; `pi`'s own `mcp.json` uses `--db-path/--bundle-root/--root`, and **`--bundle-root` changes meaning (D9)**, so it must be checked by hand. |
| The ops mixin bloats `OKFRouter`'s namespace | Ops are the only public methods; components are attributes. `dir(OKFRouter)` minus `_`-prefixed == op catalog is part of §8.1. |
| JSON-only returns (D2) annoy Python callers who want models | Components keep returning models; the docs say "ops for stable data, components for objects". |
| `isError` handling differs across MCP clients | The envelope text is the same either way; `test_mcp_stdio.py` asserts `isError` on the wire. |
| Step 2 is large | Op-by-op commits with xfail flips; each commit is green and bisectable. |

## 11. Decisions (recorded 2026-10-04)

1. **Q1 version → B: 0.10.0.** One breaking release named 0.10.0; the rename-freely ground rule stays in force afterwards (see Ground rules). No surface freeze is implied.
2. **Q2 PDF ingest default → A: `auto_import=True` everywhere.** Ingest means into the graph; convert-only stays the explicit path (`--no-auto-import --output-dir`, §3.5).
3. **Q3 `purge_deleted` clash → A: the import flag becomes `prune_missing`; the soft-delete op keeps `purge_deleted`** (§3.5 and §3.7 unchanged).
4. **Q4 D2 strictness → A: no exceptions.** `router.read(include="body")` returns a dict like every op; models stay component-layer only (`router.search_engine.get_by_id`).
5. **Q5 image reads → B: full ops on all three surfaces now.** `list_images(concept_id)` and `get_image(asset_id)` join the catalog: CLI `images` / `image` verbs, the 7th/8th MCP tools, the `UNKNOWN_ASSET` error code (§4) and parity-matrix rows (§8.2). MCP `get_image` returns metadata + base64 `data`; CLI writes bytes to `--output-path`.
6. **Q6 config.py fate — open.** Impact investigation recorded below (§11.1); decide the thin-alias vs outright-delete question before step 1 of §7.

### 11.1 Q6 impact: deleting `okfgraph/config.py` outright

Verified 2026-10-04 on `dev` @ `7e88eab`:

- **`OKFConfig` is not package-public.** `okfgraph/__init__.py` exports only `ConceptModel`, `OKFRouter`, `cli_main`; the config surface is reachable only as the submodule `okfgraph.config`.
- **One runtime consumer: `okfgraph/cli.py`** (top-level import at line 12 plus a local re-import at line 611). It uses `OKFConfig.load(bundle_root=…, cli_args=…)`, `OKFConfig._apply_cli(...)` and a bare `OKFConfig()` to assemble the router kwargs. `router.py` never imports config.
- **Six test files import it:** `test_config.py` (the Gap #11b schema-validation suite; imports all three dataclasses), `test_max_length.py`, `test_model.py` (also `DEFAULT_MODEL_ID`, which must move — its only home is config.py), `test_multiroot.py` (2 local imports), `test_pages.py`, `test_precision.py` (builds `EmbeddingConfig` to parameterize device/precision cases).
- **Zero hits in `examples/`, `docs/`, `skills/`, `README.md`, `scripts/`.** The module docstring is the only prose description.

Consequences of deleting outright (B):

- `settings.py` absorbs everything worth keeping: TOML discovery (cwd → bundle root → `~/.config/okfgraph/config.toml`), the per-key precedence merge (the presence-based fix for X5), the Gap #11b schema validation (validators become per-field checks in the settings table), and the CLI-arg application (`_apply_cli`). None of that was planned to survive as-is anyway — §5 already rewrites `_parse_toml/_load_env/_apply_cli/_merge`.
- `cli.py`'s `_router` ports `OKFConfig.load(...)` → `Settings.load(...)` (already step 4 in §7).
- The six test files port mechanically: `test_config.py` becomes the settings-table validator tests (step 1's "config tests ported"); the other five build `Settings(**fields)` or pass the same kwargs straight to `OKFRouter` — field names are 1:1 with the constructor (§3.1), so test bodies shrink more than they change.
- Nothing else can break: the name is unreachable from the package exports and unreferenced outside `cli.py` and the tests.

Conclusion: the alias (A) preserves a name that nothing outside `cli.py`/tests uses, for no behavioural continuity — the merge logic it names is rewritten regardless. **B is viable at near-zero extra cost.** The only real deltas are `DEFAULT_MODEL_ID` needing a new home in `settings.py` and one extra file deletion in step 1.
