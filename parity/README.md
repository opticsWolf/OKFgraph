# Differential parity harness

The oracle is the frozen Python 0.15 tree at `python-oracle-0.15.0`. Never run
the Python and Rust cores in one process.

## Layout

- `corpus/` — input bundles and DB fixtures.
- `fixtures/` — Python-created v10 DBs used for cross-open tests.
- `golden/pycompat.json` — Python-generated JSON/extra/content_sig/time cases.
- `scenarios/<name>.json` — settings, setup, and ordered op calls.
- `tools/run_python.py` — runs a scenario against the oracle venv.
- `tools/dump_db.py` — normalized DB dump (nodes and relationships).
- `tools/compare.py` — JSON comparer with float tolerance and rule-level
  doctor `info` ignores.
- `tools/run_matrix.py` — full differential matrix runner; exits non-zero on
  any unexpected op or DB difference.
- `tools/check_coverage.py` — verifies success and typed-error coverage for
  every public op, with documented waivers.
- `examples/import_bench.rs` / `tools/check_import_bench.py` — real-model
  filters-corpus timing smoke and unchanged-import delta check.
- `tools/gen_pycompat.py` — regenerates `golden/pycompat.json` from the oracle.
- `tools/check_rust_write.py` — verifies Python 0.15 reopens a Rust-written DB.
- `tools/check_wheel.py` — wheel import + GIL-release smoke test.
- `tools/check_mcp_spike.py` — Python MCP client for the original stdio spike.
- `tools/check_mcp.py` — Python MCP SDK client for the 11 Rust tools/annotations.
- `tools/check_mcp_connect.py` — MCP stdio proxy through a running `okf serve`.
- `tools/check_mcp_http.py` — MCP Streamable HTTP mount on `okf serve /mcp`.
- `tools/check_uds.py` — `OKFClient` over a Unix socket (POSIX only; SKIPs on Windows).
- `tools/check_jobs.py` — HTTP job submit/poll/list/cancel smoke (`okf serve`).
- `tools/check_cli.py` — 29-verb registration and `--json` differential smoke.
- `tools/check_packaging.py` — static wheel metadata/content gate.
- `tools/check_pdf_wheel.py` — clean-venv PDF add-on auto-discovery/conversion.
- `tools/check_python_matrix.py` — clean venvs for Python 3.12, 3.13 and 3.14.
- `tools/check_facade.py` — public Python `OKFRouter`/`OKFClient` smoke test.
- `tools/check_client.py` — public `OKFClient` against the Rust `okf serve`.
- `tools/make_import_oracle.py` / `tools/check_reopen_import.py` — fresh oracle
  imports and Python 0.15 reopen checks.
- `tools/compare_db.py` — semantic DB-dump comparator (normalizes Ladybug IDs
  and wall-clock graph times).
- `tools/compare_tree.py` — byte-for-byte export-directory comparator.
- `examples/dump_db.rs` — Rust DB dump for the comparator.
- `tools/gen_settings.py` — regenerate the Rust and Python settings tables.
- `tools/make_search_oracle.py` — build a real GPU-embedded v10 fixture
  (also builds the richer filter-matrix fixture with `--chunk-size 32`).
- `tools/gen_markdown.py` — regenerate chunking/YAML/frontmatter goldens.
- `tools/make_image_oracle.py` — build caption- or vision-route image fixtures.
- `tools/gen_vision.py` / `tools/compare_vision.py` — Pillow/embroider image
  golden and native vision-lane comparison.

## Scenario format

```json
{
  "name": "minimal",
  "settings": {"embedding_dim": 3},
  "setup": {"bundle": "parity/corpus/minimal"},
  "ops": [
    {"op": "get_by_id", "params": {"concept_id": "alpha"}}
  ]
}
```

## Run

```bash
uv sync --extra cpu --extra dev
uv run python parity/tools/run_python.py \
  --scenario parity/scenarios/minimal.json --out parity/out/minimal.python.json
cargo run --bin parity-run -- parity/scenarios/minimal.json > parity/out/minimal.rust.json
uv run python parity/tools/compare.py \
  --python parity/out/minimal.python.json --rust parity/out/minimal.rust.json
```

## M0 smoke checks

```bash
# Regenerate pycompat goldens with the frozen oracle
parity/.venv-oracle/Scripts/python.exe parity/tools/gen_pycompat.py
cargo test -p okfgraph --no-default-features pycompat

# lbug cross-open (Windows; vcpkg OpenSSL setup in docs/plan-rust-core.md)
export VCPKG_ROOT='C:\Users\Main\Documents\Rust\vcpkg'
export VCPKGRS_DYNAMIC=1
export PATH="$VCPKG_ROOT/installed/x64-windows/bin:$PATH"
cargo run -p okfgraph --no-default-features --example lbug_spike -- \
  parity/fixtures/oracle-v10.db inspect

# Wheel and MCP spikes
python bindings/python/build_wheel.py --release
python parity/tools/check_mcp_spike.py --exe \
  "$(pwd)/target/debug/examples/rmcp_spike.exe"
```

## M1 read-only scenario

```bash
parity/.venv-oracle/Scripts/python.exe parity/tools/run_python.py \
  --scenario parity/scenarios/read-only.json --out parity/out/read-only.python.json
cargo run -p okfgraph --bin parity-run -- parity/scenarios/read-only.json \
  > parity/out/read-only.rust.json
python parity/tools/compare.py \
  --python parity/out/read-only.python.json --rust parity/out/read-only.rust.json
```

`read-only.json` covers 14 ops (get/list/read/traverse/images/pins/lanes/doctor)
against `fixtures/oracle-v10.db`. Pydantic models are normalized with
`public_dict()` before comparison; the harness ignores the provenance-only
`runner` key. Current result: **0 differences**.

Build wheels with `python bindings/python/build_wheel.py --auditwheel=repair` so
the vcpkg OpenSSL DLLs are bundled for Python 3.8+ `add_dll_directory` behavior.

## M2 GPU search scenario

```bash
export ORT_DYLIB_PATH='D:\User\Documents\Python\OKFgraph\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll'
export PATH="/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/onnxruntime/capi:/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/nvidia/cudnn/bin:/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/nvidia/cu13/bin:$PATH"
export HF_HUB_OFFLINE=1
parity/.venv-oracle/Scripts/python.exe parity/tools/make_search_oracle.py \
  --db parity/fixtures/oracle-v10-gpu.db --corpus parity/corpus/minimal \
  --out parity/out/search-oracle.json
```

`search.json` uses that fixture with `device=cuda`, `precision=auto`, and the
real `google/embeddinggemma-2` model at dim 512. Rust host/query vectors match
the Python oracle exactly; the 15-op scenario currently reports
**0 differences**. The GPU ORT build is taken from the existing
`D:\User\Documents\Python\OKFgraph\.venv` — no download. A vendored
`esaxx-rs` patch keeps tokenizers on the MSVC dynamic CRT so it links with
Ladybug.

`parity/golden/markdown.json` is generated by `gen_markdown.py` from the
Python oracle and covers mordant chunk text/offsets, PyYAML 1.1 scalars, and
`python-frontmatter` metadata/content. Rust tests compare against all cases.

### Filter matrix

```bash
parity/.venv-oracle/Scripts/python.exe parity/tools/make_search_oracle.py \
  --db parity/fixtures/oracle-v10-filters.db --corpus parity/corpus/filters \
  --out parity/out/filter-oracle.json --chunk-size 32 --chunk-overlap 4
```

`filter-matrix.json` (37 ops) and `filter-validation.json` (20 error
envelopes) run against that fixture and both report **0 differences**. The
Python runner wraps oracle exceptions with `okfgraph.errors.envelope()` so
error payloads are compared in the same shape as Rust's `Envelope::failure`.
Recency is intentionally omitted from the differential scenario because its
wall-clock score drifts between the two sequential runs.

### Expansion, budgets, images, vision

```bash
# successful chunks:expand + max_tokens reads (uses the filter fixture above)
parity/.venv-oracle/Scripts/python.exe parity/tools/run_python.py \
  --scenario parity/scenarios/expansion-and-budget.json \
  --out parity/out/expansion-and-budget.python.json

# caption-route image fixture + real vision-route fixture
parity/.venv-oracle/Scripts/python.exe parity/tools/make_image_oracle.py \
  --db parity/fixtures/oracle-v10-images.db --corpus parity/corpus/images \
  --out parity/out/image-oracle.json
parity/.venv-oracle/Scripts/python.exe parity/tools/make_image_oracle.py \
  --db parity/fixtures/oracle-v10-vision-images.db --corpus parity/corpus/images \
  --out parity/out/vision-image-oracle.json \
  --model-id jinaai/jina-embeddings-v5-text-nano-retrieval \
  --embedding-dim 768 --image-mode omni

# Gemma vision-route fixture (artifacts already cached)
parity/.venv-oracle/Scripts/python.exe parity/tools/make_image_oracle.py \
  --db parity/fixtures/oracle-v10-gemma-vision-images.db \
  --corpus parity/corpus/images --out parity/out/gemma-vision-image-oracle.json \
  --model-id google/embeddinggemma-2 --embedding-dim 512 --image-mode omni

# native vision lane vs embroider/Pillow golden (Jina shown; Gemma below)
parity/.venv-oracle/Scripts/python.exe parity/tools/gen_vision.py \
  --images parity/corpus/images/*.png --out parity/out/vision-golden.json
cargo run -q -p okfgraph --example vision_spike -- parity/corpus/images/*.png \
  > parity/out/vision-rust.json
python parity/tools/compare_vision.py --golden parity/out/vision-golden.json \
  --rust parity/out/vision-rust.json

OKFGRAPH_TEXT_MODEL=google/embeddinggemma-2 OKFGRAPH_EMBEDDING_DIM=512 \
  cargo run -q -p okfgraph --example vision_spike -- parity/corpus/images/*.png \
  > parity/out/gemma-vision-rust.json
python parity/tools/compare_vision.py \
  --golden parity/out/gemma-vision-golden.json \
  --rust parity/out/gemma-vision-rust.json
```

`expansion-and-budget.json` (10 ops), `images.json` (6 ops),
`vision-images.json` (6 ops) and `gemma-vision-images.json` (6 ops) all report
**0 differences**. The vision spike needs cuBLAS 12 alongside the GPU DLLs
(`win-llama-cuda12-vendor-v2` in this environment); cu13 alone cannot load the
vision matmul kernel. Prepared pixels match Pillow byte-for-byte for both
slots: Jina vectors match exactly (cosine 1.0), Gemma vectors differ by at
most ~7.5e-9. No model download was needed — the Gemma vision tier was already
cached under `onnx-community/embeddinggemma-2-ONNX`.

### Stage B import

```bash
# Python oracle builds the reference DB; the scenario runner then performs the
# same fresh import in Rust and compares ops output with volatile times removed.
parity/.venv-oracle/Scripts/python.exe parity/tools/run_python.py \
  --scenario parity/scenarios/import-minimal.json \
  --out parity/out/import-minimal.python.json
parity/.venv-oracle/Scripts/python.exe parity/tools/dump_db.py \
  --db parity/out/import-minimal.db --out parity/out/import-minimal.python-dump.json
cargo run -q -p okfgraph --bin parity-run -- parity/scenarios/import-minimal.json \
  > parity/out/import-minimal.rust.json
cargo run -q -p okfgraph --example dump_db -- parity/out/import-minimal.db \
  > parity/out/import-minimal.rust-dump.json
python parity/tools/compare.py --python parity/out/import-minimal.python.json \
  --rust parity/out/import-minimal.rust.json \
  --ignore-key created_at --ignore-key updated_at --ignore-key effective_at
python parity/tools/compare_db.py \
  --python parity/out/import-minimal.python-dump.json \
  --rust parity/out/import-minimal.rust-dump.json
```

`import-minimal.json` and `import-filter.json` both report **0 differences**
on ops output and semantic DB dumps. `check_reopen_import.py` confirms Python
0.15 can reopen both Rust-written databases with working search.

Always set `PYTHONHASHSEED=0` for differential runs: Python's directory builder
iterates a `set`, so root child order is otherwise process-dependent. Scenario
string params may contain `@RUNNER@`; the harness expands it to `python` or
`rust`, letting both sides write export trees to separate paths. Compare them
with `tools/compare_tree.py`.

Additional Stage B scenarios:

- `links-repair.json` — wikilink/markdown/relation misses, `BrokenLink` rows,
  repair on import, incoming-link preservation, and a no-op second bundle
  import.
- `delete-recover.json` — soft delete, tombstone listing, recovery with full
  snapshot/embedding, incoming-link conversion to broken links, and purge.
- `export-minimal.json`, `export-filter.json`, `export-links.json` — export
  reports plus byte-for-byte `okf`/`obsidian` trees, nested indexes, and YAML
  frontmatter with `uid`, timestamps, and relations.
- `images-import.json` — caption-route image assets, deterministic asset IDs,
  content hashes, pins, and index state.
- `images-import-vision.json` — Gemma vision-route assets with explicit
  `image_precision=fp16`, plus `reindex(if_dirty=...)` and forced rebuild.
- `diff.json` — identical and different dir-vs-dir snapshots plus DB-vs-dir
  drift after import.
- `lint.json` — clean bundles and the broken-link corpus's warnings/errors.
- `reembed.json` — dry-run and full vector rebuild with post-reembed search;
  ignore the volatile `seconds` field when comparing.
- `reembed-refusal.json` — `IMAGE_PIN_MISMATCH` on a graph with image assets.
- `update-supersede.json` — file-backed `update`/`supersede` with separate
  `update-python`/`update-rust` corpus copies selected via `@RUNNER@`; ignore
  path-valued keys when comparing.
- `detach.json`, `detach-no-verify.json`, `detach-refusal.json` — verified,
  unverified, refused, and forced detach with runner-specific corpus copies;
  ignore `bundle`, `path`, `detached_at`, `since`, `last_seen`, and `message`
  when comparing volatile provenance. `compare_db.py` normalizes detach
  provenance rows.
- `produce.json` — SQLite producer, log deduplication, and unknown-producer
  errors; compare the produced trees byte-for-byte.
- `sync.json` — incremental producer cursor and import of written files.
- `ingest-md.json`, `ingest-bib.json`, `ingest-thoughts.json` — `ingest` kind
  validation plus markdown, BibTeX, and thoughts single-concept conversion.
- `ingest-pdf-no-plugin.json` — Rust-only check that a missing plugin returns
  `CONVERSION_FAILED` with the install hint while a missing file returns
  `FILE_NOT_FOUND`.

`setup.import` may contain op params such as `mode`; the Rust runner forwards
them to `import_bundle`, and the Python harness forwards them after removing
constructor-only settings. Generate the vision oracle with `parity/.venv-oracle`
(embroider 0.15.1) while keeping the GPU ORT DLL path from the main venv.

The post-review matrix covers 57 scenarios: **56 report 0 differences** on ops
output after normalizing wall-clock graph times, internal Ladybug IDs,
set-ordered tags, and runner-specific paths. The 57th,
`ingest-pdf-no-plugin`, intentionally records the Python raw-`INTERNAL` vs Rust
`CONVERSION_FAILED` missing-plugin divergence. Import, links, delete,
image, reembed, YAML/time, multi-root, prune, dimension, and write-op
scenarios also match on semantic DB dumps; export trees match byte-for-byte.
New coverage includes `import-identity-frontmatter`, `import-multi-root`,
`import-prune-missing`, `prune-absent-root`, `delete-file-backed`,
`images-prune`, `import-dim-256/768`, `ingest-materialize`,
`ingest-thoughts-*`, `ingest-bib-bundle-root`, `ingest-thoughts-errors`,
`export-dotted-id`, `yaml-block-and-nested`, `time-microseconds-offset`,
`percent-escape-nonascii`, `doctor-empty`, `doctor-params`, `op-error-matrix`,
`ingest-params`,
`import-resource-sanitize`, `ingest-pdf-no-plugin`, and
`ingest-pdf-real` / `ingest-office-real` (real PDF/DOCX fixtures through the
optional bobine add-on), and `ingest-office-formats` (XLSX conversion success
and a PPTX refusal with identical typed envelopes).
`parity/tools/check_coverage.py` asserts every public op appears in a
scenario.

Known remaining coverage gaps: external lanes are fail-closed only; old-schema
migration uses the Python 0.15 one-time step (v10-only compatibility). PDF,
DOCX and XLSX are covered; pinned bobine refuses PPTX in both runners, which
`ingest-office-formats` records as a typed conversion outcome rather than a
divergence. `doctor` parameters are covered by
`doctor-params`; portable `info` probes (index inventory, image routes, stale
lock, lanes) are now compared with `--ignore-info-rule` for only the
host-specific `ort_*`, `converter_cache`, `cache_hygiene`, `pdf_converter`,
and `db_created` entries. `check_coverage.py` now enforces success plus typed-error coverage: 30/30
successes, 21 ops with typed-error scenarios, and nine read-only ops with no
Python failure branch documented as waivers. The ledger-only DB comparisons
for thoughts/ingest-params scenarios are skipped because their materialized
frontmatter hashes depend on wall-clock ordering. Cross-process Python↔Rust lock interop and
read-only refusal are checked by `check_lock_interop.py` /
`check_read_only.py`; facade and wheel checks are `check_facade.py` /
`check_wheel.py`; serve/MCP gates are `check_client.py`, `check_mcp.py`,
`check_mcp_connect.py`, `check_mcp_http.py`, `check_uds.py`, `check_jobs.py`,
and `check_cli.py`. Full findings and the open register are in
`docs/plan-bug-fixes.md` §11.

M2 note: current parity matrix is green; Linux x86_64 and macOS arm64 remain.
