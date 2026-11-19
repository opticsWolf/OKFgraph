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
- `tools/compare.py` — JSON comparer with float tolerance.
- `tools/gen_pycompat.py` — regenerates `golden/pycompat.json` from the oracle.
- `tools/check_rust_write.py` — verifies Python 0.15 reopens a Rust-written DB.
- `tools/check_wheel.py` — wheel import + GIL-release smoke test.
- `tools/check_mcp_spike.py` — Python MCP client for the Rust stdio spike.
- `tools/check_facade.py` — public Python `OKFRouter` smoke test.
- `tools/gen_settings.py` — regenerate the Rust and Python settings tables.
- `tools/make_search_oracle.py` — build a real GPU-embedded v10 fixture.

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
the Python oracle exactly; the 13-op scenario currently reports
**0 differences**. The GPU ORT build is taken from the existing
`D:\User\Documents\Python\OKFgraph\.venv` — no download. A vendored
`esaxx-rs` patch keeps tokenizers on the MSVC dynamic CRT so it links with
Ladybug.

M2 note: PyYAML/mordant chunking parity, full filter matrix, cross rerank,
vision/image search, and budgeted reads remain. Windows x64 spikes are green;
Linux x86_64 and macOS arm64 remain.
