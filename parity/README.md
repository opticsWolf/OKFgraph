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

M0 note: `parity-run` currently emits core metadata only. Op execution is
implemented with the read ops in M1. Windows x64 M0 spikes are green; Linux
x86_64 and macOS arm64 remain.
