# Differential parity harness

The oracle is the frozen Python 0.15 tree at `python-oracle-0.15.0`. Never run
the Python and Rust cores in one process.

## Layout

- `corpus/` — input bundles and DB fixtures.
- `scenarios/<name>.json` — settings, setup, and ordered op calls.
- `tools/run_python.py` — runs a scenario against the oracle venv.
- `tools/run_rust.sh` — runs the same scenario through `parity-run --json`.
- `tools/dump_db.py` — normalized DB dump.
- `tools/compare.py` — JSON comparer with float tolerance.

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

M0 note: `parity-run` currently emits core metadata only. Op execution is
implemented with the read ops in M1.
