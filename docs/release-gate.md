# Manual release gate — OKFgraph 0.16.0

Documentation for the release is indexed in [`docs/README.md`](README.md);
operators should use [`docs/quickref.md`](quickref.md) during sign-off.

Hosted GitHub Actions runners are **smoke gates only**: format, clippy,
workspace checks, wheel build/install, facade/wheel smokes, Python version
matrix, UDS smokes and external-lane Rust unit tests. They do not have the
CUDA/ORT environment required by the full differential matrix.

The full GPU-dependent parity suite is a **manual release gate** run on the
maintainer's CUDA workstation before publishing a stable release. It must be
recorded in the release evidence. No self-hosted GitHub runner is required for
0.16.0.

## Prerequisites

- Clean checkout of the release commit on the CUDA workstation.
- Tag `python-oracle-0.15.0` available. `parity/tools/oracle.py` creates a
  cached detached worktree at `parity/tmp/oracle-0.15.0`.
- Oracle venv `parity/.venv-oracle` with `embroider 0.15.1` and `bobine 0.9.1`.
- Main venv ORT/CUDA DLLs on `PATH` and `ORT_DYLIB_PATH` set.
- Native PDF plugin built at `target/debug/okfgraph_pdf.dll` for real
  PDF/Office scenarios.
- Fixtures restored before and after the run:
  ```bash
  git checkout HEAD -- parity/fixtures parity/corpus/update-python parity/corpus/update-rust
  rm -f parity/fixtures/*.lock
  ```

## Windows CUDA environment

```bash
export VCPKG_ROOT='C:\Users\Main\Documents\Rust\vcpkg'
export VCPKGRS_DYNAMIC=1
export PATH="/c/Users/Main/Documents/Rust/vcpkg/installed/x64-windows/bin:/c/Users/Main/.lmstudio/extensions/backends/vendor/win-llama-cuda12-vendor-v2:/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/onnxruntime/capi:/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/nvidia/cudnn/bin:/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/nvidia/cu13/bin:/d/User/Documents/Python/OKFgraph/.venv/Lib/site-packages/nvidia/cu13/bin/x86_64:$PATH"
export ORT_DYLIB_PATH='D:\User\Documents\Python\OKFgraph\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll'
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
```

## Required local commands

Run from the repository root.

### Rust gates

```bash
cargo fmt --all -- --check
cargo deny check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --lib
cargo test -p okfgraph --no-default-features --lib
cargo test -p okfgraph --no-default-features --test fuzz_robustness
```

Expected: 111 all-feature tests, 86 no-default tests and 9 fuzz properties
pass.

### Full differential matrix

```bash
parity/.venv-oracle/Scripts/python.exe parity/tools/run_matrix.py \
  --pdf-plugin 'target/debug/okfgraph_pdf.dll'
```

Expected:

```text
TOTAL:73 FAILS:0
```

`ingest-pdf-no-plugin` remains the one intentional skipped divergence (72
comparable scenarios, 73 reported total). Any other op diff or DB diff fails
the release. If a
workstation standby interrupts a full run, rerun the affected scenarios with
`--only`; all must be green before sign-off.
The stable cutover rerun was performed with the root Python package deleted;
all oracle imports came from `python-oracle-0.15.0`.

### Coverage

```bash
parity/.venv-oracle/Scripts/python.exe parity/tools/check_coverage.py
```

Expected:

```text
ops: 30, success: 30, typed error: 21, documented waivers: 9
outputs scanned: 78
```

### Oracle independence

The old root Python implementation has been deleted from the default tree.
The matrix, `check_cli.py` and `check_external_lanes.py` must run without it;
the only Python 0.15 source is the cached `python-oracle-0.15.0` worktree.
Before deletion the same gates were run with the package hidden, which is the
historical evidence for this property.

### Performance gate

```bash
parity/.venv-oracle/Scripts/python.exe parity/tools/check_perf_bench.py
```

The paired benchmark must stay within the ratios in
`docs/performance-findings.md`. The local gate was green at the last
recorded run.

### Surface gates

Run the applicable smoke gates with the built binaries/wheels:

```bash
python parity/tools/check_cli.py
python parity/tools/check_external_lanes.py
python parity/tools/check_packaging.py
python parity/tools/check_wheel.py
python parity/tools/check_facade.py
python parity/tools/check_client.py
python parity/tools/check_jobs.py
python parity/tools/check_cli_pdf_batch.py --pdf-plugin target/debug/okfgraph_pdf.dll
python parity/tools/check_mcp.py
python parity/tools/check_mcp_connect.py
python parity/tools/check_mcp_http.py
python parity/tools/check_pdf_wheel.py
python parity/tools/check_python_matrix.py
```

`check_uds.py` is required on POSIX hosts only.

## Evidence record

Store a short release note beside the parity output, for example
`parity/out/release-gate.md`, containing:

```text
release: 0.16.0
python-core-in-tree: false
commit: <sha>
oracle: python-oracle-0.15.0 via parity/tools/oracle.py
matrix: TOTAL:73 FAILS:0, <duration>
coverage: 30/30 success, 21 typed errors, 9 waivers
rust tests: <all-feature count> / <no-default count>
surface gates: pass/fail list with tool names
intentional divergences: ingest-pdf-no-plugin
operator: <name>
date: <UTC date>
```

## Release condition

A stable 0.16.0 may be published only when:

- hosted CI is green on Windows x64, Linux x86_64 and macOS arm64;
- the manual GPU matrix above is green and recorded;
- the paired benchmark meets the parity gate in `docs/performance-findings.md`;
- coverage and all surface gates are green;
- the wheel contains the thin facade, the native extension and the bundled
  `okfgraph/_plugin` converter, with no Python core implementation;
- one tag publishes one PyPI project.
