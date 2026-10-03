# Plan: embroider — shared ONNX/Jina crate (bobine + okfgraph)

**Status:** active maintenance plan (2026-09-27)
**Scope:** `D:/User/Documents/Python/embroider` (now `0.2.0`) — the single shared crate both consumers build on.
**Supersedes:** nothing; complements `docs/plan-embed-spinoff.md` (Phases 0–4 done, Phase 5 deferred) and `docs/plan-onnx-alignment.md` (Phases 1–6 done, Phase 7 remaining).
**Non-goal of this file:** the OKFgraph consolidated update plan (Phases 0–7) lives in chat; this file covers the embroider side only.

Single crate, two distributions: crates.io `rlib` (default, Python-free — bobine) + PyPI wheel (`extension-module`, `JinaV5`/`TokenizerHandle` — okfgraph).
Modules: `providers` / `probe` / `policy` / `acquire` / `error` / `diag` / `jina` (`src/lib.rs`).
Pins: `ort 2.0.0-rc.13 load-dynamic + cuda/rocm/directml/openvino/coreml/tensorrt/half + ndarray`, `onnxruntime==1.29.0` (single shared binary, `ORT_DYLIB_PATH`-overridable).
CI: `cargo test --locked`, `cargo check` (default Python-free) + `--features extension-module`, `scripts/check_packaging.py` (Cargo == pyproject, readme + licenses).
Release: tag-guarded `crates.io → maturin matrix (linux/win/macOS-arm64 × 3.11–3.13)`, OIDC trusted publishing.

## Goals / non-goals

Goals: stay the boring shared layer — frozen Jina contract, explicit policy-as-data, one pinned ORT, floor-pin compatible (`okfgraph>=0.2,<0.3`, `bobine 0.1→0.2`), golden-proven vectors.

Non-goals: no vision models, no chunker (mordant stays in okfgraph — chunk mismatch = silent space fork), no slot manager (bobine owns lifetimes), no `bobine[embed]` implementation here (consumer-side), no ORT upgrade without measurement.

## What works today (bobine side)

`D:/User/Documents/Rust/bobine/Cargo.toml:45`: `embroider="0.1"`, default features (pure `rlib`, no `pyo3`).

`bobine/src/engine.rs:9,21-34`:

```rust
use embroider::{SessionPolicy, cuda_available};
SessionPolicy::ort_defaults().apply(builder) // explicit no-tuning
embroider::apply_providers(tuned, providers)  // clone-and-fallback
```

- Zero new deps (`tokenizers`/`hf-hub`/`ndarray` already in tree).
- Zero behavior change by design: vision slots keep `ort_defaults()`; measured `text_embed()` (`Level3,intra=phys/2,inter=1`) never leaks into layout/OCR/table.
- Free win banked: corrected EP-availability `cuda_available()` replaces the lax registration probe (`bobine/docs/architecture.md:187,192-200`). CPU-only dylibs stop getting CUDA prepended; commit-time fallback warnings disappear; outputs byte-identical.

`okfgraph` side (`okfgraph/router.py:169,313,328`): `import embroider` → `JinaV5.open`/`open_files`, `JinaTokenizer` via wheel. `COMPAT.md` + `fixtures/golden_jina_v5_text_small.json` pin the frozen contract.

## What is available in 0.2.0 that bobine does not use

| Module | bobine uses | Missed in 0.2.0 |
|---|---|---|
| `providers`: `apply_providers`, `apply_providers_with_arena`, `map_provider` | `apply_providers` | `with_arena` flag (0.1.5: arena-off = 8x lower RSS, 1.4x time), 0.1.2 infallible return (removes 7x error-mapping) |
| `probe`: `cuda_available()` `OnceLock` | yes | 0.1.3 panic-free when no dylib (bobine table-slot test surfaced it) |
| `policy`: `DeviceReq`, `Precision`, `SessionPolicy::apply`/`build_session` | `ort_defaults().apply()` | `Precision::Int8`, `auto`→CUDA-FP16/CPU-FP32 rule, `cpu_arena` kwarg |
| `acquire`: `parse_owner_name`, tokenizer-only fetch, `ModelSpec`/`Artifact`/`builtin_models`/`lookup_model`, `FP32`/`FP16`/`NANO_TEXT_MODEL` | no — own `hf_fetch()` per model (`engine.rs:60+`) | registry (0.2.0: small fp32 + FP16-mirror, nano fp32/fp16/int8-measured, q4 killed @0.957); `available_models()` + `NANO_TEXT_MODEL` for UIs |
| `diag`: `OrtReport`/`report()` | no | structured `ORT_DYLIB_PATH` + CUDA log line |
| `error` | maps to `BobineError` at boundary (correct) | — |
| `jina`: `JinaV5`/`TokenizerHandle`/`MAX_LENGTH`/`MODEL_MAX_TOKENS`/`NATIVE_DIM` | no — intentionally (spinoff Phase 5 deferred) | `max_length` 1..32768 (0.1.4, default 8192 bit-identical), `open_files` air-gap, golden fixture |

CHANGELOG trail: 0.1.2 infallible `apply_providers` + `SessionPolicy::apply`; 0.1.3 panic-free probe; 0.1.4 token limit; 0.1.5 precision + arena; 0.2.0 registry + `Int8` + `available_models()`.

## Phase E0 — alignment (0.2.1, ~0.5d)

- Refresh `COMPAT.md` (still `0.1.3/okfgraph 0.2.12`): `okfgraph 0.6.x / bobine 0.5.x+ / embroider 0.2.x / onnxruntime 1.29.0`. Fix floor-pin line (`>=0.1,<0.2` → `>=0.2,<0.3` for the 0.6.x line).
- Support bobine bump: `bobine/Cargo.toml:45` `0.1` → `0.2`, `cargo update -p embroider`, `Cargo.lock` in. No code needed — `0.2` is backward-compatible for `apply_providers`/`SessionPolicy`/`cuda_available`.
- Verify: `cargo publish --dry-run --locked` (no path deps), `cargo check` default Python-free + `--features extension-module`, `cargo test --locked`.
- `CHANGELOG.md:Unreleased` → `0.2.1`.

## Phase E1 — plumbing hardening (~1–2d)

Driven by consumer pain, not speculation:

- `providers`: keep infallible `apply_providers`; document `with_arena` per-slot guidance (text arena-off measured; vision keeps arena-on unless bobine benchmarks otherwise).
- `probe`: `OnceLock` + panic containment stays; surface `diag::OrtReport` in engine startup logs.
- `acquire`: unify `parse_owner_name` / tokenizer-only fetch / `ModelSpec` validation where it dedups; leave working per-model `hf_fetch` alone otherwise.
- `error`: keep `anyhow`, map at boundaries (`PyRuntimeError` / `BobineError`) — no shared Python exception.
- DoD: pure `cargo test --locked` (no dylib/network), bobine golden conversions byte-identical, okfgraph parity `≤1e-5`.

## Phase E2 — registry evolution (as-needed, minor-gated)

`acquire::{ModelSpec,Artifact,builtin_models,lookup_model}` is the pattern for adding models:

- New entry = static contract (id, native dim, ceiling, ladder, per-precision artifacts) + golden fixture + `available_models()` constant. Unknown ids keep legacy path; default id never moves.
- `Precision::Int8` stays explicit opt-in (`auto` never selects); unlisted pairs fall back to fp32, never unvalidated weights.
- Contract change = new minor + re-index notice + regenerated `fixtures/golden_jina_v5_text_small.json` (+ per-model fixture). Never silent.

## Phase E3 — conformance + perf gates (~1d)

- Per-(model,precision) golden vectors vendored in `fixtures/`; `okfgraph/tests/test_golden_vectors.py` asserts live (`abs=1e-6`) + exact token counts offline.
- Keep `README.md` sections current (Session/threading policy, Runtime discovery, Lifecycle, Precision, Arena, Explicit files, stale-DLL pitfall, Failure policy, Contract, Conformance, Testing) — it is the consumer contract.
- Perf: re-run Level3/phys-2/1 vs defaults only on workload change; commit methodology + table, not timing artifacts; model tests stay out of fast CI.

## Phase E4 — release hardening (done, keep green)

`ci.yml` + `release.yml` already correct: tag==version guard, Cargo==pyproject lock-step (`scripts/check_packaging.py`), OIDC trusted publishing (PyPI project + pending publisher pre-tag). No nightly/full-matrix drift — frozen lockfile gives no signal.

## Deferred

Vision sessions, chunking, slot management, bobine text-embed (`embed` feature / `src/embed.rs` / Python extra) — consumer-side when a concrete vector-at-conversion need appears, Python-extra first. Any new top-level module = new plan.

## Definition of done

Bobine on `0.2`, `COMPAT.md` current, goldens green in both repos, `cargo publish --dry-run` + wheel matrix green, no vector change inside a minor.
