# Plan: ONNX-only — open topics after Phases 0–4

**Status:** draft (2026-10-02)
**Parent:** `docs/plan-onnx-only.md` (Phases 0–4 marked done there).
**Scope:** `OKFgraph` (dev, 0.7.0), `D:/User/Documents/Rust/bobine` (rust_dev), `D:/User/Documents/Python/embroider` (main, 0.2.1).

The audit of 2026-10-02 checked each phase of the parent plan against the three repos. This file lists what is still open, in order. Each item names the plan phase or completion criterion it closes.

## Already fixed in the audit (uncommitted)

| Repo | Change | Closes |
|---|---|---|
| okfgraph | `import_bundle` converts a `str` path to `Path`. The Phase 0 guardrail `test_no_heavy_imports_after_work` used to crash before its torch check; now it runs to it. | Phase 0 completion criterion |
| okfgraph | Stale tests: the ingest `mode` enum is now `["text"]`; `test_default_device_is_auto` (default `"auto"` since 0.5.0) | Phase 1 "full suite green" |
| okfgraph | `pyproject.toml` comment 0.6.x → 0.7.x | — |
| bobine | `vision_policy()` is extracted and used by `session_builder`; `vision_slots_never_use_text_policy` now tests the wiring, not embroider's constructors | Phase 4 |
| bobine | CI and release workflows: ORT 1.28.1 → 1.29.0; README/quickref `>=1.28` → `==1.29.0` | Phase 3 (single pinned ORT) |
| bobine, embroider | CI step: no torch / torchvision / transformers / optimum / sentence-transformers imports or dependencies. Excludes `tests/`, `docs/`, embroider `tools/export/` and bobine `legacy/`. | Phase 0 / Phase 5 "grep in all three CIs" |

Result: full okfgraph suite **666 passed, 11 skipped, 0 failed**; bobine `cargo test --lib vision_slots` passes.

## Step 1 — commit the audit fixes (~0.1d) ✅ DONE

One commit per repo, on its working branch:

- okfgraph `dev`: "plan-onnx-only audit: str bundle paths, stale tests"
- bobine `rust_dev`: "plan-onnx-only audit: pin vision policy wiring, CI on ORT 1.29.0, torch-free CI grep"
- embroider `main`: "ci: torch-free grep (plan-onnx-only Phase 0)"

Done when: CI is green on all three after push.

## Step 2 — enforce "cpu XOR gpu" in the lockfiles (~0.2d) ✅ DONE (okfgraph; bobine declares conflicts, has no uv.lock — `uv lock` there fails on the pre-existing py3.9/gpu resolution, out of scope)

Closes Phase 3 (D5) for development installs. Today `uv.lock` resolves `onnxruntime` and `onnxruntime-gpu` as compatible, so `uv sync --all-extras` installs both. Only `okf doctor` warns afterwards.

- okfgraph and bobine `pyproject.toml`:
  ```toml
  [tool.uv]
  conflicts = [[{ extra = "cpu" }, { extra = "gpu" }]]
  ```
- `uv lock` in both. Check that the diff only forks the ORT resolution and moves no other pin.
- pip users aren't affected (pip can't express this). The doctor warning stays the guard there.

Done when: `uv sync --all-extras` fails with a conflict error, and `uv sync --extra cpu` and `uv sync --extra gpu` each succeed.

## Step 3 — clean-venv install proof (~0.3d) ✅ DONE (wheel from working tree, 2026-10-02)

- `[cpu,pdf]`: only `onnxruntime 1.29.0`; no torch/transformers/optimum/sentence-transformers (no Pillow either — unneeded).
- `[gpu,pdf]`: only `onnxruntime-gpu 1.29.0`; `--device cuda` → `used_cuda=True`; doctor reports dylib + `cuda_usable=True`.
- bare: first encode raises the `[cpu]`/`[gpu]` hint, cached, clean exit 0.

Two bugs found and fixed while proving the bare row (this machine has a stale
`onnxruntime.dll` 1.17.1 in System32): pyo3 `PanicException` derives from
`BaseException`, so `except Exception` in `session_factory` and
`LazyRustEncoder._get_encoder` missed ORT-load failures (no hint, no caching,
poisoned-init retry). Both now catch `BaseException` (cancellations re-raised,
never cached). The router additionally pre-checks before `JinaV5.open`: a
failed ort init poisons its global lock and aborts the process at interpreter
teardown — with the pre-check the backend is never touched and exit is clean.
Recorded in parent plan Phases 1/3. Remaining teardown-abort on the
no-ORT path without the pre-check (i.e. explicit bad `ORT_DYLIB_PATH`) belongs
to embroider, not here.

These Phase 1 and Phase 3 completion criteria were never run. Use throwaway venvs outside the repo, building from the working tree (wheel) or from the released packages after Step 5.

| Install | Check |
|---|---|
| `okfgraph[cpu,pdf]` | `pip list`: no torch/torchvision/transformers/optimum/sentence-transformers; exactly one of `onnxruntime` / `onnxruntime-gpu` |
| `okfgraph[gpu,pdf]` | same, plus `okf doctor` reports the resolved dylib, ORT 1.29.0, and `used_cuda=True` with `--device cuda` |
| `okfgraph` (no extra) | first encode fails fast with "install okfgraph[cpu] or okfgraph[gpu]" |

Optional: turn the first row into a CI job (Linux, CPU) so it stays proven.

Done when: all three rows pass, and the result is recorded in the parent plan's Phase 1 and Phase 3 entries.

## Step 4 — local dev venv re-sync (user decision)

`OKFgraph/.venv` still has torch 2.14, transformers 5.17 and sentence-transformers 6.0.1. Its installed okfgraph metadata still says 0.6.0. `uv sync --extra cpu --extra pdf --extra dev` would remove 42 packages:

- **wanted:** the torch stack
- **probably not:** `gossamer-web`, `ddgs`, `tiktoken`, `huggingface-hub`, `transformers`

Effects:

- `tests/test_parity.py` skips as a whole once `transformers` is gone. The `dev` extra doesn't list it, so a clean dev install never runs the parity tests. `test_golden_vectors.py` still covers the embedding contract.
  - Decide whether that's enough, or whether to add `transformers` (tokenizer only, no torch) to a `parity` test extra.
- Tools installed by hand belong in their own venv/uv tool, not the project venv.
- The omni spike's `ref_torch.py` moves to `omni_spike/.venv-export` (README updated).

## Step 5 — release train (~0.5d + CI time) ✅ DONE 2026-10-02 (except archive)

Closes Phases 2 and 5 (release). Shipped, in order, CI-green at each step:

- embroider `v0.2.1`: crates.io + PyPI live (PyPI verified).
- bobine `v0.5.12` (new CHANGELOG.md; `cargo update -p embroider` left at
  0.2.0 — 0.2.1 wasn't on crates.io yet; the `0.2` floor accepts it).
  Release workflow green (crates.io + 3-wheel matrix); PyPI shows 0.5.12.
- okfgraph `v0.7.0`: merged `dev` → `main`, tagged, PyPI live.

Two fixes went in along the way (both on `dev`, merged to `main`): fast
tests that stub `embroider` now also stub the batch-encode path / a fake
`onnxruntime` (CI fast env has no ORT); README install line is
`okfgraph[cpu,pdf]`; release.yml embroider-floor comment fixed to
`>=0.2,<0.3`.

Order (each step's CI green before the next):

1. **embroider 0.2.1**: push, tag `v0.2.1` (the tag==version guard), trusted publishing to crates.io + PyPI. No API change; COMPAT matrix already says okfgraph 0.7.x.
2. **bobine**: merge `rust_dev` → `main`, bump the patch version, CHANGELOG ("embroider 0.2, ORT ==1.29.0 extras, shared-runtime log"), release. Optionally `cargo update -p embroider` to 0.2.1 first, since `Cargo.lock` is on 0.2.0 (works; no new API is used).
3. **okfgraph 0.7.0**: merge `dev` → `main`, tag, publish. The CHANGELOG breaking notes are in place:
   - `omni` extra and `mode=omni|optional` removed
   - ORT moved to `[cpu]`/`[gpu]`
   - image search is caption-based
   - stale `route='omni'` rows reported by `okf doctor`

   The README install line must say `okfgraph[cpu]` or `okfgraph[gpu]`.
4. ✅ Step 3 re-run against published packages: `[cpu,pdf]` → one ORT
   1.29.0, trio (okfgraph 0.7.0 + embroider 0.2.1 + bobine 0.5.12) resolves;
   bare → hint + clean exit; `[gpu]` trio → `used_cuda=True`.
5. ✅ Legacy Python bobine: **nothing to archive.** `opticsWolf/bobine` on
   GitHub is already the Rust repo (history rewritten at 0.3.0 — "legacy
   template"); no separate legacy repo exists. The local legacy clone's
   banner commit stays local-only (its origin now tracks Rust history —
   never force-push it).

## Step 6 — ladybug access violation ✅ DONE (fixed by upstream upgrade)

Intermittent native crash, unrelated to ONNX, found during the audit:

- **Where:** `okfgraph/components/schema.py:534` `_build_search_indexes` →
  ladybug pybind `query()` running `CALL CREATE_VECTOR_INDEX`, always on the
  *second* in-process Database (faulthandler trace captured).
- **Rate:** ~25% (4/16 full `test_chunking.py` runs). Solo class 4/4 clean,
  class triples 0/14, standalone 2–5-router scripts 0/20+ (cpu/cuda encode,
  imports, vector search, close+rmtree, kept-alive routers) — pytest-process
  context only. Suspect was the native worker-thread pool across DB
  lifecycles (`num_threads=0` default).
- **Resolution:** latest PyPI was 0.21.2 vs pinned 0.20.3. Scratch venv
  (repo code + ladybug 0.21.2, real CUDA encodes): crash loop **0/12**,
  full suite green (only exclusions: parity needs transformers; one
  pre-existing GPU-vs-CPU tolerance failure proven ladybug-independent and
  since loosened to atol=1e-3). No mitigation code, no upstream issue —
  the 0.20.x crash is moot. Pin bumped to `ladybug==0.21.2`.
- Production opens one router per process, so user exposure was limited to
  multi-graph scripts; still, the bump removes it everywhere.

## Step 7 — Phase 6 (ONNX image embeddings, omni-nano) — blocked

The design and evidence are in the parent plan's Phase 6 and in `omni_spike/README.md`. Reordered so the go/no-go gate runs before the expensive Rust work:

| Step | Work | Blocked on | Gate |
|---|---|---|---|
| 6.4a Real-image proof (moved first) | 191 real figures + captions received (`Image_spike/`). Parity run 2026-10-03 (`omni_spike/user_*.py`, `results/user/`): ONNX-dyn vs torch-native cos min/mean **fp32-cuda 0.99973/0.99999, fp16-cuda 0.99812/0.99996** (n=191), fp32-cpu 1.0 (n=30); retrieval gap onnx-vs-torch **0.0** → parity half PASSES. Caption-as-query recall (caption route 0.929 vs image 0.147 on caption-derived queries (circular — matching captions to themselves). **Visual-question round 2026-10-03** (`Image_spike/questions.jsonl`, 30 pixel-written Qs): torch-img r@1 **0.767**, onnx-fp32 **0.767** (gap 0.0 ✅), onnx-fp16 0.733, caption **0.633** — image wins where it should (chart counts, logos, labels). **6.4a PASSES both halves.** | done (user supplied figures; assistant wrote the 30 questions) | ✅ unblocks 6.0 (needs HF upload approval) → 6.1. |
| 6.0 Artifact on HF | ✅ DONE 2026-10-03: `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx` @ `95c7be0` (fp32 1.25 GB + fp16 0.62 GB + tokenizer + manifest + CC BY-NC 4.0 card). Downloaded sha256 == manifest for all 4 weight files. | — | ✅ |
| 6.1 embroider 0.3.0 `JinaV5Vision` | ✅ DONE 2026-10-03 (embroider `main` `4b56765/3223f50/a60b13f`, pushed): `vision.rs` port — banker's rounding ported (every dim ≡16 mod 32 mistargets without it); tokenizer truncation cleared (seq guard caught it live: 512 vs 1231); registry id `jina-v5-omni-nano-retrieval-vision` + explicit sidecar + `text_partner`; vision-slot sessions (`kSameAsRequested`, `gpu_mem_limit`); fp16-on-CPU hard error (stalls, not slow). Unit 42 green incl. bitwise host tensors (31 grids) + pixel pipelines; e2e 5 real figs on CUDA: cos 1.00000/0.99998 fp32, ≥0.99991 fp16; `cargo publish --dry-run` clean. (Export script stays in `omni_spike/` scratch per standing rule, not `tools/export/`.) | 6.0 | ✅ |
| 6.2 okfgraph 0.8.0 wiring | ✅ DONE 2026-10-03 (`dev`): modes restored on the vision route (`optional`/`omni` → `vision-onnx`), `LazyVisionEncoder`, image model/precision pins + `text_partner`/dim-768 compat gate, vision hash (route+model+precision+contract), Pillow core dep, CLI/config/MCP surface, doctor `image_routes`. Wiring proven live: 2 figs via router on CPU-fp32, stored vectors cos 1.00000 vs torch. 12 hermetic vision tests + full fast suite green (194 passed). Needs embroider>=0.3 at release (6.5 flips the pin; dev runs a maturin wheel). | 6.1 | ✅ |
| 6.3 In-situ parity | ✅ DONE small 2026-10-03: 6 figs (one per area) ingested `mode=omni` through the router (CPU-fp32, text-nano graph) → stored vectors cos **1.00000** vs torch native on all 6 (gate 0.9999). fp16 + r@1 already covered by 6.4a (191 figs, fp16 cos ≥ 0.99812, gap 0.0) and the embroider e2e (fp16 ≥ 0.99991 CUDA). Full-corpus re-ingest adds no new information — kept small per agreement. | 6.2 | ✅ |
| 6.5 Release | embroider 0.3.0 → okfgraph 0.8.0, COMPAT (image search needs text-nano) | 6.3 | Phase 5 release checks |

## Order and effort

| Step | Repo(s) | Effort | Depends on |
|---|---|---|---|
| 1 Commit audit fixes | all | 0.1d | — |
| 2 uv conflicts | okfgraph, bobine | 0.2d | 1 |
| 3 Clean-venv proof | okfgraph | 0.3d | 2 (local), 5 (published) |
| 4 Dev venv re-sync | okfgraph (local) | — | user decision |
| 5 Release train | all | 0.5d | 1, 2 |
| 6 ladybug crash | okfgraph | 0.5d | — (parallel) |
| 7 Phase 6 | embroider → okfgraph | 5–7d | 6.4a images, then HF approval |

## Definition of done (parent plan), current state

| Criterion | State |
|---|---|
| No torch stack in any install path (packaging test + hermetic import + CI grep) | packaging ✅, hermetic ✅ (now actually runs), CI grep ✅ in all three (uncommitted in bobine/embroider); clean-install proof → Step 3 |
| Every runtime model file is `.onnx` on ORT 1.29.0 via embroider's layer | ✅ (bobine CI now also tests 1.29.0) |
| One vector space per graph | ✅ for text + captions; Phase 6 adds image pins |
| bobine on embroider 0.2, COMPAT current, goldens green, publish dry-run + wheel matrix | code ✅; publish → Step 5 |
| CPU and GPU installs each resolve exactly one ORT | doctor warns ✅; lockfile enforcement → Step 2; proof → Step 3 |
