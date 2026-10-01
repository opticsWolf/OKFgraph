# Plan: ONNX-only processing chain (okfgraph + bobine + embroider)

**Status:** draft (2026-09-27)
**Scope:** `OKFgraph` (0.6.x), `D:/User/Documents/Rust/bobine` (0.5.x), `D:/User/Documents/Python/embroider` (0.2.x).
**Supersedes:** `docs/plan-embroider.md` (its Phases E0–E4 are folded in below as Phases 2–5; that file stays as history).
**Goal:** one runtime for every model in the chain: ONNX Runtime `1.29.0`, loaded through embroider's session/provider layer. No torch, transformers, optimum or sentence-transformers anywhere in an install path; every vector in one graph comes from one model at one precision.

## Audit baseline (2026-09-27)

| Component | Runtime today | Status |
|---|---|---|
| Text embeddings (embroider `JinaV5`, Jina v5 text-small/nano) | Rust `ort` 2.0.0-rc.13, `load-dynamic` | ONNX ✅ |
| PDF layout (DocLayout-YOLO), OCR det/rec (PP-OCRv4), table (SlanetPlus), formula (TexTeller) in Rust bobine | Rust `ort` via `embroider::{SessionPolicy, apply_providers}` | ONNX ✅ |
| Image embeddings (`okfgraph[omni]`, `_get_omni` / `_encode_image` / `_encode_omni_text`) | `sentence-transformers` → **torch** | ❌ only torch path in the chain |
| `okfgraph/requirements.txt` | lists `torch`, `optimum`, `transformers`, `sentence-transformers` | ❌ stale, pre-embroider |
| Parity tests (`tests/test_parity.py`, `test_rust_backend.py`) | `transformers.AutoTokenizer` (no torch) | test-only, acceptable |
| Legacy Python bobine 0.2.0 (`D:/User/Documents/Python/bobine`) | Python `onnxruntime` + RapidAI | ONNX, but superseded; same PyPI name |

Consistency defects found alongside:

- **D1 — split vector space.** Image vectors always come from `omni-small` FP32 (torch); text vectors follow `--model`/`--precision`. `text-nano` or FP16/INT8 silently mixes spaces in `image_omni_idx`. `enforce_model_pin` / `enforce_precision_pin` (`okfgraph/components/embedding.py:167`) don't cover `omni_model_id`.
- **D2 — stale install hint.** `okfgraph/router.py:173` says `embroider>=0.1.5,<0.2`; `pyproject.toml` requires `>=0.2.0,<0.3`.
- **D3 — embroider version drift.** bobine `Cargo.toml:45` `embroider = "0.1"`, okfgraph on 0.2.
- **D4 — stale `COMPAT.md`** in embroider (okfgraph 0.2.12 / embroider 0.1.3 / floor `>=0.1,<0.2`).
- **D5 — no clean GPU install.** okfgraph hard-requires CPU `onnxruntime==1.29.0`; `bobine[gpu]` installs `onnxruntime-gpu`; bobine itself documents "cpu XOR gpu — never install both".
- **D6 — dead lookup entry.** `_ORT_MODULE_NAMES = ("onnxruntime", "onnxruntime-gpu")` (`embedding.py:218`): the second is not an importable module name.

## Decision needed before Phase 1

**Option A — remove the torch image path (recommended).** Images embed via caption through the text model (the `text` route already exists and is the default). Fixes D1 by construction. Loses search on image pixels; alt-text/caption search is unchanged.

**Option B — keep image-content search, move it to ONNX.** Needs an ONNX export of the Jina v5 omni vision tower, a golden fixture proving it matches the torch reference, and a vision module. Feasibility is now proven for **text-nano graphs** (omni-nano, own dynamic-grid export; spike results in Phase 6). That goes against the plan-embroider non-goal "no vision models", so it lands as its own phase (Phase 6 below), after A.

This plan assumes **A now, B as Phase 6 once A has shipped** (0.7.0 removes the torch path; 0.8.0 re-adds image content on ONNX).

## Phase 0 — guardrails first (~0.5d) ✅ DONE (dev 65f93ae)

Lock the invariant in tests before touching code so the removal can't regress.

- Extend `tests/test_rust_e2e.py:66` (hermetic import check) to also exercise the image-ingest path and `search_images_with_text`: after a full import in a subprocess, `sys.modules` must contain none of `torch`, `transformers`, `optimum`, `sentence_transformers`.
- Add a packaging test: parse `pyproject.toml`, assert no dependency or extra names a banned package.
- Add a CI grep step (all three repos): fail on `import torch` / `sentence_transformers` / `optimum` outside `tests/`, `docs/` and embroider `tools/export/` (dev-only model export scripts, never packaged; Phase 6).
- DoD: new tests fail on current `dev` (the omni extra trips the packaging test), which proves they bite.

## Phase 1 — okfgraph: remove torch (0.7.0, ~1d) ✅ DONE

Removing an extra and an ingest mode is a breaking change → minor bump.

Code:
- `okfgraph/components/embedding.py`: delete `_get_omni`, `_encode_image`, `_encode_omni_text`, the `_omni` slot and the `omni_model_id` constructor arg; keep `_truncate_normalize` only if the text path still uses it.
- `okfgraph/images.py:412` `plan_embedding`: every route → `EmbedRoute.TEXT`. `IngestMode.OMNI` / `OPTIONAL` are **refused**, not silently downgraded: `ValueError("mode 'omni' removed in 0.7.0 (torch path); images embed by caption — use mode 'text'")`.
- `okfgraph/components/image_assets.py:136,336`: drop the OMNI branch and `use_text_model=False`.
- `okfgraph/router.py`: drop `omni_model_id`, `_omni`; fix D2 (`embroider>=0.2,<0.3` in the error message).
- `okfgraph/config.py`, `cli.py` (`--omni-model-id`, `--use-omni`, `mode` choices at `:916-974`), `mcp_server.py`: drop the omni options. Unknown `omni_model_id` in `okfgraph.toml` → warn once, ignore (don't break existing configs).
- D6: `_ORT_MODULE_NAMES = ("onnxruntime",)` with a comment that `onnxruntime-gpu` installs the same module.

Packaging:
- `pyproject.toml`: delete the `omni` extra (`sentence-transformers`, `Pillow`). Keep `Pillow` only if image extraction/hashing still needs it (check `okfgraph/images.py`); if so, move it into core or `pdf`.
- `requirements.txt`: delete it (pyproject is the source of truth), or regenerate it from pyproject if something external consumes it.

Existing graphs:
- Stored image vectors from the omni route are in a different space than the text model. On open, if `ImageAsset` rows with `route='omni'` exist: warn with a count and treat them as stale → re-embed by caption on the next import (the content hash includes the route, so `_content_hash(TEXT, caption)` already differs and forces re-embedding). `okf doctor` reports the count.
- Drop `image_omni_idx` name? No — keep the index name to avoid a schema migration; rename only if a schema bump happens anyway.

Tests:
- Delete the omni cases in `tests/test_gpu_integration.py:347-374`; convert image-ingest tests to caption routing; add the `ValueError` test for `mode=omni`.

Docs:
- `README.md`, `architecture.md` (`:21`, `:183`, `:532-540`, `:1408`), `docs/implementation_status.md` (omni rows, torch 2.12.1 / optimum rows): remove or mark historical. `CHANGELOG.md` 0.7.0: breaking — `omni` extra and `mode=omni|optional` removed; image search is caption-based.

DoD: Phase 0 tests green, full suite green, `pip install okfgraph[pdf]` in a clean venv → `pip list` has no torch/transformers/optimum/sentence-transformers.

## Phase 2 — alignment across the three repos (embroider 0.2.1, ~0.5d) ✅ DONE (embroider 933e606, bobine c619bd7, py-bobine c99fc46)

(Was plan-embroider Phase E0.)

- embroider `COMPAT.md` (D4): `okfgraph 0.7.x / bobine 0.5.x+ / embroider 0.2.x / onnxruntime 1.29.0`; floor pin `>=0.2,<0.3`. Add a line: "no torch anywhere in the chain; image embeddings are caption-based".
- bobine (D3): `Cargo.toml:45` `embroider = "0.1"` → `"0.2"`, `cargo update -p embroider`, commit `Cargo.lock`. No code change: `apply_providers` / `SessionPolicy` / `cuda_available` are backward-compatible in 0.2.
- Verify: `cargo publish --dry-run --locked` (no path deps), `cargo check` default Python-free + `--features extension-module`, `cargo test --locked`; bobine golden conversions byte-identical.
- embroider `CHANGELOG.md`: `Unreleased` → `0.2.1`.
- Legacy Python bobine 0.2.0: add a README banner "superseded by the Rust bobine ≥0.5 (same PyPI name); not maintained" and archive the repo. Don't publish from it again.

## Phase 3 — one ONNX Runtime, CPU or GPU (~1d) ✅ DONE

Fixes D5. All three packages already load ORT dynamically, so the pin only needs to decide which wheel provides the DLL.

- okfgraph: move `onnxruntime==1.29.0` out of core into extras, mirroring bobine: `cpu = ["onnxruntime==1.29.0"]`, `gpu = ["onnxruntime-gpu[cuda,cudnn]==1.29.0"]`. Core without either → `resolve_ort_dylib()` returns `None` and the first encode fails fast with "install okfgraph[cpu] or okfgraph[gpu]".
- bobine: tighten its extras from `>=1.28` to `==1.29.0` so both consumers resolve the same binary; keep the "cpu XOR gpu" note.
- `okf doctor` (and bobine's startup log): report the resolved `ORT_DYLIB_PATH`, the ORT version and whether both `onnxruntime` and `onnxruntime-gpu` distributions are installed (hard warning). Use embroider `diag::OrtReport` where available (see Phase 4).
- DoD: clean-venv installs of `okfgraph[cpu,pdf]` and `okfgraph[gpu,pdf]` each end up with exactly one ORT distribution; `--device cuda` on the GPU install reports `used_cuda=True`.

## Phase 4 — embroider plumbing hardening (~1–2d) ✅ DONE

(Was plan-embroider Phase E1; driven by consumer pain, not speculation.)

- `providers`: keep infallible `apply_providers`; document `apply_providers_with_arena` per-slot guidance (text: arena-off measured at 8× lower RSS, 1.4× time; vision slots keep arena-on unless bobine benchmarks otherwise).
- `probe`: `OnceLock` + panic containment stays; surface `diag::OrtReport` in bobine engine startup logs and okfgraph `doctor` (feeds Phase 3).
- `acquire`: dedupe `parse_owner_name` / tokenizer-only fetch / `ModelSpec` validation where it removes code; leave bobine's working per-model `hf_fetch` (`engine.rs:55`) alone otherwise.
- `error`: keep `anyhow`, map at boundaries (`PyRuntimeError` / `BobineError`); no shared Python exception.
- Session policy stays split on purpose: vision slots `SessionPolicy::ort_defaults()`, text `text_embed()` (Level3, intra=phys/2, inter=1). Add a bobine test asserting the vision slots never get the text policy.
- DoD: pure `cargo test --locked` (no dylib, no network), bobine goldens byte-identical, okfgraph parity ≤1e-5.

## Phase 5 — registry, conformance, release (ongoing)

(Was plan-embroider Phases E2–E4.)

Registry (`acquire::{ModelSpec, Artifact, builtin_models, lookup_model}`), minor-gated:
- New model = static contract (id, native dim, ceiling, precision ladder, per-precision artifacts) + golden fixture + `available_models()` constant. Unknown ids keep the legacy path; the default id never moves.
- `Precision::Int8` stays explicit opt-in (`auto` never picks it); unlisted (model, precision) pairs fall back to fp32, never to unvalidated weights.
- Only `.onnx` artifacts are admissible in the registry; a spec pointing at `.pt`/`.bin`/`.safetensors` is a build-time test failure.
- A contract change means a new minor + re-index notice + regenerated golden fixtures. Never silent.

Conformance:
- Per-(model, precision) golden vectors in embroider `fixtures/`; `okfgraph/tests/test_golden_vectors.py` asserts live (`abs=1e-6`) plus exact offline token counts.
- Keep embroider `README.md` sections current (session/threading policy, runtime discovery, lifecycle, precision, arena, explicit files, stale-DLL pitfall, failure policy, contract, conformance, testing); it is the consumer contract.
- Perf: rerun Level3/phys-2/1 vs defaults only on a workload change; commit the method and table, not timing artifacts; model tests stay out of fast CI.

Release (done, keep green): tag==version guard, Cargo==pyproject lock-step (`scripts/check_packaging.py`), OIDC trusted publishing, maturin matrix linux/win/macOS-arm64 × py3.11–3.13. Add the Phase 0 banned-package grep to all three CIs.

## Phase 6 — ONNX image embeddings: omni-nano, dynamic grid (okfgraph 0.8.0, embroider 0.3.0, ~5–7d)

Re-adds search on image pixels without torch, for **text-nano graphs only**. The design is verified by the spike in `D:/User/Documents/Python/omni_spike` (`README.md`, `results/REPORT.md`, `results/dyn/REPORT.md`, 2026-09-27, RTX 3090, ORT 1.29.0).

### Evidence (spike)

| Finding | Number | Consequence |
|---|---|---|
| omni-nano's text tower **is** text-nano | cos 1.0000 on every query (embroider text-nano vs omni text tower) | Image vectors land in an existing text-nano graph's space. No text re-embed. |
| omni-small (text-small's partner) has no ONNX export | only safetensors/GGUF/MLX on HF | text-small graphs: no image-content route (refuse). Porting the export recipe to omni-small is a separate, unverified task. |
| Community export `onnx-community/…-omni-nano-ONNX@e3bbd1a` is correct but **fixed at a 32×32 grid** (static `[1024, 1536]`, 271 tokens) | matches torch at the same resolution (cos 0.99997), but r@1 **0.824** vs **1.0** at native resolution; stop-sign text image cos 0.40 vs native | Not usable: every image is squashed to 512×512. |
| Own **dynamic-grid** export | torch-free ORT vs torch native: cos ≥ 0.99996; r@1 **1.0**; reproduces the community graph at 32×32 (cos 0.99998) | This is the artifact. |
| fp16 (CUDA) | cos ≥ 0.9998, r@1 1.0 — **only** with RMSNorm + rotary kept fp32 | A naive fp16 conversion overflows (cos down to 0.40). |
| q4f16 (community) | cos 0.954, r@3 drops | Not offered. |
| Latency per image, 256 → 1280 image tokens | fp16 CUDA 0.027 → 0.106 s; fp32 CUDA 0.030 → 0.194 s; fp32 CPU 0.60 → 6.7 s | CPU works but is slow on large images (see batch import note below). |

Limits of the evidence: 16 synthetic images / 17 queries. Gate 6.4 below repeats the proof on real okfgraph images before release.

### Design

**Artifact (the one new model file).** `jina-embeddings-v5-omni-nano-retrieval` exported by us from Jina's torch weights (`b7287f6`) with `torch.onnx.export(dynamo=True)`, opset 18, as one graph (vision tower + merger + EuroBERT text tower + last-token pool + L2 norm; retrieval adapter already merged in the source weights). fp32 1.25 GB, fp16 0.62 GB.

The one-time export needs torch; the runtime never does. Keep the export script plus its manifest (source rev, torch/transformers versions, sha256) in embroider `tools/export/` as a dev-only script outside the package. Exclude that path in the Phase 0 CI grep.

Graph contract (one image per call; `seq = patches/4 + 15`):

| Input | Type/shape | Produced by (host) |
|---|---|---|
| `input_ids`, `attention_mask` | int64 `[1, seq]` | prompt `<\|im_start\|>user\n` + `<image>`×(patches/4) + `<\|im_end\|>\n`, omni tokenizer, **truncation off** |
| `pixel_values` | f32 `[patches, 1536]` | resize → /255 → (x−0.5)/0.5 → Qwen2-VL patchify (temporal ×2) |
| `vision_pos_ids` | int64 `[patches, 2]` | (row, col) per patch, spatial-merge-block order |
| `interp_indices`, `interp_weights` | int64 / f32 `[patches, 4]` | bilinear taps (align_corners) into the learned 48×48 position table |
| → `sentence_embedding` | f32 `[1, 768]` | L2-normalised; Matryoshka-truncate + renormalise like text |

The last three inputs are exactly what transformers' `vision_utils` computes from `grid_thw` with loops and `.tolist()`, which is why the community trace froze them. Moving them to the host is what makes the grid dynamic. They are pure integer and float32 arithmetic; the spike's `grid.py` is bit-identical to transformers on 43,431 sizes and all 3,319 reachable grids.

Resolution contract (part of the model spec; changing any value changes the vectors, which means a new model id):
- `smart_resize`: factor 32, `min_pixels` 262,144 (512²), `max_pixels` 1,310,720 (1,280 image tokens). Aspect ratio is preserved, and ratios above 200:1 are rejected.
- Resize: bicubic via Pillow, RGB. Pillow is the reference, and bit-exact parity was measured against it.

**Where it lives: split at the resize.**
- **embroider 0.3.0**, new `JinaV5Vision` class (Rust + PyO3). This deliberately reverses the "no vision models" non-goal (minor bump + `COMPAT.md` note). Contents:
  - `vision_target_size(h, w) -> (rh, rw)`
  - `JinaV5Vision.open(model_id, precision, device)` / `open_files(...)`
  - `encode_image(rgb: uint8 ndarray [rh, rw, 3], truncate_dim) -> list[float]`, which does normalisation, patchify, host tensors and prompt ids, then runs the graph.

  Sessions, providers, the registry, precision fallback and `diag` stay in one place.
- **okfgraph** does only decode + `convert("RGB")` + Pillow bicubic resize to `vision_target_size`. Porting PIL's resampler to Rust is not worth the parity risk. embroider rejects arrays whose shape isn't a valid target size.

**Registry entry (Phase 5 rules).**
- id `jina-v5-omni-nano-retrieval-vision`
- native dim 768, same Matryoshka ladder as text-nano
- `text_partner = jina-v5-text-nano-retrieval`
- artifacts: fp32 and fp16 `.onnx` + `.onnx.data` + omni `tokenizer.json`, pinned by HF revision + sha256
- `Precision::Auto` picks fp16 on CUDA and fp32 on CPU; no int8/q4.
- fp16 is produced by the recorded recipe:
  - ORT `float16` converter with `keep_io_types`
  - `Pow`/`ReduceMean`/`Sqrt`/`Reciprocal`/`Cos`/`Sin` blocked (kept fp32)
  - the converter's duplicate Cast nodes deduped
- Golden fixtures per precision:
  - host-tensor vectors (sizes and grids from the spike)
  - end-to-end image vectors

**Session policy.** Use the vision-slot policy (not `text_embed()`). On CUDA, set `arena_extend_strategy = kSameAsRequested`. The spike saw +8.4 GB (fp32) and +3.4 GB (fp16) of arena growth across varying shapes with the default. Expose `gpu_mem_limit`.

**okfgraph wiring (0.8.0).**
- New `EmbedRoute.VISION` with route id `vision-onnx`, distinct from the removed `omni`. `IngestMode.OMNI`/`OPTIONAL` come back, mapped to it.
- The content hash covers route + image model id + precision + resolution contract, so a change re-embeds only the image vectors.
- Pins: `MetaText` gains `image_model_id` + `image_precision` alongside the text pins (closes D1). `enforce_model_pin` refuses:
  - an image route on a graph whose text model isn't the vision model's `text_partner` ("image-content search needs a text-nano graph; this graph uses text-small — use mode=text (captions)")
  - a second image model or precision in the same graph

  Image vectors use the graph's `truncate_dim`.
- Stale rows: `route='omni'` vectors from ≤0.6 stay stale (Phase 1 rule). `okf doctor` reports image model, precision and the count per route.
- Batch import on CPU:
  - log the expected cost (about 0.6–6.7 s per image)
  - keep captions as the fallback route when `mode=optional`
  - add no hidden CPU downscaling: a lower `max_pixels` would be a different contract, i.e. a new model id if ever wanted.

### Steps and gates

| Step | Work | Gate (must pass before the next step) |
|---|---|---|
| 6.0 Licence + artifact | **Checked 2026-09-27:** omni-nano(-retrieval) is **CC BY-NC 4.0**, the same as every Jina v5 model already in the chain (text-small, text-nano, omni-small, the community ONNX, and the existing `opticsWolf/…-text-small-retrieval-onnx-fp16` mirror). Adapting and redistributing is allowed non-commercially with attribution, the licence notice and a statement of changes, so hosting is fine on the same terms as the existing mirror: `license: cc-by-nc-4.0`, `base_model: jinaai/jina-embeddings-v5-omni-nano-retrieval`, a changes section (ONNX export, grid inputs moved to host, fp16 with fp32 norms), and a commercial-use-needs-Jina-licence line. Publish fp32 + fp16 + tokenizer + manifest at a pinned commit. Separately (applies today, not only Phase 6): okfgraph/embroider code is MIT/Apache but the weights they fetch are NC, and no README/COMPAT says so. Add a "Model licences" section. | Model card as above; the sha256 of the downloaded files equals the manifest; the model-licence note is in the okfgraph README + embroider COMPAT. |
| 6.1 embroider `JinaV5Vision` | Port `grid.py` host math to Rust; tokenizer with truncation off; registry entry; fixtures from the spike (`test_grid` cases → JSON). | Rust host tensors bit-identical to fixtures; end-to-end cos ≥ 0.9999 (fp32) / 0.999 (fp16) vs golden; pure `cargo test --locked`. |
| 6.2 okfgraph wiring | Route, pins, doctor, CLI/MCP modes; Pillow in core or `pdf` (Phase 1 check). | Hermetic import test (Phase 0) covers the image path: no torch. Refusal tests for a text-small graph and for mixed image precision. |
| 6.3 Parity in-situ | okfgraph ingest of the spike corpus vs torch native vectors. | cos ≥ 0.9999 fp32, ≥ 0.999 fp16; r@1 = torch. |
| 6.4 Real-image proof | ≥ 100 real figures/screenshots/photos from okfgraph PDFs with human-written queries; compare image-content vs caption routes vs torch native. | r@1 within 0.05 of torch native; image-content beats captions on the question set it's meant for (text-in-image, charts). Otherwise ship nothing and keep captions. |
| 6.5 Release | embroider 0.3.0 → okfgraph 0.8.0, CHANGELOG, COMPAT, docs (image search needs text-nano). | Phase 5 release checks green. |

Out of scope: omni-small (no export; the recipe may port, but it is unverified), batched image inference (one image per call is 0.03–0.1 s on GPU), audio/video, q4 variants.

## Definition of done

- No torch/transformers/optimum/sentence-transformers in any install path of okfgraph, bobine or embroider (packaging test + hermetic import test + CI grep).
- Every model file loaded at runtime is `.onnx`, run by ORT 1.29.0 through embroider's policy/provider layer.
- One vector space per graph, enforced by the model + precision pins.
- bobine on embroider 0.2, `COMPAT.md` current, goldens green in all repos, `cargo publish --dry-run` + wheel matrix green, no vector change inside a minor.
- CPU and GPU installs each resolve exactly one ORT distribution.

## Order and effort

| Phase | Repo(s) | Effort | Depends on |
|---|---|---|---|
| 0 Guardrails | okfgraph (+CI in all) | 0.5d | — |
| 1 Remove torch path | okfgraph → 0.7.0 | 1d | 0, decision A |
| 2 Alignment | embroider 0.2.1, bobine | 0.5d | — (parallel with 1) |
| 3 One ORT, CPU/GPU | okfgraph, bobine | 1d | 1, 2 |
| 4 Plumbing hardening | embroider, bobine | 1–2d | 2 |
| 5 Registry/conformance/release | all | ongoing | 4 |
| 6 ONNX vision (omni-nano, dynamic grid) | embroider 0.3.0 → okfgraph 0.8.0 | 5–7d | 1, 3, 5; gate 6.4 on real images |
