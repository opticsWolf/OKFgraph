# Plan: ONNX-runtime alignment with bobine

## Status

In progress. Phases 1 (cross-platform ORT discovery), 2 (provider
fallback + extension-module feature), 3 (lazy encoder + tokenizer-only
counts), 4 (explicit local model files), and 5 (extension-module feature,
landed with Phase 2) are implemented; Phases 6–7 remain open.

This plan ports the strongest parts of bobine’s ONNX integration into
OKFgraph/`okf-embed` without importing bobine as a required dependency and
without expanding the five-tool MCP surface.

## Goal

Make OKFgraph’s ONNX behavior as robust as bobine’s while keeping
`okf-embed` narrow, deterministic, and agent-surface slim:

- One shared ONNX Runtime binary whenever possible.
- Cross-platform runtime discovery.
- Graceful accelerator fallback.
- Lazy model/session initialization.
- Explicit local model-file support.
- Clear Cargo feature separation for pure-Rust tests versus Python wheels.
- A measured session/threading policy instead of inherited tuning.

## Non-goals

- No new MCP tools.
- No public provider-list schema unless a later phase proves it necessary.
- No replacement of the Jina v5 embedding contract.
- No silent vector-space changes.
- No torch, transformers, optimum, OpenCV, or other heavy runtime dependencies.
- No copying of bobine’s multi-model routing, quantization matrix, OCR batching,
  formula decoding, or document-conversion configuration.
- No GPU-specific CI requirement; GPU behavior must remain optional and
  non-fatal.

## Background and current state

### What bobine already does well

Bobine’s ONNX integration is built around a multi-model lazy engine:

- Same `ort 2.0.0-rc.13` pin as `okf-embed`.
- Dynamic runtime loading through `ORT_DYLIB_PATH`.
- Broad execution-provider features:
  - CUDA
  - ROCm
  - DirectML
  - OpenVINO
  - CoreML
  - TensorRT
  - half
- Base provider list plus per-model overrides.
- Automatic CUDA use for layout and OCR when the loaded runtime registers CUDA.
- CPU pinning for table recognition because SLANet is slower on CUDA.
- Builder clone-and-fallback:
  - Try accelerator providers.
  - Fall back to the pristine CPU-capable builder on registration failure.
- Unknown provider names warn and are skipped.
- Lazy model lifecycle:
  - Engine construction is cheap.
  - TexTeller, layout, OCR, and table models load only when required.
- Explicit separation between:
  - `from_pretrained...`
  - `load_from_paths...`
- Cross-platform Python runtime discovery.
- Windows GPU DLL warming through `onnxruntime.preload_dlls()`.
- Optional Cargo `extension-module` feature:
  - Pure Rust crate by default.
  - Python extension only when requested.

Relevant bobine sources:

- `D:/User/Documents/Rust/bobine/Cargo.toml`
- `D:/User/Documents/Rust/bobine/src/engine.rs`
- `D:/User/Documents/Rust/bobine/src/config.rs`
- `D:/User/Documents/Rust/bobine/src/tex_teller.rs`
- `D:/User/Documents/Rust/bobine/src/rapid_layout.rs`
- `D:/User/Documents/Rust/bobine/src/rapid_ocr.rs`
- `D:/User/Documents/Rust/bobine/src/rapid_table.rs`
- `D:/User/Documents/Rust/bobine/python/bobine/__init__.py`
- `D:/User/Documents/Rust/bobine/pyproject.toml`
- `D:/User/Documents/Rust/bobine/docs/benchmarks.md`
- `D:/User/Documents/Rust/bobine/docs/quickref.md`

### What `okf-embed` already does well

`okf-embed` is narrower and more deterministic:

- Exact Jina v5 pipeline:
  - Task prefix.
  - Tokenization with an 8192-token limit.
  - ONNX forward pass.
  - Last-token pooling.
  - L2 normalization.
  - Matryoshka truncation.
  - Renormalization.
- Robust ONNX contract discovery:
  - Requires `input_ids` and `attention_mask`.
  - Feeds `token_type_ids` only when declared.
  - Prefers `last_hidden_state`, otherwise uses the first output.
- Sequential encoding by design for variable-length documents.
- GIL released during encoding.
- CUDA treated as opportunistic, never fatal.
- `used_cuda` reports the effective device.
- Pure Rust unit tests plus Python parity and end-to-end coverage.

Relevant OKFgraph sources:

- `rust/okf-embed/src/lib.rs`
- `rust/okf-embed/Cargo.toml`
- `rust/okf-embed/pyproject.toml`
- `rust/okf-embed/README.md`
- `okfgraph/components/embedding.py`
- `okfgraph/router.py`
- `tests/test_router.py`
- `tests/test_rust_backend.py`
- `tests/test_rust_e2e.py`
- `tests/test_packaging.py`
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`

### Known gaps in OKFgraph’s ONNX integration

1. `resolve_ort_dylib()` is effectively Windows-only.
   - It hardcodes `onnxruntime.dll`.
   - It has no macOS/Linux candidate search.
   - It does not warm GPU DLLs.
   - It does not expose the resolved runtime for diagnostics.
2. `okf-embed` supports only CUDA as an accelerator concept.
   - No ROCm/DirectML/OpenVINO/CoreML registration path.
   - No generic unknown-provider handling.
3. CUDA availability is probed with `CUDA::default().is_available()`.
   - Bobine instead probes provider registration on a session builder.
   - Registration is the operation that fails against a CPU-only runtime.
4. The text encoder opens eagerly during router construction.
   - Model-free operations still pay model-initialization cost.
   - This is especially costly for cold CLI workflows.
5. There is no explicit local model-file constructor.
   - HF cache directory and revision are supported.
   - Exact model/tokenizer paths are not.
6. The Cargo packaging relies on Maturin to inject `pyo3/extension-module`.
   - Bobine’s explicit optional Cargo feature is clearer.
7. Session/thread tuning was inherited from another project’s Jina runner.
   - It has not been benchmarked against ORT defaults for this workload.

## Invariants

All implementation phases must preserve these invariants:

- Same five MCP tools.
- No new required runtime dependencies.
- No new required deployment services.
- Deterministic embedding numerics.
- No silent fallback between embedding stacks.
- Fail fast at encode time.
- Explicit user configuration always beats automatic discovery.
- CPU-only systems keep working.
- Missing accelerators never become fatal errors.
- Pure Rust tests stay free of network, tokenizer downloads, and ORT dylib requirements.
- Model-dependent tests stay marked and outside the fast CI path.
- Public agent-facing token surface stays slim.

## Phase 0 — Baseline and characterization

Effort: 0.5 day.  
Dependencies: none.

### Tasks

1. Record the exact current behavior for:
   - `ORT_DYLIB_PATH` precedence.
   - Missing runtime behavior.
   - `auto`, `cpu`, and `cuda` device handling.
   - CUDA fallback logging.
   - Eager encoder initialization.
   - Model acquisition order:
     - `model.onnx`.
     - Optional `model.onnx_data`.
     - `tokenizer.json`.
   - Session configuration:
     - Optimization level.
     - Intra-op threads.
     - Inter-op threads.
2. Add or update characterization tests only where behavior is currently implicit.
3. Capture a cold-start baseline for representative operations:
   - `search --rank ppr`.
   - `read --include context`.
   - `diff`.
   - `doctor`.
   - First text encode.
4. Record OS, CPU, RAM, GPU, ORT build, model revision, dimensions, and cache state.

### Acceptance criteria

- Baseline behavior is documented.
- No production code has changed.
- Later phases can prove they did not alter embedding numerics.

## Phase 1 — Cross-platform ORT discovery

Effort: 1–2 days.  
Dependencies: Phase 0.

This is the highest-value, lowest-risk phase.

### Design

Port bobine’s resolver behavior into OKFgraph without making bobine required.

Resolution precedence:

1. Existing `ORT_DYLIB_PATH`.
2. Pip-installed `onnxruntime` or `onnxruntime-gpu`.
3. OS loader path.
4. `None` when no runtime is installed.

Platform candidates:

- Windows:
  - `<package>/capi/onnxruntime.dll`.
- macOS:
  - `<package>/capi/libonnxruntime.dylib`.
- Linux:
  - Versioned `<package>/capi/libonnxruntime.so.*`, preferred.
  - Unversioned `<package>/capi/libonnxruntime.so`, fallback.

Additional behavior:

- On Windows, add the runtime `capi` directory to DLL resolution.
- If the installed Python runtime exposes CUDA, call its DLL preload helper.
- Never raise during discovery merely because no runtime exists.
- Only session creation or encoding may fail for a missing runtime.
- Resolve before importing the native embedding module.
- Expose the resolved path through router/model diagnostics.

### Implementation files

- `okfgraph/components/embedding.py`
- `okfgraph/router.py`
- `tests/test_packaging.py` or a new focused `tests/test_ort.py`
- `rust/okf-embed/README.md`
- `README.md`, only if user-facing runtime behavior changes
- `CHANGELOG.md`

### Test strategy

Separate pure candidate selection from filesystem and import effects.

Test with fakes/mocks:

- Explicit environment variable always wins.
- Correct Windows candidate.
- Correct macOS candidate.
- Correct Linux versioned `.so` preference.
- Missing runtime returns `None` without raising.
- GPU preload helper is best-effort and never breaks import.
- Existing Windows behavior is unchanged.

Do not require:

- A GPU.
- A downloaded model.
- An ORT shared library.

### Acceptance criteria

- Existing Windows resolution behavior is unchanged.
- Linux and macOS use the appropriate runtime candidates.
- Explicit `ORT_DYLIB_PATH` always wins.
- GPU DLL warming cannot break import or CPU operation.
- Resolved runtime is visible in diagnostics.
- Fast tests cover the new logic.

### Risks and mitigations

- Preloading native libraries can have process-global effects.
  - Mitigation: narrow preload to GPU runtimes and guard every call.
- Importing `onnxruntime` for discovery has a cost.
  - Mitigation: do it once, cache the result, and keep it lazy where possible.

## Phase 2 — Provider fallback abstraction

Effort: 2–3 days.  
Dependencies: Phases 0–1.

This phase adopts bobine’s fallback mechanics without adopting its large
provider-configuration surface.

### Design

Add an internal provider helper with behavior equivalent to bobine’s
`apply_providers`:

1. Normalize provider names case-insensitively.
2. Recognize aliases such as:
   - `cpu` and `CPUExecutionProvider`.
   - `cuda` and `CUDAExecutionProvider`.
3. Warn and skip unknown providers.
4. Treat an empty or CPU-only list as “use the incoming builder unchanged.”
5. Clone the session builder before attempting accelerators.
6. Attempt accelerator registration.
7. On failure, warn and return the pristine builder.
8. Never fail model initialization solely because an accelerator is unavailable.

Map the existing simple device surface internally:

- `cpu` → CPU only.
- `cuda` → CUDA first, CPU fallback.
- `auto` → CUDA first when available, otherwise CPU.

Do not initially expose a public provider list through the router, CLI, or MCP.

### Optional Cargo work in this phase

Enable bobine’s broader execution-provider features if compilation and runtime
behavior remain clean:

- `rocm`
- `directml`
- `openvino`
- `coreml`
- `tensorrt`
- `half`

Bobine documents these features as registration-only under dynamic loading.
Verify that claim for `okf-embed` rather than assuming it.

If any feature materially affects the wheel, binary size, supported Python
versions, or CPU behavior, leave it disabled and record the reason.

### CUDA probe improvement

Replace or supplement `CUDA::default().is_available()` with a bobine-style
registration probe:

- Create a session builder.
- Attempt CUDA provider registration.
- Cache the boolean result.
- Treat builder-creation failure as unavailable.
- Do not commit a session during probing.
- Do not download a model during probing.

### Test strategy

Pure Rust tests:

- Provider-name normalization.
- Alias handling.
- Unknown-provider skipping.
- CPU-only short-circuit.
- Explicit override precedence, if introduced.

Python tests:

- Existing CPU behavior is unchanged.
- Requested CUDA without CUDA still falls back.
- Fallback warning remains deterministic and observable.
- No GPU or model download is required.

### Acceptance criteria

- Public device options remain `auto`, `cpu`, and `cuda`.
- Unknown providers cannot crash initialization.
- Accelerator registration failure degrades to CPU.
- Embedding numerics are unchanged.
- No model download is needed for provider unit tests.

### Explicit non-goal for this phase

No per-model provider overrides. `okf-embed` has one embedding session, so
bobine’s layout/OCR/table provider matrix would be unnecessary complexity.

## Phase 3 — Lazy encoder initialization

Effort: 3–5 days.  
Dependencies: Phases 0–2.

This is the most architecturally valuable and riskiest phase.

### Problem

The router currently opens the embedding session eagerly. Consequently,
model-free workflows may still incur model-download and session-startup costs.

Bobine avoids the analogous problem by constructing a cheap engine and loading
each ONNX model only when its capability is actually required.

### Design

Separate three concerns:

1. Embedding configuration.
2. Encoder availability.
3. Actual encoding.

Proposed behavior:

- Router construction remains fail-fast if the `okf-embed` wheel is missing.
- Router construction validates embedding configuration eagerly.
- `JinaV5.open` is deferred until the first operation that truly needs vectors.
- The live encoder is cached after the first successful open.
- Encoder initialization is thread-safe.
- Model-free operations never trigger model acquisition.

A Python-level lazy holder should:

- Store all parameters needed for `JinaV5.open`.
- Hold an initially empty encoder slot.
- Use a lock around initialization.
- Cache initialization failure or retry safely and explicitly.
- Avoid opening the encoder for:
  - Model-free PPR search.
  - Stored-body reads.
  - Chunk/document reconstruction from stored data.
  - Structural diff.
  - Health checks.
  - Directory traversal.

### Important logging constraint

Do not log `used_cuda` during router construction if obtaining that value
would open the session. Either:

- Defer device reporting until first encode, or
- Report requested versus effective device separately:
  - `device_requested`: `auto`, `cpu`, or `cuda`.
  - `device_active`: unknown until initialization, then CPU or CUDA.

### Tokenizer question

`count_tokens` currently uses the Rust tokenizer. Lazy encoder initialization
creates three options:

A. Lazy everything:
   - Simplest.
   - `count_tokens` falls back to `chars/4` until first encode.
   - Risk: token-budget behavior changes before first encode.

B. Eager tokenizer, lazy session:
   - Best cold-start behavior for budgeted reads.
   - More Rust/Python API work.
   - Tokenizer acquisition may still require network access.

C. Lazy everything and always open for token counting:
   - Preserves current token-count behavior.
   - Largely defeats lazy initialization for budgeted workflows.

Recommended starting point:

- Implement option A only if token-budget behavior can be preserved.
- Otherwise, implement the simple fully lazy encoder first and treat tokenizer
  separation as a follow-up.
- Do not silently change budgeted-read truncation behavior.

### Test strategy

Add stub/fake-encoder tests proving:

- Router construction does not call the encoder factory.
- Model-free PPR search does not call the factory.
- Stored reads do not call the factory.
- First encode calls the factory exactly once.
- Concurrent first encodes still create only one live encoder.
- Initialization errors are explicit and deterministic.

Update existing tests that implicitly assume eager initialization.

### Acceptance criteria

- Cold PPR search does not acquire or open a model.
- Cold structural commands do not acquire or open a model.
- First encoding initializes the encoder exactly once.
- Thread-safe initialization is tested.
- Embedding numerics and budget behavior are unchanged after initialization.
- Missing-wheel behavior remains explicit.

### Risks and mitigations

- Deferred errors can surprise users.
  - Mitigation: keep wheel/import validation eager; defer only model/session work.
- Logging may accidentally trigger initialization.
  - Mitigation: audit every access to encoder properties.
- Tests may rely on eager side effects.
  - Mitigation: update tests to assert the new lifecycle explicitly.

## Phase 4 — Explicit local model-file loading

Effort: 1–2 days.  
Dependencies: Phases 0–3.

### Design

Follow bobine’s separation between acquisition and session construction by
adding an explicit-path constructor, conceptually:

- `JinaV5.open(...)`
  - Existing HF acquisition path.
- `JinaV5.open_files(...)`
  - New explicit local-file path.

Explicit inputs should include at least:

- `model_path`
- `tokenizer_path`
- `truncate_dim`
- `device`

Optional follow-up inputs:

- Explicit external-data sidecar path.
- Explicit revision metadata.
- Read-only enforcement for input files.

Validation order:

1. Required files exist.
2. Tokenizer loads.
3. Session contract discovery succeeds.
4. No network access occurs.

The HF path should internally share as much session-construction logic as
possible with the explicit path.

### Router and CLI surface

Recommended minimal exposure:

- Router or embedding configuration accepts optional local paths.
- Advanced CLI/config support may follow.
- No MCP schema change.
- No new MCP tool.
- No change to the default acquisition behavior.

This keeps the agent surface slim while supporting air-gapped and reproducible
deployments.

### Test strategy

- Missing model file fails before network access.
- Missing tokenizer file fails before network access.
- Explicit valid files bypass HF acquisition.
- Invalid ONNX contract still produces a clear error.
- Existing HF behavior is unchanged.

### Acceptance criteria

- Air-gapped loading works with pre-staged files.
- No HF network call occurs on the explicit path.
- Session contract discovery remains mandatory.
- Default behavior is unchanged.

## Phase 5 — Explicit Cargo extension-module feature

Effort: 0.5 day.  
Dependencies: none, but coordinate release timing with other Rust changes.

### Design

Adopt bobine’s clearer Cargo pattern:

```toml
[features]
default = []

extension-module = [
  "dep:pyo3",
  "pyo3/extension-module",
]
```

Make `pyo3` optional.

Update Maturin configuration so wheels continue to build with the extension
feature enabled.

Preserve the existing behavior:

- `cargo test --locked` works without Python linking complications.
- Wheel builds produce the Python extension module.
- No runtime behavior changes.

### Acceptance criteria

- `cargo test --locked` passes.
- Wheel metadata remains correct.
- PyPI packaging tests remain green.
- No embedding behavior changes.

## Phase 6 — Session, threading, and batching benchmark

Effort: 2–4 days.  
Dependencies: Phases 0–3.

Do not change tuning based solely on inheritance. Measure it.

### Benchmark matrix

Compare at least:

- ORT default optimization versus explicit Level 3.
- Default thread settings versus `intra = physical-cores/2, inter = 1`.
- Other reasonable intra/inter combinations.
- Sequential encoding versus padded batch encoding.
- Short, medium, and near-context-limit documents.
- CPU and CUDA, where available.
- Cold session creation versus steady-state encoding.

Record for every run:

- Hardware and OS.
- ORT build and provider.
- Model ID and revision.
- Embedding dimensions.
- Cache state.
- Thread settings.
- Optimization level.
- Batch strategy.
- Wall time.
- Tokens per second.
- Peak memory, if available.
- Numerical equivalence hash or max deviation.

### Constraints

- Do not add permanent heavyweight benchmark dependencies without separate approval.
- Prefer temporary local benchmark scripts.
- Commit only the methodology and results, not noisy timing artifacts.
- Keep model-dependent benchmarks outside fast CI.

### Acceptance criteria

- Current tuning is either confirmed or replaced with measurements.
- Results are recorded in the repository.
- The final policy is documented in `rust/okf-embed/README.md`.
- Embedding outputs remain numerically equivalent across tuning changes.

## Phase 7 — Hardening, documentation, and release

Effort: 1–2 days.  
Dependencies: Phases 1–6.

### Code hardening

- Audit every ORT error path.
- Preserve fail-fast encode behavior.
- Keep user-facing errors actionable.
- Consider an internal typed error taxonomy only if it improves diagnostics.
- Python exception behavior may remain unchanged.

### Documentation updates

Update at least:

- `rust/okf-embed/README.md`
  - Runtime discovery order.
  - Supported platforms.
  - Device and fallback behavior.
  - Lazy lifecycle, if implemented.
  - Explicit local-file loading, if implemented.
  - Final threading policy.
- Main `README.md`
  - Only user-visible behavior changes.
- `CHANGELOG.md`
  - One entry per user-visible behavior change.
- Relevant architecture notes
  - Without attempting the unrelated full `architecture.md` rewrite here.
- Skills
  - Only if CLI/config behavior gains user-facing options.
  - No MCP skill change is expected.

### Test updates

- Keep pure Rust tests network- and dylib-free.
- Keep model tests marked outside fast CI.
- Add regression coverage for:
  - Runtime discovery.
  - Provider fallback.
  - Lazy initialization.
  - Explicit file loading.
  - Version/feature synchronization.
- Update the CI metadata check if wheel metadata changes.

### Versioning and release guidance

- Python-only internal changes follow the normal patch/minor process.
- Any `okf-embed` Rust change that alters wheels requires:
  - Synchronized Python/Cargo versions.
  - Updated `Cargo.lock`.
  - Updated `uv.lock`.
  - Successful Rust tests.
  - Successful wheel builds.
  - Successful PyPI metadata validation.
- A coordinated `okfgraph` + `okf-embed` release is required when both change.
- Do not publish a release solely to update internal comments.

### Final acceptance criteria

- Fast CI passes.
- Rust tests pass.
- Targeted model tests pass where applicable.
- Wheel metadata validation passes.
- No embedding parity regression.
- No MCP surface expansion.
- Documentation matches implemented behavior.

## Suggested execution order

1. Phase 1: cross-platform ORT discovery.
2. Phase 2: provider fallback abstraction.
3. Phase 5: Cargo feature cleanup, bundled with the next Rust release if convenient.
4. Phase 3: lazy encoder initialization.
5. Phase 4: explicit local model files.
6. Phase 6: benchmark and final tuning policy.
7. Phase 7: hardening, documentation, and release.

Phases 1, 2, and 5 can land as safe incremental improvements. Phase 3 should
not start until provider fallback behavior is stable and well tested.

## Effort estimate

| Phase | Estimate |
|---|---|
| Baseline | 0.5 day |
| Cross-platform ORT discovery | 1–2 days |
| Provider fallback | 2–3 days |
| Lazy encoder | 3–5 days |
| Explicit local files | 1–2 days |
| Cargo feature cleanup | 0.5 day |
| Benchmark/policy | 2–4 days |
| Hardening/docs/release | 1–2 days |
| **Total** | **Approximately 10–17 days** |

A realistic calendar with review, CI, release overhead, and unrelated project
work is approximately three weeks.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Native library preload side effects | Process-global DLL behavior | Narrow preload to GPU runtimes; guard every call |
| Deferred initialization errors | Confusing first-encode failures | Keep wheel validation eager; make errors explicit |
| Logging triggers model load | Lazy lifecycle defeated | Audit property access and logging paths |
| Broad EP features affect wheels | Packaging or compatibility regression | Verify wheel metadata and platform behavior |
| Benchmark noise leads to wrong tuning | Performance regression | Use repeated runs, fixed fixtures, and recorded methodology |
| Behavior drift between HF and explicit paths | Reproducibility failure | Share session-construction code and contract checks |
| Overexposing provider options | Agent token-surface bloat | Keep provider lists internal unless proven necessary |

## Rollback strategy

Each phase should land as a separately revertible commit or small commit series.

- Phase 1 can be rolled back to Windows-only discovery.
- Phase 2 can be rolled back to CUDA-only handling.
- Phase 3 can be rolled back to eager initialization.
- Phase 4 can be rolled back by removing the explicit-path constructor.
- Phase 5 can be rolled back to Maturin-injected features.
- Phase 6 should not change behavior without a separate tuning commit.
- No phase should require a database migration.
- No phase should invalidate existing embeddings.

## Open decisions

1. Should a missing `okf-embed` wheel fail at router construction or first encode?
   - Recommended: fail at construction for the wheel, defer only model/session work.
2. Should tokenizer loading remain coupled to session loading?
   - Recommended: investigate separation, but do not change budgeted-read behavior silently.
3. Should provider selection ever become public configuration?
   - Recommended: no, unless a concrete ROCm/DirectML/CoreML/OpenVINO need arises.
4. Should OKFgraph add a GPU-oriented optional dependency?
   - Recommended: no while the core pins one CPU ORT build; retain explicit `ORT_DYLIB_PATH` override.
5. Should TensorRT and half features be enabled?
   - Recommended: only after verifying wheel, compatibility, and performance effects.
6. Should runtime discovery move into a shared helper crate?
   - Recommended: no; duplicate the small resolver with attribution rather than coupling
     `okf-embed` to optional bobine packaging.

## Definition of done

The implementation is complete when:

- Linux, macOS, and Windows resolve an appropriate ORT runtime.
- Explicit runtime configuration always wins.
- Accelerator failures fall back to CPU without failing conversion or encoding setup.
- Model-free workflows do not initialize the embedding model.
- Explicit local model files work without network access.
- Cargo distinguishes pure-Rust and extension-module builds clearly.
- Session/thread settings are measured and documented.
- Embedding numerics remain unchanged.
- Fast CI, Rust tests, packaging checks, and release validation all pass.
- Documentation and changelog describe only implemented behavior.
