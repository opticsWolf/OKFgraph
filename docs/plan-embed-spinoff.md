# Plan: spin out the embedder into a unified module

Design doc — no code changed. Goal: extract OKFgraph's `rust/okf-embed`
into a standalone project that serves **both** OKFgraph and bobine,
without breaking either release pipeline or changing any vector space.

## 1. What each side actually has today

| Concern | `okf-embed` (in OKFgraph) | bobine 0.5.9 |
|---|---|---|
| Text embedding | Jina v5 (`JinaV5`, `JinaTokenizer`, last-token pool, Matryoshka 32–1024) | **none** — pipeline.rs explicitly touches no embedding model |
| ONNX sessions | 1 text session, `Level3` + intra=phys/2 + inter=1 (measured) | 4 vision sessions (layout/OCR det+rec/table/TexTeller), **ORT defaults** (no thread/opt tuning anywhere) |
| Provider fallback | `apply_providers` clone-and-fallback | near-identical `apply_providers` (duplicated) |
| CUDA probe | EP availability check (builder probe tried, reverted — too lax in ort 2.0.0-rc.13) | **still the builder probe** (`engine.rs:70`) — i.e. bobine currently reports CUDA on CPU-only boxes |
| HF acquisition | `hf-hub` blocking, `parse_owner_name`, tokenizer fetch | own `hf_fetch` per model (same pattern, duplicated) |
| ORT pin | `ort 2.0.0-rc.13`, `load-dynamic` | **same pin, same features** (already aligned) |
| Python surface | `okf_embed.JinaV5 / JinaTokenizer` (maturin, `extension-module` feature) | `bobine._native` (maturin, `extension-module` feature) |
| Releases | wheels built from OKFgraph tag, `okf-embed` 0.2.0 on PyPI | `cargo publish` (crates.io) + maturin wheels (PyPI), tag-guarded `v*` |
| Consumers | `okfgraph` via `uv.sources` path dep + `okf-embed>=0.1` | — |

Key insight: **there is no embedding duplication to merge** — bobine
doesn't embed text. What *is* duplicated (and already diverging) is the
**ONNX plumbing**: provider mapping, fallback, CUDA probing, HF fetch,
session policy. The spin-out therefore has two layers:

- **Layer A — shared plumbing crate.** The true "unified module serving
  both". Kills the duplication and fixes bobine's stale probe for free.
- **Layer B — standalone text-embedding project.** `okf-embed` gets its
  own repo + releases; OKFgraph switches path-dep → version pin. Bobine
  adopts Layer A now; text embedding in bobine (if ever) becomes an
  *optional* consumer later, not a migration requirement.

## 2. Target architecture

New standalone repo (recommended: `okf-embed`, same name the user
already owns on PyPI). Two crates in one repo, one version:

```
okf-embed-repo/
  crates/
    embed-core/      # pure Rust, no pyo3. Publishes to crates.io.
    okf-embed/       # PyO3 bindings only (extension-module feature).
                     # Publishes to crates.io (lib) + PyPI (wheels).
  python/            # thin .pyi + README for the wheel
  .github/workflows/ # ci.yml (cargo test + clippy) + release.yml
```

### 2.1 `embed-core` (new, Layer A)

Pure-Rust helpers both projects build on. No model weights, no Python:

```rust
// providers: name → dispatch mapping + clone-and-fallback application
pub fn map_provider(name: &str) -> Option<ExecutionProviderDispatch>; // None = cpu-implicit/unknown
pub fn apply_providers(builder: SessionBuilder, providers: &[impl AsRef<str>])
    -> ort::Result<SessionBuilder>;

// probing: the corrected availability check, OnceLock-cached
pub fn cuda_available() -> bool;

// session policy: explicit, per-workload — NOT one hardcoded tuning.
// Text (measured): Level3, intra=phys/2, inter=1.
// Vision (bobine today): ORT defaults. Policy is data, chosen by caller.
pub struct SessionPolicy { pub opt_level: GraphOptimizationLevel,
                           pub intra_threads: Option<usize>, // None = ORT default
                           pub inter_threads: Option<usize> }
impl SessionPolicy {
    pub fn text_embed() -> Self;   // the measured policy, with its benchmark table in docs
    pub fn ort_defaults() -> Self; // what bobine uses today — adopt without behavior change
}
pub fn build_session(onnx_path: &Path, policy: &SessionPolicy,
                     providers: &[impl AsRef<str>]) -> ort::Result<Session>;

// acquisition: the hf-hub blocking pattern both sides reimplemented
pub fn hf_fetch(cache_dir: &Path, repo: (&str, &str), filename: &str) -> Result<PathBuf>;

// observation (shared vocabulary for both projects' logs/diagnostics)
pub struct OrtReport { pub dylib_path: Option<PathBuf>, pub cuda_usable: bool }
pub fn report() -> OrtReport;
```

Deliberately **excluded** from core: image preprocessing (`image`,
`ndarray` — bobine's vision concern), tokenizers (only needed by text
models; keep `tokenizers` + `hf-hub` as core deps anyway — they're small
and both text consumers need them — *or* gate behind a `text` feature;
decision: plain deps, the crate is still tiny vs. any model).

Error type: core defines `pub enum EmbedError` (`thiserror`, no pyo3);
each project's bindings map it to their own error (`PyRuntimeError` /
`BobineError`) at the boundary. No shared Python exception — keeps both
tracebacks project-local.

### 2.2 `okf-embed` crate (moved, Layer B)

Today's `rust/okf-embed/src/lib.rs`, slimmed to: `TokenizerHandle`,
`JinaV5` (contract: prefix → tokenize@8192 → forward → last-token →
L2 → Matryoshka-truncate → renorm — **frozen**), `open`/`open_files`,
`encode_one`/`encode_many`, PyO3 classes. Everything else (providers,
probe, policy, fetch) delegates to `embed-core`. The 21 pure unit tests
move with it; session/model tests stay network/dylib-gated as today.

Python surface is **byte-for-byte stable**: module `okf_embed`, classes
`JinaV5` / `JinaTokenizer`, same signatures. OKFgraph's
`LazyRustEncoder` needs zero changes.

### 2.3 Why two crates, not one

Bobine publishes a pure-Rust crate to crates.io and must stay
`pyo3`-free by default (it already does this via its own
`extension-module` feature — same pattern). A single combined crate
would force `pyo3` (optional or not) into bobine's dependency tree for
plumbing it uses from Rust. `embed-core` (no pyo3, ever) is adoptable
from bobine's `engine.rs` with zero Python involvement.

## 3. Dependency & versioning rules

1. **One pinned ORT.** `ort 2.0.0-rc.13` + `load-dynamic` in core; both
   consumers inherit the pin via the core dep. A version skew between
   consumers fails at *resolve* time (good — loud, not silent).
2. **Independent semver, conservative pins.** Core and bindings share one
   version in the spin-out repo (they release together). Consumers pin
   `embed-core = "0.1"` (bobine, crates.io) and `okf-embed>=0.3,<0.4`
   (okfgraph, PyPI) — same discipline as today's `ladybug==` /
   `onnxruntime==` pins.
3. **Jina contract is versioned by tests, not hope.** The golden parity
   vectors (≤1e-5 vs. reference) move into the spin-out repo as fixture
   data; any contract change fails CI before it can ship.
4. **No path deps across repos.** `cargo publish` rejects path
   dependencies — and bobine runs `cargo publish --locked`. From the
   moment bobine adopts core, core **must be on crates.io**. (OKFgraph
   has no such constraint but gets the same released version anyway —
   both consumers on the same artifact is the point.)
5. **One wheel per Python module, ever.** The `okf_embed` wheel ships
   only from the spin-out repo. Bobine must never vendor or repack its
   `.pyd`. If bobine ever exposes text embedding to Python, it declares
   `okf-embed` as an optional PyPI dependency (`bobine[embed]`), not a
   bundle.

## 4. Release pipeline (new repo)

Mirror what already works in both projects (tag guard, OIDC):

- `ci.yml`: `cargo test --locked` (+ `libpython3-dev` equivalent),
  clippy, Python fast tests (packaging/pinning), maturin build check.
- `release.yml` on `v*`: tag==version guard → `cargo publish`
  (`embed-core` first, then `okf-embed` lib) → maturin matrix
  (linux/win/macOS-arm64 × py3.11–3.13) → PyPI trusted publishing.
- First-time setup lessons carried over: create PyPI project + pending
  publisher *before* the first tag (OIDC can't create projects), declare
  `readme` in both manifests, keep Python/Cargo versions in lock-step
  (existing `test_packaging.py` pattern moves to the new repo).

## 5. Migration phases
### Phase 0 — Audit & freeze (0.5 day)
- Inventory every cross-boundary item: the 9 `pub fn`s in
  `okf-embed/src/lib.rs`, `resolve_ort_dylib` + `LazyRustEncoder` call
  sites in OKFgraph, `apply_providers`/`cuda_available`/`hf_fetch` call
  sites in bobine.
- Freeze the Jina contract doc (prefixes, 8192, pooling, Matryoshka
  range, default dim) as `CONTRACT.md` in the new repo — quoted from
  current code, not rewritten.
- DoD: inventory checked in; both projects' suites green as baseline.

### Phase 1 — Stand up the repo + `embed-core` (2–3 days)
- New repo, crates `embed-core` + `okf-embed` (moved code, history
  preserved via `git subtree split`/`filter-repo` if worth it, else a
  clean move with a pointer commit — prefer history preservation).
- Core starts as the extracted plumbing: providers, probe (corrected),
  policy, fetch, report, error type. Unit tests move with the code.
- Publish `embed-core 0.1.0` to crates.io (manual first version is
  fine; pipeline takes over after).
- DoD: `cargo test --locked` green; docs show the text/vision policy
  split with the existing benchmark table.

### Phase 2 — Bobine adopts core (1–2 days)
- `engine.rs`: delete local `apply_providers` + `cuda_available`,
  depend on `embed-core 0.1`. Policy choice: `SessionPolicy::ort_defaults()`
  — **zero behavior change** (bobine's sessions keep ORT defaults; the
  measured text policy is documented but not applied to vision models
  without its own benchmark).
- Side effect (the free win): bobine's CUDA probe becomes the corrected
  one; its `used_cuda`-style reporting stops lying on CPU boxes.
- DoD: bobine suite green, `cargo publish --dry-run` passes (no path
  deps), conversion outputs byte-identical on golden fixtures.

### Phase 3 — OKFgraph switches to released wheels (1 day)
- `pyproject.toml`: drop `[tool.uv.sources]` path dep →
  `okf-embed>=0.3` (or whatever the first spun-out release is);
  `rust/okf-embed/` deleted from OKFgraph; `uv.lock` re-resolved.
- Rust-side tests (`test_rust_backend/e2e`) now run against the
  released wheel — unchanged, they already treat it as a binary.
- CI: remove the maturin rebuild job from OKFgraph's release workflow
  (it no longer builds embed wheels); keep the `okf-embed>=` floor
  check in the fast suite.
- DoD: full non-slow suite green on a released wheel; `cargo test` job
  removed from OKFgraph CI (nothing Rust left to test there).

### Phase 4 — Cross-project conformance (1 day)
- Golden parity vectors live in the spin-out repo; OKFgraph's parity
  test reads them (submodule or vendored fixture — prefer vendored,
  deterministic, no network).
- A `COMPAT.md` matrix: okfgraph version × okf-embed version ×
  onnxruntime version, updated per release. The single-pinned-ORT rule
  (§3.1) keeps this a 1×1×1 table in practice.
- DoD: spinning a new embed release without updating OKFgraph is a
  *supported* state (pins allow it) and CI proves the floor version
  still passes.

### Phase 5 — Bobine text embedding: DEFERRED (scoped, not scheduled)

Decision: do not build. Scope below exists so the work can be picked up
without re-analysis when a concrete consumer appears. Revisit triggers:
a downstream pipeline wanting vectors at conversion time (e.g. an
okf-ingest-style flow embedding bobine's markdown), or Python users
asking for one-pip-install convert+embed.

**Critical compatibility note for either option:** vectors are only
interchangeable with OKFgraph's index if the *chunking policy* also
matches (mordant GFM chunking, same overlap/prefix rules). Embedding
with a different chunker produces valid vectors that silently misalign
with okfgraph expectations. Any future option MUST either reuse the
chunker contract or document the resulting index as a separate vector
space. The Jina contract itself (prefixes, pooling, Matryoshka range)
comes free via the shared crate — chunking does not.

**Option (a) — Rust-side** (bobine depends on the `okf-embed` rlib
behind a Cargo feature):
- Work: new `embed` feature + `src/embed.rs` (lazy session via OnceLock,
  provider config from `ConverterConfig`, error mapping into
  `BobineError`); a Rust text chunker matching the mordant contract
  (new code — bobine has none); pure unit tests + model-gated
  integration tests; optional `py_bindings` exposure.
- Impact: new public API (semver minor); `cargo publish` unaffected
  (crates.io dep); dependency tree barely grows (`tokenizers`/`hf-hub`
  already bobine deps); ~0.4 GB first-use model download; cold-start
  session cost on first embed; golden-fixture output grows.
- Effort: ~2–4 days incl. tests/docs.

**Option (b) — Python-side** (`bobine[embed]` extra → `okf-embed` wheel):
- Work: `pyproject` optional-dependencies entry + floor pin; lazy
  `embed_texts()` in `python/bobine/__init__.py` importing `okf_embed`
  on first use (mirrors OKFgraph's lazy pattern); clear error when the
  extra is missing; one model-gated test; docs.
- Impact: zero Rust change, zero crates.io impact, no new required
  deps, `import bobine` stays cold, wheel size unchanged. Only new
  coupling: the floor pin on `okf-embed`, managed like anydep.
- Effort: ~0.5–1 day.

Pre-approved order when revisited: **(b) first** (cheap, reversible),
**(a) only if** a Rust-native consumer needs embeddings without Python.
Either way: no repackaging of the wheel, no new top-level module (§3.5).

## 6. What does NOT change

- **No vector-space change.** Same bytes in → same vectors out, pinned
  by golden fixtures. Any tuning delta (cf. the Level1-vs-Level3 bit
  divergence found in Phase 6) is a major-version event with re-embed
  guidance, never silent.
- **No OKFgraph surface change.** 5 MCP tools, CLI verbs, skills, and
  `LazyRustEncoder` semantics untouched. The spin-out is invisible to
  agents (same doctrine as bobine behind `DocumentConverter`).
- **No new required dependencies** for either consumer. Core's Rust
  deps (`ort`, `tokenizers`, `hf-hub`, `ndarray`, `thiserror`) are a
  subset of what both projects already ship.
- **No ORT upgrade.** Stays on `2.0.0-rc.13` / `onnxruntime==1.29.0`
  until a measured reason appears (the alignment plan's standing rule).

## 7. Risks

| Risk | Mitigation |
|---|---|
| `cargo publish` blocked by path deps | Core hits crates.io in Phase 1, before any consumer adopts it |
| OIDC can't create the PyPI project | Manual first upload + pending publisher pre-tag (documented lesson) |
| Two release trains drift (embed 0.x vs consumers) | Conservative pins + floor-version CI + COMPAT matrix; drift is loud, never silent |
| Vision sessions regress under shared policy | Bobine adopts `ort_defaults()` — policy is explicit data, not a forced default; text policy never leaks into vision without its own benchmark |
| Wheel/module conflict (`okf_embed` shipped twice) | §3.5: single publisher, optional-dep-only consumption |
| History loss on move | `filter-repo`/`subtree split` preferred; fallback is a pointer commit — decided in Phase 0, not mid-migration |

## 8. Effort & order

~1 week total (0.5 + 2–3 + 1–2 + 1 + 1 days), suggested order 0 → 1 →
2 → 3 → 4, Phase 5 deferred indefinitely. Rollback at any phase is
"revert the dep line": both consumers keep working pinned artifacts at
every step, and no phase changes vectors.

## 9. Decisions (recorded)

1. **Repo home**: new standalone `okf-embed` repo. Both consumers equal;
   independent release trains.
2. **Crate split**: `embed-core` + `okf-embed`. Publishing split:
   `embed-core` → crates.io only (pure Rust lib, bobine's Rust dep);
   `okf-embed` → crates.io (rlib with the PyO3 bindings) **and** PyPI
   (compiled maturin wheels — what OKFgraph installs). crates.io
   ships source for Rust builds; PyPI ships binaries for Python
   installs — they are two distributions of the same crate, not two
   crates. `embed-core` never touches PyPI (nothing Python imports it
   directly).
3. **History**: clean move. (`filter-repo` would have replayed
   `rust/okf-embed/` commits into the new repo to preserve blame;
   rejected — a pointer commit in OKFgraph noting the move origin is
enough, and the new repo starts with readable history.)
4. **Bobine text embedding**: deferred; scope + impact recorded in
   Phase 5. Pre-approved order when revisited: Python-side extra first,
   Rust-native only on concrete need.
