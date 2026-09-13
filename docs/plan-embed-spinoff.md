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
session policy. The spin-out therefore keeps two *layers* — shared
plumbing vs. Jina text model — as **internal modules of a single crate**,
not separate packages (package surface stays minimal: one crates.io
package, one PyPI wheel — §9.2).

## 2. Target architecture

New standalone repo `okf-embed` (name the user already owns on PyPI).
**One crate, one version, two distributions:**

```
okf-embed-repo/
  src/
    lib.rs        # re-exports; extension-module-gated pyo3 bindings
    providers.rs  # name → dispatch mapping + clone-and-fallback
    probe.rs      # corrected CUDA availability check (OnceLock)
    policy.rs     # SessionPolicy: text_embed() | ort_defaults()
    acquire.rs    # hf-hub blocking fetch (parse_owner_name, tokenizer)
    error.rs      # EmbedError (thiserror, no pyo3)
    jina.rs       # TokenizerHandle + JinaV5 (frozen contract)
    diag.rs       # OrtReport (dylib path, cuda_usable)
  python/            # thin .pyi + README for the wheel
  .github/workflows/ # ci.yml (cargo test + clippy) + release.yml
```

- crates.io (`okf-embed`): the whole crate as an rlib. Pure-Rust
  consumers (bobine) depend on it with **default features** — no pyo3
  in their tree (the `extension-module` feature stays opt-in, exactly
  the pattern both projects already use).
- PyPI (`okf-embed`): maturin wheels built with `features =
  ["extension-module"]` — what OKFgraph installs.

One registry entry per ecosystem. No `-core`/`-sys`/`-bindings`
sprawl.

### 2.1 Plumbing modules (the unified part)

`providers` + `probe` + `policy` + `acquire` + `error` + `diag` — today's
duplicated helpers, extracted once. `providers::apply_providers` and
`probe::cuda_available` replace bobine's `engine.rs` copies (including
the free win: bobine's stale builder probe becomes the corrected
availability check). `policy::SessionPolicy` keeps the per-workload
split explicit — `text_embed()` (the measured Level3/phys-2/1 tuning)
vs. `ort_defaults()` (what bobine uses today; adopted with zero
behavior change). No model weights, no Python in these modules.

`error::EmbedError` (`thiserror`) is mapped to each project's own error
at the boundary (`PyRuntimeError` / `BobineError`). No shared Python
exception — tracebacks stay project-local.

### 2.2 Text-model + bindings (OKFgraph's part, untouched)

`jina.rs` holds `TokenizerHandle` + `JinaV5` (contract frozen — §6);
`lib.rs` holds the PyO3 classes behind the existing
`extension-module` feature. Python surface stays byte-for-byte stable
(module `okf_embed`, same signatures); OKFgraph's `LazyRustEncoder`
needs zero changes.

### 2.3 Why a single crate, not `embed-core` + `okf-embed`

The split was considered and rejected: the only thing it buys is
keeping `pyo3` out of bobine's tree, which the opt-in
`extension-module` feature already achieves, and the only deps the
plumbing shares (`tokenizers`, `hf-hub`, `ndarray`) are **already in
bobine's tree** for TexTeller/OCR — so a split saves bobine zero
dependencies while doubling the registry/package/CI surface. If a
consumer ever appears that needs plumbing without the text-model deps,
revisit then; until that day, modules — not packages — are the
layering mechanism.

## 3. Dependency & versioning rules

1. **One pinned ORT.** `ort 2.0.0-rc.13` + `load-dynamic` in core; both
   consumers inherit the pin via the core dep. A version skew between
   consumers fails at *resolve* time (good — loud, not silent).
2. **Independent semver, conservative pins.** One version in the spin-out
   repo (single crate). Consumers pin `okf-embed = "0.3"` (bobine,
   crates.io, default features) and `okf-embed>=0.3,<0.4` (okfgraph,
   PyPI) — same discipline as today's `ladybug==` / `onnxruntime==`
   pins.
3. **Jina contract is versioned by tests, not hope.** The golden parity
   vectors (≤1e-5 vs. reference) move into the spin-out repo as fixture
   data; any contract change fails CI before it can ship.
4. **No path deps across repos.** `cargo publish` rejects path
   dependencies — and bobine runs `cargo publish --locked`. From the
   moment bobine adopts the crate, `okf-embed` **must be on crates.io**.
   (OKFgraph has no such constraint but gets the same released version
   anyway — both consumers on the same artifact is the point.)
5. **One wheel per Python module, ever.** The `okf_embed` wheel ships
   only from the spin-out repo. Bobine must never vendor or repack its
   `.pyd`. If bobine ever exposes text embedding to Python, it declares
   `okf-embed` as an optional PyPI dependency (`bobine[embed]`), not a
   bundle.

## 4. Release pipeline (new repo)

Mirror what already works in both projects (tag guard, OIDC):

- `ci.yml`: `cargo test --locked` (+ `libpython3-dev` equivalent),
  clippy, Python fast tests (packaging/pinning), maturin build check.
- `release.yml` on `v*`: tag==version guard → single `cargo publish`
  → maturin matrix (linux/win/macOS-arm64 × py3.11–3.13) → PyPI trusted
  publishing.
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

### Phase 1 — Stand up the repo + extract plumbing (2–3 days)
- New repo, single `okf-embed` crate (moved code, clean move per §9.3).
  First refactor: extract `providers`/`probe`/`policy`/`acquire`/`error`/
  `diag` modules out of the Jina code; `jina.rs` + bindings delegate to
  them. Unit tests move with the code.
- Publish `okf-embed 0.3.0` to crates.io (manual first version is
  fine; pipeline takes over after).
- DoD: `cargo test --locked` green; docs show the text/vision policy
  split with the existing benchmark table.

### Phase 2 — Bobine adopts the plumbing (1–2 days)
- `engine.rs`: delete local `apply_providers` + `cuda_available`,
  depend on `okf-embed 0.3` with default features (no pyo3 enters
  bobine's tree; `tokenizers`/`hf-hub`/`ndarray` are already there, so
  the adoption adds **zero new dependencies**). Policy choice:
  `SessionPolicy::ort_defaults()` — **zero behavior change** (bobine's
  sessions keep ORT defaults; the measured text policy is documented
  but not applied to vision models without its own benchmark).
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

**Option (a) — Rust-side** (use the `okf-embed` rlib bobine already
depends on — no new dependency at all):
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
2. **Package surface**: one crates.io package + one PyPI wheel, both
   named `okf-embed` (single crate, internal module layering). The
   `-core` split was rejected: it saves bobine zero dependencies (its
   tree already contains the shared deps) while doubling registry, CI,
   and release-train surface. Revisit only if a consumer appears that
   needs plumbing without the text-model deps.
3. **History**: clean move. (`filter-repo` would have replayed
   `rust/okf-embed/` commits into the new repo to preserve blame;
   rejected — a pointer commit in OKFgraph noting the move origin is
enough, and the new repo starts with readable history.)
4. **Bobine text embedding**: deferred; scope + impact recorded in
   Phase 5. Pre-approved order when revisited: Python-side extra first,
   Rust-native only on concrete need.
