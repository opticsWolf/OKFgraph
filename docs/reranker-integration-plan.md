# okfgraph reranker integration — implementation plan

**Status:** proposed, unimplemented. Target: okfgraph 0.11.0 (minor — new
opt-in search stage, no behavior change when off). Scheduled in
`docs/issue-solving-implementation-plan.md` §10.1, which lists the
changes below against the 0.11.0 draft.
**Depends on:** ~~`embroider>=0.4,<0.5`~~ **met (2026-11-01):** `main`
floors `embroider>=0.12,<0.13`, and 0.12.0 ships `embroider.Reranker`
plus both cross-encoders as revision-pinned registry entries, each with a
`q8` tier (embroider 0.11.0). Phase 0 needs no pin bump.
**Sources:** embroider `docs/rerank-typed-decision-plan.md` (contract,
verification numbers, rejected precisions); this repo's
`okfgraph/components/search.py` (`search_chunks`, `search_hybrid`,
`rerank_with_hub_score`), `okfgraph/ops/query.py` (`QueryOps.search`
dispatch + HONOURS), `okfgraph/components/embedding.py`
(`LazyRustEncoder`, pin regime), `okfgraph/router.py` (session factories),
`okfgraph/settings.py`, `okfgraph/mcp_server.py`.

## 0. Background

Two sides already exist; this plan joins them.

**embroider 0.4.x ships the engine.** `Reranker::open/open_files`,
`score_one/score_many` (one padded session run per batch call),
`score_long` (sliding windows + MaxP for documents past the pair cap),
`count_tokens` (truncation-free, shared helper with JinaV5), registry
entries for the twin spikes (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
English; `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, multilingual;
both Apache-2.0, fp32, golden-pinned ≤1e-6 on the ORT 1.29 CPU runtime,
torch parity 2.9e-6/7.7e-6). Measured on this hardware: 1–3 ms/pair CPU,
CUDA ~1.5x at batch via `score_batch`, fp16 rejected (slower + drift),
int8 gated on a wider harness, Qwen3-seq-cls rejected (100x CPU cost).
Python surface: `embroider.Reranker` with `open/open_files/score/
score_batch/score_long/count_tokens`. Nothing further is needed
embroider-side for the integration below.

**okfgraph has the result surfaces and the patterns.**

- `search_chunks` (chunk-level RRF) and `search_hybrid` (concept-level
  RRF) return `List[Dict]`; `rerank_with_hub_score` establishes the
  post-retrieval contract: take result dicts, add a score key
  (`final_score`), sort descending.
- `QueryOps.search` dispatches by strict paths with an `_HONOURS` map —
  a param a path ignores raises `BAD_VALUE`, never silently drops.
- `LazyRustEncoder` establishes session ownership: lazy open,
  thread-safe, fail-fast-once, `session_factory` injected in `router.py`.
- Model/precision Meta pins guard *stored vector spaces*
  (`MODEL_META_KEY`, `IMAGE_MODEL_META_KEY`, …).

## 1. Decisions (locked before implementation)

1. **Caller parameter, not a new tool.** Rerank is a stage
   (retrieve-wide → reorder top-N), reached through `rerank: bool`
   (chunks path, beside `hub_rerank`) and `rank='cross'` (concepts
   path, beside `'hub'`/`'ppr'`). Precedence on chunks:
   retrieve → hub-blend → cross-encoder reorders top-N. Model and
   threshold selection stay server-side (`rerank.*` settings); the
   caller asks *whether*, the server decides *how*. (A standalone
   `rerank_pairs` tool for non-graph content is out of scope —
   see §7.)
2. **Separate lazy session.** New `LazyReranker` mirroring
   `LazyRustEncoder` (lazy open, lock, fail-fast-once,
   `session_factory` in `router.py`). Not bolted onto the embedding
   engine: different model, different lifecycle, and cold paths stay
   cold (no rerank weights downloaded unless a rerank path is used).
3. **No pin changes.** The reranker writes no vectors, so no Meta
   pin keys. Provenance (rerank model id + revision) goes in call
   metadata/logs, not Meta.
4. **Off by default.** `rerank=False`, `rank` default unchanged.
   Existing latency, downloads, and result order are untouched
   unless requested.
5. **Retrieve-then-rerank, never rerank-then-retrieve.** The
   cross-encoder reorders the RRF top-N (candidate cap setting,
   default 30); it never sees the full corpus. Logits are not
   probabilities and are never averaged with RRF/hub scales —
   reorder by cross-encoder rank, optionally RRF-fuse the two orders.
6. **Thresholds are earned on okfgraph labels.** No threshold ships
   in this plan (Phase 3 builds the harness; promotion needs a
   measured win on held-out okfgraph labels, error at equal review
   coverage — the befailproof discipline).

## 2. Engine method

`SearchEngine.rerank_with_cross_encoder(chunk_results, query, *, top_n,
model=None) -> List[Dict]`, beside `rerank_with_hub_score`:

- Input: result dicts from either level. **Full text required:**
  `search_chunks` truncates `chunk_text` to 500 chars for presentation,
  so the method re-fetches full `ch.chunk_text` in one bulk query
  (same Stage-4 pattern) keyed by `chunk_id`. Concept-level hits use
  `score_long` per concept (bodies exceed the cap); chunk-level hits
  use one `score_batch` over the candidate list (query + title prefix
  + chunk text — see §3), then map back.
- Title/heading prefix: pair text is
  `f"{parent_title}: {chunk_text}"` (parent_title already rides the
  result dicts; ingest builds `heading_context` the same way for
  embeddings). Keeps chunks self-contained for the cross-encoder.
- Output: same mutate-and-sort contract — adds `cross_score`
  (native logit) and re-sorts descending; keeps `rrf_score`/`hub_score`
  keys intact for fusion and debugging.
- Long-document rule: chunk text past the pair cap goes through
  `score_long` (window default = session `max_len`, stride default);
  `count_tokens` is the rerank tokenizer's, not Jina's (§3).

## 3. Text-supply notes (the two real gotchas)

1. **The 500-char cut.** Forgetting it silently scores truncated
   text. The bulk re-fetch is load-bearing — pin it with a test that
   feeds a >500-char chunk and asserts the scored text is whole
   (e.g. a planted tail marker outranks its head-only variant).
2. **Tokenization mismatch.** Ingest caps chunks at ~512 *Jina*
   tokens; the rerank tokenizers (WordPiece/Unigram) count
   differently, so the longest chunks lose a few tail tokens under
   OnlySecond. Measure the overflow rate on the target graph during
   Phase 1: if material, lower `chunk_size` or pre-truncate with the
   rerank tokenizer — do NOT silently accept unknown truncation.

## 4. Settings, dispatch, surfaces

- **Settings** (`settings.py`, validated like `embedding.*`):
  `rerank.model_id` (default MiniLM-L-6; mMARCO for multilingual
  graphs), `rerank.device` (`auto`), `rerank.precision` (`auto`),
  `rerank.top_n` (candidates rescored, default 30),
  `rerank.window`/`rerank.stride` (`None` = engine defaults).
- **Dispatch** (`ops/query.py`): `search(..., rerank: bool = False)`
  honoured by `chunks:plain` and `concepts:none` (and composable with
  `hub_rerank`/`rank='hub'` as pre-filter); `rank` gains `'cross'`
  for concepts. HONOURS entries + BAD_VALUE coverage for the rest.
- **MCP** (`mcp_server.py`): `rerank: bool = False` on the `search`
  tool with the same path rules in the description; result dicts
  gain `cross_score` (documented as native logit, not probability).
- **CLI**: render `cross_score` where scores render; no new command.
- **Dependency**: none to add. `embroider>=0.12,<0.13` already carries
  the contract. `rerank.precision` takes `auto | fp32 | q8` (`auto` →
  fp32; embroider never resolves `auto` to q8).

## 5. Work plan

### Phase 0 — dependency + lazy session + settings (0.5 day)

(No floor bump: already on embroider 0.12.) Add `LazyReranker` + `rerank.*` settings group
(with validation bounds); wire `session_factory` in `router.py`
following the vision block. No behavior change: nothing calls it yet.
Tests: settings validation bounds, lazy-never-opens (no network in
unit tests — factories uncalled), fail-fast-once cached error.

### Phase 1 — engine method + fixtures (1–1.5 days)

Implement `rerank_with_cross_encoder` (§2) against a stubbed scorer
first (deterministic fake via `session_factory` injection — no
weights in unit tests), then live against the pinned spikes. Add a
frozen fixture: fixed query + 8–10 fixed chunk dicts (short, long,
>500-char, multilingual, adversarial-tail) with expected ordering
(and per-model expected scores recorded, not asserted to 1e-6 —
okfgraph pins *order*, embroider pins *values*). Measure the §3.2
overflow rate on the dev graph here.

### Phase 2 — dispatch + MCP + CLI (0.5–1 day)

`QueryOps.search` params + HONOURS + BAD_VALUE tests;
MCP `search` param + descriptions; CLI render. Full matrix: every
path × rerank on/off asserting honoured/ignored correctly.

### Phase 3 — eval harness on okfgraph labels (1–2 days, gates promotion)

Held-out query→expected-ordering set from the target graph's own
content and language mix; report per-query wins/losses vs RRF-only
and RRF+hub, latency at top_n, and the multilingual slice split
(weak slices hide in averages). No thresholds ship; the harness
records the data promotion needs. mMARCO-vs-MiniLM choice for the
default comes out of this phase, not out of priors.

### Phase 4 — release (0.5 day)

CHANGELOG, compat-matrix row (embroider 0.12.x floor), docs touch-up
(`architecture.md` search-modes line). Opt-in only — no migration,
no reindex, no pin changes.

## 6. Test matrix

| Layer | What | Weights? |
|---|---|---|
| Settings validation | bounds, defaults off | no |
| Lazy session | never-opens cold, fail-fast-once, thread-safety | no |
| Engine vs stub | reorder contract, key preservation, top_n cap, full-text re-fetch (>500-char probe) | no |
| Engine live | fixture ordering + recorded scores, both spikes | yes (env-gated, like embroider's e2e) |
| Dispatch | HONOURS/BAD_VALUE matrix, MCP passthrough | no |
| Eval | held-out ordering win/loss, latency, language slices | yes |

## 7. Out of scope (explicitly not this plan)

- Production-reranker bakeoff (bge-v2-m3, jina-v2-multilingual) —
  decided by Phase 3 numbers, separate plan.
- Qwen3-seq-cls (rejected: 100x CPU cost, no CUDA relief — see
  embroider plan), bf16 (dead), fp16 (dead). The int8 gate was passed
  upstream as the weight-only `q8` tier (embroider 0.11.0); it is an
  allowed setting here, but recommending it waits for Phase 3 numbers.
- Typed decisions (`decide.rs` track) — separate plan.
- `rerank_pairs` tool for non-graph content — needs its own
  justification, not smuggled in here.
- Thresholds, auto-enable, or default-on rerank — none ship without
  Phase 3 evidence.
