# Plan: the embedder-loader seam (`docs/plan-embedder-loader.md`)

**Status:** draft v2, 2026-11-05 — embedder seam + reranker seam,
updated for embroider 0.13.0 (EmbeddingGemma 2).
First workstream of the 0.12 line.
**Supersedes in scope:** the `plan-http-embedder` thought (the same lane,
re-scoped from "HTTP backend" to "loader seam" after reading Weave Silk's
loader surface — the HTTP endpoint is one consumer of the seam, not the
seam itself).
**Sources of grounding:** `Silk/functions/embeddings.py`,
`Silk/functions/model_pool.py`, `Weave/weave/plugins/silk/functions/
okf_graph.py` (all read 2026-11-03); okfgraph `components/embedding.py`
(PRECISION_META_KEY, MODEL_META_KEY, `LazyRustEncoder`), `router.py`
(`_open_session`, ImportManager wiring), Q8 layer-split facts (`docs/
issue-solving-implementation-plan.md` §11); embroider 0.13.0 CHANGELOG
+ `src/jina.rs`/`src/acquire.rs` (EmbeddingGemma 2 family branch — the
in-crate precedent for declared pooling/prefixes).

---

## 1. Problem statement

okfgraph's embedding is hardcoded to one engine: `LazyRustEncoder` → the
`embroider` crate (Jina v5 via ONNX, task prefixes + last-token pooling +
Matryoshka truncation). Everything downstream consumes that one object
(`Router.embed_engine`, `ImportManager`, budget counting via
`encoder.count_tokens`) and every graph pin (`embedding_model`,
`embedding_precision`) ultimately describes *that* contract.

Hosts that want to embed differently — concretely: **Weave + Silk**, whose
loader surface ships two non-crate embedders — cannot plug in. The
plan defines a **loader seam**: a small, provider-agnostic protocol the
router accepts, so embeddings can come from

1. the crate (default, unchanged),
2. an HTTP endpoint that speaks an OpenAI-compatible `/embeddings` route
   (Silk's llama.cpp server behind its pool; any TEI/Ollama/openai-fixture
   server),
3. an **in-process loader object** supplied by the host process at router
   construction (for embedding hosts that run okfgraph in one process).

Everything else — chunking, budgets, pins, reembed, the Q8 hardening —
stays exactly as is, re-targeted to whatever engine sits behind the seam.

## 1a. embroider 0.13.0 — the in-crate family branch (read 2026-11-05)

`google/embeddinggemma-2` joins the builtin registry: `native_dim 768`,
`max_tokens 8192`, MRL ladder `128/256/512/768`, fp32 from the
`onnx-community` export (pin `daa72c5`, `onnx/model.onnx` + sidecar),
fp16 from the `opticsWolf/embeddinggemma-2-onnx-fp16` mirror (pin
`296b2a6`, CUDA-only, CPU parity 2.1e-4), `int8: None`. `JinaV5` gains a
per-id family branch: **mean pooling** + **Gemma prompts**
(`task: search result | query:` / `title: none | text:`); every other id
keeps the frozen last-token + `Query:`/`Document:` contract. Gemma runs
take empty `[0,512]` multimodal feature feeds and read the native
`sentence_embedding` head (projection + L2 in-graph).

Two consequences for this plan:

1. **The crate is the reference implementation of the seam's honesty
   rules.** `pooling_for(id)` / `gemma_prefixed()` is precisely the
   `EngineSpec.pooling` / `prefixes` pattern — declared per id inside the
   engine, not probed by callers. The external seam generalizes this to
   engines the crate does not own. W1's fixtures should mirror the crate's
   family-branch shape so both lanes stay legible.
2. **Adoption in okfgraph is a floor bump, not a feature:** okfgraph has
   no per-model registry of its own. `model_id` is a free string that the
   crate resolves (`DEFAULT_MODEL_ID` stays text-small; the registry
   comment is explicit — *adding models never moves existing graphs*).
   Bump `embroider>=0.13` + ORT 1.31.1 (+gpu) extras in pyproject, verify
   the pin/refusal tests pass unchanged, add an EmbeddingGemma-2 row to
   the eval plan (it is a crate-family candidate, so Phase 0 can measure
   it on the reviewed labels with zero new plumbing). Pages/stack rows
   follow. EmbeddingGemma 2 is therefore **not** an external-loader
   consumer — it widens the default lane; worth keeping the two apart in
   every doc and release note.

## 2. The external surface to cover (what Weave/Silk gives and expects)

Silk's loader (`Silk/functions/embeddings.py`) is a duck-typed embedder:

- `Embedder.embed(text) -> Optional[list[float]]` — fails silent: it
  disables itself once (with a logged reason) and memory stays FTS-only.
  Timeout 30 s on the wire path; width learned from the first vector and
  frozen (`DimMismatchError` discipline).
- Implementations: `LlamaEmbedder` (in-process `llama_cpp.Llama` or
  shape-alike), `ServerEmbedder` (an OpenAI-compatible `/embeddings`
  endpoint behind the model pool, pool-shared headers), `embedder_for(
  handle)` as the factory (inspects `handle["model"]` / `handle["pool"]`,
  asks the object, never the metadata).
- Identity discipline: one model = one vector space; Silk names the table
  after `model_name()` (Macrame-legal `[a-z][a-z0-9_]*` ≤ 48) — two
  spellings of one model would silently split the index.

Silk's okfgraph node (`Weave/weave/plugins/silk/functions/okf_graph.py`)
already runs okfgraph **out of process** (`okf serve` + `OkfClient` →
`POST /ops/<op>`, auto-discovery via `<db>.serve.json`) for the same
reasons the crash-log found (no second ONNX in someone else's process;
single open of the DB file), and already consumes the **pin +
reembed discipline** (`OkfGraph.pins()`, `reembed` under a 6 h budget).
`OkfSpec` (db/executable/bundle_root/device/model_id) is the node's saved
config — the loader lane extends *this struct*, not the protocol.

## 3. The `EmbedEngine` protocol (the required surface, general by design)

```python
class EmbedEngine(Protocol):
    def embed(self, text: str, task: str = "Document") -> list[float]: ...
    def embed_batch(self, texts: list[str], task: str = "Document") -> list[list[float]]: ...
    def count(self, text: str) -> int: ...          # token-grade, budget-grade
    def spec(self) -> EngineSpec: ...               # the family declaration
```

`EngineSpec` (all graph-pin-relevant truth in one object, **declared,
never guessed**):

| Field | Meaning | Example |
|---|---|---|
| `family` | free-form identity string | `"llama.cpp:qwen3-embedding-0.6b"` |
| `model_id` | the graph pin value | Silk's `model_name()` slug |
| `native_dim` | width the engine returns | 1024 |
| `matryoshka` | truncation ladder or `None` | `[512, 768, 1024]` or `None` (= native-only) |
| `max_tokens` | the engine's true input ceiling | 8192 |
| `pooling` | declared, not probed | `"last"` / `"mean"` / `"cls"` |
| `prefixes` | task-prefix table or `None` | `"jina-v5"` map ingest: task → exact prefix text |
| `counter` | where `count()` comes from | `"tokenizer"` / `"gguf-meta"` / `"heuristic"` |

Contract clauses:

- **Determinism:** same text+task → same vector; batch order must not
  change a row's vector. The loader owns retries/timeouts; okfgraph's ops
  stay fail-fast (`LOAD_ERROR`/`RUNTIME` semantics as today).
- **No silent half:** Silk's fail-to-silence is a *host* policy. The
  engine protocol fails loudly; whoever wraps it (node, CLI, ops call)
  decides degradation. okfgraph keeps importing/serving, it does not
  guess an engine into service.
- **Matryoshka:** if the engine declares no ladder, the router's
  `embedding_dim` must equal `native_dim` or init fails (`CONFIG_INVALID`
  naming the mismatch). The crate's slice+renormalize math stays the
  reference implementation.
- **Prefixes:** task prefixes are the *engine's* business. `prefixes=None`
  means raw text; the router never prepends anything behind an external
  engine's back.

## 4. Config, pins, injection

**Settings** (`[embedding]`, all validated in `Settings.checks()`):

```toml
[embedding]
backend   = "embroider"   # "embroider" (default) | "external"
# only for backend = "external":
endpoint_url = ""         # e.g. http://127.0.0.1:8100  (Silk's pool server)
api_key_env  = ""         # name of env var carrying the bearer, if any
family       = ""         # EngineSpec.family (validated non-empty)
model_name   = ""         # the graph pin value (Silk's model_name() slug)
dim          = 0          # native_dim; = embedding_dim unless matryoshka says otherwise
max_tokens   = 32768      # ceiling
pooling      = ""         # last | mean | cls
prefixes     = "none"     # none | jina-v5
counter      = "heuristic" # tokenizer | gguf-meta | heuristic
```

- CLI/TOML/env parity per the one settings table (`OKFGRAPH_EMBEDDING_
  BACKEND`, ...). `router_kwargs()` sprays it verbatim.
- **In-process injection:** `OKFRouter(..., embed_engine=<obj>)` accepts
  any protocol implementer (a Silk `LlamaEmbedder` wrapped by the host).
  Settings `backend = "embroider"` + an injected engine = init fails
  (choose one truth). The injected engine's `spec()` feeds the pin.
- **Pins:** `MODEL_META_KEY` gains the external identity
  (`family:model_name`), `PRECISION_META_KEY` keeps its meaning
  (numeric identity of what was *stored*). Mixed-backend vectors are
  refused at session open exactly like mixed precisions, with the remedy
  = `reembed` (graph-wide) or `import --all --force`. The one-backend-per-
  graph rule: a pin authored under one backend refuses sessions from the
  other. **Landed shape (2026-11-05):** the external lane enforces the
  model pin at router boot (`_enforce_pins`; external carries no numeric
  precision so the precision pin is skipped there) and the reembed op is
  the sanctioned escape: boot with `OKFGRAPH_ALLOW_REPIN=1` (loud warning)
  to open a graph whose pin disagrees with the configured model, run
  `reembed`, re-pin on success.
- **When a switch is legal:** change settings → `okf reembed` re-computes
  every vector under the *current* engine and re-pins on success
  (existing op; the fp16/fp32 path is the precedent). `okf reembed
  --dry-run` reports the plan first. The shipped op refuses dim
  migrations (`DIM_MISMATCH`) and multimodal graphs
  (`IMAGE_PIN_MISMATCH`); each step auto-commits, so a crash mid-way is
  recoverable by re-running.

## 5. Counting port (the part budgets live or die on)

`count()` is token-grade today via the crate (`encoder.count_tokens`),
consumed by: `_token_bucketed_encode`, `chunk_size` caps, the
context-window guard, and `_clamp_unsplittable_payloads` (the Q8
hardening). The protocol carries `count`; sources in order of preference:

1. `"tokenizer"` — the engine exposes a real tokenizer (llama_cpp's
   `tokenize`, a `tokenizer.json` via the session-free counting handle).
2. `"gguf-meta"` — Silk's `gguf_meta.py` path (vocab ceil from the file).
3. `"heuristic"` — a `/4` chars-per-token estimate with `max(8, len)`
   floor; **the spec must flag it** (`counter = heuristic` → the router
   logs once per process: budgets are approximate, the context-window
   guard degrades to advisory).

Chunking itself is mordant-based (no engine), so chunk boundaries are
engine-independent; only their *row caps* read `count()`.

## 6. Work items

- **W1 `EmbedEngine` protocol + `ExternalEmbedEngine`** (HTTP transport
  first: batch request, per-row timeout, size cap, no retries inside).
  Fixture engines mirror the crate's family-branch shape (§1a).
- **W2 settings + validation + `router_kwargs`** + init-time cross-check
  (dim vs matryoshka ladder; `MODEL_META_KEY`/`backend` pin read).
- **W3 pins/refusals:** pin-on-write unchanged (session's first embed
  writes what `spec()` says); `enforce_backend_pin` mirrors
  `enforce_precision_pin`; `reembed` re-pins from `spec()`. **Note
  (verified 2026-11-05): okfgraph 0.11.0 has no `reembed` op** — Silk's
  `OkfGraph.reembed()` (`okf_graph.py`) already calls `POST /ops/reembed`,
  so shipping W3's op is a live host-compat fix, not a nicety. The pin
  storage split (numeric `Meta` vs string `MetaText`) is also now a
  confirmed harness fact (`eval_search.py` had to adopt both).
- **W4 counting port** (three `counter` sources; heuristic flagged).
- **W5 `POST /ops/embed`** (small, unlocks the inverse lane: Silk's ledger
  riding okfgraph's own pinned encoder; `_HONOURS` guarded).
- **W6 tests:** a fixture engine (scripted vectors) covering the protocol;
  pin refusals; budget fences under each counter; batch-order
  determinism; `ops/embed` wire; reembed-under-external happy path.
- **W7 docs:** README settings chapter + quickref block, architecture
  "settings" table row (and the "all numerics live in the crate" line
  amends to "all ONNX numerics — external loaders own their numerics"),
  compat row, CHANGELOG under 0.12.0 Unreleased.
- **W8 Silk wiring (host side, in the Weave repo):** `OkfSpec` gains
  `backend`, `family`, `endpoint_url`; the node maps its pool server and
  `embedder_for(handle)` onto the new settings; pins warnings on every
  evaluation (already coded) keep working unchanged.
- **W9 reranker seam (same treatment, second stage).** The reranker is
  currently `LazyReranker` → `embroider.Reranker` (mMARCO mMiniLM cross-
  encoder, `rerank_model_id` default, its own `rerank_device`, no pins —
  it writes no vectors). The seam extends the same protocol shape to the
  scoring stage: a `RerankEngine` perception of `score_batch(pairs) ->
  logits`, `score_long(q, doc)`, `count(text)`, plus a stage scheduling
  spec (`family`, `model_id`, `pair_cap`, `precision identity`). The
  external transport is the standard rerank wire (`POST /v1/rerank` with
  `{model, query, documents, top_n}` → `{results: [{index, score}]}`),
  which Silk's pool can front with llama.cpp's reranking support or any
  TEI-shaped server — the same `endpoint_url` + pool-header pattern as
  the embedder. `LazyReranker` keeps the fallback rule: the external
  engine failing its first score disables rerank for the session (rerank
  = an optional half; keyword+vector search still answers). Settings:
  `[rerank] backend = "embroider" | "external"` with the same field
  shapes the embedder grew (W2). No graph pins (rerank scores are
  order-only); a *model identity* note lands in `search_meta`-style
  provenance only. Silk gains `OkfSpec.rerank_backend` and maps its pool
  the same way (W8 extension).

## 7. Phases

- **Phase 0 — decision data (bounded, quick-eval rules) — DONE
  2026-11-05 (first slice).** Evidence: `.okf-suite-1102/phase0-embedder/`
  (reports + the surgical-reembed probe). Findings:

  1. **The reviewed-labels harness needed pin adoption** — the model pin
     lives under a *separate node label* (`MetaText`, not `Meta`) and the
     router refuses an explicit `model_id=None`; the harness now adopts
     both pins and accepts `--model-id` (`26b2388`).
  2. **Fresh-repo-import evals of the okfgraph labels collapse to
     ~0.12–0.14 for every config, both models** — the labels' concept ids
     (`reports/…`, `thoughts/…`) exist only in the dogfood ledger, not in
     a repo-tree import. Model-vs-model on a fresh import is a tie
     (gemma2 0.131 vs jina 0.130 rrf) but versus the recorded 0.808 it is
     meaningless. **Lanes for model comparison must re-embed the same
     graph, not re-import the corpus.**
  3. **The `reembed` op is implementable as a bounded probe**: drop the
     two vector indexes (`DROP_VECTOR_INDEX` after `LOAD EXTENSION
     vector`), SET the embeddings row by row with the new engine, recreate
     the indexes with the schema lane's exact DDL, re-pin `MetaText
     embedding_model`. Whole dogfood copy (46 concepts + 637 chunks) took
     **11.8 s on CUDA fp16**. HNSW dropped/recreated cleanly; FTS stayed
     untouched. This probe is W3's blueprint (the op version must do the
     same inside one transaction and refuse multi-tenant shapes the probe
     ignores, e.g. image tables).
  4. **EmbeddingGemma 2 verdict (dogfood copy re-embed @512, CUDA fp16):
     at-or-above the text-small baseline on every vector row**, hub row
     bit-identical (lexical, model-free — the calibration proof):

     | config | gemma2@512 | jina-small@512 (0.11.0 record) |
     |---|---|---|
     | rrf | 0.673 / 0.746 / 0.766 | 0.673 / 0.747 / 0.795 |
     | hub | 0.673 (identical — model-free) | 0.673 |
     | cross | 0.775 / 0.887 / 0.872 | 0.776 / 0.883 / 0.864 |
     | cross:mmarco | **0.821 / 0.921 / 0.917** | 0.808 / 0.903 / 0.897 |

     (nDCG / MRR / recall.) The gemma2 lane also scores ~0.10 s/concept
     encode on CUDA vs ~0.30 s for jina-v5-small — the fp16 mirror is
     574 MB and already in the hub cache.
  5. **Q8 note:** gemma2's encode economics change the CPU-lane math
     (model is ~2× smaller than text-small; CUDA lane measured 20×+ per
     row). A CPU gemma2 import is worth one bounded timing probe before
     more Q8 work lands.

  6. **The external llama.cpp lane carries crate-level quality**
     (2026-11-05, first external-family slice). Machine infra: LM Studio's
     llama.cpp runtime (`~/.lmstudio`) serving the bundled
     `nomic-embed-text-v1.5-Q4_K_M.gguf` on `127.0.0.1:8931`/v1/embeddings
     — the exact wire the pool's `ServerEmbedder` uses, and the wire W1
     will consume. Offline vector-KNN scorer (`offline_trio_nomic.py`),
     identical pool (dogfood's 637 chunks + 46 concepts) and identical
     scoring (grade-2/1 per concept, first-appearance-wins), chunk-target
     label rows (n=13:

     | lane | nDCG@10 | MRR | recall |
     |---|---|---|---|
     | gemma2@512 (crate, CUDA fp16) | **0.810** | **0.910** | 0.859 |
     | jina-v5-small@512 (crate) | 0.781 | 0.846 | 0.910 |
     | nomic-v1.5@768 (llama.cpp server) | 0.784 | 0.846 | 0.872 |

     The external family lands **at parity with the crate baseline** on
     its first try. Nomic also validates the whole EngineSpec posture:
     different family = mean pooling + **mandatory per-side prefixes**
     (`search_query:` / `search_document:`) + native dim 768 with its own
     ladder — exactly the per-family declaration the seam is designed
     around. W1's stub fixtures should encode the nomic contract as the
     first real reference, jina-v5's as the second (in-crate).

  7. **The reranker lane closes Phase 0** (2026-11-05). Three reranks
     over identical pure-VS top-30 pools (gemma2 retriever, chunk-target
     label rows, n=13; probes in the evidence dir):

     | rerank lane | nDCG@10 | MRR | recall | p50/p95 s |
     |---|---|---|---|---|
     | none (VS order) | 0.830 | 0.923 | 0.859 | — |
     | crate mmarco cross-encoder | 0.859 | 0.923 | 0.923 | 0.06 / 0.78 |
     | bge-reranker-v2-m3 Q8, llama.cpp `/v1/rerank` | 0.839 | 0.910 | 0.885 | 0.12 / 0.21 |

     The external rerank wire (llama.cpp b12110 CUDA13, official release,
     `--rerank`) lands **at parity with the crate cross-encoder at its
     latency profile** — and the external lane is even steady at p95.
     Two hard facts ride with it: (a) **Silk's pool cannot rerank at all**
     — `llama_cpp` 0.3.34 has no `LlamaReranker` and no `/v1/rerank`
     route, so Silk's reranker future rides the llama.cpp binary server
     lane (or the crate), never the python pool; (b) family contracts
     matter as much here as for embedding: the community **Qwen3-Reranker
     0.6B Q8 GGUF scored 0.424 / 0.541 / 0.538 — a red datum for the
     conversion, not the wire** (the toy 3-doc sanity call ranks
     correctly; official safetensors conversions are part of any real
     adoption). Also recorded: long pair inputs need `-ub 1024` (the
     512-token physical batch 500-errors on ~657-token chunks).
  8. **Phase 1 slice 1 = the seam, landed** (2026-11-05, `ead53ab`).
     `ExternalEmbedEngine` (OpenAI-shaped `/embeddings`, `EngineSpec`, the
     client-side Matryoshka slice+renorm fallback, one-shot heuristic-
     counter warning, endpoint normalization `<base>/v1/embeddings`),
     settings rows `[embedding] backend/endpoint_url/api_key_env/
     embed_family/embed_pooling/embed_prefixes/embed_max_tokens/
     embed_counter` with checks + `router_kwargs` spray, `OKFRouter(
     embed_backend="external")` wiring (crate optional on this lane, pin
     plumbing via `encoder.precision`), offline stub-server tests
     (13 green) and a live smoke through LM Studio's llama.cpp nomic
     lane: end-to-end `ingest` + `search` with the family-contract lane
     live (nomic prefixes, heuristic counter, MRL fallback logged once).
  10. **Phase 1 slices 2–3 = the graph-side ops, landed** (2026-11-05,
      `afd0454` …).
      `reembed(dry_run)` (W3): drop both vector indexes → re-encode every
      Concept/Chunk with the active engine (import-lane text shape) →
      recreate indexes from the schema specs → re-pin
      ``embedding_model`` (+ numeric precision when the engine owns one);
      refuses dim migrations (`DIM_MISMATCH`) and multimodal graphs
      (`IMAGE_PIN_MISMATCH`); reachable across a pin change via the
      `OKFGRAPH_ALLOW_REPIN=1` boot escape (loud warning; pin refusals
      stay fail-closed without it; crate + external enforce through one
      shared `_enforce_pins`, external = model pin only). `embed`
      (W5): the inverse lane — any caller may ride the graph's OWN
      pinned encoder via `POST /ops/embed` with `{texts, task, dim}`;
      per-call Matryoshka dims slice+renormalise client-side (no server
      cooperation needed), the response carries provenance
      (`renormalised`, `honours_dimensions`, `counter` — the `_HONOURS`
      guard), caps = 256 texts / 1 MiB, refusals = BAD_VALUE. Both ops
      are CLI/HTTP (`OPS` tuple + client proxy); MCP keeps search/read.
      W5 smoke: serve route + stub endpoint, dim 256 round-trip \\\\
      `renormalised=True`. **W7 landed** (`8fc2afc`): `okf reembed
      [--dry-run]` CLI, README ops/maintenance rows, CHANGELOG 0.12
      section, architecture settings note, CLI skill row. **W6 tail
      landed** (`bcb8ffe`): the greedy token bins respect the batch cap
      and budget under the external heuristic counter.
  8b. **Phase 1 slice 4 = the reranker seam, landed** (W9,
      `dev_0.12.0`). `ExternalRerankEngine`
      (`okfgraph/components/external_reranker.py`): the
      `LazyReranker` duck-type (`score_batch`/`score_long`/
      `count_tokens`/`max_length`/`model_id`/`used_cuda`) over the
      standard rerank wire, `POST <base>/v1/rerank`
      `{model, query, documents, top_n}` → `{results: [{index,
      relevance_score}]}` (the llama.cpp/TEI shape; one call = one
      query group; `score_batch` groups pairs by query, input order
      preserved; full index coverage + finite scores validated).
      Settings `[search] rerank_backend='external'` +
      `rerank_endpoint_url` / `rerank_api_key_env` / `rerank_family` /
      `rerank_max_tokens` (declared per-pair ceiling; the short/long
      split point). The optional half's contract: the external engine
      failing its first score **disables rerank for the session**
      (`down` + one loud warning); `SearchEngine._cross_scores`
      answers that search and all later ones unranked (returns None →
      no cross_score keys, raw order). The crate lane's fail-fast
      semantics unchanged. No graph pins — identity rides boot logs +
      `spec_report()`; windowing = None in the spec (window/stride
      arguments accepted and ignored, warned once). Tests
      (`test_rerank_external.py`, 7 green): grouping/order, the
      one-shot window warning, broken-index coverage → trip → the
      fast-path skip, the router smoke (first failure degrades the
      search, later searches make no rerank calls; a healthy host
      orders with `cross_score`), settings validation, factory.
      Remaining Phase-1 scope: the W4 counters
      (`server-tokenize` first, GGUF calibrated prior second),
      0.11.x backlog.
  8c. **Phase 1 slice 5 = `pin_info` + the host half of the node flow
      (2026-11-05, `9a7f850`).** `OKFRouter.pin_info()` (in `OPS`): what
      the vectors are pinned to (`MetaText` model, `Meta` numeric
      precision, the column width) against what the server booted with
      (model, expected precision, backend, family) — the Softer-side
      pins display flows over this. ``embedding_dim`` never mismatches
      (the width is a create-time fact; a re-embed cannot move it),
      `reembed`'s report gains `pins_after`. Rust-side tests: the
      mismatch across the repin escape, cleared by the re-embed.
  8d. **Phase 1 slice 6 = W8, the Silk/Weave wiring (2026-11-05).** In
      the Weave plugin's silk mirror (the module tree the runtime
      imports; the whole okf layer = working-tree files there):
      `OkfSpec` gains `embedding_dim` / `backend` / `embed_family` /
      `endpoint_url` / `api_key_env` / `allow_repin`; the external-lane
      rows and the re-embed escape ride `OKFGRAPH_*` env at launch
      (`--embedding-dim` carries the width), `OkfEmbedder` rides
      `POST /ops/embed` on the graph's pinned encoder
      (`embedder_for` accepts an `OkfGraph` handle; the inherited
      fail-to-silence), `OkfGraph.repinned()` = the declared pin-exempt
      boot → re-embed → pin-clean restart cycle, and the Knowledge
      Graph node gains vector-dim / backend / endpoint / family /
      key-env rows (node_version 2) with the honest pin display and
      the 0.12 report shapes. Seam tests 4 green; the touched Weave
      suites 55 green. Remaining: the W4 counters (`server-tokenize`
      first), docs passes, the 0.11.x backlog.
  8e. **Phase 1 slice 7 = W4, the counting ladder (2026-11-05).**
      `embed_counter` declares where external token-count budgets come
      from: `server-tokenize` (exact — the host's own `POST /tokenize`,
      which llama.cpp serves at the server root; probed once per process,
      degrading to the heuristic with one loud warning when absent),
      `gguf-meta` + `embed_counter_gguf` (a measured chars/token prior
      read from the model's GGUF header — pure Python, no `tokenizers`
      wheel: the header's vocabulary is greedily longest-match probed
      against a fixed English sample), `declared` + `embed_chars_per_token`
      (0.5..64) and `heuristic` (`max(8, len/4)`, logged once). A failed
      probe lands on `heuristic-fallback`; `POST /ops/embed` reports the
      **effective** source. Live on this machine (nomic-v1.5 Q4_K_M at a
      142-char text): true `/tokenize` = 25 tokens (5.68 chars/token),
      heuristic = 35 (+40% over-count, the safe direction), gguf-meta
      prior 4.35 → ~33 (+32%), header scan 32 ms. LM Studio / TEI serve
      no `/tokenize` and degrade as designed. Tests: `test_counter_gguf.py`
      (15 — synthetic GGUFs, wrapped `EngineSpec` usage, the root-vs-/v1
      wire, the once-only degrade), `test_external_embedder.py` +3
      (settings checks incl. the crate-lane refusal, the env float
      parse, the kwargs spray), `test_ops_embed.py` +1 (effective
      counter provenance). Touched offline suites 86 + 109 green.
      Remaining Phase-1 scope: docs passes, the 0.11.x backlog.
  8f. **Phase 1 slice 8 = interruptible re-embed: the job lane (2026-11-05,
      the Weave integration's blocking gap).** The re-embed rode ONE
      blocking HTTP request with a 6 h timeout; Silk could only abandon
      it by dropping the connection — and the op kept the graph lock
      while search/ingest/pin_info queued behind it. ``okf serve`` now
      owns a job lane (0.12.1): ``POST /ops/reembed {"mode": "job"}``
      answers a job id inline, ``GET /jobs/<id>`` reports status +
      phase/done/total (the reembed accepts a ``progress`` hook called
      per batch), and ``POST /jobs/<id>/cancel`` stops the op AT ITS
      NEXT BATCH BOUNDARY (``reembed`` takes a ``cancelled`` callable
      checked between batches only). The cancel itself never queues
      behind the op lock, so cancelling a 6 h re-embed answers now.
      A cancelled run returns ``cancelled: true`` with partial counts
      ``resumable: true`` — the same crash-like state the docstring
      recorded (vector indexes stay dropped until a re-run converges;
      a re-run re-encodes FROM SCRATCH, honestly noted, since no
      per-row rewrite marker exists). Sync default unchanged (mode
      absent = the blocking call), a second submit (job OR sync) now
      refuses ``REEMBED_IN_PROGRESS`` instead of silently queueing (a
      route precheck + an in-op check under the lock as the race
      backstop; new state code). Silk-side: ``OkfGraph.reembed`` rides
      the job lane (poll between ``OkfGraph.REEMBED_POLL`` seconds,
      ``on_progress`` per poll, ``should_stop()`` between polls →
      cancel), ``OkfGraph.repinned`` inherits it, and
      ``OkfClient.job/cancel_job`` carry the GET/POST surface; each
      dropped poll retries without killing the loop, and a window that
      expires cancels the job server-side so nothing queues. Tests:
      ``tests/test_reembed_jobs.py`` (5 — real-server-on-a-thread:
      progress, cancel-between-batches with the partial counts + the
      re-run converges + search restored, the double-submit refusal,
      unknown-job/terminal-cancel idempotence, the mode list; the
      sleepy stub encoder × 40 rows makes the batch boundaries real).
      Plus a cross-repo live smoke (the Silk OkfClient polling a real
      0.12.1 owner: submit → poll → done → report). Touched suites
      green; sync-contract regression suites green.
  9. **Phase 0 = closed.** Defaults going into Phase 1: gemma2@512 as
     the crate default family, nomic-v1.5 as the reference external
     embedder contract (datum 6), bge-reranker-v2-m3 as the reference
     external rerank contract (datum 7), mmarco cross-encoder stays the
     in-crate default reranker. jina-v3 / Qwen3-Embedding GGUFs can ride
     the offline scorer opportunistically when their GGUFs land on disk;
     they are no longer blocking.
- **Phase 1 — the seam:** W1–W5 + tests, HTTP backend only.
- **Phase 2 — Silk integration:** W8 + docs; default answer for Weave is
  the HTTP backend (their node is out-of-process by design).
- **Phase 3 — in-process injection + registry growth:** `embed_engine=`
  param shipped but behind a documented use warning (the second-ONNX
  story), registry entries for declared families.

## 8. Open questions

| # | Question | Default if no answer by implementation time |
|---|---|---|
| ELQ1 | Pin value format for external models: `family:model_name` vs registry-only? | `family:model_name`, free string, refused on mismatch |
| ELQ2 | `counter = heuristic` allowed at all, or exact-tokenizers only? | allowed + logged-off advisory; Q8 guard keeps its 90%-window warn |
| ELQ3 | Matryoshka for external families: declare ladders, or native-only? | native-only, per family, opt-in ladder in `EngineSpec` |
| ELQ4 | `backend` value names: `embroider`/`external` vs `onnx`/`external`? | `embroider`/`external` (names the supplier, not the implement) |
| ELQ5 | Ship `ops/embed` in the same release as the seam? | yes (small; `_HONOURS`-guarded) |
| ELQ6 | Silk lane default: HTTP first or in-process injection first? | HTTP (matches Silk's out-of-process node decree) |
| ELQ7 | Rerank wire shape: `/v1/rerank` top_n-only, or also expose raw pairwise `/score`? | `/v1/rerank` only; `score_long` rides pair re-scoring |
| ELQ8 | Rerank provenance: which record keeps the model identity for nDCG audits? | `search_meta`-style note on the op response, no new Meta node |

## 9. Non-goals

- No torch/transformers anywhere (loads still arrive as vectors; GGUF via
  silk's loader = llama_cpp, not torch).
- No multi-backend-per-graph, no per-model tables inside okfgraph (Silk's
  ledger owns that pattern; okfgraph stays one pinned space per graph).
- No change to chunking, Q8 hardening, or the crate's Jina semantics —
  the default lane is byte-identical to 0.11.0 behavior.
- The in-process injection is **not** supported for hosts whose process
  already runs an ONNX session (the crash-log's second-ONNX story):
  inject llama_cpp-class engines only, or separate processes.

  9. **Phase 1 slice 9 = external rerank hardening (0.12.1): TEI
      dialect + gated exact counting, with the measured overhead.**
      Two follow-ups on the W9 rerank lane. (a) The wire dialect probes once per session: the
      standard `results[] / relevance_score` shape first, then the TEI
      shape — bare `[{"index","score"}]` over
      `{query, texts, truncate: true}` at the host root; rows accept
      either score key, irrelevant of shape; the answering dialect
      latches; neither → the combined error names both attempts (the
      lane trips down as always). (b) `rerank_count` picks the
      short/long split-counting source: `heuristic` (len/4, warned
      once — the default, unchanged) or `server-tokenize`: exact counts
      from the rerank host's own `POST /tokenize` (llama.cpp serves it
      at the root; timeout min(timeout, 10)), **gated** — the split is
      a threshold check, so `count_tokens(text, cap=)` (the caller now
      passes the effective `min(max_length, window)`; the crate lane's
      exact counts ignore it) fast-paths texts at most `(cap-4)/2`
      chars (chars bound tokens from above for BPE/SPM/Unigram: the
      combined pair can never cross `cap`, zero calls) and texts over
      `cap` chars (upper bound → over-cautiously long, zero calls);
      only the ambiguous middle band asks the host, after a one-time
      probe. Any counting failure latches to `heuristic-fallback` with
      one warning (the search never waits twice on a counted lane).
      Offline: `test_rerank_external.py` 12 (7 W9 + the TEI probe/
      latch, the score alias, the gate's 0-call bands + exact band
      counts + window-cap, the degrade-once latch, factory + settings
      checks). Touched suites 163 + 1 skip green.
      **Live + measured** (llama-b12110 `--rerank -ngl 99`, CUDA,
      bge-reranker-v2-m3 GGUF Q4_K_M, ctx-size 8192, :8935): binary
      relevance ordering intact (laptop 5.71 vs -10.99/-11.0); the gate
      asked **8 of 30** candidates (fast-path 22, `cap=512`); the warm
      `/tokenize` route costs **~0.9 ms/call** loopback (first call
      hosts the cold BPE path: ~8 ms), so counting adds **~7 ms over
      the ~136 ms cross-encoder pass ≈ 5%** for a search-shaped
      workload with zero split flips on English text — while the exact
      counts expose the CJK drift the heuristic hides (304 chars:
      exact 241/243 tokens vs len/4's 76, a 3.2× underestimate, which
      is exactly the silent-truncate case the gate removes). Probe
      script `$LOCALAPPDATA/Temp/okf-p0/w9_rerank_live.py`; GGUF in
      `$LOCALAPPDATA/Temp/okf-p0/gguf/`; server killed after.
