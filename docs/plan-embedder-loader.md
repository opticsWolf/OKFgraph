# Plan: the embedder-loader seam (`docs/plan-embedder-loader.md`)

**Status:** draft v1, 2026-11-03 — first workstream of the 0.12 line.
**Supersedes in scope:** the `plan-http-embedder` thought (the same lane,
re-scoped from "HTTP backend" to "loader seam" after reading Weave Silk's
loader surface — the HTTP endpoint is one consumer of the seam, not the
seam itself).
**Sources of grounding:** `Silk/functions/embeddings.py`,
`Silk/functions/model_pool.py`, `Weave/weave/plugins/silk/functions/
okf_graph.py` (all read 2026-11-03); okfgraph `components/embedding.py`
(PRECISION_META_KEY, MODEL_META_KEY, `LazyRustEncoder`), `router.py`
(`_open_session`, ImportManager wiring), Q8 layer-split facts (`docs/
issue-solving-implementation-plan.md` §11).

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
  other.
- **When a switch is legal:** change settings → `okf reembed` re-computes
  every vector under the *current* engine and re-pins on success
  (existing op; the fp16/fp32 path is the precedent). `okf reembed
  --dry-run` reports the plan first.

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
- **W2 settings + validation + `router_kwargs`** + init-time cross-check
  (dim vs matryoshka ladder; `MODEL_META_KEY`/`backend` pin read).
- **W3 pins/refusals:** pin-on-write unchanged (session's first embed
  writes what `spec()` says); `enforce_backend_pin` mirrors
  `enforce_precision_pin`; `reembed` re-pins from `spec()`.
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

## 7. Phases

- **Phase 0 — decision data (bounded, quick-eval rules):** stand up two
  last-token-pooling embedders Silk already trusts: jina-v3 GGUF and
  Qwen3-Embedding-0.6B GGUF (llama.cpp `llama-server`, `/embeddings`).
  Run the reviewed-labels harness (`scripts/eval_search.py`) against the
  ONNX baseline on the same copy DB. Datum = nDCG/MRR/recall deltas per
  family. Decide Phase-1 defaults from numbers, not vibes.
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
