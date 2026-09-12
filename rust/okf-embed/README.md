# okf-embed — Jina v5 text embeddings (Rust core, PyO3)

Exact port of `EmbeddingEngine._encode`: task prefix → tokenize (8192) →
ONNX forward → last-token pooling → L2 → Matryoshka truncate → re-normalise.
Pinned against the optimum/numpy path by `tests/test_parity.py` (≤1e-5).

## Build & install

Needs a Rust toolchain (1.85+) and maturin. (`maturin develop` needs pip,
which uv venvs lack — build the wheel and install it instead.)

```bash
cd rust/okf-embed
maturin build --release
uv pip install --python <venv> target/wheels/okf_embed-*.whl --reinstall
```

## Runtime: ONNX Runtime discovery

`ort` loads dynamically (`load-dynamic`, same pin as bobine: `2.0.0-rc.13`).
Resolution order: `ORT_DYLIB_PATH` first (user override always wins), else the
OS loader path. OKFgraph's `resolve_ort_dylib()` points the var at the
pip-installed `onnxruntime` build when unset, so both bobine and okf-embed
share **one** ORT binary — no version/CUDA drift between ingest and import.

## Backend selection (`embedding_backend`)

`auto` (default) → Rust when the wheel is importable, else optimum.
`rust` → require the wheel (`RuntimeError` otherwise).
`optimum` → require optimum + transformers (+ torch).

## Fallback policy

| Level | Behaviour |
|---|---|
| Install | `auto` resolves once at router construction; missing stacks raise a clear `RuntimeError`, never an `ImportError` from deep inside. |
| Device | CUDA is opportunistic: `auto`/`cuda` use it when the loaded ORT registers the EP, else warn (stderr) + CPU. `used_cuda` reports the outcome. Never fatal. |
| Encode | **Fail fast.** No rust→optimum fallback at encode time — a mid-run stack switch would silently mix vector spaces in one index. |
| Tokenizer | No transformers in rust mode, anywhere: internal tokenize + `count_tokens()` (== `tokenizer.encode(t, add_special_tokens=False)`, pinned by `test_rust_backend.py`) feed the context-window guard; `SearchEngine` only ever stored the tokenizer, never called it. |

## Contract notes

- Session IO is discovered at load (`input_ids` + `attention_mask` required,
  `token_type_ids` fed only if declared — v5's export doesn't declare it,
  which is where generic runners fail). Output prefers `last_hidden_state`.
- `truncate_dim` validated like the router (32–1024, warning off the
  Matryoshka ladder). `MAX_LENGTH` (8192) is exposed for the window guard.
- Batch encoding is sequential by design (padded batches waste attention
  compute on variable-length docs). GIL is released during encode.
