# okf-embed — Jina v5 text embeddings (Rust core, PyO3)

Exact port of `EmbeddingEngine._encode`: task prefix → tokenize (8192) →
ONNX forward → last-token pooling → L2 → Matryoshka truncate → re-normalise.
Pinned against a numpy/transformers replication of that pipeline by
`tests/test_parity.py` (max abs diff ≤ 1e-5, cosine ≥ 0.999999).

**The only embedding backend.** There is no Python fallback stack, no
`embedding_backend` selector, and no optimum/transformers in the runtime
path — a mid-run stack switch would silently mix vector spaces in one
index, so the design is fail-fast instead.

## Install

Published to PyPI — `okfgraph` pulls it in automatically (platform wheels
for Linux / Windows / macOS-arm64, Python 3.11–3.13):

```bash
uv sync          # editable path source, builds via maturin
```

From source (needs a Rust toolchain 1.85+ and maturin; `maturin develop`
needs pip, which uv venvs lack — build the wheel and install it instead):

```bash
cd rust/okf-embed
maturin build --release
uv pip install --python <venv> target/wheels/okf_embed-*.whl --reinstall
```

## Runtime: ONNX Runtime discovery

`ort` loads dynamically (`load-dynamic`, same pin as bobine: `2.0.0-rc.13`).
Resolution order: `ORT_DYLIB_PATH` first (user override always wins), else the
pip-installed `onnxruntime`/`onnxruntime-gpu` build when unset: Windows uses
`capi/onnxruntime.dll`, macOS uses `capi/libonnxruntime.dylib`, and Linux
prefers versioned `capi/libonnxruntime.so.*` with `capi/libonnxruntime.so` as
fallback. GPU DLL warming and Windows DLL-directory setup are best-effort and
never fatal. OKFgraph's `resolve_ort_dylib()` runs before the native module is
imported and exposes the choice as `OKFRouter.ort_dylib`, so both bobine and okf-embed
share **one** ORT binary — no version/CUDA drift between ingest and import.

## Failure policy

| Level | Behaviour |
|---|---|
| Install | The wheel is a core dependency of OKFgraph; if it is missing or fails to import, the router raises a clear `RuntimeError` with the install hint — never an `ImportError` from deep inside, never a silent fallback. |
| Device | CUDA is opportunistic: `auto`/`cuda` use it when the loaded ORT registers the EP, else warn (stderr) + CPU. `used_cuda` reports the outcome. Never fatal. |
| Encode | **Fail fast.** No fallback at encode time — vectors must stay bit-comparable within one index. |
| Tokenizer | No transformers in the runtime path, anywhere: internal tokenize + `count_tokens()` (== `tokenizer.encode(t, add_special_tokens=False)`) feed the context-window guard. |

## Contract notes

- Session IO is discovered at load (`input_ids` + `attention_mask` required,
  `token_type_ids` fed only if declared — v5's export doesn't declare it,
  which is where generic runners fail). Output prefers `last_hidden_state`.
- `truncate_dim` validated like the router (32–1024, warning off the
  Matryoshka ladder). `MAX_LENGTH` (8192) is exposed for the window guard.
- Batch encoding is sequential by design (padded batches waste attention
  compute on variable-length docs). GIL is released during encode.
- `input_ids`/`attention_mask` feed as int64; pooling takes the last
  attended token (`mask_sum - 1`, clamped ≥ 0).

## Testing

- **Rust unit tests** (13, pure — no network, no dylib, no tokenizer file):
  device parsing, task-prefix idempotence, the L2 → truncate → re-normalise
  math, contract constants, and `open()` validation firing before I/O.

  ```bash
  cd rust/okf-embed && cargo test --locked
  ```

  Runs in CI (Ubuntu, `--locked`) alongside OKFgraph's pytest jobs.
- **Python parity** (`tests/test_parity.py`, marked `slow`): Rust output vs
  a numpy/transformers replication across dims × tasks × texts, ≤ 1e-5.
  Needs the `omni` extra (transformers rides in via sentence-transformers).
- **Python e2e** (`tests/test_rust_backend.py`, `tests/test_rust_e2e.py`):
  wheel import, count_tokens contract, encode against the real model.
