# OKFgraph implementation plan: 0.13.1 release → 0.14.x

Status, 2026-11-13: A released as 0.13.1. B, C, D and H done and
released as 0.14.0. E and F done and **released as 0.14.1** (`v0.14.1` on
`main`, wheel + sdist on PyPI; built against bobine 0.9.1). G1 and G2
decided and **released as 0.14.2**; G3 is a design note
(`docs/plan-declared-image-identity.md`), no code yet.

This plan covers the okfgraph side of the ONNX unification
(`plugins/silk/docs/ONNX_UNIFICATION_PLAN.md` in Weave: items **O3** and
**O6**) plus the follow-ups left open by 0.13.1. Each step names the files
it touches, the tests and docs it owes, and what "done" means.

| Step | Version | Depends on | Size |
|---|---|---|---|
| A. Release 0.13.1 | 0.13.1 | green suite | small |
| B. External image lane (O3) | 0.14.0 | embroider ≥ 0.15 (`serve` vision route, `prepare_image`) | medium |
| C. In-process image prep on `embroider.prepare_image` (E6 tail) | 0.14.0 | B's encoder seam | small |
| H. Caption fallback covers the encode call (declared-lane hardening) | 0.14.0 | B's encoder seam | small |
| D. Typed conversion errors | 0.14.0 | — | small |
| E. bobine device/precision vocabulary (O6, part 1) | 0.14.1 | bobine 0.9 (B3) | small |
| F. Warm PDF converter with a bounded lifetime (O6, part 2) | 0.14.1 | bobine 0.9 (B4) | medium |
| G. Open questions | — | a decision | — |

Steps B–D and H can ship together without bobine. E and F wait for bobine
0.9 and ship together, because both change the converter construction.

Standing rules that apply to every step:

- Release tags are pushed on `main` only: fast-forward `main` to the
  release commit first, then tag `main`'s HEAD.
- ONNX never loads in a host process that asked for an external lane
  (Weave runs okfgraph that way). An external lane must not import
  `embroider`'s native module, directly or through a helper.
- The CHANGELOG, README, `pages/index.html` badge and the skills move
  with the version.

---

## A. Release 0.13.1

What's in it (already committed): flags before the command, `--pdf-path`
taking several files, `okf setup skills`, skills in the wheel, help that
names where things live, read commands refusing a missing database, the
db's parent folder created on init/import/ingest, the `bobine>=0.8.1`
floor.

1. Full suite green in the locked venv (`okflocked`: okfgraph 0.13.1,
   bobine 0.8.1, embroider 0.15.0). Baseline was 1234 passed, 13 skipped
   plus the 23 new tests. Watch for older tests that run a read command
   (search, read, doctor, export, shell, traverse, …) against a database
   path that doesn't exist yet. Those now get `FILE_NOT_FOUND`; fix
   such a test by running `init` first, not by loosening the rule.
2. CI: `.github/workflows/ci.yml` passes on the PR (torch grep path is now
   `okfgraph/ examples/`).
3. PR `dev_silk` → `main`. Body: the CHANGELOG `[0.13.1]` entry.
4. After merge: fast-forward `main`, tag `v0.13.1` on `main`'s HEAD, push
   the tag. The release workflow builds the wheel and sdist; check both
   carry `okfgraph/skills/*/SKILL.md` (three files).
5. Check PyPI shows 0.13.1 and `pip install "okfgraph[cpu,pdf]"` in a
   clean venv pulls bobine ≥ 0.8.1 and runs `okf setup skills --to
   <tmp>`.
6. Pages deploy shows the 0.13.1 badge.
7. Silk's plan status block: mark okfgraph 0.13.1 released.

Done when: tag on `main`, wheel on PyPI, pages updated.

---

## B. External image lane (O3)

### Why

Text embedding and reranking can already run out of process
(`backend='external'`, `rerank_backend='external'`). The image lane
cannot: `LazyVisionEncoder` always opens an embroider vision session in
the caller's process. A host that must not load ONNX (Weave) therefore
can't embed image content and falls back to captions. embroider's
`serve` already answers `POST /v1/images/embeddings` for a `vision`
model, decodes and resizes on its side (`embroider.prepare_image`), and
publishes the model's identity on `GET /v1/models` (`kind`, `dim`,
`precision`, `revision`, `text_partner`, `space`). okfgraph only needs a
client for it.

### Settings (`okfgraph/settings.py`)

New keys in the `embedding` group, mirroring the text and rerank lanes:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `image_backend` | str | `embroider` | `embroider` (in process) or `external` |
| `image_endpoint_url` | str | `""` | http(s) URL of an embroider vision host; required for `external` |

- Validation (in the same block that checks `backend` / `rerank_backend`,
  around line 546):
  - `image_backend` must be one of the two values.
  - `external` needs an http(s) `image_endpoint_url`.
  - `image_endpoint_url` set with `image_backend='embroider'` is a
    `CONFIG_INVALID` ("external-backend setting — set
    embedding.image_backend='external' or drop it"), as for the other lanes.
- There is **no** `image_identity = declared` mode. A declared image lane
  could not prove which text model its vectors align with, and image
  vectors are only useful against their text partner. The host must be
  embroider and must answer `/v1/models`; otherwise the lane refuses to
  open (see below). This keeps the first version small; a declared mode
  can follow if a non-embroider vision host ever matters — that is a
  **stated long-term goal** (see H, "Long-term goal: vision on the GGUF
  lane"), not a closed door.
- `image_model_id` stays meaningful: when set, the host's `id` must equal
  it. `image_precision` likewise: when not `auto`, the host's
  `precision` must equal it.
- Wire the two keys through `to_router_kwargs` (around line 674) and
  the CLI/env/TOML layers like every other setting (they get
  `OKFGRAPH_IMAGE_BACKEND` / `OKFGRAPH_IMAGE_ENDPOINT_URL` and
  `--image-backend` / `--image-endpoint-url` for free from the
  `Setting` table; check `cli_signal()` picks them up).

### The engine (`okfgraph/components/external_vision.py`, new)

```python
class ExternalVisionEngine:
    def __init__(self, *, endpoint_url, identity, timeout=60.0, api_key=None): ...
    model_id: str            # identity["id"]
    dim: int                 # identity["dim"]
    precision: str           # identity["precision"]
    space: str | None        # identity["space"]
    text_partner: str        # identity["text_partner"]
    used_cuda: bool          # identity["device"] == "cuda"
    is_loaded = True
    def encode_image_bytes(self, data: bytes) -> list[float]: ...
    def encode_images_bytes(self, items: list[bytes]) -> list[list[float]]: ...
    def spec_report(self) -> dict: ...
```

- Reuse the probe and HTTP helpers from `external_embedder.py`
  (`probe_identity`, `_normalize_endpoint`, the `urllib` request with
  timeout and the optional API key header). Move whatever is shared into a
  small `components/_http.py` instead of copying it; the text and
  rerank engines keep their behaviour.
- Probe `GET /v1/models` at construction. Refuse (as `ValueError`,
  which the router turns into `CONFIG_INVALID`, like the text lane) when:
  - the endpoint is not embroider (no identity),
  - `kind != "vision"` ("serves a 'embed' model; image_endpoint_url needs
    a vision host"),
  - `image_model_id` is set and differs from `id`,
  - `image_precision` is not `auto` and differs from `precision`,
  - `dim` differs from the graph's `embedding_dim`. The image route
    has no `dimensions` parameter, so the host must be opened at the
    graph's width (`python -m embroider.serve --dim N`); name that in the
    remedy.
- `encode_image_bytes` posts `{"image": [base64(original bytes)]}` to
  `/v1/images/embeddings` and reads `data[0].embedding`. Original bytes,
  not resized RGB: the host owns the resize contract, which is the point
  of the lane. Batch up to 8 images per request (bounded by
  `serve.MAX_BODY`, 32 MiB; split by size, not only count).
- Errors: an HTTP error or unreachable host raises
  `ExternalEmbeddingError` (the type the text lane uses). Image import
  already catches per-image failures and falls back to the caption, so one
  bad image doesn't stop an import. A host that is down for the whole
  import must not silently turn every image into a caption, though:
  after the first connection error the engine marks itself broken and
  the import raises a typed `OKFError("EXTERNAL_UNAVAILABLE", ...)`, with
  the endpoint in `fields` and the remedy "start the vision host or set
  image_backend='embroider'". This is a new code in `okfgraph/errors.py`
  (none exists today; the text lane surfaces `ExternalEmbeddingError`
  untyped). Give the text and rerank lanes the same code in the same
  change.
- Width check on every answer (`len(vector) == dim`), like
  `ExternalEmbedder` at line 485.
- **Never import `embroider` in this module.** The text partner comes
  from the identity, not from `embroider.available_models()`.

### Router wiring (`okfgraph/router.py`, vision session factory, ~641–691)

Today: `_require_ort` → `enforce_vision_compat` → `loader.open(...)`,
then the image model and precision pins.

New branch at the top of the factory:

```python
if settings.image_backend == "external":
    engine = build_external_vision(endpoint_url=..., image_model_id=...,
                                   image_precision=..., embedding_dim=...)
    enforce_vision_compat_external(conn, text_model_id=..., embedding_dim=...,
                                   partner=engine.text_partner)
    enforce_image_model_pin(conn, engine.model_id)
    enforce_image_precision_pin(conn, engine.precision)
    enforce_image_space_pin(conn, engine.space, adopt=...)
    return engine
```

- **No `_require_ort` on this branch.** That is the test that proves the
  lane: with `onnxruntime` absent (or `sys.modules["onnxruntime"] = None`)
  an import with images through the external lane succeeds.
- `enforce_vision_compat` today calls `vision_text_partner()`, which
  imports embroider. Split it into the registry lookup and a pure check
  `_check_vision_compat(conn, text_model_id, embedding_dim, partner)`;
  the in-process branch keeps calling the registry, the external branch
  passes `engine.text_partner`. Same error codes
  (`VISION_INCOMPATIBLE`), same caption-fallback remedy.
- `LazyVisionEncoder` keeps wrapping the factory, so the probe happens on
  first image encode, not at router boot. Text-only imports never contact
  the vision host. Exception: `doctor` and `model-info` report the lane
  without opening it (they show the configured endpoint and
  `identity=unopened`).

### Image space pin (`okfgraph/components/embedding.py`)

- New `IMAGE_SPACE_META_KEY = "embedding_image_space"` and
  `enforce_image_space_pin(conn, space, *, adopt)`, the image twin of
  `enforce_space_pin` (line 313). Same rules:
  - A stored space plus a landing session in another space is refused
    with `IMAGE_PIN_MISMATCH`, naming the differing fields
    (`_space_differences`; `max_length` is always 0 for vision).
  - A graph with no image vectors re-pins.
  - With none stored, the space is recorded when the graph has no image
    vectors, or when the model and precision pins just agreed (`adopt`).
- The in-process lane gets the pin too. The embroider vision session
  already reports `space` (embroider ≥ 0.15 `Model.describe`). Read it
  off the opened session if the native object exposes it; otherwise the
  in-process lane passes `None` and checks nothing, as the text lane does
  for sessions that can't name their space.
- "Graph has no image vectors" check: `MATCH (i:ImageAsset) WHERE
  i.route = 'vision' RETURN i.id LIMIT 1`. Check the actual
  property names in the schema before writing it.

### The encoder seam (`okfgraph/components/image_assets.py`)

Today the import path is `_prepare_vision(img)` → `(rgb, h, w)` →
`_encode_vision_image(rgb, h, w)` → `ve.encode_image(rgb, h, w)`. The
external engine takes original bytes, so move the preparation behind the
encoder:

- `LazyVisionEncoder.encode_image_bytes(data)`: prepares (step C) and
  calls the session's `encode_image(rgb, h, w)`.
- `ExternalVisionEngine.encode_image_bytes(data)`: posts the bytes.
- `image_assets` calls `ve.encode_image_bytes(img.data)` only. The
  `_prepare_vision` call at line 117 exists to catch undecodable bytes
  early and fall back to a caption. Keep that early check as a cheap
  decode probe (Pillow `Image.open(...).verify()`, or `prepare_image` for
  the in-process lane), and catch the same `(ValueError, RuntimeError)`
  around the encode call too, so a host answering 400 for a bad image
  also falls back to the caption.
- `_vision_precision()` (line 217) reads the engine's `precision` when the
  lane is external. It must not call `embroider.cuda_available()`
  then.
- `vision_content_hash` stays the same. The external lane hashes
  `(image model id, host precision, VISION_CONTRACT, bytes)`. Equal hashes
  across lanes mean an in-process graph can switch to the external host
  of the same model and precision with no re-embed. That is intended,
  and the space pin guards it.

### Ops and reports

- `ops/admin.py` `lanes()` gains an `image` lane (`model_id`,
  `endpoint` or in-process, `identity`, `precision`, `space`). `doctor`
  then prints `image_lane` with no further change (line 188 loops over
  `lanes()`).
- `model-info` shows the image pins and the image space.
- MCP `doctor` / `model_info` tools pick it up through the same ops.

### Tests (`tests/test_external_vision.py`, new)

Use the fake-host pattern from `tests/test_external_identity.py` (a
`ThreadingHTTPServer` on port 0 answering `/v1/models` and the route).

- Settings: `image_backend` validation; `image_endpoint_url` without
  `external` is `CONFIG_INVALID`; env/TOML/CLI precedence for both keys.
- Probe refusals: not embroider, `kind='embed'`, model id mismatch,
  precision mismatch, dim mismatch, each with its message and remedy.
- Encode: posts base64 of the original bytes (assert on the fake's
  received body); batches split by size; width check.
- Import end to end with a fake vision host returning fixed vectors, and
  `onnxruntime` blocked in `sys.modules`: assets land with
  `route='vision'`, the pins and the image space are written, and
  `embroider._native` was never imported (assert on `sys.modules`).
- Re-import reuses (hash equal), no second POST.
- Host down mid-import → `EXTERNAL_UNAVAILABLE`, nothing half-written for
  that concept.
- A host answering 400 for one image → that image falls back to the
  caption, the rest embed.
- Compat: a graph whose text model isn't the host's `text_partner` →
  `VISION_INCOMPATIBLE` with the caption remedy.
- Space pin: stored space differs → `IMAGE_PIN_MISMATCH` naming the field;
  empty image set re-pins; pre-0.14 graph adopts.
- `doctor` shows `image_lane` for both lanes.
- Image search (`--target images`, text query): unchanged path. One test
  that it still ranks the fake vectors.

### Docs

- README: the external lanes section gains the image lane, with the host
  command (`python -m embroider.serve --kind vision --model <id> --dim
  <graph dim>`; check the flag spelling against `serve.open_model`).
- `docs/harness-integration.md`: "all three lanes external" is how a
  host that must not load ONNX runs okfgraph.
- Skill `okfgraph-cli`: the doctor row names `image_lane`.
- CHANGELOG `[0.14.0]`.

Done when: an import with images works end to end with `onnxruntime`
absent from the process, and the suite is green.

---

## C. In-process image prep on `embroider.prepare_image`

`_prepare_vision` resizes with Pillow (`prepare_vision_rgb` plus
`vision_target_size` / `gemma_vision_target_size`), duplicating the
contract embroider now owns in `prepare_image(data, model_id)`. Two copies
of a resize contract drift.

1. Equivalence first: a test that runs both on a fixed set of images
   (JPEG, PNG with alpha, palette PNG, tiny, very wide, very tall, EXIF
   rotated) for both slots (jina omni, gemma vision) and compares the RGB
   buffers and sizes byte for byte.
2. If they match: `LazyVisionEncoder.encode_image_bytes` calls
   `embroider.prepare_image`; delete `prepare_vision_rgb`'s resize half
   and the per-slot `target_size_fn` branch (lines 252–270).
   `VISION_CONTRACT` stays, so no re-embed.
3. If they differ: decide which is right with the embroider maintainer
   (embroider's is the reference the host uses). Switch anyway and bump
   `VISION_CONTRACT`. Hashes then miss once and image vectors re-embed on
   the next import (captions don't). Say so in the CHANGELOG.
4. Keep Pillow only for the early decode probe, or drop it if
   `prepare_image` raises `ValueError` on undecodable bytes (check).

---

## H. Caption fallback covers the encode call (declared-lane hardening)

Ships with B and C in 0.14.0. It is the declared (GGUF) lane's half of
B's robustness, and it must land in the same release: B fixes the
external-embroider branch of the image lane, while the failure below is
the *declared* branch's, and a `mode=omni` import on a GGUF graph would
otherwise keep dying exactly as it does today.

### The defect (measured on 0.13.1)

The caption fallback wraps preparation only (`image_assets.py:139`,
`except (ValueError, RuntimeError)` around `self._prepare_vision(img)`).
The encode call (`:165`, `self._encode_vision_image(rgb, rh, rw)`) sits
outside any try, and every open-side refusal is a `StateError` — which
subclasses `RuntimeError`, so it *would* be caught if the try reached
fifteen lines further:

- `NO_ORT_RUNTIME` (no runtime in the process),
- `VISION_INCOMPATIBLE` (the graph's text model is not the vision
  model's `text_partner` — always true on a declared lane, whose model
  has no partner at all),
- `IMAGE_PIN_MISMATCH`,
- `ExternalEmbeddingError` from a vision host (B).

Reproduced against a llama.cpp-shaped fake host with
`backend='external'`, `embed_family='nomic-v1.5'`, a 640×480 PNG and
`mode=omni`:

```
RAISED: StateError VISION_INCOMPATIBLE | image-content search needs a
jinaai/jina-embeddings-v5-text-nano-retrieval graph; this graph uses
owner/text-model — use mode=text (captions)
```

So the error names a remedy the code does not take. What the user sees
then depends on the caller: a warning plus that concept's images dropped
(`import_.py:1184`, bundle import), silently zero images
(`import_.py:1534`, `except Exception: pass`), or a hard error
(`import_.py:2258`, `okf ingest`). No test combines an external backend
with images, which is why this survived 0.8 → 0.13.

### What changes

The caption decision moves out to cover the whole vision attempt, and
the vision hash is computed only for an image that actually embeds as
vision:

```python
if route is EmbedRoute.VISION:
    try:
        embedding = ve.encode_image_bytes(img.data)   # prepare + encode
        content_hash = vision_content_hash(ve.model_id, self._vision_precision(), img.data)
    except (ValueError, RuntimeError) as exc:
        self._note_vision_fallback(exc)               # one summary line per import
        route = EmbedRoute.TEXT
        caption = img.alt_text if img.has_alt_text else fallback_caption(...)
        content_hash = self._content_hash(route, caption.encode("utf-8"))
        embedding = self.embed_engine._encode(caption or img.filename, task="Document")
```

Three details that are easy to get wrong:

1. **Hash ordering.** The vision hash must not be stored for a row whose
   vector is a caption. Computing it inside the try (after a successful
   encode) keeps `content_hash` and `embedding` describing the same
   thing; the reuse check then still works, and a later import on a
   working lane re-embeds by hash miss.
2. **One warning, not N.** Record the first failure's message and the
   count; log a single line at the end of `_ingest_concept_images`
   (`"%d image(s) embedded as captions: %s"`). A hundred-image document
   on a declared lane must not write a hundred identical warnings.
3. **A dead host is not a per-image failure (B's rule).** Only open and
   per-image refusals degrade to captions. A connection failure stays
   `EXTERNAL_UNAVAILABLE` and aborts the import — silently turning every
   image into a caption because the host is down is the failure mode B
   exists to prevent.

### Tests

- Fake external vision host, `mode=omni`, bytes-bearing alt-less image
  → captions land, `route='text'`, no exception, one summary warning.
- The same through each caller (bundle import, single file, `okf ingest`)
  → identical outcome, no swallowed-exception path still hides one.
- Hash correctness: the stored `content_hash` equals the text hash, and
  re-importing while the lane is broken reuses (no churn).
- In-process lane with `sys.modules["onnxruntime"] = None` → captions,
  not `NO_ORT_RUNTIME`.
- Vision host unreachable mid-import → `EXTERNAL_UNAVAILABLE` (B), not a
  caption for every image.
- `mode=omni` on a graph whose text model *is* a valid partner → still
  `route='vision-onnx'` (the fallback must not swallow a healthy lane).

### Long-term goal: vision on the GGUF lane

The end state is deliberately **not** "declared lanes are caption-only
forever". The target is a declared image lane as well:

> A graph whose text model comes from a GGUF host should be able to embed
> images through a declared vision host, exactly as it embeds text
> through a declared text host.

What that requires, and why it is not in 0.14.0:

- **A declared partner.** Image vectors live in the text model's space
  (one `ImageAsset.embedding` column, one `image_omni_idx`), so a
  declared vision lane needs an explicit `image_text_partner` claim —
  "these two models were trained together" — in place of the registry's
  `text_partner`. Without it, mixing vectors is silent garbage.
- **A declared space.** The image space pin (`IMAGE_SPACE_META_KEY`, B)
  is cheap once a host names its `space`; a declared lane must supply
  the string itself, and the pin then compares declarations, not proofs.
- **A host route.** `POST /v1/images/embeddings` answered by a non-
  embroider host (llama.cpp's multimodal / mtmd path, a TEI vision
  endpoint), with the resize contract declared by whoever serves it.
- **The honest failure mode.** Where B refuses a vision lane it cannot
  verify, the declared lane accepts it with `identity=declared` and the
  same fail-closed pins over the declared values.

Until that lands, H makes the declared lane *safe* (captions, one
warning, correct hashes) rather than *broken*, which is the whole point
of shipping it now.

---

## D. Typed conversion errors

Today a bobine failure inside `BobineConverter.convert` (a corrupt PDF, an
unsupported file) surfaces as `INTERNAL` and writes a crash log, which
reads like an okfgraph bug.

- New error code `CONVERSION_FAILED` (exit 1, a state/outcome error) in the
  error table, with `fields={"path": ..., "converter": "bobine"}` and the
  bobine message as text.
- `BobineConverter.convert` wraps `bobine.ingest_document` and maps
  bobine's exception types. Check which ones the Python binding raises
  (`ValueError`, `OSError`, `RuntimeError`, a custom `bobine.Error`?) and
  map those only. Anything else stays `INTERNAL`.
- A missing file stays `FILE_NOT_FOUND`, checked before calling bobine.
- `_ingest_many_pdfs` needs no change: it already reports the per-file
  error code.
- Tests: a fake converter raising each mapped type; a real corrupt PDF
  (a few bytes of `%PDF-`) behind the `pdf` marker.

---

## E. bobine device/precision vocabulary (O6, part 1)

**Landed in 0.14.1.** Settings `pdf_device` / `pdf_precision` (validated
against the local `DEVICES` / `PRECISIONS` tuples, pinned to bobine's exports
by `tests/test_pdf_converter.py::test_vocabulary_never_drifts_from_bobine`),
`BobineConverter(..., device, precision)`, the `pdf` floor at `bobine>=0.9`,
and the resolved plan in `converter_status()` / `okf model-info --converters`
(and the doctor’s `pdf_converter` line). One deliberate deviation: the
settings table validates with local tuples instead of reading bobine at load
time — settings load on machines without the pdf extra, but the drift test
(and the plan row) are the guard.

**Unblocked: bobine 0.9.0 is published** (crates.io + PyPI, 2026-11-11,
tag `v0.9.0`). Raise the `pdf` extra floor to `bobine>=0.9` as part of this
step. What 0.9 exposes, checked against its docs and code:

- `ConverterConfig(device="auto|cpu|cuda", precision="auto|fp32|fp16|int8")`
  — parsed through embroider's `DeviceReq`, so the words mean what they
  mean in embroider and okfgraph. `q8` is refused by name (bobine has no
  q8 artifact).
- `bobine.DEVICES` / `bobine.PRECISIONS` — the tuples to validate against
  (read them; no local copy).
- `HybridConverter.describe()` → `{device_requested, precision_requested,
  passes: {encoder, decoder, layout, ocr, table, potatr: {providers,
  policy, precision, landed}}}`; `providers` is `None` for an auto-CUDA
  slot before it opens.
- `bobine.runtime_info()` → `{dylib, source, version, set_by, cuda_dirs}`
  (useful for the converter status row when bobine is installed).

- Settings in the `ingest` group: `pdf_device` (default `auto`) and
  `pdf_precision` (default `auto`), validated against bobine's
  vocabulary. Read the allowed values from bobine if it exposes them,
  else from a local tuple that a test checks against bobine. **bobine 0.9
  will expose them** (`bobine.DEVICES`, `bobine.PRECISIONS`, per its
  `docs/plan-0.9.md` §3), so read the source and keep no duplicate tuple.
  The precision words are `auto|fp32|fp16|int8` — bobine has no `q8`
  artifact, so don't offer one.
- `BobineConverter(routing_mode, extract_images, device, precision)`,
  passed into `ConverterConfig`.
- `converter_status()` (line 74) reports the resolved device and
  precision once bobine reports them — from **`describe()` on the
  converter**, not from `bobine.model_status()`, which is cache status and
  stays cache-only. `describe()` is answerable before the first
  conversion (requested filled, landed `None`) and after.
- The `pdf` extra floor becomes `bobine>=0.9`. That is okfgraph's own
  extra, so it's okfgraph's call. Weave's floors are raised separately,
  in Weave.
- Tests: settings validation; the config receives both values (fake
  `bobine` module).

---

## F. Warm PDF converter with a bounded lifetime (O6, part 2)

**Landed in 0.14.1.** `BobineConverter` holds one `HybridConverter` and calls
`ingest_document`; `threading.Lock` serialises conversions; `pdf_idle_seconds`
arms one daemon timer per held converter (0 = release after each call);
`OKFRouter.close()` releases it (CLI, serve, shell all reach it through
`_close_routers` / `serve`’s `finally`, so no `with router.pdf_converter()`
was needed); a knob change closes the warm converter instead of holding two;
`okf doctor` reports warm/released, the resolved words, the last conversion
and the pending release as an informational `pdf_converter` row.

**Unblocked: shipped in bobine 0.9.0.** `HybridConverter.ingest_document(path,
output_dir, on_page=, should_continue=)` reuses the sessions it already
opened; `close()` (also `with`, and dropping the object) releases them and
is idempotent. Two contracts from the bobine side that F has to respect:

- A closed converter **refuses** work (`RuntimeError: converter was closed;
  construct a new HybridConverter`) instead of silently rebuilding — so
  okfgraph's `close()` then rebuild on key change is exactly right.
- One document at a time per converter; a concurrent call raises rather
  than waiting (bobine cannot block on a mutex while holding the GIL), so
  okfgraph's `threading.Lock` is load-bearing, not redundant.
- `describe()` keeps answering after `close()` (`landed` goes false), so
  the `pdf_converter` doctor row can render at any time.

### The rule (confirmed)

- **CLI:** one converter for the files of one command, released when the
  command ends. `okf ingest --kind pdf --pdf-path a.pdf b.pdf c.pdf` loads
  the layout/OCR models once, not three times; nothing outlives the
  process anyway.
- **`okf serve` / `okf-mcp` / `okf shell`:** keep the converter after an
  ingest, release it after `pdf_idle_seconds` without use (new setting in
  the `ingest` group, default 300; 0 = release after every call).
  Long-running hosts must not hold layout and OCR models indefinitely.

### Code

- `BobineConverter` gains a lazily built `self._hybrid`, keyed on
  `(routing_mode, extract_images, device, precision)`. A key change
  releases and rebuilds. A `threading.Lock` guards build/use/release:
  serve answers on several threads, and one bobine converter must not
  run two documents at once.
- `convert()` uses `self._hybrid.ingest_document(...)`; `close()` releases
  it. `__enter__/__exit__` call `close()`.
- `last_used` timestamp updated after each convert. A
  `threading.Timer` (or a check at the start of each serve request plus a
  daemon sweep, whichever `okf serve` already has a hook for) calls
  `close()` once `now - last_used >= pdf_idle_seconds` and the lock is
  free. The timer is cancelled and re-armed per use; no thread runs
  while no converter is held.
- The router owns one `BobineConverter` (today `_ingest` builds one per
  call). `OKFRouter.close()` closes it, so `okf shell` exiting and serve
  shutting down both release.
- CLI: `_ingest_many_pdfs` and the single-file path run inside
  `with router.pdf_converter():` so the release happens even on errors
  and Ctrl-C.
- `doctor` / `model-info` info row: `pdf_converter` held or idle, and
  for how long.

### Tests

- Fake bobine with a counting `HybridConverter`:
  - three files in one CLI command → one construction, one close;
  - a failing file in the middle still closes at the end;
  - a key change (another routing mode) → close and rebuild;
  - idle eviction with `pdf_idle_seconds=0.1` and a fake clock (no
    sleeps in tests; inject the clock);
  - `pdf_idle_seconds=0` releases after each call;
  - concurrent converts serialise (two threads, the fake records
    overlap);
  - router `close()` releases.
- One real bobine test behind the `pdf` marker: two PDFs in one command,
  converter built once (spy on the class).

---

## G. Open questions (G1, G2 decided in 0.14.2; G3 a design note)

1. **Should `okf-mcp` / `okf serve` keep creating a missing database on
   first boot?** Today they do, unlike CLI read commands. A mistyped
   `OKFGRAPH_DB_PATH` in an MCP config then silently starts an empty graph,
   and the agent sees "no results" instead of an error. Options:
   - (a) keep creating, but log the created path at WARNING and put
     `created: true` in the first `doctor` / health answer;
   - (b) refuse unless `--create` / `OKFGRAPH_CREATE=1`.

   **Decided, 0.14.2: (a), plus the agent-facing half.** The server still
   creates the database; while that new graph is empty, every MCP tool
   envelope (okf-mcp and okf serve's `/mcp`) carries a "graph created at …
   still empty" warning, `/health` reports `created`, `okf doctor` leads
   with a `db_created` row, and boot logs a WARNING. (b) stays a 1.0
   question.
2. **Folders as `--pdf-path` input.** **Decided, 0.14.2:** a folder
   expands to the convertible files directly inside it (no recursion) and
   `*`/`?` patterns expand in-process (PowerShell does not glob), filtered
   to bobine's extensions, Office owner (`~$…`) and hidden files skipped,
   duplicates converted once; nothing matched is `FILE_NOT_FOUND`. CLI
   only — MCP ingest stays one file per call.
3. **Declared image identity.** Deferred out of 0.14.0 by decision (B:
   "no `image_identity = declared` mode"), then re-opened deliberately:
   llama-server can embed images through `/v1/embeddings` once loaded with
   `--mmproj`, so a GGUF graph needs no embroider pairing at all and a
   same-model declaration is true by construction. Shape and open questions
   live in **`docs/plan-declared-image-identity.md`**; no code until a
   current build passes the two-image probe and one GGUF model + mmproj is
   chosen. Until then H keeps the declared lane safe (captions, one warning,
   correct hashes).

---

## Local housekeeping (not code, no release)

- Repair `D:/User/Documents/Python/OKFgraph/.venv` once its `okf-mcp` is
  stopped:
  `uv pip install --python .venv/Scripts/python.exe --reinstall
  "onnxruntime-gpu[cuda,cudnn]==1.31.1" "embroider==0.15.0"`.
- Delete the crash logs from the 0.13.1 investigation:
  `%LOCALAPPDATA%/okfgraph/crash/okf-20261111-19370*.log`.
