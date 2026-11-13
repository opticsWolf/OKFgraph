# Declared image identity for GGUF text graphs — design note

Status: **shape decided, no code.** This note replaces the earlier
"embroider vision pairing" sketch: a GGUF graph can keep both lanes on
llama.cpp, so no pairing table and no alignment test are involved.

Related: `docs/implementation-plan-0.14.md` §G3 (decision), the image-lane
gateway invariant in §H (one `ImageAsset.embedding` column, one index, one
space).

## What changed

The earlier proposal assumed llama.cpp could not embed images and paired a
GGUF text lane with an embroider ONNX vision model. The pair would then need
a *learned* shared space and an alignment measurement.

llama-server can embed images itself. The current server README documents it:

- `POST /v1/embeddings`: for models loaded with `--mmproj`, each element of
  `input` may be an object with a `content` array using the chat-completions
  parts — `{"type": "text", "text": "..."}` and
  `{"type": "image_url", "image_url": {"url": "..."}}` (remote URL, base64
  data URI, or `file://` with `--media-path`). **Each object yields one
  embedding.**
- Requirements: `--embedding` (or `--embeddings`) and a pooling type other
  than `none` (`--pooling`), plus the model's image component (`--mmproj`).
- `/models` (and `/v1/models`) report a `multimodal` capability; the README
  tells clients to check it before sending multimodal input.
- Local check: `llama-server` build 12110 (commit `cd959bae9`) exposes
  `--embedding`, `--mmproj`/`--mmproj-url`, `--pooling`, `--media-path`.

So a multimodal GGUF model can produce text *and* image vectors in one
runtime. If it is the same model the graph already embeds text with, the two
kinds of vectors are in one space **by construction** — the property that
made the pairing idea expensive.

## Shape

| Setting | Value |
|---|---|
| `image_backend` | `"external"` (exists) |
| `image_identity` | `"declared"` (new; today only embroider `auto` is possible) |
| `image_endpoint_url` | the llama-server base URL |
| `image_model_id` | the declared model name — must equal the graph's text `model_id` |

The declared lane is only valid when the image model **is** the graph's text
model: same GGUF, same image component, one server or two servers loading
the same pair. Nothing weaker is accepted; a declared model that differs
from the text lane stays refused with captions as the remedy.

## Checks (all fail closed)

1. **Name agreement.** The graph declares `image_model_id == model_id`, and
   the host's `/v1/models` reports that same name. llama-server names its
   model from `-m`/the GGUF, which usually differs from a hub-style id, so
   the remedy is explicit: start it with `--alias <model_id>` (llama-server
   supports aliasing the served name). A host that reports another name is
   refused before any image is sent.
2. **Capability.** `/v1/models` must report the model as multimodal. This
   is a host claim, not a proof, but it catches the common mistake of
   pointing at a text-only server.
3. **Two-image probe.** At startup, embed two different built-in test
   images through the same route. If the vectors are identical, refuse the
   lane (`EXTERNAL_UNAVAILABLE`-style refusal with the caption remedy).
   This is the safety net for
   [llama.cpp discussion #13666](https://github.com/ggml-org/llama.cpp/discussions/13666):
   llama-server used to return one embedding for every image. `#15108`
   changed the `/embeddings` prompt shape (the non-OpenAI route), and later
   comments report EmbeddingGemma-2 working on recent builds — but that is
   *not* confirmed for `/v1/embeddings` on a current release. The probe
   makes the question moot at runtime: a broken host is refused, a working
   one is accepted, and the answer is cached per endpoint.
4. **Vector normalization.** Image vectors are truncated to
   `embedding_dim` and L2-normalized exactly as the text lane does before
   they are stored. Same space means same dimension *and* same scaling;
   skipping this would silently mis-rank images against text.

## Pins: honest, not stronger

`embedding_image_model` and `embedding_image_precision` keep working, keyed
on the declared id. The space pin (`IMAGE_SPACE_META_KEY`) does **not**
record an `embroider/...` space for a declared lane — `parse_space` only
understands embroider's proof format, and a llama.cpp host proves
nothing. A declared lane therefore has no space string to compare, and the
pin stays unset rather than pretending; the model/precision/dimension pins
plus the probe are the protection. This is the honest limit of
`identity=declared`, and it is why it is only allowed for the same-model
case.

## Fallback

Everything else keeps today's behaviour: `mode=text` captions, one warning
per run naming the remedy (`use mode=text (captions)`, or point at an
embroider vision host). A graph sized for a GGUF text model is not
re-embedded by adding images.

## Open questions — confirm on a current llama.cpp build

1. **Which GGUF models ship an embedding-quality image component.** The
   discussion uses `vdr-2b-multi-v1` (Q5_K_M + `mmproj-Q8_0`); EmbeddingGemma-2
   is reported working on a pre-release build (Oct 2026). Nothing here is a
   recommendation yet — pick one, document its mmproj, verify its license.
2. **Does `/v1/embeddings` merge text and image into one vector for a mixed
   content element** (e.g. a caption + the image), or is each part separate?
   The README says one embedding per element; the probe should also compare
   a text-only element against an image+caption element to see how the model
   actually composes them.
3. **Is the identical-vector bug fixed on `/v1/embeddings`?** The probe
   answers it at runtime; a fixed build is worth recording in the docs.
4. **Pooling and normalization defaults.** Embedding models need
   mean/cls pooling and Euclidean normalization; confirm the GGUF's default
   `--pooling` matches the text lane's and that `--embd-normalize 2` is in
   effect.

## Decision

Write the code only after (1) a current build passes the two-image probe,
(2) one GGUF model + mmproj is chosen with its license checked, and (3) the
`--alias` name check works against `/v1/models`. Then the change is small:
one settings row, a declared branch in the external vision engine (identity
from the declaration, probe at open), the pins above, and tests with a fake
llama-server — no embroider work and no alignment study.
