# OKF v4 "Unified Omni" — Implementation Notes

Implements multimodal image ingestion on top of the existing LadybugDB +
Jina-v5 knowledge graph, with a user-selectable **mode** that controls when the
heavy multimodal model is used.

## Image ingestion modes

Selected with `--mode` on `import` (default `text`):

| Mode       | Image **with** alt-text          | Image **without** alt-text                |
|------------|----------------------------------|-------------------------------------------|
| `text`     | text-embed(alt-text)             | text-embed(`filename + image-number`)     |
| `optional` | text-embed(alt-text)             | **omni**-embed(image bytes)               |
| `omni`     | **omni**-embed(image bytes)      | **omni**-embed(image bytes)               |

Both encoders write into **one** vector space / index (`image_omni_idx` on
`ImageAsset.embedding`). This is supported because jina-embeddings-v5
text-small-retrieval and omni-small-retrieval share a vector space — you can
index with text and query with image (or vice-versa) without reindexing.

`text` mode never loads the omni model. The omni model is **lazy-loaded** on
first actual use (vision tower only, `modality="vision"`), so text-only
pipelines pay none of its cost.

When `omni` is requested but the raw bytes are unavailable (e.g. an
`http(s)://` URL, which is not fetched unless `--allow-remote-images` is set, or
a missing local file), the plan **degrades gracefully** to the text path
(alt-text, else filename fallback) so ingestion never hard-fails.

## Two correctness changes that REQUIRE re-importing existing databases

1. **Pooling fix (mean → last-token).** Jina-v5 uses *last-token* pooling; the
   previous `_encode` used *mean* pooling, which puts text vectors in a
   different space than the omni image vectors — silently breaking the unified
   index (and degrading concept search quality). Fixed in `_encode`, with
   truncation now followed by L2 re-normalisation.

2. **Default embedding dimension 384 → 512.** 384 is not an official Matryoshka
   dimension for these models; 512 is (and matches the v4 architecture). A
   warning is emitted for any non-Matryoshka dimension. The valid set is
   `{32, 64, 128, 256, 512, 768, 1024}`.

Because vectors and the fixed-length `embedding FLOAT[dim]` column both change,
**existing `okfgraph.db` files must be re-created and re-imported.** Doing the
pooling fix and dimension bump together means a single re-import covers both.

## Schema additions

- `ImageAsset` node table: `id, file_name, mime_type, alt_text, caption,
  embed_route, content_hash, data BLOB, embedding FLOAT[dim]`.
- `INCLUDES_ASSET` rel table: `Concept -> ImageAsset`.
- `image_omni_idx` cosine vector index on `ImageAsset.embedding`.

`embed_route` / `caption` record how each asset was embedded (provenance).
`content_hash` is a reuse key: on re-import, unchanged images are **not**
re-embedded (important for the costly omni path), and images removed from a
document are pruned.

## New API / CLI surface

- `OKFRouter.import_from_okf(path, mode=...)` and
  `OKFRouter.import_bundle(..., mode=...)`.
- `OKFRouter.search_images_with_text(query, use_text_model=True, limit=10)` —
  text→image search over the unified index. Defaults to the lightweight text
  model for the query (no omni load needed to search).
- `OKFRouter.list_images(concept_id)` / `get_image_data(asset_id)`.
- CLI: `okf import ... --mode {text,optional,omni} [--allow-remote-images]`,
  `okf search-images <query> [--use-omni]`, plus `--omni-model-id`. Interactive
  shell gains `import/import-bundle ... [mode]`, `search-images`, and `images`.
- `images.py` exposes `IngestMode` (also re-exported from the package).

## Verification performed

- `py_compile` passes for every package file.
- `tests/test_images.py`: **10/10 pass** (mode routing for all three modes incl.
  graceful fallbacks, markdown extraction order/alt-text, mime sniffing,
  deterministic asset ids + `okf-asset://` passthrough, local + inline-data
  resolution).
- The full model + LadybugDB pipeline was **not** executed here (torch,
  optimum, sentence-transformers, and ladybug are not installed in the build
  sandbox). Logic that does not depend on those was unit-tested directly.

## Scoped out (clean follow-ups, intentionally not done to avoid destabilising
the tested concept path)

- **Concept temporal dual-tracking** (`created_date`/`modified_date` internal,
  single `timestamp` on export). The existing single-`timestamp` behaviour is
  unchanged; ImageAsset likewise uses `content_hash`-based change detection
  rather than timestamps.
- **`okf-asset://` link rewriting on ingest.** Concept bodies are stored
  verbatim (markdown round-trips), and asset ids are derived deterministically,
  so rewriting can be added later without a data migration.
