# Bench — EmbeddingGemma 2 precision, width, and database size

Measured 2026-11-14 on the RTX 3090 machine, against 20 real OKFgraph
databases built from a real-text corpus. It answers three questions the
precision/width discussion kept guessing at:

1. Does int8 (q8) cost retrieval quality, and does it change the graph
   file? (No, and no.)
2. What does a narrower MRL width actually save in the file? (26% at
   256d; nothing more at 128d.)
3. Is the fp16 precision pin safe for this model? (Yes — 6.5e-7 mean
   cosine drift, identical rankings.)

Companion documents: `embroider/docs/plan-eg2-int8-audio.md` (the plan
this is evidence for) and `embroider/docs/eg2-int8-audio-results.md`
(same tables plus the audio/int8 lane context).

## Setup

- **Corpus**: 1,500 long Wikipedia articles — 50 curated topics fetched
  through the REST API, 1,450 from `wikimedia/wikipedia` 20231101.en —
  capped at 4,000 chars and split into ~1,000-char chunks.
  **5.2 M chars, 3,477 chars/doc, 3.72 chunks/doc.**
- **Queries**: 50 hand-written questions, one target article each
  ("which planet is closest to the Sun" → Mercury (planet)); the other
  1,450 articles are distractors.
- **Prompts**: the publisher's Gemma contract, identical in every tier:
  `task: search result | query: …` and `title: {title} | text: …`.
- **Tiers**: `fp32` and `fp16` through embroider (CUDA); `int8` (q8),
  `q4` and `q4f16` through the publisher's `model_quantized`,
  `model_q4` and `model_q4f16` graphs on ONNX Runtime 1.31 CUDA
  (embroider's `prepare_cuda_runtime()` loads cuDNN).
- **Retrieval**: exact cosine over concept vectors, so the numbers
  isolate the tier and width. The 20 DBs also carry real HNSW + FTS
  indexes and were spot-checked through `QUERY_VECTOR_INDEX`.
- **DBs**: real OKFgraph schema (`SchemaManager`), 1,500 concepts +
  5,587–5,623 chunks, one per tier × width (768/512/256/128).

## Precision parity (same documents, 768d)

| Tier | mean 1−cos | p99 | max |
|---|---|---|---|
| fp16 | 6.5e-7 | 1.7e-6 | 3.9e-6 |
| int8 (q8) | 8.5e-4 | 2.9e-2 | 6.6e-2 |
| q4 | 2.2e-2 | 4.7e-2 | 8.5e-2 |
| q4f16 | 2.2e-2 | 4.7e-2 | 8.4e-2 |

fp16 is effectively exact for these vectors; int8 is ~25× closer to
fp32 than the 4-bit tiers.

## Retrieval (50 queries)

| Tier | 768 R@1 / R@5 / NDCG | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | .88 / 1.0 / .956 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .80 / .96 / .905 |
| fp16 | .88 / 1.0 / .956 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .80 / .96 / .905 |
| int8 | .88 / 1.0 / .956 | .88 / 1.0 / .956 | .84 / 1.0 / .941 | .82 / 1.0 / .929 |
| q4 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .84 / 1.0 / .941 | .82 / .98 / .918 |
| q4f16 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .84 / 1.0 / .941 | .80 / .98 / .911 |

Ranking agreement with fp32@768 (top-1 / top-10 overlap):

| Tier | 768 | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | 1.00 / 1.00 | 1.00 / 0.87 | 0.94 / 0.76 | 0.92 / 0.50 |
| fp16 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| int8 | 1.00 / 0.88 | 1.00 / 0.83 | 0.96 / 0.75 | 0.94 / 0.53 |
| q4 | 1.00 / 0.80 | 0.98 / 0.78 | 0.96 / 0.73 | 0.94 / 0.55 |
| q4f16 | 1.00 / 0.81 | 0.98 / 0.78 | 0.96 / 0.72 | 0.92 / 0.55 |

One query is 2 points: only gaps above ~4–6 points mean anything here.
Within that: **width moves retrieval more than precision**; int8 is
indistinguishable at 768/512; q4/q4f16 lose a step at 512/256 and drift
10–20× further from fp32 than int8.

## Database size (real schema, 1,500 concepts, ~5,600 chunks)

| Tier | 768 | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | 79.7 MB | 73.5 MB | 59.1 MB | 59.0 MB |
| fp16 | 80.4 MB | 73.6 MB | 59.1 MB | 59.0 MB |
| int8 | 80.0 MB | 73.6 MB | 64.3 MB | 59.1 MB |

The int8/q4/q4f16 DBs have 36 more chunks (0.6%): the corpus file was
rewritten between the fp32/fp16 and quantized runs. The concept documents
are identical (fp16 parity 1e-6), so retrieval is unaffected.

Composition (`db-size` reporter):

| Part | 768 | 256 | 128 |
|---|---|---|---|
| File | 76.0 MB | 56.3 MB | 56.2 MB |
| Vector columns | 20.8 MB | 6.9 MB | 3.5 MB |
| Text | 9.6 MB | 9.6 MB | 9.6 MB |
| Indexes / pages | ~45 MB | ~40 MB | ~43 MB |

**A precision tier does not change the file.** Re-embedding under int8
keeps a `FLOAT[dim]` column, so the vector payload is byte-identical;
only the values move. **768 → 256 saves 26%** of the file; **256 → 128
saves nothing measurable**, because HNSW/FTS and page overhead do not
scale with width (the vector column halves, 6.9 → 3.5 MB, into a ~40 MB
index/text floor).

An earlier synthetic dummy (2,000 concepts, 20,000 chunks, 300 images)
showed the same shape with image bytes in the mix: 260.5 MB at 768 vs
214.1 MB at 256 (−18%), with vectors 25% → 10% and image BLOBs ~25% of
the file at both widths. Image storage policy was not changed.

## What this means for okfgraph

- **The fp16 precision pin is safe** for this model: 6.5e-7 mean drift
  and identical top-10 rankings. The existing `fp16`/`fp32` pin codec
  needs nothing here.
- **int8 embeddings have no database-size story** (same column, same
  file) and their quality is within noise at 768/512 on this 50-query
  set. The blocker remains the pin: `_PRECISION_CODE` has only
  `fp16: 16` and `fp32: 32`, so an int8 graph would carry no precision
  pin and could be reopened under another precision. That codec change
  is still the prerequisite, and a labelled query set is still the gate.
- **256d is a real, supported lever** for a fresh import (a width change
  is a new database, not a re-embed): 26% smaller file, ~1–2 points of
  R@1 on this set, officially ~98.5% text / ~95% multimodal quality.
  128d saves no additional file space and costs 4–8 points — not worth
  it unless a downstream index (not this one) shrinks with width.
- **The file is indexes and text, not vectors.** At 3.5 KB/doc, vectors
  are 27% at 768d. Storage-side quantization of vectors would only make
  sense after a measurement shows vectors dominate (a chunk-heavy graph,
  not this one), and it is an index-format project, not a precision
  setting.
- **`okf db-size` is worth having** as an option: the reporter used here
  prints row counts, vector bytes by table, BLOB bytes and largest
  assets, text bytes, index/page overhead, and the same at a candidate
  width. It answers "does 256d pay for this graph" without building one.

## Throughput (RTX 3090, docs/s)

| Tier | docs/s | chunks/s | queries/s |
|---|---|---|---|
| fp32 | 38.4 | 43.5 | 36.7 |
| fp16 | 33.2 | 45.8 | 40.9 |
| int8 | 38.6 | 162.5 | 740.3 |
| q4 | 35.0 | 134.0 | 380.4 |
| q4f16 | 56.7 | 215.2 | 240.8 |

The quantized graphs run on the CUDA EP with cuDNN loaded. The same int8
graph on CPU was ~0.6 docs/s (≈60× slower), so "int8 is a CPU tier" is a
portability statement, not a hardware one.

## Caveats

- 50 queries with one target each: a 2-point difference is one query.
  This is a signal, not a shipping gate; a labelled set (the cached BeIR
  suites are a candidate) is the next step.
- Exact cosine, not HNSW recall: the 20 DBs have indexes and were
  spot-checked, but approximate-search recall is not measured across the
  matrix.
- One run, one machine; fp32/fp16 and quantized embeds ran in
  overlapping jobs, so the throughput column is indicative.
- Text tower only; no image or audio vectors.
- The raw vectors, DBs and scripts are in the temporary bench
  (`%TEMP%/okf-eg2-bench/`): `RESULTS.md`, `corpus.jsonl`,
  `queries.json`, `emb_<tier>.npz`, `db/`, `results.json`,
  `report.md`, and the four scripts. They are reproducible but not
  versioned.
