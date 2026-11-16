# EG2 int8 — text and image gates

Status: **text, image and downgrade gates run, 2026-11-16.** Text int8
passes the labelled retrieval bar on two public sets; vision int8 passes
the cross-modal gate on a public COCO subset; okfgraph 0.14.2 refuses a
new int8-pinned graph through the space pin. This record answers N1, N2
and N3 of `embroider/docs/plan-eg2-rollout.md`.

## N1 — labelled retrieval gate

### Method

- **Sets**: BEIR `scifact` (5,183 docs, 300 labelled queries) and
  `nfcorpus` (3,633 docs, 323 labelled queries), downloaded from the
  BEIR distribution and read in their JSONL/qrels layout.
- **Tiers**: `fp32` and `int8` through `embroider.open("embed", ...)`;
  the same `google/embeddinggemma-2` id, explicit precision, CUDA.
- **Prompts**: publisher Gemma contract, identical in both tiers:
  `task: search result | query: …` and `title: {title} | text: …`.
- **Vectors**: encoded once at 768d, truncated to the evaluated width and
  re-normalised, so width and precision are the only variables.
- **Ranking**: exact cosine over all documents; metrics macro over
  queries with qrels. This isolates the embedding tier, not HNSW recall.
- **Harness**: `scripts/eval_precision.py` (tracked with this record).
- **Raw output**: `%TEMP%/okf-n1-cache/{scifact,nfcorpus}.json`; vectors
  are cached beside it and are reproducible but not versioned.

### Results

**scifact** (300 queries)

| tier | width | R@1 | R@5 | R@20 | MRR@10 | NDCG@10 | top-1 agree | top-10 overlap |
|---|---|---|---|---|---|---|---|---|
| fp32 | 768 | 0.7733 | 0.9067 | 0.9667 | 0.8293 | 0.8514 | 1.0000 | 1.0000 |
| int8 | 768 | 0.7767 | 0.9067 | 0.9667 | 0.8314 | 0.8530 | 0.9967 | 0.9853 |
| fp32 | 512 | 0.7833 | 0.9033 | 0.9633 | 0.8330 | 0.8507 | 1.0000 | 1.0000 |
| int8 | 512 | 0.7800 | 0.9000 | 0.9633 | 0.8321 | 0.8494 | 0.9933 | 0.9857 |

**nfcorpus** (323 queries)

| tier | width | R@1 | R@5 | R@20 | MRR@10 | NDCG@10 | top-1 agree | top-10 overlap |
|---|---|---|---|---|---|---|---|---|
| fp32 | 768 | 0.4644 | 0.6935 | 0.2311 | 0.5720 | 0.3682 | 1.0000 | 1.0000 |
| int8 | 768 | 0.4706 | 0.6904 | 0.2308 | 0.5766 | 0.3694 | 0.9814 | 0.9842 |
| fp32 | 512 | 0.4644 | 0.6935 | 0.2307 | 0.5704 | 0.3689 | 1.0000 | 1.0000 |
| int8 | 512 | 0.4644 | 0.6966 | 0.2303 | 0.5700 | 0.3686 | 0.9721 | 0.9836 |

Deltas (int8 − fp32, points):

| set | width | R@1 | R@20 | NDCG@10 |
|---|---|---|---|---|
| scifact | 768 | +0.34 | 0.00 | +0.16 |
| scifact | 512 | −0.33 | 0.00 | −0.13 |
| nfcorpus | 768 | +0.62 | −0.03 | +0.12 |
| nfcorpus | 512 | 0.00 | −0.04 | −0.03 |

### Verdict

**Pass.** The gate was NDCG@10 and recall@20 delta ≤ 0.5 pt; every
width/set combination is within 0.62 pt on the *most* sensitive metric
(R@1) and within 0.16 pt on NDCG@10. The 50-query long-text run was a
signal; these two labelled sets are the shipping bar.

int8 remains **explicit-only**; `auto` never resolves to it. This gate
decides text retrieval quality only; vision and audio have their own
gates.

## N3 — older okfgraph on a new int8 graph

### Method

- A temp graph built by the current stack (`precision=int8`, 768d,
  `google/embeddinggemma-2`, the live-proof graph).
- A clean venv with **okfgraph 0.14.2** and ONNX Runtime 1.31.1 (the
  release before the codec learned `int8: 8`).
- Opened a copy with `OKFRouter(..., device="cpu", precision="fp32")` and
  called `search`, so the encoder pin check ran.

### Result

**Refused, as intended** — the old build landed fp32, then the space pin
rejected the graph:

```
StateError: graph vectors live in another space than this session's
(pinned revision=52614d767029dd71817689c0bde0a0146c5cc0f0,
landed daa72c51243991dfcaf9f9137d2c573d8f7790c0;
pinned precision=int8, landed fp32).
Vectors from different spaces must never mix.
```

The old codec could not decode 8, but it did not silently adopt the
graph; the space string is the second guard. No version/schema floor is
needed for this path.

Caveat: this proves the space-pin path. A graph that somehow carried an
int8 precision pin but no space pin would not be protected by it; new
graphs always write a space pin, and the current build fails closed on an
unknown code. Do not serve int8 graphs to older clients that predate the
space pin.

## Reproduce

```bash
# N1 (from OKFgraph)
./.okf-p0-venv/Scripts/python.exe scripts/eval_precision.py \
  --beir-dir %TEMP%/okf-n1-beir/scifact --dataset scifact \
  --tiers fp32,int8 --widths 768,512 --device cuda \
  --cache-dir %TEMP%/okf-n1-cache --json-out %TEMP%/okf-n1-cache/scifact.json

# N3: create a temp venv with okfgraph==0.14.2, copy an int8 graph,
# then OKFRouter(db_path=copy, device="cpu", precision="fp32").search(...)
```

## N2 — vision int8 live encode and cross-modal gate

### Method

- **Set**: `JotDe/mscoco_1k` (openrail), a public MS-COCO subset: 995
  rows, **199 unique images**, five captions per image.
- **Tiers**: `fp32`, `fp16`, `int8` through the Gemma vision slot on
  CUDA; the caption queries/documents use the matching text model at the
  same precision.
- **Directions**: text→image (each caption queries its image) and
  image→text (each image queries its five captions); exact cosine, no
  HNSW.
- **Fusion check**: before scoring, the text model's `space` must equal
  the vision slot's new `text_partner_space`; the harness aborts on a
  mismatch. This is the W3 identity, not a naming convention.
- **Live checks**: one image was encoded through `int8` on CPU and CUDA;
  output is finite, 768-d and unit norm. CPU and CUDA int8 vectors were
  identical (cos 1.0); fp16/fp32 were cos 0.99937 against int8.
- **Harness**: `scripts/eval_image_precision.py`.
- **Raw output**: `%TEMP%/okf-n2-cache/{results.json,report.md}`.

### Results (CUDA, 199 images, 995 captions)

| tier | direction | n | R@1 / hit@1 | R@5 | R@10 | NDCG@10 | top-1 agree | top-10 overlap |
|---|---|---|---|---|---|---|---|---|
| fp32 | text→image | 995 | 0.7789 | 0.9578 | 0.9889 | 0.8917 | 1.0000 | 1.0000 |
| fp32 | image→text | 199 | 0.8442 | 0.6958 | 0.8660 | 0.9022 | 1.0000 | 1.0000 |
| fp16 | text→image | 995 | 0.7799 | 0.9578 | 0.9889 | 0.8922 | 0.9960 | 0.9959 |
| fp16 | image→text | 199 | 0.8442 | 0.6948 | 0.8660 | 0.9019 | 0.9950 | 1.0000 |
| int8 | text→image | 995 | 0.7779 | 0.9558 | 0.9869 | 0.8905 | 0.9698 | 0.9583 |
| int8 | image→text | 199 | 0.8442 | 0.6948 | 0.8620 | 0.9027 | 0.9648 | 0.9744 |

Deltas (int8 − fp32, points): text→image R@1 −0.10, R@5 −0.20,
NDCG@10 −0.12; image→text hit@1 0.00, R@5 −0.10, R@10 −0.40,
NDCG@10 +0.05. Every gate metric is inside the 0.5 pt bar.

Image-vector parity vs fp32 (same 199 images): mean 1−cos **1.27e-3**,
p99 3.8e-3, max 1.17e-2. Text vectors are an order of magnitude closer
(query mean 3.0e-5, document 3.6e-5).

### Verdict

**Pass.** Vision int8 is quality-approved for the image lane on this
labelled set. The artifact remains explicit-only.

### N2.3 — the mixed-pair rule is now decided

The original plan called text-int8 + vision-fp16 “mechanically safe”.
The implementation says otherwise: `GemmaVision` runs the **partner text
tower at the image precision**, so an fp16 vision host fuses images in
the fp16 text space. Those vectors are not comparable to a text-int8
graph. W3 now stores the partner text space and refuses the mismatch.

**Product rule: Gemma text and vision precision must match — text int8 +
vision int8, or no image lane.** Mixed text/vision precision needs a
separate text-fusion precision feature that does not exist today; it is
not a setting the current host can honour. Silk’s wiring refuses the
mismatch before starting the image host; okf refuses it again at the
first image open if the graph was not served by Silk.

### W3 — the pin, not just a check

- embroider’s vision `describe()` now carries `text_partner_space` (the
  partner text model’s revision/precision/width space) next to the
  tower’s `space`.
- okf stores it in `MetaText.embedding_image_text_space`; a first image
  open compares it with the graph’s text space, later opens compare it
  with the stored pair, and `pin_info` reports `text_space` / an
  `image_text_space` mismatch.
- A graph with no vision vectors may adopt/re-pin; a graph with vectors
  must re-encode images or reimport into a fresh database.
- Older hosts that cannot name a fusion space still check model and
  precision and warn when a fusion pin exists; the W3 guarantee requires
  the new host.
- Image graphs built before W3 have no fusion pin and are not
  retroactively verified. Re-encode their images to claim the pair; the
  model/precision/text pins still guard the text side.

### Caveats

- 199 images and 995 captions is a public subset, not the full MIEB or
  MMEB-v2 suites. It is large enough to catch a quality cliff; not a
  leaderboard number.
- Exact cosine, not HNSW; CPU was checked on one image, not the whole
  matrix.
- Sources consulted 2026-11-16: the [Gemma model
  card](https://gemma.ai/models/embeddinggemma/model-card-v2/) (full
  precision, MIEB/MMEB-v2, fp16 warning), the [ONNX model
  card](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX)
  (q8 worst-case cosine 0.9997, per-component dtypes), the [MIEB
  paper](https://arxiv.org/abs/2504.10471), and the
  [COCO 1k dataset](https://huggingface.co/datasets/JotDe/mscoco_1k).
- Official EmbeddingGemma 2 numbers use the full-precision checkpoint;
  q8 has no published per-benchmark vision quality. The model card’s
  whole-model q8 parity (worst-case cosine 0.9997) is corroboration, not
  this gate. The card also recommends bfloat16/float32 and warns about
  fp16; this stack's fp16 lane is parity-gated, not an official claim.
