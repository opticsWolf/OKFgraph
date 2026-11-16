# EG2 int8 — labelled retrieval gate and downgrade check

Status: **both gates run, 2026-11-16.** The text int8 tier passes the
shipping bar on two public labelled sets; okfgraph 0.14.2 refuses a new
int8-pinned graph through the space pin. This record answers N1 and N3 of
`embroider/docs/plan-eg2-rollout.md`.

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
