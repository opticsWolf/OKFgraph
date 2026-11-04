# Search eval — scripts/eval_labels_okfgraph.jsonl

28 queries, k=10, db `C:/Users/Main/AppData/Local/Temp/okf-p0/scratch/graph.db`, baseline `hub`.

| config | n | nDCG@k | MRR | recall | distinct | W/L/T vs base | p50 ms | p95 ms | long texts | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| `rrf` | — | error: StateError: graph is pinned to model=google/embeddinggemma-2 but the session opened with model=jinaai/jina-embeddings-v5-text-small-retrieval: different weights live in different vector spaces and must never mix. Reimport into a fresh database with model=jinaai/jina-embeddings-v5-text-small-retrieval, or reopen with model=google/embeddinggemma-2. |||||||||
| `hub` | 43 | 0.049 | 0.058 | 0.112 | 0.80 | 0/0/43 | 101 | 113 | — | 0 |

## nDCG@k by slice (target/lang)

| config | chunks/de | chunks/en | concepts/de | concepts/en |
|---|---|---|---|---|
| `hub` | 0.000 (n=7) | 0.090 (n=21) | 0.000 (n=4) | 0.019 (n=11) |
