# Search eval — scripts/eval_labels_okfgraph.jsonl

28 queries, k=10, db `C:/Users/Main/AppData/Local/Temp/okf-p0/scratch-jina/graph.db`, baseline `rrf`.

| config | n | nDCG@k | MRR | recall | distinct | W/L/T vs base | p50 ms | p95 ms | long texts | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| `rrf` | 43 | 0.130 | 0.304 | 0.159 | 0.75 | 0/0/43 | 47 | 59 | — | 0 |
| `cross` | 43 | 0.126 | 0.279 | 0.186 | 0.80 | 6/6/31 | 135 | 2485 | 36% | 0 |
| `cross:mmarco` | 43 | 0.117 | 0.273 | 0.194 | 0.80 | 5/8/30 | 166 | 3493 | 37% | 0 |

## nDCG@k by slice (target/lang)

| config | chunks/de | chunks/en | concepts/de | concepts/en |
|---|---|---|---|---|
| `rrf` | 0.043 (n=7) | 0.190 (n=21) | 0.000 (n=4) | 0.121 (n=11) |
| `cross` | 0.035 (n=7) | 0.165 (n=21) | 0.015 (n=4) | 0.148 (n=11) |
| `cross:mmarco` | 0.024 (n=7) | 0.154 (n=21) | 0.046 (n=4) | 0.133 (n=11) |

## Losses vs baseline

- `cross`: q08/chunks, q08/concepts, q13/chunks, q16/chunks, q17/chunks, q22/chunks
- `cross:mmarco`: q01/chunks, q08/chunks, q08/concepts, q13/chunks, q15/concepts, q16/chunks, q17/chunks, q26/chunks
