# Search eval — scripts/eval_labels_okfgraph.jsonl

28 queries, k=10, db `C:/Users/Main/AppData/Local/Temp/okf-p0/kb-gemma.db`, baseline `rrf`.

| config | n | nDCG@k | MRR | recall | distinct | W/L/T vs base | p50 ms | p95 ms | long texts | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| `rrf` | 43 | 0.673 | 0.746 | 0.766 | 0.72 | 0/0/43 | 30 | 39 | — | 0 |
| `hub` | 43 | 0.673 | 0.746 | 0.766 | 0.72 | 0/0/43 | 35 | 41 | — | 0 |
| `cross` | 43 | 0.775 | 0.887 | 0.872 | 0.77 | 17/16/10 | 88 | 608 | 17% | 0 |
| `cross:mmarco` | 43 | 0.821 | 0.921 | 0.917 | 0.78 | 20/15/8 | 123 | 868 | 17% | 0 |

## nDCG@k by slice (target/lang)

| config | chunks/de | chunks/en | concepts/de | concepts/en |
|---|---|---|---|---|
| `rrf` | 0.630 (n=7) | 0.624 (n=21) | 0.832 (n=4) | 0.738 (n=11) |
| `hub` | 0.630 (n=7) | 0.624 (n=21) | 0.832 (n=4) | 0.738 (n=11) |
| `cross` | 0.528 (n=7) | 0.865 (n=21) | 0.613 (n=4) | 0.818 (n=11) |
| `cross:mmarco` | 0.712 (n=7) | 0.868 (n=21) | 0.712 (n=4) | 0.840 (n=11) |

## Losses vs baseline

- `cross`: q01/concepts, q07/concepts, q09/concepts, q11/concepts, q13/chunks, q13/concepts, q14/chunks, q17/chunks, q22/chunks, q22/concepts, q23/concepts, q24/chunks, q25/concepts, q26/chunks, q27/chunks, q27/concepts
- `cross:mmarco`: q01/concepts, q07/concepts, q09/chunks, q09/concepts, q11/concepts, q12/chunks, q13/concepts, q14/chunks, q15/concepts, q16/chunks, q17/chunks, q22/concepts, q26/chunks, q27/chunks, q27/concepts
