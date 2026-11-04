"""Phase-0 reranker lane: three reranks over identical pure-VS top-30 pools.

Pool = the dogfood copy's 637 chunks (kb-gemma.db). Retriever = gemma2@512
(CUDA fp16, crate). Rerank lanes on the same top-30 per chunk-target query:
  (a) none (VS order — baseline),
  (b) crate mmarco cross-encoder (in-process, Jina API pair scores),
  (c) Qwen3-Reranker-0.6B via llama.cpp /v1/rerank (:8932, external wire).
Grading identical to offline_trio_nomic.py (grade 2/1 per concept,
first-appearance-wins). Also reports per-call latency.
"""
import json, os, sys, time, urllib.request
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OKFGRAPH_DEVICE"] = "cuda"; os.environ["OKFGRAPH_NO_SERVER"] = "1"
sys.path.insert(0, r"D:/User/Documents/Python/OKFgraph")
WORK = r"C:/Users/Main/AppData/Local/Temp/okf-p0"
DB = WORK + r"/kb-gemma.db"

import ladybug as lb
import embroider
from okfgraph.components.embedding import (LazyRustEncoder, LazyReranker,
                                           resolve_ort_dylib)

resolve_ort_dylib(warm_gpu=True)
gemma = LazyRustEncoder(model_id="google/embeddinggemma-2", truncate_dim=512,
                        device="cuda",
                        session_factory=lambda: embroider.JinaV5.open(
                            "google/embeddinggemma-2", cache_dir=None,
                            device="cuda", truncate_dim=512),
                        tokenizer_factory=lambda: None)
mmarco = LazyReranker(model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                      device="cuda",
                      session_factory=lambda: embroider.Reranker.open(
                          "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                          cache_dir=None, device="cuda", precision="auto"))

h = lb.Database(DB); c = lb.Connection(h)
chunks = c.execute("MATCH (n:Chunk) RETURN n.id AS id, n.parent_doc_id AS doc, n.chunk_text AS t").rows_as_dict().get_all()
c.close(); h.close()
chunk_texts = [r["t"] or "" for r in chunks]
chunk_docs = [r["doc"] for r in chunks]
print(f"pool: {len(chunks)} chunks", flush=True)

labels = [json.loads(l) for l in open(r"D:/User/Documents/Python/OKFgraph/scripts/eval_labels_okfgraph.jsonl", encoding="utf-8")
          if l.strip() and not l.startswith("//")]
labels = [r for r in labels if (r.get("target") or "chunks") == "chunks"]
print(f"chunk-target labels: {len(labels)}", flush=True)

t0 = time.time()
Q = np.asarray([gemma.encode(r["query"], task="Retrieva-Query") if False else gemma.encode(r["query"], task="Retrieval-Query") for r in labels], dtype=np.float32)
C = np.asarray([gemma.encode(t, task="Document") for t in chunk_texts], dtype=np.float32)
Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
SIM = Q @ C.T
print(f"VS encoded in {time.time()-t0:.1f}s", flush=True)

def grade(order_indices, relevant):
    seen, dcg, mrr, hits, rank = set(), 0.0, 0.0, set(), 0
    for idx in order_indices[:10]:
        key = chunk_docs[idx]
        if key in seen:
            continue
        seen.add(key); rank += 1
        g = relevant.get(key, 0)
        if g:
            dcg += (2.0 if g == 2 else 1.0) / np.log2(rank + 1)
            if not mrr: mrr = 1.0 / rank
            hits.add(key)
    rels = sum(1 for g in relevant.values() if g)
    ideal = sum((2.0 if g == 2 else 1.0) / np.log2(i + 1) for i, g in zip(range(1, 11), sorted(relevant.values(), reverse=True)[:10]))
    return dcg / (ideal + 1e-9), mrr, len(hits) / (rels or 1)

def qwen_rerank(query, docs_idx, texts):
    body = json.dumps({"model": "qwen3-reranker", "query": query,
                       "documents": texts}).encode()
    req = urllib.request.Request("http://127.0.0.1:8932/v1/rerank", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())["results"]
    dt = time.time() - t0
    order = [docs_idx[x["index"]] for x in res]
    scores = [x["relevance_score"] for x in res]
    return order, scores, dt

K_POOL = 30
rows = {"none": [], "mmarco": [], "qwen3": []}
lat = {"mmarco": [], "bge": []}
for qi, row in enumerate(labels):
    relevant = row.get("relevant") or {}
    order = np.argsort(-SIM[qi])[:K_POOL].tolist()
    idxs = [int(i) for i in order]
    texts = [chunk_texts[i] for i in idxs]
    rows["none"].append(grade(idxs, relevant))
    t0 = time.time()
    scores = mmarco.score_batch([(row["query"], t) for t in texts])
    lat["mmarco"].append(time.time() - t0)
    m_order = [int(idxs[j]) for j in np.argsort(-np.asarray(scores, dtype=np.float32)).tolist()]
    rows["mmarco"].append(grade(m_order, relevant))
    q_order, _s, dt = qwen_rerank(row["query"], idxs, texts)
    lat["qwen3"].append(dt)
    rows["qwen3"].append(grade(q_order, relevant))
    print(".", end="", flush=True)
print()

print(f"\n=== rerank lanes over identical VS-top-{K_POOL} pools (gemma2 retriever) ===")
for name, arr in rows.items():
    a = np.asarray(arr)
    extra = f"  p50={np.percentile(lat[name],50):.2f}s p95={np.percentile(lat[name],95):.2f}s" if name in lat else ""
    print(f"{name:7s} n={len(arr)} nDCG@10={a[:,0].mean():.3f} MRR={a[:,1].mean():.3f} recall={a[:,2].mean():.3f}{extra}")
