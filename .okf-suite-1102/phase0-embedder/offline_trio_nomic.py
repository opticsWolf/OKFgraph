"""Offline vector-KNN scorer for three embedder families on one corpus.

Pool = the dogfood kb's 46 concepts + 637 chunks (read from kb-gemma.db,
the reembedded copy; identical TEXTS in all three lanes).
Families: gemma2@512 + jina-small@512 (crate, fresh CUDA fp16 encode) and
nomic-embed-text-v1.5@768 (llama.cpp server on 127.0.0.1:8931, with the
v1.5 prefix contract: `search_query: ` on queries, `search_document: `
on documents).
Scores per label row (target = chunks | concepts) with the same grade
semantics: hits are graded per concept (grade 2 > 1), only the first
appearance of a concept scores in its slot (a repeat wastes the slot).
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
from okfgraph.components.embedding import LazyRustEncoder, resolve_ort_dylib

def lane(Symbol, *, model, dim, device="cuda", factory_extra=None):
    def factory():
        kw = dict(cache_dir=None, device=device, truncate_dim=dim)
        if factory_extra: kw.update(factory_extra)
        return Symbol.open(model, **kw)
    return LazyRustEncoder(model_id=model, truncate_dim=dim, device=device,
                           session_factory=factory,
                           tokenizer_factory=lambda: None)

# crates
dylib = resolve_ort_dylib(warm_gpu=True)
gemma = lane(embroider.JinaV5, model="google/embeddinggemma-2", dim=512)
jina = lane(embroider.JinaV5, model="jinaai/jina-embeddings-v5-text-small-retrieval", dim=512)

# nomic via the llama.cpp server lane
NOMIC_URL = "http://127.0.0.1:8931/v1/embeddings"
def nomic(texts, prefix):
    out = []
    for t in texts:
        body = json.dumps({"model": "text-embedding-nomic-embed-text-v1.5",
                           "input": f"{prefix}{t}" if t else prefix.strip()}).encode()
        req = urllib.request.Request(NOMIC_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            v = np.asarray(json.loads(r.read())["data"][0]["embedding"], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        out.append(v)
    return np.vstack(out)

# pool
h = lb.Database(DB); c = lb.Connection(h)
chunks = c.execute("MATCH (n:Chunk) RETURN n.id AS id, n.parent_doc_id AS doc, n.chunk_text AS t ORDER BY n.id").rows_as_dict().get_all()
concepts = c.execute("MATCH (n:Concept) RETURN n.id AS id, n.title AS title, n.description AS d, n.body AS b ORDER BY n.id").rows_as_dict().get_all()
c.close(); h.close()
chunk_texts = [r["t"] or "" for r in chunks]
chunk_docs = [r["doc"] for r in chunks]
concept_texts = [f"{r['title'] or ''} {r['d'] or ''} {r['b'] or ''}" for r in concepts]
concept_ids = [r["id"] for r in concepts]
doc_to_ci = {cid: i for i, cid in enumerate(concept_ids)}
print(f"pool: {len(chunks)} chunks / {len(concepts)} concepts", flush=True)

labels = [json.loads(l) for l in open(r"D:/User/Documents/Python/OKFgraph/scripts/eval_labels_okfgraph.jsonl", encoding="utf-8") if l.strip() and not l.startswith("//")]
print(f"labels: {len(labels)}", flush=True)

def dcg_mrr_recall(similarities, flatDocs, flatConcepts, relevant):
    order = np.argsort(-similarities)[:10]
    got, seen, dcg, mrr, hitset = [], set(), 0.0, 0.0, set()
    for rank, idx in enumerate(order, 1):
        key = flatConcepts[idx]
        if key in seen:
            continue
        seen.add(key); got.append(key)
        g = relevant.get(key, 0)
        if g:
            dcg += (2.0 if g == 2 else 1.0) / np.log2(rank + 1)
            if not mrr: mrr = 1.0 / rank
            hitset.add(key)
    rels = sum(1 for g in relevant.values() if g)
    ideal = sum((2.0 if g == 2 else 1.0) / np.log2(i + 1) for i, g in zip(range(1, 11), sorted(relevant.values(), reverse=True)[:10]))
    return dcg / (ideal + 1e-9), mrr, len(hitset) / (rels or 1)

def gather():
    qs = [r["query"] for r in labels]
    return qs

qs = gather()
t0 = time.time()
Qs_gemma = [gemma.encode(q, task="Retrieval-Query") for q in qs]; Cd_gemma = [gemma.encode(t, task="Document") for t in chunk_texts]; Ct_gemma = [gemma.encode(t, task="Document") for t in concept_texts]
Qs_jina = [jina.encode(q, task="Retrieval-Query") for q in qs]; Cd_jina = [jina.encode(t, task="Document") for t in chunk_texts]; Ct_jina = [jina.encode(t, task="Document") for t in concept_texts]
Qs_nomic = nomic(qs, "search_query: "); Cd_nomic = nomic(chunk_texts, "search_document: "); Ct_nomic = nomic(concept_texts, "search_document: ")
print(f"all lanes encoded in {time.time()-t0:.1f}s", flush=True)

def qnorm(Q): Q = np.asarray(Q, dtype=np.float32); return Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
def score(name, Q, C, K):
    Q, C, K = qnorm(Q), qnorm(C), qnorm(K)
    out = {"chunks": [], "concepts": []}
    for i, row in enumerate(labels):
        target = row.get("target", "chunks"); relevant = row.get("relevant") or {}
        if target == "chunks":
            n, m, r = dcg_mrr_recall(Q[i] @ C.T, None, chunk_docs, relevant); out["chunks"].append((n, m, r))
        elif target == "concepts":
            n, m, r = dcg_mrr_recall(Q[i] @ K.T, None, concept_ids, relevant); out["concepts"].append((n, m, r))
    print(f"\n=== {name} ===")
    for tag, rows in out.items():
        if rows:
            a = np.asarray(rows)
            print(f"{tag}: n={len(rows)} nDCG@10={a[:,0].mean():.3f} MRR={a[:,1].mean():.3f} recall={a[:,2].mean():.3f}")

score("gemma2@512 (crate, CUDA fp16)", Qs_gemma, Cd_gemma, Ct_gemma)
score("jina-v5-small@512 (crate, CUDA fp16)", Qs_jina, Cd_jina, Ct_jina)
score("nomic-embed-v1.5@768 (llama.cpp server, prefix lane)", Qs_nomic, Cd_nomic, Ct_nomic)
