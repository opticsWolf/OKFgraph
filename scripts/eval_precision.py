#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precision gate on a labelled retrieval set (BEIR layout).

Compares embedding tiers at the same width with exact cosine ranking.  The
corpus is the BEIR JSONL layout::

    scifact/corpus.jsonl
    scifact/queries.jsonl
    scifact/qrels/test.tsv

The prompts are the publisher's Gemma contract, identical in every tier.
Vectors are encoded once at 768d and truncated (then re-normalised) per
width, so the only variable is the precision tier.

Example::

    ./.okf-p0-venv/Scripts/python.exe scripts/eval_precision.py \
        --beir-dir %TEMP%/okf-n1-beir/scifact --dataset scifact \
        --tiers fp32,int8 --widths 768,512 --device cuda \
        --out docs/eval-eg2-int8.md

This is not a pytest file.  It never opens a graph database.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_TIERS = ("fp32", "int8")
DEFAULT_WIDTHS = (768, 512)
MODEL = "google/embeddinggemma-2"
BATCH = 16


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    """query-id -> {corpus-id: grade}."""
    out: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            out.setdefault(qid, {})[cid] = float(row["score"])
    return out


def load_set(root: Path) -> Tuple[List[dict], List[dict], Dict[str, Dict[str, float]]]:
    corpus = read_jsonl(root / "corpus.jsonl")
    queries = read_jsonl(root / "queries.jsonl")
    qrels = read_qrels(root / "qrels" / "test.tsv")
    return corpus, queries, qrels


def prompts(corpus: List[dict], queries: List[dict]) -> Tuple[List[str], List[str]]:
    docs = []
    for row in corpus:
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        docs.append(f"title: {title} | text: {text}")
    asks = [f"task: search result | query: {str(row.get('text') or '')}"
            for row in queries]
    return docs, asks


def encode(texts: Sequence[str], tier: str, device: str, cache: Path | None,
           task: str) -> Tuple[np.ndarray, float]:
    if cache is not None and cache.is_file():
        data = np.load(cache)
        return np.asarray(data["vectors"], dtype=np.float32), float(data["seconds"])
    import embroider
    model = embroider.open("embed", MODEL, device=device, precision=tier,
                           truncate_dim=768)
    out: List[np.ndarray] = []
    started = time.time()
    for start in range(0, len(texts), BATCH):
        batch = list(texts[start:start + BATCH])
        out.extend(np.asarray(model.encode_batch(batch, task=task),
                              dtype=np.float32))
        if start and start % (BATCH * 25) == 0:
            done = len(out)
            print(f"  {tier} {done}/{len(texts)} "
                  f"({done / (time.time() - started):.1f}/s)", flush=True)
    vectors = np.asarray(out, dtype=np.float32)
    seconds = time.time() - started
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, vectors=vectors, seconds=seconds)
    return vectors, seconds


def l2_width(matrix: np.ndarray, width: int) -> np.ndarray:
    part = np.asarray(matrix, dtype=np.float32)[:, :width]
    return part / np.clip(np.linalg.norm(part, axis=1, keepdims=True), 1e-12, None)


def metrics(scores: np.ndarray, relevant: List[Dict[str, float]],
            doc_ids: List[str], k: int = 10) -> dict:
    """Binary/graded retrieval metrics, macro over queries with qrels."""
    rr, r1, r5, r20, ndcg = [], [], [], [], []
    for row, rel in zip(scores, relevant):
        order = np.argsort(-row)
        ranked = [doc_ids[int(x)] for x in order]
        gold = set(rel)
        found = [i for i, cid in enumerate(ranked) if cid in gold]
        first = found[0] + 1 if found else None
        rr.append(1.0 / first if first else 0.0)
        r1.append(1.0 if first == 1 else 0.0)
        r5.append(1.0 if first and first <= 5 else 0.0)
        r20.append(len([i for i in found if i < 20]) / max(len(gold), 1))
        gains = [rel.get(cid, 0.0) for cid in ranked[:k]]
        dcg = sum((2.0 ** g - 1.0) / math.log2(i + 2.0)
                  for i, g in enumerate(gains))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum((2.0 ** g - 1.0) / math.log2(i + 2.0)
                   for i, g in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0.0)
    return {
        "n": len(rr),
        "recall@1": float(np.mean(r1)),
        "recall@5": float(np.mean(r5)),
        "recall@20": float(np.mean(r20)),
        "mrr@10": float(np.mean(rr)),
        "ndcg@10": float(np.mean(ndcg)),
    }


def top_agreement(scores: np.ndarray, baseline: np.ndarray) -> Tuple[float, float]:
    a = np.argsort(-scores, axis=1)
    b = np.argsort(-baseline, axis=1)
    top1 = float(np.mean(a[:, 0] == b[:, 0]))
    top10 = float(np.mean([
        len(set(a[i, :10]) & set(b[i, :10])) / 10.0 for i in range(len(a))
    ]))
    return top1, top10


def markdown(rows: List[dict], meta: dict) -> str:
    lines = [
        f"# Precision gate — {meta['dataset']}",
        "",
        f"{meta['docs']} documents, {meta['queries']} labelled queries; "
        f"model `{MODEL}` on `{meta['device']}`. Exact cosine, "
        "vectors encoded at 768d and truncated/re-normalised per width.",
        "",
        "| tier | width | n | R@1 | R@5 | R@20 | MRR@10 | NDCG@10 | "
        "top-1 agree | top-10 overlap |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {tier} | {width} | {n} | {recall@1:.4f} | {recall@5:.4f} | "
            "{recall@20:.4f} | {mrr@10:.4f} | {ndcg@10:.4f} | "
            "{top1_agree:.4f} | {top10_overlap:.4f} |".format(**row))
    lines += ["", "`top-1 agree` / `top-10 overlap` compare with fp32 at the "
              "same width; fp32 shows 1.0 by construction.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    ap.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.beir_dir)
    corpus, queries, qrels = load_set(root)
    if args.limit_docs:
        # Keep query targets in a smoke slice, or every qrel is outside it
        # and the run measures nothing.
        targets = {cid for rel in qrels.values() for cid in rel}
        head = [row for row in corpus if str(row["_id"]) in targets]
        seen = {str(row["_id"]) for row in head}
        head += [row for row in corpus if str(row["_id"]) not in seen]
        corpus = head[:args.limit_docs]
    keep = [q for q in queries if str(q.get("_id")) in qrels]
    if args.limit_queries:
        keep = keep[:args.limit_queries]
    # Limit docs before building ids: a qrel target outside the slice is
    # unretrievable by construction and would make the smoke numbers lie.
    corpus_ids = {str(row["_id"]) for row in corpus}
    queries = [q for q in keep
               if set(qrels[str(q["_id"])]) & corpus_ids]
    qrels = {str(q["_id"]): {c: g for c, g in qrels[str(q["_id"])].items()
                             if c in corpus_ids} for q in queries}
    if not queries:
        raise SystemExit("no labelled query has a target in this corpus slice")
    docs, asks = prompts(corpus, queries)
    print(f"{args.dataset}: {len(corpus)} docs, {len(queries)} labelled queries",
          flush=True)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    vectors: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]] = {}
    for tier in tiers:
        print(f"{tier}: document embeddings", flush=True)
        doc_cache = cache_dir / f"{args.dataset}_{tier}_docs_768.npz" if cache_dir else None
        doc_vecs, doc_s = encode(docs, tier, args.device, doc_cache, "document")
        print(f"{tier}: query embeddings", flush=True)
        qry_cache = cache_dir / f"{args.dataset}_{tier}_queries_768.npz" if cache_dir else None
        qry_vecs, qry_s = encode(asks, tier, args.device, qry_cache, "query")
        vectors[tier] = (doc_vecs, qry_vecs, doc_s, qry_s)

    rel = [qrels[str(q["_id"])] for q in queries]
    doc_ids = [str(row["_id"]) for row in corpus]
    rows: List[dict] = []
    by_width: Dict[int, Dict[str, np.ndarray]] = {}
    for width in widths:
        by_width[width] = {}
        for tier in tiers:
            doc_vecs, qry_vecs, doc_s, qry_s = vectors[tier]
            scores = l2_width(qry_vecs, width) @ l2_width(doc_vecs, width).T
            by_width[width][tier] = scores
        baseline = by_width[width].get("fp32")
        for tier in tiers:
            scores = by_width[width][tier]
            m = metrics(scores, rel, doc_ids)
            top1, top10 = ((1.0, 1.0) if baseline is None or tier == "fp32"
                           else top_agreement(scores, baseline))
            rows.append({
                "tier": tier, "width": width, **{k: round(v, 4)
                                                 for k, v in m.items()},
                "top1_agree": round(top1, 4),
                "top10_overlap": round(top10, 4),
            })
            print(rows[-1], flush=True)

    meta = {"dataset": args.dataset, "docs": len(corpus),
            "queries": len(queries), "device": args.device,
            "model": MODEL, "tiers": tiers, "widths": widths}
    report = markdown(rows, meta)
    print("\n" + report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n",
                       encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
