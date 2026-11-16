#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-modal precision gate on a public image/text set.

Uses `JotDe/mscoco_1k` (995 rows, 199 unique MS-COCO images, 5 captions
each). Development-only dependency: `pyarrow` (parquet reader);
`huggingface_hub` is already a project dependency.  For every tier it embeds the images with the Gemma vision slot
(which fuses through the partner text tower) and the captions with the
matching text model, then measures both retrieval directions with exact
cosine:

* text -> image: each caption is a query for its own image;
* image -> text: each image is a query for its five captions.

The tier's text space must equal the vision slot's `text_space`; that is
the W3 fusion identity (text tower revision/precision + width), and a
mismatch aborts before any retrieval number is reported.

Example::

    ./.okf-p0-venv/Scripts/python.exe scripts/eval_image_precision.py \
        --tiers fp32,fp16,int8 --device cuda \
        --out docs/eval-eg2-image-int8.md

This is not a pytest file.  It never opens a graph database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

MODEL_TEXT = "google/embeddinggemma-2"
MODEL_VISION = "google/embeddinggemma-2-vision"
BATCH = 16
DATASET = "JotDe/mscoco_1k"


def load_rows(parquet: Path) -> List[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(parquet).to_pylist()


def unique_images(rows: List[dict]) -> List[Tuple[bytes, List[str]]]:
    """One entry per image, with all its captions (row order preserved)."""
    seen: Dict[str, int] = {}
    out: List[Tuple[bytes, List[str]]] = []
    for row in rows:
        image = row.get("image") or {}
        data = image.get("bytes") or b""
        text = str(row.get("text") or "").strip()
        if not data or not text:
            continue
        key = hashlib.sha256(data).hexdigest()
        if key not in seen:
            seen[key] = len(out)
            out.append((data, []))
        out[seen[key]][1].append(text)
    return out


def encode_texts(texts: Sequence[str], tier: str, device: str,
                 task: str, cache: Path | None) -> np.ndarray:
    if cache is not None and cache.is_file():
        return np.asarray(np.load(cache)["vectors"], dtype=np.float32)
    import embroider
    model = embroider.open("embed", MODEL_TEXT, device=device, precision=tier,
                           truncate_dim=768)
    out: List[np.ndarray] = []
    started = time.time()
    for start in range(0, len(texts), BATCH):
        out.extend(np.asarray(
            model.encode_batch(list(texts[start:start + BATCH]), task=task),
            dtype=np.float32))
        if start and start % (BATCH * 25) == 0:
            print(f"  text {tier} {len(out)}/{len(texts)} "
                  f"({len(out) / (time.time() - started):.1f}/s)", flush=True)
    vectors = np.asarray(out, dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, vectors=vectors)
    return vectors


def encode_images(images: Sequence[bytes], tier: str, device: str,
                  cache: Path | None) -> Tuple[np.ndarray, str, str]:
    if cache is not None and cache.is_file():
        data = np.load(cache)
        return (np.asarray(data["vectors"], dtype=np.float32),
                str(data["space"]), str(data["text_space"]))
    import embroider
    model = embroider.open("vision", MODEL_VISION, device=device,
                           precision=tier, truncate_dim=768)
    identity = model.describe()
    text_space = identity.get("text_partner_space") or ""
    out: List[np.ndarray] = []
    started = time.time()
    for start, data in enumerate(images):
        rgb, h, w = embroider.prepare_image(data, MODEL_VISION)
        out.append(np.asarray(model.encode_image(rgb, h, w), dtype=np.float32))
        if start and start % 25 == 0:
            print(f"  image {tier} {len(out)}/{len(images)} "
                  f"({len(out) / (time.time() - started):.1f}/s)", flush=True)
    vectors = np.asarray(out, dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, vectors=vectors,
                            space=str(identity.get("space") or ""),
                            text_space=text_space)
    return vectors, str(identity.get("space") or ""), text_space


def l2(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True),
                            1e-12, None)


def ndcg(binary: Sequence[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2.0) for i, r in enumerate(binary[:k]) if r)
    ideal = sum(1.0 / math.log2(i + 2.0) for i in range(min(sum(binary), k)))
    return dcg / ideal if ideal else 0.0


def text_to_image(queries: np.ndarray, images: np.ndarray,
                  targets: Sequence[int]) -> dict:
    scores = l2(queries) @ l2(images).T
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.where(row == t)[0][0])
                      for row, t in zip(order, targets)])
    return {
        "n": len(targets),
        "recall@1": float(np.mean(ranks < 1)),
        "recall@5": float(np.mean(ranks < 5)),
        "recall@10": float(np.mean(ranks < 10)),
        "ndcg@10": float(np.mean([
            ndcg([1 if int(c) == t else 0 for c in order[i, :10]])
            for i, t in enumerate(targets)])),
        "order": order,
    }


def image_to_text(images: np.ndarray, docs: np.ndarray,
                  relevant: Sequence[Sequence[int]]) -> dict:
    scores = l2(images) @ l2(docs).T
    order = np.argsort(-scores, axis=1)
    hit1, rec5, rec10, ndcgs = [], [], [], []
    for row, gold in zip(order, relevant):
        gold = set(gold)
        ranked = [int(x) for x in row]
        hit1.append(1.0 if ranked[0] in gold else 0.0)
        rec5.append(len([x for x in ranked[:5] if x in gold]) / len(gold))
        rec10.append(len([x for x in ranked[:10] if x in gold]) / len(gold))
        ndcgs.append(ndcg([1 if x in gold else 0 for x in ranked[:10]]))
    return {
        "n": len(relevant),
        "hit@1": float(np.mean(hit1)),
        "recall@5": float(np.mean(rec5)),
        "recall@10": float(np.mean(rec10)),
        "ndcg@10": float(np.mean(ndcgs)),
        "order": order,
    }


def agreement(order: np.ndarray, baseline: np.ndarray) -> Tuple[float, float]:
    top1 = float(np.mean(order[:, 0] == baseline[:, 0]))
    top10 = float(np.mean([
        len(set(order[i, :10]) & set(baseline[i, :10])) / 10.0
        for i in range(len(order))]))
    return top1, top10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--tiers", default="fp32,fp16,int8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    if args.parquet:
        parquet = Path(args.parquet)
    else:
        from huggingface_hub import hf_hub_download
        parquet = Path(hf_hub_download(
            args.dataset, "data/train-00000-of-00001-1c3511b9df1f3a58.parquet",
            repo_type="dataset"))
    pairs = unique_images(load_rows(parquet))
    if args.limit_images:
        pairs = pairs[:args.limit_images]
    images = [p[0] for p in pairs]
    captions = [text for _, texts in pairs for text in texts]
    image_of_caption = [i for i, (_, texts) in enumerate(pairs)
                        for _ in texts]
    relevant = [[sum(len(t) for t in [x[1] for x in pairs[:i]])
                 + j for j in range(len(texts))]
                for i, (_, texts) in enumerate(pairs)]
    print(f"{args.dataset}: {len(images)} images, {len(captions)} captions",
          flush=True)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    results: Dict[str, dict] = {}
    for tier in tiers:
        print(f"{tier}: images", flush=True)
        image_cache = (cache_dir / f"{args.dataset.replace('/', '_')}_{tier}_"
                       f"images_768.npz") if cache_dir else None
        image_vecs, image_space, vision_text_space = encode_images(
            images, tier, args.device, image_cache)
        print(f"{tier}: caption queries", flush=True)
        query_cache = (cache_dir / f"{args.dataset.replace('/', '_')}_{tier}_"
                       f"queries_768.npz") if cache_dir else None
        query_vecs = encode_texts(
            [f"task: search result | query: {c}" for c in captions],
            tier, args.device, "query", query_cache)
        print(f"{tier}: caption documents", flush=True)
        doc_cache = (cache_dir / f"{args.dataset.replace('/', '_')}_{tier}_"
                     f"docs_768.npz") if cache_dir else None
        doc_vecs = encode_texts(
            [f"title: none | text: {c}" for c in captions],
            tier, args.device, "document", doc_cache)
        # W3: the image vectors were fused through this text space.
        import embroider
        text_model = embroider.open("embed", MODEL_TEXT, device=args.device,
                                   precision=tier, truncate_dim=768)
        text_space = text_model.space
        if vision_text_space and text_space and vision_text_space != text_space:
            raise SystemExit(
                f"{tier}: fusion space {vision_text_space} != text space "
                f"{text_space}")
        results[tier] = {
            "text2image": text_to_image(query_vecs, image_vecs,
                                        image_of_caption),
            "image2text": image_to_text(image_vecs, doc_vecs, relevant),
            "image_space": image_space,
            "text_space": text_space,
            "vision_text_space": vision_text_space,
        }

    baseline = results.get("fp32")
    rows = []
    for tier in tiers:
        for direction in ("text2image", "image2text"):
            m = results[tier][direction]
            base = baseline[direction]["order"] if baseline else m["order"]
            top1, top10 = ((1.0, 1.0) if baseline is None or tier == "fp32"
                           else agreement(m["order"], base))
            row = {"tier": tier, "direction": direction,
                   **{k: round(v, 4) for k, v in m.items()
                      if k not in ("order", "n")},
                   "n": m["n"], "top1_agree": round(top1, 4),
                   "top10_overlap": round(top10, 4)}
            rows.append(row)
            print(row, flush=True)

    lines = [
        f"# Image/text precision gate — {args.dataset}",
        "",
        f"{len(images)} unique images, {len(captions)} captions; "
        f"`{MODEL_VISION}` on `{args.device}`; exact cosine. "
        "Text->image queries are the captions; image->text targets each "
        "image's caption set.",
        "",
        "| tier | direction | n | R@1 / hit@1 | R@5 | R@10 | NDCG@10 | "
        "top-1 agree | top-10 overlap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        first = r.get("recall@1", r.get("hit@1"))
        lines.append(
            "| {tier} | {direction} | {n} | {first:.4f} | {recall@5:.4f} | "
            "{recall@10:.4f} | {ndcg@10:.4f} | {top1_agree:.4f} | "
            "{top10_overlap:.4f} |".format(first=first, **r))
    lines += ["", "`top-1 agree` / `top-10 overlap` compare with fp32 in the "
              "same direction; fp32 shows 1.0 by construction.", ""]
    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dataset": args.dataset, "rows": rows,
                                   "spaces": {t: {
                                       "image": results[t]["image_space"],
                                       "text": results[t]["text_space"],
                                       "vision_text": results[t]["vision_text_space"]}
                                       for t in tiers}}, indent=2) + "\n",
                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
