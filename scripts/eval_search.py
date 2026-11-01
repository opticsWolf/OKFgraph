"""Search evaluation on a graph's own labels (reranker plan, Phase 3).

Runs each labelled query under several search configs and reports
ranking quality, per-query wins/losses against the first config, and
latency. Not a pytest file; run it against a copy of a real graph:

    .venv/Scripts/python.exe scripts/eval_search.py LABELS.jsonl --db kb.db --copy \
        --configs rrf,hub,cross,cross:mmarco,rrf+dedupe --md report.md

Labels (JSONL, one query per line)::

    {"id": "q01", "query": "how are changed files detected?",
     "relevant": {"docs/architecture": 2, "docs/chunking-and-graph-retrieval": 1},
     "target": "chunks", "lang": "en", "split": "heldout"}

``relevant`` maps concept ids to a grade (2 = the answer is here,
1 = related); a plain list means grade 1 each. ``target`` is
``chunks`` (default), ``concepts`` or ``both``. ``lang`` and ``split``
are free-form slices; ``--split`` keeps one of them.

Relevance is judged per concept: a chunk hit counts for its concept, and
only its first appearance scores (a repeated passage wastes a slot, which
is what ``dedupe`` addresses).

Configs: ``rrf`` (plain RRF), ``hub`` (concepts ``rank='hub'``, chunks
``hub_rerank``), ``cross[:MODEL][@PRECISION]`` (cross-encoder rerank;
MODEL is a registry id or ``minilm`` / ``mmarco``, PRECISION ``fp32`` /
``q8``), ``noul[:MODEL][@PRECISION]`` (an embroider typed-decision
classifier as the reranker: one yes/no question per passage, scored by
P(true); MODEL is a registry id or ``decider2b`` / ``laya`` /
``laya-ml``), each optionally ``+dedupe`` (chunks only). No thresholds
ship from this: the report records the numbers a default needs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}
NOUL_MODELS = {
    "decider2b": "Mapika/decider-2b",
    "laya": "convaiinnovations/laya-typed-decisions",
    "laya-ml": "convaiinnovations/laya-multilingual",
}
# The relevance question, as embroider's decider-2b/Laya goldens phrase it.
NOUL_INSTRUCTIONS = "Does the passage answer the query: {query}"
NOUL_TRUE = "The passage answers the query."
NOUL_FALSE = "The passage does not answer the query."
TARGETS = ("chunks", "concepts")


# ── labels ───────────────────────────────────────────────────────────────

@dataclass
class Label:
    id: str
    query: str
    relevant: Dict[str, int]
    targets: List[str]
    lang: str = "?"
    split: str = ""


def load_labels(path: str, split: Optional[str] = None) -> List[Label]:
    """Read and validate a JSONL label file; ``split`` keeps one slice."""
    labels, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        row = json.loads(line)
        rel = row.get("relevant")
        if isinstance(rel, list):
            rel = {cid: 1 for cid in rel}
        if not row.get("query") or not isinstance(rel, dict) or not rel:
            raise ValueError(f"{path}:{n}: needs 'query' and non-empty 'relevant'")
        if any(not isinstance(g, int) or g < 0 for g in rel.values()):
            raise ValueError(f"{path}:{n}: grades must be integers >= 0")
        target = row.get("target", "chunks")
        targets = list(TARGETS) if target == "both" else [target]
        if any(t not in TARGETS for t in targets):
            raise ValueError(f"{path}:{n}: target must be chunks, concepts or both")
        qid = str(row.get("id") or f"q{n:03d}")
        if qid in seen:
            raise ValueError(f"{path}:{n}: duplicate id {qid!r}")
        seen.add(qid)
        label = Label(qid, row["query"], rel, targets,
                      str(row.get("lang", "?")), str(row.get("split", "")))
        if split is None or label.split == split:
            labels.append(label)
    return labels


# ── metrics ──────────────────────────────────────────────────────────────

def score_ranking(ids: List[str], relevant: Dict[str, int], k: int) -> Dict[str, float]:
    """nDCG@k (graded, gain 2^g - 1), MRR@k, recall@k and distinct@k.

    ``ids`` are concept ids in rank order, repeats allowed: only the first
    appearance of a concept earns gain; later ones fill a slot for nothing.
    """
    top = ids[:k]
    seen, dcg, rr, found = set(), 0.0, 0.0, set()
    for rank, cid in enumerate(top, 1):
        if cid in seen:
            continue
        seen.add(cid)
        grade = relevant.get(cid, 0)
        if grade > 0:
            dcg += (2 ** grade - 1) / math.log2(rank + 1)
            found.add(cid)
            if not rr:
                rr = 1.0 / rank
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    wanted = sum(1 for g in relevant.values() if g > 0)
    return {
        "ndcg": dcg / idcg if idcg else 0.0,
        "mrr": rr,
        "recall": len(found) / wanted if wanted else 0.0,
        "distinct": len(seen) / len(top) if top else 0.0,
    }


# ── configs ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    name: str
    base: str                      # rrf | hub | cross | noul
    model: Optional[str] = None
    precision: Optional[str] = None
    dedupe: bool = False
    max_length: Optional[int] = None   # noul: Decider2B context cap

    def search_kwargs(self, target: str) -> Optional[Dict[str, Any]]:
        """Search params for ``target``; None when the config has no
        meaning there (dedupe on concepts)."""
        if self.dedupe and target != "chunks":
            return None
        kw: Dict[str, Any] = {"target": target}
        if self.base == "hub":
            kw.update({"rank": "hub"} if target == "concepts" else {"hub_rerank": True})
        elif self.base in ("cross", "noul"):
            kw.update({"rank": "cross"} if target == "concepts" else {"rerank": True})
        if self.dedupe:
            kw["dedupe"] = True
        return kw


def parse_config(spec: str) -> Config:
    """``rrf`` / ``hub`` / ``cross[:MODEL][@PRECISION]`` /
    ``noul[:MODEL][@PRECISION]``, optional ``+dedupe``."""
    text = spec.strip()
    dedupe = text.endswith("+dedupe")
    if dedupe:
        text = text[: -len("+dedupe")]
    text, at, precision = text.partition("@")
    base, colon, model = text.partition(":")
    scored = base in ("cross", "noul")
    if base not in ("rrf", "hub", "cross", "noul") or ((colon or at) and not scored):
        raise ValueError(f"unknown config {spec!r}")
    if not scored:
        return Config(spec.strip(), base, dedupe=dedupe)
    precision = precision or "fp32"
    if precision not in ("fp32", "q8"):
        raise ValueError(f"precision must be fp32 or q8 in {spec!r}")
    if base == "noul":
        model = NOUL_MODELS.get(model or "decider2b", model)
    else:
        model = MODELS.get(model or "minilm", model)
    return Config(spec.strip(), base, model, precision, dedupe)


class CountingReranker:
    """Wraps a ``LazyReranker``: counts batch pairs vs windowed long texts."""

    def __init__(self, inner):
        self.inner = inner
        self.pairs = 0
        self.long = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def score_batch(self, pairs):
        pairs = list(pairs)
        self.pairs += len(pairs)
        return self.inner.score_batch(pairs)

    def score_long(self, query, doc, window=None, stride=None):
        self.long += 1
        return self.inner.score_long(query, doc, window=window, stride=stride)


class NoulSession:
    """A typed-decision classifier behind the reranker session interface.

    Each (query, passage) pair is one ``noul`` question over the passage
    as state; the score is P(true). Laya scores long passages with
    ``predict_long`` (strongest window); Decider2B has no window mode and
    truncates the state to its ``max_length``.
    """

    OVERHEAD = 64  # instructions + criteria tokens on top of query + passage

    def __init__(self, inner, precision: str):
        self.inner = inner
        self.precision = precision
        self.windowed = hasattr(inner, "predict_long")

    @property
    def used_cuda(self) -> bool:
        return bool(self.inner.used_cuda)

    @property
    def max_length(self) -> int:
        return max(1, int(self.inner.max_length) - self.OVERHEAD)

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def _question(self, query: str):
        ins = NOUL_INSTRUCTIONS.format(query=query)
        if self.windowed:
            return ("noul", ins, NOUL_TRUE)
        return ("noul", ins, NOUL_TRUE, NOUL_FALSE)

    @staticmethod
    def _p_true(decision) -> float:
        return float(dict(decision[1])["true"])

    def score_batch(self, pairs):
        return [self._p_true(self.inner.decide(doc, [self._question(q)])[0])
                for q, doc in pairs]

    def score_long(self, query, doc, window=None, stride=None):
        if not self.windowed:
            return self.score_batch([(query, doc)])[0]
        return self._p_true(self.inner.predict_long(
            doc, [self._question(query)], window=window, stride=stride)[0])


def make_reranker(router, config: Config):
    """A lazy reranker session for one config (opened on first use)."""
    import embroider

    from okfgraph.components.embedding import LazyReranker

    device = getattr(router.reranker, "_device", "auto")
    cache_dir = getattr(router, "cache_dir", None)
    model = config.model
    # Explicit: precision=None means 'auto', which is fp16 on CUDA.
    precision = config.precision
    if config.base == "noul":
        def factory():
            if "decider-2b" in model:
                inner = embroider.Decider2B.open(
                    model, cache_dir=cache_dir, device=device,
                    max_length=config.max_length, precision=precision)
            else:
                inner = embroider.LayaDecider.open(
                    model, cache_dir=cache_dir, device=device, precision=precision)
            return NoulSession(inner, config.precision)
    else:
        def factory():
            return embroider.Reranker.open(
                model, cache_dir=cache_dir, device=device, precision=precision)
    return CountingReranker(LazyReranker(
        model_id=model, device=device, session_factory=factory))


# ── run ──────────────────────────────────────────────────────────────────

@dataclass
class Run:
    config: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    pairs: int = 0
    long: int = 0


def run_config(router, config: Config, labels: List[Label], k: int,
               rerankers: Dict[tuple, Any]) -> Run:
    """Every (label, target) under one config; first query warms up."""
    run = Run(config.name)
    rr = None
    if config.base in ("cross", "noul"):
        key = (config.base, config.model, config.precision)
        rr = rerankers.get(key) or rerankers.setdefault(
            key, make_reranker(router, config))
        router.search_engine.reranker = rr
    jobs = [(lb, t) for lb in labels for t in lb.targets
            if config.search_kwargs(t) is not None]
    try:
        if jobs:  # warm-up: model loads stay out of the latency numbers
            lb, t = jobs[0]
            router.search(lb.query, limit=k, **config.search_kwargs(t))
        base_pairs, base_long = (rr.pairs, rr.long) if rr else (0, 0)
        for lb, t in jobs:
            t0 = time.perf_counter()
            hits = router.search(lb.query, limit=k, **config.search_kwargs(t))
            ms = (time.perf_counter() - t0) * 1000
            ids = [h.get("id") or h.get("concept_id") for h in hits]
            run.rows.append({
                "id": lb.id, "target": t, "lang": lb.lang, "ms": ms, "hits": ids,
                "collapsed": sum(len(h.get("collapsed") or ()) for h in hits),
                **score_ranking(ids, lb.relevant, k),
            })
        if rr:
            run.pairs, run.long = rr.pairs - base_pairs, rr.long - base_long
    except BaseException as exc:  # noqa: BLE001 — report and go on
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        run.error = f"{type(exc).__name__}: {exc}"
    return run


def _mean(rows, key):
    return statistics.fmean(r[key] for r in rows) if rows else float("nan")


def _pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(runs: List[Run]) -> Dict[str, Any]:
    """Per-config means (overall and per slice), latency, and per-query
    nDCG wins/losses against the first config that ran."""
    base = next((r for r in runs if not r.error), None)
    base_by = {(x["id"], x["target"]): x["ndcg"] for x in base.rows} if base else {}
    out = []
    for run in runs:
        entry: Dict[str, Any] = {"config": run.config, "error": run.error}
        if not run.error:
            rows = run.rows
            slices: Dict[str, Dict[str, float]] = {}
            for key in sorted({f"{r['target']}/{r['lang']}" for r in rows}):
                part = [r for r in rows if f"{r['target']}/{r['lang']}" == key]
                slices[key] = {"n": len(part), "ndcg": _mean(part, "ndcg")}
            wins = losses = ties = 0
            lost = []
            for r in rows:
                b = base_by.get((r["id"], r["target"]))
                if b is None:
                    continue
                if r["ndcg"] > b + 1e-9:
                    wins += 1
                elif r["ndcg"] < b - 1e-9:
                    losses += 1
                    lost.append(f"{r['id']}/{r['target']}")
                else:
                    ties += 1
            ms = [r["ms"] for r in rows]
            entry.update({
                "n": len(rows),
                "ndcg": _mean(rows, "ndcg"), "mrr": _mean(rows, "mrr"),
                "recall": _mean(rows, "recall"), "distinct": _mean(rows, "distinct"),
                "p50_ms": _pct(ms, 0.5), "p95_ms": _pct(ms, 0.95),
                "collapsed": sum(r["collapsed"] for r in rows),
                "wins": wins, "losses": losses, "ties": ties, "lost": lost,
                "slices": slices,
                "long_share": (run.long / (run.long + run.pairs)
                               if run.long + run.pairs else None),
            })
        out.append(entry)
    return {"baseline": base.config if base else None, "configs": out}


def markdown(summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """The summary as a Markdown report."""
    lines = [f"# Search eval — {meta['labels']}", "",
             f"{meta['queries']} queries, k={meta['k']}, db `{meta['db']}`, "
             f"baseline `{summary['baseline']}`.", "",
             "| config | n | nDCG@k | MRR | recall | distinct | W/L/T vs base "
             "| p50 ms | p95 ms | long texts | collapsed |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in summary["configs"]:
        if c["error"]:
            lines.append(f"| `{c['config']}` | — | error: {c['error']} |||||||||")
            continue
        long_share = "—" if c["long_share"] is None else f"{c['long_share']:.0%}"
        lines.append(
            f"| `{c['config']}` | {c['n']} | {c['ndcg']:.3f} | {c['mrr']:.3f} | "
            f"{c['recall']:.3f} | {c['distinct']:.2f} | "
            f"{c['wins']}/{c['losses']}/{c['ties']} | {c['p50_ms']:.0f} | "
            f"{c['p95_ms']:.0f} | {long_share} | {c['collapsed']} |")
    slices = sorted({s for c in summary["configs"] if not c["error"]
                     for s in c["slices"]})
    if slices:
        lines += ["", "## nDCG@k by slice (target/lang)", "",
                  "| config | " + " | ".join(slices) + " |",
                  "|---|" + "---|" * len(slices)]
        for c in summary["configs"]:
            if c["error"]:
                continue
            cells = []
            for s in slices:
                v = c["slices"].get(s)
                cells.append("—" if v is None else f"{v['ndcg']:.3f} (n={v['n']})")
            lines.append(f"| `{c['config']}` | " + " | ".join(cells) + " |")
    lost = [(c["config"], c["lost"]) for c in summary["configs"]
            if not c["error"] and c["lost"]]
    if lost:
        lines += ["", "## Losses vs baseline", ""]
        lines += [f"- `{name}`: {', '.join(ids)}" for name, ids in lost]
    return "\n".join(lines) + "\n"


def open_router(db: str, copy: bool):
    """Open ``db`` (or a temp copy of it, WAL included)."""
    from okfgraph.router import OKFRouter

    if copy:
        tmp = Path(tempfile.mkdtemp(prefix="okf-eval-"))
        shutil.copy2(db, tmp / Path(db).name)
        if Path(db + ".wal").exists():
            shutil.copy2(db + ".wal", tmp / (Path(db).name + ".wal"))
        db = str(tmp / Path(db).name)
    return OKFRouter(db_path=db, bundle_root=str(Path(db).parent))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", help="JSONL label file")
    ap.add_argument("--db", required=True, help="graph database to search")
    ap.add_argument("--copy", action="store_true",
                    help="search a temp copy (the live db may be held by a server)")
    ap.add_argument("--configs", default="rrf,hub,cross",
                    help="comma-separated configs, the first is the baseline "
                    "(default: rrf,hub,cross)")
    ap.add_argument("--split", default=None, help="only labels with this split")
    ap.add_argument("-k", type=int, default=10, help="cutoff (default: 10)")
    ap.add_argument("--noul-max-length", type=int, default=2048,
                    help="Decider2B context cap in tokens; it has no window "
                    "mode, so longer passages are truncated (default: 2048)")
    ap.add_argument("--offline", action="store_true",
                    help="never download model weights (HF_HUB_OFFLINE=1)")
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--md", default=None, help="write the Markdown summary here")
    args = ap.parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    labels = load_labels(args.labels, args.split)
    configs = [parse_config(s) for s in args.configs.split(",") if s.strip()]
    for cfg in configs:
        if cfg.base == "noul":
            cfg.max_length = args.noul_max_length
    router = open_router(args.db, args.copy)
    try:
        own = router.search_engine.reranker
        runs = []
        for cfg in configs:
            print(f"… {cfg.name}", file=sys.stderr, flush=True)
            # One session per config, dropped after it: kept sessions (and
            # their grown CUDA arenas) slowed later configs past CPU speed.
            runs.append(run_config(router, cfg, labels, args.k, {}))
            router.search_engine.reranker = own
            gc.collect()
    finally:
        router.close()
    summary = summarize(runs)
    meta = {"labels": args.labels, "db": args.db, "k": args.k,
            "queries": len(labels), "split": args.split}
    report = markdown(summary, meta)
    print(report)
    if args.md:
        Path(args.md).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"meta": meta, "summary": summary,
             "runs": [run.__dict__ for run in runs]}, indent=1, default=str),
            encoding="utf-8")
    return 1 if any(r.error for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
