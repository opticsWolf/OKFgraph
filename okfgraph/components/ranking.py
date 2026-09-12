"""Model-free retrieval: lexical seeds + exact Personalized PageRank.

Mirrors the okf-ingest ``seeds`` → multi-seed PPR cascade: a query is scored
lexically against concept metadata, the top seeds weight an exact
power-iteration PPR over the resolved ``LINKS_TO`` graph, and the ranked
concepts come back — with no embedding model loaded. Over a sparse graph this
degrades gracefully to seed order (== the lexical baseline).

Determinism rules (all load-bearing for the golden fixtures):
  - seed ties break by concept id ascending;
  - edges iterate in sorted ``(src, dst)`` index order;
  - the L1 convergence sum accumulates in ascending node order;
  - scores round half-even to 10 decimals (Python ``round``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Fixed stopword list — keep in sync with tests/fixtures, never grow silently.
STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "with", "that", "this",
    "from", "how", "what", "when", "where", "which", "does", "did",
    "can", "could", "should", "would", "will", "has", "have", "had",
    "not", "its", "our", "your", "their", "about", "into", "over",
    "under", "why", "who", "whom",
})

TITLE_HIT = 3
META_HIT = 2   # description or tags
BODY_HIT = 1


def query_tokens(query: str) -> List[str]:
    """Lowercase alphanumeric tokens, len >= 3, minus stopwords (sorted)."""
    return sorted({
        m for m in TOKEN_RE.findall((query or "").lower())
        if len(m) >= 3 and m not in STOPWORDS
    })


def seeds(
    concepts: Sequence[Mapping[str, Any]],
    query: str,
    k: int = 20,
) -> List[Tuple[str, float]]:
    """Lexically score concepts against the query.

    ``+3`` per distinct token in the title, ``+2`` in description/tags,
    ``+1`` in the body. Returns ``(concept_id, score)`` sorted by score
    desc, id asc, truncated at ``k``. Pure function — no DB, no model.
    """
    toks = query_tokens(query)
    if not toks:
        return []
    out: List[Tuple[str, float]] = []
    for c in concepts:
        cid = c.get("id", "")
        if not cid:
            continue
        title = str(c.get("title") or "").lower()
        desc = str(c.get("description") or "").lower()
        tags = c.get("tags") or []
        tags_text = " ".join(str(t) for t in tags).lower() if isinstance(tags, list) else str(tags).lower()
        body = str(c.get("body") or "").lower()
        score = 0
        for t in toks:
            if t in title:
                score += TITLE_HIT
            if t in desc or t in tags_text:
                score += META_HIT
            if t in body:
                score += BODY_HIT
        if score > 0:
            out.append((cid, float(score)))
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out[:k]


def ppr(
    nodes: Sequence[str],
    directed_edges: Sequence[Tuple[str, str]],
    starts: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    damping: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 200,
    k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Exact Personalized PageRank over the undirected link graph.

    ``nodes`` need not be sorted (sorted internally); ``directed_edges`` are
    symmetrized, self-loops dropped. Dangling mass returns to the seed
    distribution. Unknown start ids raise ``KeyError``.
    """
    ids = sorted(set(nodes))
    idx = {nid: i for i, nid in enumerate(ids)}
    missing = [s for s in starts if s not in idx]
    if missing:
        raise KeyError(f"start concept not found: {', '.join(missing)}")
    w = list(weights) if weights is not None else [1.0] * len(starts)
    if len(w) != len(starts) or sum(w) <= 0 or any(x < 0 for x in w):
        raise ValueError("weights must be non-negative, same length as starts, positive sum")

    n = len(ids)
    edges: Set[Tuple[int, int]] = set()
    for s, d in directed_edges:
        if s == d or s not in idx or d not in idx:
            continue
        edges.add((idx[s], idx[d]))
        edges.add((idx[d], idx[s]))
    ordered = sorted(edges)
    deg = [0] * n
    for s, _ in ordered:
        deg[s] += 1

    seed = [0.0] * n
    for s, x in zip(starts, w):
        seed[idx[s]] += x
    total = sum(seed)
    seed = [x / total for x in seed]

    p = list(seed)
    for _ in range(max_iter):
        contrib = [0.0] * n
        for s, d in ordered:  # fixed order -> deterministic fp
            if p[s] != 0.0:
                contrib[d] += p[s] / deg[s]
        dangling = sum(p[i] for i in range(n) if deg[i] == 0)
        nxt = [
            (1.0 - damping) * seed[i] + damping * (contrib[i] + dangling * seed[i])
            for i in range(n)
        ]
        delta = sum(abs(nxt[i] - p[i]) for i in range(n))  # ascending order
        p = nxt
        if delta < tol:
            break

    rows = [
        (nid, round(p[i], 10))
        for i, nid in enumerate(ids)
        if round(p[i], 10) > 0.0
    ]
    rows.sort(key=lambda kv: (-kv[1], kv[0]))
    return rows[:k] if k is not None else rows


def seed_ranked_ppr(
    concepts: Sequence[Mapping[str, Any]],
    directed_edges: Sequence[Tuple[str, str]],
    query: str,
    seed_k: int = 20,
    k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """One-call cascade: lexical seeds (as weights) → multi-seed PPR."""
    found = seeds(concepts, query, k=seed_k)
    if not found:
        return []
    ids = [cid for cid, _ in found]
    weights = [s for _, s in found]
    known = {c.get("id") for c in concepts}
    return ppr(
        [c.get("id", "") for c in concepts if c.get("id") in known],
        directed_edges, ids, weights, k=k,
    )
