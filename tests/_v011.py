"""Shared helpers for the 0.11.0 acceptance tests (``test_v011_*.py``).

The 0.11 changes are about identity, links, change detection, filters and
process ownership — none of them depends on what the embedding model says.
So these tests replace the encoder with a deterministic bag-of-words stand-in
(no model download, no ONNX) and count every text it encodes, which is what
the O2 acceptance criteria are stated in ("embeds exactly that file").

Vector and FTS search still run through Ladybug's real extensions.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

_WORD = re.compile(r"[A-Za-z0-9]+")


def _vec(text: str, dim: int) -> List[float]:
    acc = np.zeros(dim, dtype="float64")
    for w in _WORD.findall((text or "").lower()) or ["_empty_"]:
        h = hashlib.sha256(w.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        acc += rng.standard_normal(dim)
    n = np.linalg.norm(acc) or 1.0
    return (acc / n).astype("float32").tolist()


class EncodeCounter:
    """How many texts the (stub) encoder has embedded since ``reset``."""

    def __init__(self) -> None:
        self.count = 0

    def reset(self) -> None:
        self.count = 0


@pytest.fixture
def stub_encoder(monkeypatch) -> EncodeCounter:
    """Deterministic encoder; returns a counter of encoded texts."""
    from okfgraph.components import embedding as E

    counter = EncodeCounter()

    def _dim(self) -> int:
        return int(getattr(self, "_truncate_dim", 512) or 512)

    def encode(self, text, task="Document"):
        counter.count += 1
        return _vec(text, _dim(self))

    def encode_batch(self, texts, task="Document"):
        counter.count += len(texts)
        return [_vec(t, _dim(self)) for t in texts]

    monkeypatch.setattr(E.LazyRustEncoder, "encode", encode)
    monkeypatch.setattr(E.LazyRustEncoder, "encode_batch", encode_batch)
    monkeypatch.setattr(E.LazyRustEncoder, "count_tokens",
                        lambda self, text: len((text or "").split()))
    monkeypatch.setattr(E.LazyRustEncoder, "is_loaded", property(lambda self: True))
    return counter


def write(root: Path, rel: str, text: str) -> Path:
    """Write ``text`` to ``root/rel`` (creating directories)."""
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def note(title: str, body: str, **fm) -> str:
    """A markdown note with YAML frontmatter."""
    import yaml
    meta = {"title": title, **fm}
    return "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---\n" + body + "\n"


def links_into(router, target: str) -> List[tuple]:
    """``[(source, kind)]`` for every LINKS_TO edge into ``target``."""
    rows = router.conn.execute(
        "MATCH (a:Concept)-[l:LINKS_TO]->(b:Concept {id: $t}) "
        "RETURN a.id AS a, l.kind AS k", {"t": target},
    ).rows_as_dict().get_all()
    return sorted((r["a"], r["k"]) for r in rows)


def edges_from(router, source: str) -> List[tuple]:
    """``[(target, kind)]`` for every LINKS_TO edge out of ``source``."""
    rows = router.conn.execute(
        "MATCH (a:Concept {id: $s})-[l:LINKS_TO]->(b:Concept) "
        "RETURN b.id AS b, l.kind AS k", {"s": source},
    ).rows_as_dict().get_all()
    return sorted((r["b"], r["k"]) for r in rows)


def concept_ids(router) -> List[str]:
    rows = router.conn.execute("MATCH (c:Concept) RETURN c.id AS id").rows_as_dict().get_all()
    return sorted(r["id"] for r in rows)


def broken(router) -> List[Dict]:
    return [
        {k: b[k] for k in ("source", "target", "kind", "reason", "candidates")}
        for b in router.list_broken_links()
    ]
