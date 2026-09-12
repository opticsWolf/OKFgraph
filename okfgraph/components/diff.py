"""Structural diff: knowledge-structure comparison, not text hunks.

Compares two states — a bundle directory (parsed, never imported) or the live
graph — and reports concepts added/removed/changed, retitles/retypes, edge
deltas, and newly broken vs fixed links. Pure hash/set comparison; every list
sorted. Exit-code friendly (``identical`` flag for CI gates).

Drift mode (graph vs its bundle dir) answers "what changed since last import";
snapshot mode (dir vs dir) previews an import or diffs two vaults.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from okfgraph.components.import_ import parse_source_file
from okfgraph.components.links import (
    build_name_index,
    extract_md_links,
    extract_wikilinks,
    is_external,
    normalize_path_link,
    resolve_wiki,
)

logger = logging.getLogger(__name__)

SOURCE_EXTS = (".md", ".markdown", ".txt")


def content_hash(body: str) -> str:
    """SHA-256 over newline-normalized, stripped body text."""
    norm = (body or "").replace("\r\n", "\n").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@dataclass
class DiffState:
    """One diffable side: concept meta, resolved edges, unresolved links."""

    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    broken: Set[Tuple[str, str]] = field(default_factory=set)


def state_of_dir(bundle_dir: Path) -> DiffState:
    """Parse a bundle directory into a DiffState (no database, no import)."""
    state = DiffState()
    files = sorted(
        fp for fp in Path(bundle_dir).rglob("*")
        if fp.is_file() and fp.suffix.lower() in SOURCE_EXTS
    )
    raw_links: Dict[str, Dict[str, List[str]]] = {}
    index_concepts = []
    for fp in files:
        try:
            concept, body, cid = parse_source_file(fp, Path(bundle_dir))
        except Exception as e:
            logger.warning("diff: skipping unparsable %s: %s", fp, e)
            continue
        extra = concept.model_extra or {}
        state.concepts[cid] = {
            "title": concept.title,
            "type": concept.type,
            "hash": content_hash(body),
        }
        index_concepts.append({
            "id": cid,
            "title": concept.title,
            "uid": extra.get("uid"),
            "aliases": extra.get("aliases"),
            "alias": extra.get("alias"),
        })
        raw_links[cid] = {
            "md": extract_md_links(body),
            "wiki": extract_wikilinks(body),
        }
    maps, _ = build_name_index(index_concepts)
    known = set(state.concepts)
    for cid, links in raw_links.items():
        for raw in links["md"]:
            if is_external(raw):
                continue
            target = normalize_path_link(raw.split("#", 1)[0])
            if target in known:
                state.edges.add((cid, target))
            else:
                state.broken.add((cid, target))
        for raw in links["wiki"]:
            if not raw or is_external(raw):
                continue
            target = resolve_wiki(raw, maps, known)
            if target is not None:
                state.edges.add((cid, target))
            else:
                state.broken.add((cid, raw))
    return state


class DiffManager:
    """Diffs involving the live graph (dir-vs-dir uses ``state_of_dir``)."""

    def __init__(self, conn):
        self.conn = conn

    def state_of_db(self) -> DiffState:
        """Read the live graph into a DiffState."""
        state = DiffState()
        for r in self.conn.execute(
            "MATCH (c:Concept) RETURN c.id, c.title, c.type, c.body"
        ).rows_as_dict().get_all():
            state.concepts[r["c.id"]] = {
                "title": r.get("c.title"),
                "type": r.get("c.type"),
                "hash": content_hash(r.get("c.body") or ""),
            }
        for r in self.conn.execute(
            "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) "
            "RETURN a.id AS src, b.id AS dst"
        ).rows_as_dict().get_all():
            state.edges.add((r["src"], r["dst"]))
        for r in self.conn.execute(
            "MATCH (bl:BrokenLink) RETURN bl.source_id AS source, bl.target_id AS target"
        ).rows_as_dict().get_all():
            state.broken.add((r["source"], r["target"]))
        return state

    @staticmethod
    def compare(a: DiffState, b: DiffState) -> Dict[str, Any]:
        """Pure comparison of two states (a = old, b = new). All lists sorted."""
        a_ids, b_ids = set(a.concepts), set(b.concepts)
        added = sorted(b_ids - a_ids)
        removed = sorted(a_ids - b_ids)
        changed, retitled, retyped = [], [], []
        for cid in sorted(a_ids & b_ids):
            old, new = a.concepts[cid], b.concepts[cid]
            if old["hash"] != new["hash"]:
                changed.append(cid)
            if (old["title"] or "") != (new["title"] or ""):
                retitled.append({"id": cid, "old": old["title"], "new": new["title"]})
            if (old["type"] or "") != (new["type"] or ""):
                retyped.append({"id": cid, "old": old["type"], "new": new["type"]})
        edges_added = sorted(b.edges - a.edges)
        edges_removed = sorted(a.edges - b.edges)
        broken_new = sorted(b.broken - a.broken)
        broken_fixed = sorted(a.broken - b.broken)
        identical = not (
            added or removed or changed or retitled or retyped
            or edges_added or edges_removed or broken_new or broken_fixed
        )
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "retitled": retitled,
            "retyped": retyped,
            "edges_added": [list(e) for e in edges_added],
            "edges_removed": [list(e) for e in edges_removed],
            "broken_new": [list(e) for e in broken_new],
            "broken_fixed": [list(e) for e in broken_fixed],
            "identical": identical,
        }

    def diff_dirs(self, old: Path, new: Path) -> Dict[str, Any]:
        """Snapshot mode: two bundle directories, no database needed."""
        return self.compare(state_of_dir(old), state_of_dir(new))

    def diff_db_dir(self, bundle_dir: Path) -> Dict[str, Any]:
        """Drift mode: live graph (old) vs bundle directory (new)."""
        return self.compare(self.state_of_db(), state_of_dir(bundle_dir))
