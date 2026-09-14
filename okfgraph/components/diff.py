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
from typing import Any, Dict, List, Optional, Set, Tuple

from okfgraph.components.import_ import (
    in_skipped_dir,
    is_concept_file,
    parse_source_file,
)
from okfgraph.components.roots import qualify_alias_link
from okfgraph.components.links import (
    build_name_index,
    extract_md_links,
    extract_wikilinks,
    is_external,
    normalize_path_link,
    resolve_wiki,
)

logger = logging.getLogger(__name__)

# NOTE: file enumeration uses is_concept_file() from import_ (extensions +
# reserved-name skip) so diff and import agree on what a concept file is.


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


def _parse_dir(bundle_dir: Path, alias: str = ""):
    """Parse one tree into (concepts, raw_links, index_concepts)."""
    concepts: Dict[str, Dict[str, Any]] = {}
    raw_links: Dict[str, Dict[str, List[str]]] = {}
    index_concepts = []
    files = sorted(fp for fp in Path(bundle_dir).rglob("*")
                    if is_concept_file(fp) and not in_skipped_dir(fp))
    for fp in files:
        try:
            concept, body, cid = parse_source_file(fp, Path(bundle_dir), alias)
        except Exception as e:
            logger.warning("diff: skipping unparsable %s: %s", fp, e)
            continue
        extra = concept.model_extra or {}
        concepts[cid] = {
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
    return concepts, raw_links, index_concepts


def _resolve_state(concepts, raw_links, index_concepts, aliases=()) -> DiffState:
    """Resolve parsed links against the union name index."""
    state = DiffState()
    state.concepts = concepts
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
            target = resolve_wiki(qualify_alias_link(raw, aliases), maps, known)
            if target is not None:
                state.edges.add((cid, target))
            else:
                state.broken.add((cid, raw))
    return state


def state_of_dir(bundle_dir: Path, alias: str = "", aliases=()) -> DiffState:
    """Parse a bundle directory into a DiffState (no database, no import)."""
    concepts, raw_links, index_concepts = _parse_dir(bundle_dir, alias)
    own = {alias} if alias else set()
    return _resolve_state(concepts, raw_links, index_concepts, own | set(aliases))


class DiffManager:
    """Diffs involving the live graph (dir-vs-dir uses ``state_of_dir``)."""

    def __init__(self, conn, roots=None, primary=None):
        self.conn = conn
        # Multi-root drift (§2.8): {alias: Path} + primary tree. {} means
        # legacy single-tree (explicit dir or primary).
        self.roots = {a: Path(p) for a, p in (roots or {}).items()}
        self.primary = Path(primary) if primary is not None else None

    def _alias_for(self, bundle_dir) -> str:
        """Namespace alias for an explicit drift tree ("" = legacy)."""
        from okfgraph.components.roots import resolve_alias_for_path
        try:
            rp = Path(bundle_dir).resolve()
        except OSError:
            return ""
        for alias, rpath in self.roots.items():
            if rp == Path(rpath):
                return alias
        return resolve_alias_for_path(rp, self.roots) or ""

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

    def diff_dirs(self, old: Path, new: Path, alias: str = "") -> Dict[str, Any]:
        """Snapshot mode: two bundle directories, no database needed."""
        return self.compare(state_of_dir(old, alias), state_of_dir(new, alias))

    def diff_db_dir(
        self, bundle_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Drift mode: live graph (old) vs bundle directorie(s) (new).

        Multi-root (0.4.0, Phase 2 §2.8): no explicit dir drifts against
        EVERY configured tree (primary + named roots, alias-aware parse);
        an explicit dir drifts against that tree alone (alias resolved).
        """
        old_state = self.state_of_db()
        if bundle_dir is not None:
            alias = ""
            if self.roots:
                alias = self._alias_for(bundle_dir)
            return self.compare(old_state, state_of_dir(bundle_dir, alias))
        if not self.roots:
            return self.compare(old_state, state_of_dir(self.primary))
        # Union state: parse every present tree, resolve links once
        # against the combined index (cross-root links are not drift).
        concepts: Dict[str, Dict[str, Any]] = {}
        raw_links: Dict[str, Dict[str, List[str]]] = {}
        index_concepts = []
        for alias, root in [("", self.primary)] + list(self.roots.items()):
            if root is None or not Path(root).is_dir():
                continue  # unmounted ≠ drift
            c, l, i = _parse_dir(root, alias)
            concepts.update(c)
            raw_links.update(l)
            index_concepts.extend(i)
        merged = _resolve_state(concepts, raw_links, index_concepts,
                                set(self.roots))
        return self.compare(old_state, merged)
