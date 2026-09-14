from __future__ import annotations

import base64
import hashlib
import heapq
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse

import mordant
import numpy as np
import yaml
import frontmatter
from okfgraph.models import ChunkModel, ConceptModel
from okfgraph.images import IngestMode
from okfgraph.components.roots import (
    namespaced_id,
    resolve_alias_for_path,
)
from okfgraph.components.links import (
    build_name_index,
    extract_md_links,
    extract_wikilinks,
    is_external,
    normalize_path_link,
    resolve_wiki,
)

logger = logging.getLogger(__name__)

#: Bulk import skips these filenames: generated navigation files are graph
#: noise (self-linking ``*/index`` nodes that dilute PPR), never knowledge.
#: Explicit single-file ``import <path>/index.md`` still works — explicit
#: beats implicit. Mirrors google-okf's ``RESERVED_FILENAMES``.
RESERVED_FILENAMES = frozenset({"index.md"})

#: Extensions import (and diff/delta/lint) treat as concept sources.
SOURCE_EXTS = (".md", ".markdown", ".txt")

#: Directory names never walked for concept sources (0.4.0): tool output,
#: caches, and dependency trees are graph noise — and can be enormous
#: (.venv LICENSE files, target/ build products). Any dot-dir is skipped
#: (.git, .obsidian, .pytest_cache, ...); named non-hidden tool dirs join
#: them. Shared by import, detach-verify, diff, lint, and delta hashing so
#: all agree on the bundle's file set.
SKIP_DIR_NAMES = frozenset({
    "target", "node_modules", "__pycache__", "venv", "dist", "build",
})


def in_skipped_dir(fp: Path) -> bool:
    """True when any path component is a skipped directory."""
    return any(
        part.startswith(".") or part in SKIP_DIR_NAMES
        for part in fp.parts
    )


def is_concept_file(fp: Path) -> bool:
    """True when ``fp`` is a bulk-importable concept file.

    Single choke point shared by import, diff, delta, and lint so all agree
    on what a "concept file" is (drift mode would otherwise flag every
    exported bundle as changed: import skips index.md, diff counts it).
    """
    return (
        fp.is_file()
        and fp.suffix.lower() in SOURCE_EXTS
        and fp.name.lower() not in RESERVED_FILENAMES
    )


_USERINFO_RE = re.compile(
    r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<userinfo>[^/@]*@)(?P<rest>.*)$",
)


def sanitize_resource(uri: Any) -> Any:
    """Strip credentials from a ``resource:`` URI (``u:p@host`` → ``***@host``).

    Connection strings landing verbatim in frontmatter would persist secrets
    into the graph, exports, and vaults. Only the authority section
    (between ``://`` and the next ``/``) is rewritten, so bare paths,
    ``mailto:``, anchor-only targets, and ``okf-asset://`` (UUID, no
    userinfo) pass through untouched. Non-strings pass through as-is.
    """
    if not isinstance(uri, str):
        return uri
    m = _USERINFO_RE.match(uri.strip())
    if not m:
        return uri
    return f"{m.group('scheme')}***@{m.group('rest')}"

def parse_source_file(
    file_path: Path, root: Path, alias: str = ""
) -> Tuple["ConceptModel", str, str]:
    """Parse a .md/.txt source into ``(ConceptModel, body, concept_id)``.

    Module-level so the structural diff can parse bundle directories without
    a database-backed manager. ``ImportManager._parse_source_file`` delegates
    here; behaviour is identical (path-derived id, ``type``/``title``
    synthesis, frontmatter ``id:`` preserved as ``uid``).

    Multi-root (0.4.0, Phase 2 §2.1): a non-empty ``alias`` namespaces the
    path-derived ID (``@alias/rel``); the outside-root bare-stem fallback
    stays bare — it carries no root relationship.
    """
    post = frontmatter.load(file_path)
    body = post.content
    fm = dict(post.metadata)

    rel_path = file_path.relative_to(root) if file_path.is_relative_to(root) else None
    if rel_path is not None:
        # with_suffix("") strips only the final extension (.md/.txt/.markdown),
        # avoiding the old str.replace(".md","") which could corrupt paths.
        concept_id = str(rel_path.with_suffix("")).replace("\\", "/")
        concept_id = namespaced_id(alias, concept_id)
    else:
        # File lives outside bundle_root (common when a GUI writes each .md
        # next to its source). Fall back to the bare stem so the import
        # doesn't crash with a relative_to ValueError.
        concept_id = file_path.stem

    if not fm.get("type"):
        fm["type"] = "note"
    if not fm.get("title"):
        stem = file_path.stem.replace("_", " ").replace("-", " ").strip()
        fm["title"] = stem or concept_id

    # A frontmatter ``id:`` (Foam/Dendron-style stable identity) must not
    # overwrite the path-derived concept id — but dropping it would lose
    # a wikilink name. Preserve it as ``uid`` (extra MAP): the name index
    # resolves it, and export writes it back as ``id:``.
    fm_id = fm.pop("id", None)
    if fm_id is not None and str(fm_id) != concept_id:
        fm["uid"] = str(fm_id)

    # Hygiene: never persist credentials hiding in ``resource:`` URIs
    # (connection strings would otherwise land in graph → export → vault).
    if fm.get("resource") is not None:
        fm["resource"] = sanitize_resource(fm["resource"])

    concept = ConceptModel.model_validate({**fm, "id": concept_id, "body": body})
    return concept, body, concept_id


def _length_bucketed_encode(
    texts: List[str],
    encode_batch_size: int,
    encode_fn,
    progress=None,
    token_budget: Optional[int] = None,
    count_fn=None,
) -> List[Any]:
    """Encode texts shortest-first in small buckets, restoring input order.

    Concept search_texts span three orders of magnitude (a one-line stub
    vs a 32K-truncated architecture doc). The encoder pads every sequence
    to the batch max, so naive batching inflates 31 small docs sharing a
    batch with one giant into a 32x8192-token forward (tens of GB, tens
    of minutes on CPU). Sorting by length bounds each bucket's padding to
    its own max; vectors are identical (padding is masked out), only the
    compute order changes. ``progress(done, total, longest)`` is called
    after each bucket for visibility on large imports.

    With ``token_budget`` (and ``count_fn``), buckets are additionally
    capped by total tokens (greedy bins over token-sorted texts, at least
    one text per bin): a long text can no longer drag 7 others into a
    giant padded forward, which bounds peak ORT memory regardless of the
    configured truncation ceiling.
    """
    if token_budget is not None and count_fn is not None:
        return _token_bucketed_encode(
            texts, encode_batch_size, encode_fn, progress,
            token_budget, count_fn,
        )
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: List[Any] = [None] * len(texts)
    total = (len(order) + encode_batch_size - 1) // encode_batch_size
    for b, start in enumerate(range(0, len(order), encode_batch_size)):
        idx = order[start:start + encode_batch_size]
        embs = encode_fn([texts[i] for i in idx])
        for i, emb in zip(idx, embs):
            out[i] = emb
        if progress is not None:
            progress(b + 1, total, max(len(texts[i]) for i in idx))
    return out


def _token_bucketed_encode(
    texts: List[str],
    encode_batch_size: int,
    encode_fn,
    progress,
    token_budget: int,
    count_fn,
) -> List[Any]:
    """Greedy token-capped bins over token-sorted texts (see above)."""
    counts = [int(count_fn(t)) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: counts[i])
    bins: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0
    for i in order:
        if cur and (len(cur) >= encode_batch_size
                     or cur_tokens + counts[i] > token_budget):
            bins.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += counts[i]
    if cur:
        bins.append(cur)
    out: List[Any] = [None] * len(texts)
    for b, idx in enumerate(bins):
        embs = encode_fn([texts[i] for i in idx])
        for i, emb in zip(idx, embs):
            out[i] = emb
        if progress is not None:
            progress(b + 1, len(bins), max(len(texts[i]) for i in idx))
    return out


class ImportManager:
    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")
    def __init__(self, conn, bundle_root, _write_lock_ctx, enable_chunking,
                 schema_mgr, delta_mgr, embed_engine, image_mgr, purge_mgr,
                 token_counter, context_window=8192, db_path=None, roots=None):
        self.conn = conn
        self.bundle_root = bundle_root
        # Multi-root (0.4.0, Phase 2 §2.1): {alias: resolved Path} for trees
        # beyond the primary bundle_root (validated by the router). {}
        # means legacy single-root: every behaviour below is unchanged.
        self.roots = {a: Path(p) for a, p in (roots or {}).items()}
        # One delta detector per namespace, sharing the connection. The
        # primary ("") detector is the router-owned delta_mgr, so
        # single-root call sites keep working untouched.
        from okfgraph.components.delta import DeltaDetector as _DD
        self._delta_by_alias = {"": delta_mgr}
        for _a, _rp in self.roots.items():
            self._delta_by_alias[_a] = _DD(conn, _rp, _a)
        self._write_lock_ctx = _write_lock_ctx
        # Database location for detach's WILL-NOT-SURVIVE walk (0.2.16): the
        # graph's own files (.db + ladybug sidecars) are never source content.
        try:
            self._db_path = Path(db_path).resolve() if db_path else None
        except OSError:
            self._db_path = None
        # Token counting for the context-window guard comes from the Rust
        # encoder (count_tokens); no transformers tokenizer exists anymore.
        self.token_counter = token_counter
        self.context_window = context_window
        self.enable_chunking = enable_chunking
        self.schema_mgr = schema_mgr
        self.delta_mgr = delta_mgr
        self.embed_engine = embed_engine
        self.image_mgr = image_mgr
        self.purge_mgr = purge_mgr

    def _merge_directory_contains(self, parent_id: str, child_id: str):
        """Create a Directory→Directory CONTAINS edge without crashing Ladybug.

        Ladybug 0.20.3 segfaults when one Cypher statement combines an
        existing-node ``MERGE``, a new-node ``MERGE``, and a relationship
        ``MERGE`` on a long-lived connection. Separate idempotent statements
        preserve the same graph while avoiding that planner path.
        """
        self.conn.execute("MERGE (p:Directory {id: $id})", {"id": parent_id})
        self.conn.execute("MERGE (d:Directory {id: $id})", {"id": child_id})
        self.conn.execute("""
            MATCH (p:Directory {id: $parent}), (d:Directory {id: $child})
            MERGE (p)-[:CONTAINS]->(d)
        """, {"parent": parent_id, "child": child_id})

    def _merge_concept_contains(self, parent_id: str, child_id: str):
        """Create a Directory→Concept CONTAINS edge without crashing Ladybug.

        See ``_merge_directory_contains``: a single three-clause ``MERGE`` can
        segfault ladybug 0.20.3 after matching an existing Directory and
        creating the next sibling Concept. The separate statements below are
        semantically equivalent.
        """
        self.conn.execute("MERGE (d:Directory {id: $id})", {"id": parent_id})
        self.conn.execute("MERGE (c:Concept {id: $id})", {"id": child_id})
        self.conn.execute("""
            MATCH (d:Directory {id: $parent}), (c:Concept {id: $child})
            MERGE (d)-[:CONTAINS]->(c)
        """, {"parent": parent_id, "child": child_id})

    def _batch_build_directories(self, cids: List[str]):
        """Build directory hierarchy for a batch of concept IDs.

        Collects all unique directory paths and creates them in order
        (shallowest first) to ensure parents exist before children.
        """
        # Collect all unique directory paths
        dir_paths = set()
        for cid in cids:
            parts = cid.split("/")
            if len(parts) > 1:
                for i in range(1, len(parts)):
                    dir_paths.add("/".join(parts[:i]))

        # Sort by depth (shallowest first)
        sorted_dirs = sorted(dir_paths, key=lambda d: d.count("/"))

        # Create directory hierarchy
        for d in sorted_dirs:
            parent = "/".join(d.split("/")[:-1]) if "/" in d else None
            if parent and parent in dir_paths:
                self._merge_directory_contains(parent, d)
            elif parent:
                # Parent is root (not a directory node)
                self.conn.execute("""
                    MERGE (d:Directory {id: $child})
                """, {"child": d})
            else:
                self.conn.execute("""
                    MERGE (d:Directory {id: $child})
                """, {"child": d})

        # Link each concept to its parent directory
        for cid in cids:
            parts = cid.split("/")
            if len(parts) > 1:
                parent_dir = "/".join(parts[:-1])
                self._merge_concept_contains(parent_dir, cid)


    def _load_link_index(self):
        """Load the known-id set + wikilink name index over all concepts.

        Single pair of queries feeding every link-resolution site (batch,
        single, repair). ``aliases``/``alias``/``uid``/``title`` come from
        the ``extra`` MAP (JSON-decoded when needed); ambiguous names resolve
        to nothing (see ``okfgraph.components.links``).
        """
        rows = self.conn.execute(
            "MATCH (c:Concept) RETURN c.id, c.title, c.extra"
        ).rows_as_dict().get_all()
        concepts = []
        for r in rows:
            extra = r.get("c.extra") or {}
            entry: Dict[str, Any] = {"id": r["c.id"], "title": r.get("c.title")}
            for key in ("uid", "aliases", "alias"):
                val = extra.get(key)
                if isinstance(val, str) and val.startswith("["):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                if val is not None:
                    entry[key] = val
            concepts.append(entry)
        maps, ambiguous = build_name_index(concepts)
        if ambiguous:
            shown = sorted(ambiguous)[:10]
            logger.warning(
                "wikilinks: %d ambiguous name(s), resolving to nothing (e.g. %s)",
                len(ambiguous), ", ".join(shown),
            )
        return maps, {c["id"] for c in concepts}

    def _record_links(self, links: List[Tuple[str, Optional[str], str]]):
        """Persist resolved/unresolved ``(source, target_or_None, raw)`` links.

        Resolved targets get ``MERGE (source)-[:LINKS_TO]->(target)``;
        misses become ``BrokenLink`` records for ``repair_links``/doctor.
        """
        for source_id, target_id, raw in links:
            if target_id is not None:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": source_id, "target": target_id})
            else:
                link_id = f"{source_id}\u2192{raw}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": source_id, "target": raw, "ts": now})

    def _resolve_body_links(
        self, source_id: str, body: str, maps, known_ids
    ) -> List[Tuple[str, Optional[str], str]]:
        """Resolve one body to ``(source, target_or_None, raw)`` triples.

        Markdown ``](path.md)`` links resolve by path; ``[[wikilinks]]`` by
        name (uid → alias → title → stem). External URLs are skipped.
        """
        out: List[Tuple[str, Optional[str], str]] = []
        for raw in extract_md_links(body):
            if is_external(raw):
                continue
            target = normalize_path_link(raw.split("#", 1)[0])
            out.append((source_id, target if target in known_ids else None, target))
        for raw in extract_wikilinks(body):
            if not raw or is_external(raw):
                continue
            out.append((source_id,
                         resolve_wiki(self._qualify_alias_link(raw),
                                      maps, known_ids), raw))
        return out

    def _qualify_alias_link(self, raw: str) -> str:
        """Rewrite `[[alias/rest]]` → `[[@alias/rest]]` for known aliases.

        Multi-root (0.4.0, Phase 2 §2.4): the human-friendly form without
        `@` resolves via this pre-step; the `@`-form hits resolve_wiki's
        exact-id probe for free. Rule shared with diff snapshots
        (:func:`okfgraph.components.roots.qualify_alias_link`).
        """
        from okfgraph.components.roots import qualify_alias_link as _q
        return _q(raw, self.roots)

    def _batch_extract_links(self, parsed: List[Dict[str, Any]]):
        """Extract and create LINKS_TO relationships for a batch of concepts.

        Collects markdown links (by path) and wikilinks (by name) over the
        batch, resolves against every concept in the graph, and persists
        hits/misses in bulk via ``_record_links``.
        """
        maps, known_ids = self._load_link_index()
        all_links: List[Tuple[str, Optional[str], str]] = []
        for item in parsed:
            all_links.extend(
                self._resolve_body_links(item["cid"], item["body"], maps, known_ids)
            )
        if all_links:
            self._record_links(all_links)


    def _batch_upsert_concepts(
        self,
        parsed: List[Dict[str, Any]],
        all_embeddings: List[List[float]],
    ):
        """Upsert all concepts in one transaction.

        Deletes existing concepts, then creates new ones with embeddings.
        """
        # Collect IDs for bulk delete
        cids = [p["cid"] for p in parsed]
        if cids:
            # Bulk delete existing concepts. Use a bound parameter (never string
            # interpolation) and DETACH DELETE so concepts that already have
            # edges (LINKS_TO / CONTAINS / INCLUDES_ASSET) can be replaced on
            # re-import instead of raising a duplicated-primary-key error.
            self.conn.execute(
                "MATCH (c:Concept) WHERE c.id IN $ids DETACH DELETE c",
                {"ids": cids},
            )

        # Create all concepts
        for item, emb in zip(parsed, all_embeddings):
            concept = item["concept"]
            body = item["body"]
            concept_id_val = item["cid"]
            all_data = concept.model_dump()
            all_data.pop("body", None)
            all_data.pop("id", None)
            all_data.pop("embedding", None)  # embedding is passed separately, must not leak into extra MAP

            core = {
                "type": all_data.pop("type"),
                "title": all_data.pop("title", None),
                "description": all_data.pop("description", None),
                "resource": all_data.pop("resource", None),
                "tags": all_data.pop("tags", []),
                "timestamp": all_data.pop("timestamp", None),
            }

            extra = {
                k: json.dumps(v) if not isinstance(v, str) else v
                for k, v in all_data.items()
            }
            extra_keys = list(extra.keys())
            extra_values = list(extra.values())

            if isinstance(core["timestamp"], datetime):
                core["timestamp"] = core["timestamp"].isoformat()

            params: Dict[str, Any] = {
                "id": concept_id_val,
                "body": body,
                "embedding": emb,
                **core,
            }
            if extra_keys:
                params["extra_keys"] = extra_keys
                params["extra_values"] = extra_values
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding,
                        extra: MAP($extra_keys, $extra_values)
                    })
                """, params)
            else:
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding
                    })
                """, params)

        # Concepts changed -> search indexes are now stale. Bumping inside the
        # caller's transaction means the marker rolls back with the data if the
        # import aborts, keeping dirty-state consistent with what's committed.
        if parsed:
            self.schema_mgr._bump_write_epoch()


    def _extract_links_for_concept(self, concept_id: str, body: str):
        """Extract and create LINKS_TO relationships for a single concept.

        Handles markdown links [text](file.md) by path and [[wikilinks]] by
        name (uid → alias → title → stem); misses become BrokenLink records.
        """
        maps, known_ids = self._load_link_index()
        links = self._resolve_body_links(concept_id, body, maps, known_ids)
        if links:
            self._record_links(links)

    def _get_property(self, cid: str, prop: str) -> Any:
        """Retrieve a single property from a concept node."""
        result = self.conn.execute(
            f"MATCH (c:Concept {{id: $id}}) RETURN c.{prop}",
            {"id": cid},
        )
        row = result.rows_as_dict().get_all()
        return row[0][f"c.{prop}"] if row else None


    def _absent_roots(self) -> List[str]:
        """Aliases ("" = primary) whose tree is not currently present."""
        absent = []
        if not Path(self.bundle_root).is_dir():
            absent.append("")
        for alias, rpath in self.roots.items():
            if not Path(rpath).is_dir():
                absent.append(alias)
        return absent

    def _alias_for_root(self, root) -> str:
        """Namespace alias for an import tree: exact root match wins,
        else longest-prefix match, else "" (legacy bare IDs)."""
        try:
            rp = Path(root).resolve()
        except OSError:
            return ""
        for alias, rpath in self.roots.items():
            if rp == Path(rpath):
                return alias
        return resolve_alias_for_path(rp, self.roots) or ""

    def _detector_for(self, alias: str):
        """Delta detector for a namespace (primary "" = router-owned)."""
        return self._delta_by_alias.get(alias) or self._delta_by_alias[""]

    def _import_bundle_inner(
        self,
        bundle_path: Optional[Path],
        batch_size: int,
        mode: "str | IngestMode",
        purge_deleted: bool,
        alias: Optional[str] = None,
    ) -> List[str]:
        """Inner implementation of import_bundle (called under write lock)."""
        mode = IngestMode.coerce(mode)
        root = bundle_path or self.bundle_root
        if alias is None:
            alias = self._alias_for_root(root)
        det = self._detector_for(alias)
        # Purge is fail-closed in multi-root graphs (§2.3): consuming
        # tombstones while a root's state is unknown could purge live
        # concepts as "deleted". Single-root graphs keep legacy behaviour.
        # Suspended work-dir imports never purge (existing guard below),
        # so the gate skips them too — PDF auto-import must not refuse
        # over an unrelated unmounted root.
        if purge_deleted and self.roots and not det._suspended:
            absent = self._absent_roots()
            if absent:
                names = [a or "<primary>" for a in absent]
                raise RuntimeError(
                    f"purge refused: root(s) {names} not present — "
                    "remount them so every tree's state is known, then retry."
                )
        # Single walk, partitioned: reserved names (index.md, ...) are graph
        # noise, never knowledge — but counted in the log so silent loss is
        # impossible. Shared predicate with diff/delta/lint (is_concept_file).
        candidates = sorted(
            fp for fp in root.rglob("*")
            if fp.is_file() and fp.suffix.lower() in SOURCE_EXTS
            and not in_skipped_dir(fp)
        )
        source_files = [fp for fp in candidates if is_concept_file(fp)]
        if len(source_files) != len(candidates):
            logger.info(
                "skipping %d reserved file(s) (index.md, ...)",
                len(candidates) - len(source_files),
            )
        if not source_files and not Path(root).is_dir():
            # Absent tree (unmounted root, mistyped path): no baseline can
            # be computed — return before detection so nothing reads as
            # deleted (liveness invariant, §2.3; the loud skip lands there).
            # A PRESENT-but-empty tree is genuine (its last file was
            # deleted): fall through so detection tombstones it and
            # --purge-deleted can consume it. Without this, emptying a
            # root's final file could never purge (the old early return).
            return []
        if not alias:
            # Reserved-prefix rule (§2.1): a legacy tree containing top-level
            # `@*` entries would mint IDs that look namespaced but aren't
            # (usually a re-imported multi-root export — harmless, but a
            # future `alias` of the same name would collide). Warn, don't refuse.
            for _fp in source_files:
                try:
                    _rel = _fp.relative_to(root)
                except ValueError:
                    continue
                if _rel.parts and _rel.parts[0].startswith("@"):
                    logger.warning(
                        "bundle contains top-level '@'-prefixed entry %s: its "
                        "concept ID will look namespaced; avoid adding a root "
                        "alias with the same name",
                        _rel.parts[0],
                    )
                    break

        _t0 = time.monotonic()

        # Phase 0: Delta detection — directory-level hash aggregation.
        # Skips entire subtrees when directory hash is unchanged. Detection
        # is side-effect-free (0.2.15): dir_updates are persisted only after
        # concepts commit (Phase 3b), so a crash can never leave hashes
        # describing state newer than the graph.
        _t1 = time.monotonic()
        changed, deleted, dir_updates = det._changed_directories(source_files)
        logger.info("directory-delta: %d changed, %d deleted (%.1fs)", len(changed), len(deleted), time.monotonic() - _t1)

        # Record deletions durably: tombstones survive no-purge runs so a
        # later --purge-deleted still sees them.
        if deleted:
            det._record_deletions(deleted)

        # Purge deleted concepts if requested — consumes this run's
        # detections plus tombstones left by earlier no-purge runs.
        # Skipped for suspended (work-dir) imports: a temp import must
        # never tombstone real concepts.
        if purge_deleted and not det._suspended:
            pending = det._load_pending_deletions()
            if pending:
                purged = 0
                cid_map = det._load_file_hash_concept_ids()
                for path in pending:
                    cid = cid_map.get(path)
                    if cid and self.purge_mgr._purge_concept(cid):
                        purged += 1
                    else:
                        # Stale tombstone (already purged, or wedged with no
                        # concept): resolve it so it stops re-reporting.
                        det._clear_deletion(path)
                logger.info("purged %d deleted concept(s)", purged)
                if len(pending) > purged:
                    logger.warning(
                        "purge: %d tombstone(s) had no matching concept and were cleared",
                        len(pending) - purged,
                    )

        if not changed:
            return []
        source_files = changed

        # Phase 1: Parse all files. Failures are per-file warnings; failed
        # files get no hash state so they retry next run (their directories
        # are dropped from dir_updates in Phase 3b).
        _t1 = time.monotonic()
        parsed: List[Dict[str, Any]] = []
        failed_files: List[Path] = []
        for fp in source_files:
            try:
                concept, body, cid = self._parse_source_file(fp, root, alias)
                search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
                parsed.append({
                    "concept": concept,
                    "search_text": search_text,
                    "body": body,
                    "cid": cid,
                    "dir": fp.parent,
                    "fp": fp,
                })
            except Exception as e:
                failed_files.append(fp)
                logger.warning("parse failed %s: %s", fp.name, e)
        if failed_files:
            logger.warning(
                "parse: %d file(s) failed — excluded from this import, will retry next run",
                len(failed_files),
            )

        # No-op guard: if nothing parsed successfully, do no work (no encode, no
        # transaction, no index rebuild).
        if not parsed:
            return []

        logger.info("parsed %d concept(s)", len(parsed))

        # Phase 2: Batch encode (length-bucketed; see _length_bucketed_encode).
        # batch_size drives DB writes; encoding uses small buckets so one
        # giant doc cannot pad 31 small ones into a 32x8192-token forward.
        # The token budget additionally caps each forward (~16K tokens ≈
        # 1GB ORT arena), so raising --max-length cannot OOM the box.
        _t1 = time.monotonic()
        all_search_texts = [p["search_text"] for p in parsed]
        encode_batch_size = min(batch_size, 8)
        def _encode_progress(done, total, longest):
            logger.info(
                "encode: bucket %d/%d (longest %d chars, %.1fs)",
                done, total, longest, time.monotonic() - _t1,
            )
        all_embeddings: List[List[float]] = _length_bucketed_encode(
            all_search_texts,
            encode_batch_size,
            lambda bucket: self.embed_engine._encode_batch(bucket, task="Document"),
            progress=_encode_progress,
            token_budget=16384,
            count_fn=self.embed_engine.count_tokens,
        )
        logger.info("encode: %d texts in %.1fs", len(all_search_texts), time.monotonic() - _t1)

        # Phase 3: Batch upsert all concepts in a single transaction
        _t1 = time.monotonic()
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self._batch_upsert_concepts(parsed, all_embeddings)
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        logger.info("upsert: %d concepts in %.1fs", len(parsed), time.monotonic() - _t1)

        # Phase 3b: Crash-consistency (0.2.15) — persist hashes only for
        # files whose concepts committed, and only now. Directories holding
        # failed files are dropped from dir_updates so the next run
        # re-walks them (plus their successfully parsed siblings — wasted
        # work only, never a silent skip).
        file_hashes: Dict[str, str] = {}
        for p in parsed:
            fp = p["fp"]
            rel = str(fp.relative_to(root))
            file_hashes[rel] = det._file_hash(fp)
        det._store_file_hashes(file_hashes)
        for fp in failed_files:
            # dir_updates keys are DB (namespaced) keys — pop via the
            # detector's key space, not the native rel.
            from okfgraph.components.roots import prefix_key as _pk
            dir_updates.pop(_pk(det.alias, str(fp.parent.relative_to(root))), None)
        det._store_directory_hashes(dir_updates)

        # Phase 3.5: Chunk all documents (NEW) — per-concept error isolation
        _import_chunk_errors: List[Tuple[str, Exception]] = []
        _t1 = time.monotonic()
        if self.enable_chunking:
            for p in parsed:
                try:
                    self._import_chunks_for_concept(p)
                except Exception as e:
                    _import_chunk_errors.append((p["cid"], e))
                    logger.warning(
                        "chunk import failed for %s: %s",
                        p["cid"], e,
                    )

            if _import_chunk_errors:
                logger.warning(
                    "chunk import: %d concept(s) failed out of %d",
                    len(_import_chunk_errors), len(parsed),
                )

            logger.info("chunk: %d concepts in %.1fs", len(parsed), time.monotonic() - _t1)
        else:
            _t1 = time.monotonic()

        # Phase 4: Batch directory hierarchy (collected from all concept IDs)
        _t1 = time.monotonic()
        self._batch_build_directories([p["cid"] for p in parsed])
        logger.info("directories: %d in %.1fs", len(parsed), time.monotonic() - _t1)

        # Phase 5: Batch link extraction
        _t1 = time.monotonic()
        self._batch_extract_links(parsed)
        logger.info("links: %d concepts in %.1fs", len(parsed), time.monotonic() - _t1)

        # Phase 6: Image ingestion (per concept, honouring the selected mode)
        _import_image_errors: List[Tuple[str, Exception]] = []
        _t1 = time.monotonic()
        for p in parsed:
            try:
                self.image_mgr._ingest_concept_images(p["cid"], p["body"], p["dir"], mode)
            except Exception as e:
                _import_image_errors.append((p["cid"], e))

        if _import_image_errors:
            logger.warning(
                "image ingestion: %d concept(s) failed out of %d",
                len(_import_image_errors), len(parsed),
            )
        logger.info("images: %d concepts in %.1fs", len(parsed), time.monotonic() - _t1)

        # Phase 7: rebuild vector + FTS indexes once so every concept/image
        # written above is searchable (indexes reflect table contents at build
        # time, not subsequent inserts).
        _t1 = time.monotonic()
        self.schema_mgr._build_search_indexes(rebuild=True)
        logger.info("reindex: %.1fs", time.monotonic() - _t1)

        # Aggregate failure report
        total_failures = len(_import_chunk_errors) + len(_import_image_errors)
        if total_failures > 0:
            logger.warning(
                "import_bundle: %d concept(s) had non-fatal errors "
                "(chunks: %d, images: %d) out of %d total",
                total_failures,
                len(_import_chunk_errors),
                len(_import_image_errors),
                len(parsed),
            )

        elapsed = time.monotonic() - _t0
        logger.info(
            "import_bundle: %d concept(s) in %.1fs",
            len(parsed), elapsed,
        )

        return [p["cid"] for p in parsed]


    def _import_chunks_for_concept(self, parsed_item: Dict[str, Any]) -> None:
        """Split, encode, and upsert chunks for a single concept.

        Isolated so that a failure for one concept doesn't block the rest of
        the bundle. Also checks context-window occupancy and logs a warning
        when a chunk exceeds 90%% of the tokenizer's limit.
        """
        body = parsed_item["body"]
        cid = parsed_item["cid"]

        # Delete old chunks for this document (re-import). Match by
        # parent_doc_id, NOT by PART_OF edge: concept replacement (Phase 3
        # DETACH DELETE) orphans chunks by dropping edges first, so an
        # edge-joined delete finds nothing and the re-CREATE dies on
        # duplicate primary keys (0.2.20: 22 stale-chunk docs on reimport).
        self.conn.execute(
            "MATCH (ch:Chunk {parent_doc_id: $id}) DETACH DELETE ch",
            {"id": cid},
        )

        # Token-measured cap: the same counter that sizes the payloads
        # below also sizes the split, so the cap is in true tokens.
        chunks = self.embed_engine._split_into_chunks(
            body, cid, count_tokens=self.token_counter)
        if not chunks:
            return

        # Compute overlap payloads for embedding
        payloads = self.embed_engine._compute_overlap_payloads(chunks)
        texts = [p["text"] for p in payloads]

        # --- Context-window warning (Gap #14a) ---
        ctx_window = self.context_window
        threshold = int(ctx_window * 0.9)
        for i, t in enumerate(texts):
            token_count = self.token_counter(t)
            if token_count >= threshold:
                logger.warning(
                    "chunk %s#%d is %d tokens (%.0f%% of context window %d). "
                    "Consider reducing chunk_size or splitting the document.",
                    cid, i, token_count,
                    100.0 * token_count / ctx_window,
                    ctx_window,
                )

        embeddings = self.embed_engine._encode_batch(texts, task="Document")

        self.conn.execute("BEGIN TRANSACTION")
        try:
            for payload, emb in zip(payloads, embeddings):
                chunk_id = payload["chunk_id"]
                # Find the original chunk for metadata
                orig_chunk = next(
                    c for c in chunks
                    if c["chunk_index"] == int(chunk_id.split(":")[-1])
                )
                self.conn.execute("""
                    CREATE (ch:Chunk {
                        id: $id, parent_doc_id: $doc_id,
                        chunk_index: $idx, chunk_text: $text,
                        block_type: $block_type,
                        start_offset: $start, end_offset: $end_offset,
                        embedding: $emb
                    })
                """, {
                    "id": chunk_id,
                    "doc_id": cid,
                    "idx": orig_chunk["chunk_index"],
                    "text": orig_chunk["chunk_text"],
                    "block_type": orig_chunk["block_type"],
                    "start": orig_chunk["start_offset"],
                    "end_offset": orig_chunk["end_offset"],
                    "emb": emb,
                })
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        self.schema_mgr._bump_write_epoch()

        # Create PART_OF relationships between Concept and its Chunks
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for chunk in chunks:
                chunk_id = f"{cid}#chunk:{chunk['chunk_index']}"
                self.conn.execute("""
                    MATCH (d:Concept {id: $doc})
                    MATCH (ch:Chunk {id: $cid})
                    MERGE (d)-[:PART_OF]->(ch)
                """, {"doc": cid, "cid": chunk_id})
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise


    def _import_single_concept(
        self,
        concept: "ConceptModel",
        body: str,
        mode: "str | IngestMode" = IngestMode.TEXT,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Import a single concept using the full pipeline (encode, upsert, chunk, etc.).

        This is the core shared logic between ingest_md() and ingest_thoughts().
        """
        mode = IngestMode.coerce(mode)
        # Rootless addressed write: force bypasses the detached refusal,
        # but never re-attaches (no mirror relationship to verify).
        self._require_attached(force)

        # Phase 2: Encode (Document prefix)
        search_text = (
            f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
        )
        concept.embedding = self.embed_engine._encode(search_text, task="Document")

        # Phase 3: Upsert in transaction
        self._insert_concept(concept, body, concept.id)

        # Phase 3.5: Chunk
        chunk_count = 0
        if self.enable_chunking:
            # Delete old chunks for this document (re-import) — by
            # parent_doc_id, not PART_OF edge (see _import_chunks_for_concept:
            # concept replacement orphans chunks first).
            self.conn.execute(
                "MATCH (ch:Chunk {parent_doc_id: $id}) DETACH DELETE ch",
                {"id": concept.id},
            )

            chunks = self.embed_engine._split_into_chunks(
                body, concept.id, count_tokens=self.token_counter)
            if chunks:
                # Compute overlap payloads for embedding
                payloads = self.embed_engine._compute_overlap_payloads(chunks)
                texts = [p["text"] for p in payloads]
                embeddings = self.embed_engine._encode_batch(texts, task="Document")

                self.conn.execute("BEGIN TRANSACTION")
                try:
                    for payload, emb in zip(payloads, embeddings):
                        chunk_id = payload["chunk_id"]
                        # Find the original chunk for metadata
                        orig_chunk = next(
                            c for c in chunks
                            if c["chunk_index"] == int(chunk_id.split(":")[-1])
                        )
                        self.conn.execute("""
                            CREATE (ch:Chunk {
                                id: $id, parent_doc_id: $doc_id,
                                chunk_index: $idx, chunk_text: $text,
                                block_type: $block_type,
                                start_offset: $start, end_offset: $end_offset,
                                embedding: $emb
                            })
                        """, {
                            "id": chunk_id,
                            "doc_id": concept.id,
                            "idx": orig_chunk["chunk_index"],
                            "text": orig_chunk["chunk_text"],
                            "block_type": orig_chunk["block_type"],
                            "start": orig_chunk["start_offset"],
                            "end_offset": orig_chunk["end_offset"],
                            "emb": emb,
                        })
                    self.conn.execute("COMMIT")
                except Exception:
                    try:
                        self.conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                self.schema_mgr._bump_write_epoch()

                chunk_count = len(chunks)

                # Create PART_OF relationships
                self.conn.execute("BEGIN TRANSACTION")
                try:
                    for chunk in chunks:
                        chunk_id = f"{concept.id}#chunk:{chunk['chunk_index']}"
                        self.conn.execute("""
                            MATCH (d:Concept {id: $doc})
                            MATCH (ch:Chunk {id: $cid})
                            MERGE (d)-[:PART_OF]->(ch)
                        """, {"doc": concept.id, "cid": chunk_id})
                    self.conn.execute("COMMIT")
                except Exception:
                    try:
                        self.conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise

        # Phase 5: Links
        self._extract_links_for_concept(concept.id, body)

        # Phase 6: Images
        image_count = 0
        try:
            image_count = len(
                self.image_mgr._ingest_concept_images(concept.id, body, Path("."), mode)
            )
        except Exception:
            pass

        # Phase 7: Rebuild indexes
        self.schema_mgr._build_search_indexes(rebuild=True)

        return {
            "concept_id": concept.id,
            "title": concept.title,
            "description": concept.description,
            "tags": concept.tags,
            "chunk_count": chunk_count,
            "image_count": image_count,
        }


    def _insert_concept(
        self,
        concept: ConceptModel,
        body_text: str,
        concept_id_val: str,
    ) -> str:
        """Internal helper: upsert a single concept into the graph."""
        all_data = concept.model_dump()
        embedding_vec = all_data.pop("embedding", None)
        all_data.pop("body", None)
        all_data.pop("id", None)

        core = {
            "type": all_data.pop("type"),
            "title": all_data.pop("title", None),
            "description": all_data.pop("description", None),
            "resource": all_data.pop("resource", None),
            "tags": all_data.pop("tags", []),
            "timestamp": all_data.pop("timestamp", None),
        }

        extra = {
            k: json.dumps(v) if not isinstance(v, str) else v
            for k, v in all_data.items()
        }
        extra_keys = list(extra.keys())
        extra_values = list(extra.values())

        if isinstance(core["timestamp"], datetime):
            core["timestamp"] = core["timestamp"].isoformat()

        # Atomic upsert
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.execute(
                "MATCH (c:Concept {id: $id}) DETACH DELETE c",
                {"id": concept_id_val},
            )
            params: Dict[str, Any] = {
                "id": concept_id_val,
                "body": body_text,
                "embedding": embedding_vec,
                **core,
            }
            if extra_keys:
                params["extra_keys"] = extra_keys
                params["extra_values"] = extra_values
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding,
                        extra: MAP($extra_keys, $extra_values)
                    })
                """, params)
            else:
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding
                    })
                """, params)
            self.schema_mgr._bump_write_epoch()  # concept changed -> indexes dirty
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        # Build Directory Hierarchy
        path_parts = concept_id_val.split("/")
        if len(path_parts) > 1:
            for i in range(1, len(path_parts)):
                parent = "/".join(path_parts[:i])
                child = "/".join(path_parts[: i + 1])
                if i == len(path_parts) - 1:
                    self._merge_concept_contains(parent, child)
                else:
                    self._merge_directory_contains(parent, child)

        # NOTE: link extraction intentionally lives outside _insert_concept
        # (callers run _extract_links_for_concept after the node exists), so
        # path-links and [[wikilinks]] share one resolution path.

        return concept_id_val


    def _parse_source_file(
        self, file_path: Path, root: Path, alias: str = ""
    ) -> Tuple[ConceptModel, str, str]:
        """Parse a .md/.txt source into ``(ConceptModel, body, concept_id)``.

        Frontmatter is used when present. Plain-text files (and Markdown lacking
        frontmatter) get a synthesized ``type`` ('note') and a ``title`` derived
        from the filename, so the simplified text-only pipeline can ingest .txt
        alongside .md without every file needing OKF frontmatter.

        Delegates to :func:`parse_source_file` (shared with the diff tool).
        """
        return parse_source_file(file_path, root, alias)


    def import_bundle(
        self,
        bundle_path: Optional[Path] = None,
        batch_size: int = 32,
        mode: "str | IngestMode" = IngestMode.TEXT,
        purge_deleted: bool = False,
        force: bool = False,
        alias: Optional[str] = None,
    ) -> List[str]:
        """Import an entire OKF bundle directory with batched encoding.

        Walks the bundle directory, parses all .md files, generates
        embeddings in batched ONNX forward passes, and upserts them.

        Args:
            bundle_path: Root directory of the OKF bundle (defaults to constructor bundle_root).
            batch_size: Number of texts per ONNX forward pass.
            mode: Image ingestion mode (``text`` | ``optional`` | ``omni``).
            purge_deleted: If True, concepts whose source files were deleted
                from disk are removed from the graph (including chunks,
                links, and orphaned image assets).
            force: Bypass the detached-graph refusal (0.2.16). The root must
                match the recorded source tree; on success the graph
                re-attaches (detached state cleared, baseline rebuilt).

        Returns:
            List of imported concept IDs.
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            if bundle_path is not None or not self.roots:
                # Explicit tree, or legacy single-root graph: one import.
                # `alias` is the PDF work-dir ingest namespace (§2.1); it
                # never re-attaches (a temp dir never matches provenance).
                # NOTE: an explicit --bundle pins ONE tree even with --all;
                # omit --bundle to import every configured root.
                reattach = self._require_attached(
                    force, bundle_path or self.bundle_root
                )
                _one = self._import_bundle_inner(
                    bundle_path, batch_size, mode, purge_deleted,
                    alias=alias,
                )
                if self.roots:
                    logger.warning(
                        "import: single tree (%s): %d concept(s) — other "
                        "roots untouched (omit --bundle for all roots)",
                        bundle_path or self.bundle_root, len(_one),
                    )
                ids = _one
                if reattach:
                    self.delta_mgr.clear_detached()
                return ids
            # Multi-root (Phase 2 §2.1–§2.3): primary tree (bare IDs) plus
            # one import per named root (namespaced IDs). Absent roots are
            # SKIPPED with a loud warning — unmounted ≠ deleted (§2.3).
            # A --force re-attach requires the FULL configured root set to
            # match the recorded provenance (open question 3, strict
            # reading): a partial mirror must never clear detached state.
            if force and self.delta_mgr.is_detached():
                absent = self._absent_roots()
                if absent:
                    names = [a or "<primary>" for a in absent]
                    raise RuntimeError(
                        f"re-attach refused: root(s) {names} not present — "
                        "remount the full recorded source tree, then retry."
                    )
                recorded = {
                    r.get("alias", ""): r.get("path")
                    for r in (self.delta_mgr.get_detached_state() or {}).get("roots", [])
                }
                configured = {"": str(Path(self.bundle_root).resolve())}
                configured.update({a: str(Path(p).resolve())
                                   for a, p in self.roots.items()})
                if recorded != configured:
                    raise RuntimeError(
                        "re-attach refused: configured roots differ from the "
                        f"recorded source tree ({recorded}). Create a new "
                        "database for a different tree."
                    )
            all_ids: List[str] = []
            for _alias, _root in [("", self.bundle_root)] + list(self.roots.items()):
                if not Path(_root).is_dir():
                    logger.warning(
                        "import: root %s (%s) not present — skipped, "
                        "nothing tombstoned (unmounted is not deleted)",
                        _alias or "<primary>", _root,
                    )
                    continue
                reattach = self._require_attached(force, _root)
                _ids = self._import_bundle_inner(
                    _root, batch_size, mode, purge_deleted, alias=_alias
                )
                logger.info(
                    "import: root %s: %d concept(s)",
                    _alias or "<primary>", len(_ids),
                )
                all_ids.extend(_ids)
                if reattach:
                    self.delta_mgr.clear_detached()
            return all_ids


    def _require_attached(self, force: bool, root: Optional[Path] = None) -> bool:
        """Refuse mirror writes on a detached graph (0.2.16).

        Args:
            force: Bypass the refusal (explicit re-attach / addressed write).
            root: Bundle root of this call, for provenance matching. None
                for rootless writes (single md / thoughts / file import),
                which carry no mirror relationship to verify.

        Returns True when the caller must clear detached state on success
        (force re-attach with a matching root). Raises RuntimeError otherwise.
        """
        if not self.delta_mgr.is_detached():
            return False
        if not force:
            raise RuntimeError(
                "graph is detached from its bundle (see `okf detach`): mirror "
                "writes are refused. Re-attach with --force (the bundle root "
                "must match the recorded source tree)."
            )
        if root is None:
            return False
        want = str(Path(root).resolve())
        roots = (self.delta_mgr.get_detached_state() or {}).get("roots", [])
        if not any(r.get("path") == want for r in roots):
            known = roots[0]["path"] if roots else "<unknown>"
            raise RuntimeError(
                f"graph was detached from a different source tree ({known}); "
                f"refusing --force import from {want}. Create a new database "
                "for a different tree."
            )
        return True

    def detach(
        self,
        bundle_path: Optional[Path] = None,
        verify: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """End the mirror relationship between this graph and its bundle (0.2.16).

        After detach the database is the artifact: reads, searches,
        traverses, exports, recover, and index rebuilds keep working, but
        mirror writes refuse until a `--force` re-attach against the same
        source tree. The FileHash/DirHash/DeletedPath baseline is dropped -
        there is no baseline without a mirror.

        Args:
            bundle_path: Source tree to verify against (defaults to the
                constructor bundle_root). Pass explicitly when the working
                directory is not the original tree.
            verify: Compare sources against graph content first (default on).
            force: Proceed despite mismatches / untracked files / source-only
                artifacts (the user declares the database the artifact).

        Returns a report dict with the verify lists and recorded provenance.
        Raises RuntimeError when verification fails without --force, the
        bundle is missing with verify on, or the graph is already detached.
        """
        if self.delta_mgr.is_detached():
            state = self.delta_mgr.get_detached_state() or {}
            since = state.get("detached_at")
            raise RuntimeError(
                "graph is already detached"
                + (f" (since epoch {since})" if since else "")
                + "."
            )
        # Multi-root whole-graph detach (§2.5): every configured tree is
        # verified (with its alias) and recorded as its own SourceRoot row.
        # An explicit bundle_path keeps the legacy single-tree meaning.
        if self.roots and bundle_path is None:
            return self._detach_multi(verify=verify, force=force)
        root = Path(bundle_path) if bundle_path is not None else Path(self.bundle_root)
        report: Dict[str, Any] = {
            "bundle": str(root),
            "verified": False,
            "mismatched": [],
            "untracked": [],
            "non_source_files": [],
            "already_sourceless": 0,
            "file_count": 0,
        }
        # file_count fallback when there is no bundle to walk.
        baseline_rows = self.conn.execute(
            "MATCH (f:FileHash) RETURN count(f) AS n"
        ).rows_as_dict().get_all()
        baseline_count = baseline_rows[0]["n"] if baseline_rows else 0
        if verify:
            if not root.is_dir():
                raise RuntimeError(
                    f"bundle '{root}' not found - nothing to verify against. "
                    "Pass --bundle pointing at the source tree, or --no-verify "
                    "to detach an already-removed tree without verification."
                )
            self._verify_detach_fidelity(root, report)
            summary = (
                f"{len(report['mismatched'])} mismatched, "
                f"{len(report['untracked'])} untracked, "
                f"{len(report['non_source_files'])} source-only"
            )
            if report["mismatched"] and not force:
                sample = ", ".join(
                    f"{m['path']} [{', '.join(m['fields'])}]"
                    for m in report["mismatched"][:5]
                )
                raise RuntimeError(
                    f"detach refused: {summary}. Re-import first so the graph "
                    "matches the files, or pass --force to declare the "
                    f"database the artifact anyway. e.g. {sample}"
                )
            if (report["untracked"] or report["non_source_files"]) and not force:
                raise RuntimeError(
                    f"detach refused: {summary} - these files have no counterpart "
                    "in the graph and would not survive source deletion "
                    "(WILL-NOT-SURVIVE). Pass --force to acknowledge."
                )
            report["verified"] = True
        else:
            report["file_count"] = baseline_count
        # Drop the mirror baseline (no baseline without a mirror).
        self.conn.execute("MATCH (f:FileHash) DELETE f")
        self.conn.execute("MATCH (d:DirHash) DELETE d")
        self.conn.execute("MATCH (d:DeletedPath) DELETE d")
        report["provenance"] = self.delta_mgr.set_detached(
            str(root.resolve()), report["file_count"]
        )
        return report

    def _detach_multi(self, verify: bool, force: bool) -> Dict[str, Any]:
        """Whole-graph detach over primary + named roots (Phase 2 §2.5)."""
        targets = [("", Path(self.bundle_root))] + [
            (a, Path(p)) for a, p in self.roots.items()
        ]
        report: Dict[str, Any] = {
            "bundle": ", ".join(str(r) for _, r in targets),
            "verified": False,
            "mismatched": [],
            "untracked": [],
            "non_source_files": [],
            "already_sourceless": 0,
            "file_count": 0,
            "roots": {},
        }
        baseline_rows = self.conn.execute(
            "MATCH (f:FileHash) RETURN count(f) AS n"
        ).rows_as_dict().get_all()
        baseline_count = baseline_rows[0]["n"] if baseline_rows else 0
        per_root_counts: Dict[str, int] = {}
        if verify:
            for alias, root in targets:
                name = alias or "<primary>"
                if not root.is_dir():
                    raise RuntimeError(
                        f"root {name} ({root}) not found - nothing to verify "
                        "against. Remount it, or pass --no-verify to detach "
                        "without verification."
                    )
                before = report["file_count"]
                self._verify_detach_fidelity(root, report, alias)
                per_root_counts[alias] = report["file_count"] - before
                report["roots"][name] = {
                    "path": str(root),
                    "file_count": per_root_counts[alias],
                }
            summary = (
                f"{len(report['mismatched'])} mismatched, "
                f"{len(report['untracked'])} untracked, "
                f"{len(report['non_source_files'])} source-only"
            )
            if report["mismatched"] and not force:
                raise RuntimeError(
                    f"detach refused: {summary}. Re-import first so the graph "
                    "matches the files, or pass --force to declare the "
                    "database the artifact anyway."
                )
            if (report["untracked"] or report["non_source_files"]) and not force:
                raise RuntimeError(
                    f"detach refused: {summary} - these files have no counterpart "
                    "in the graph and would not survive source deletion "
                    "(WILL-NOT-SURVIVE). Pass --force to acknowledge."
                )
            report["verified"] = True
        else:
            report["file_count"] = baseline_count
        self.conn.execute("MATCH (f:FileHash) DELETE f")
        self.conn.execute("MATCH (d:DirHash) DELETE d")
        self.conn.execute("MATCH (d:DeletedPath) DELETE d")
        provenance = []
        for alias, root in targets:
            count = per_root_counts.get(alias, 0) if verify else 0
            provenance.append(self.delta_mgr.set_detached(
                str(root.resolve()), count, alias
            ))
        report["provenance"] = provenance
        return report

    def _is_db_file(self, fp: Path) -> bool:
        """True for the graph's own database + ladybug sidecars (.lock/.wal/...).

        Detach's source-only walk must not flag the database the user is
        detaching (it commonly lives inside the bundle root). Matches the db
        file itself plus same-directory `<dbname>.*` / `<dbname>-*` sidecars.
        """
        if self._db_path is None:
            return False
        try:
            rp = fp.resolve()
        except OSError:
            return False
        if rp == self._db_path:
            return True
        if rp.parent != self._db_path.parent:
            return False
        base = self._db_path.name
        return rp.name.startswith(base + ".") or rp.name.startswith(base + "-")

    def _verify_detach_fidelity(
        self, root: Path, report: Dict[str, Any], alias: str = ""
    ) -> None:
        """Fill the detach report by comparing bundle files to graph content.

        - concept file whose derived id has no stored concept -> `untracked`
          (never imported, or unparseable: content is NOT in the graph)
        - concept file whose content differs from the stored concept ->
          `mismatched` (dirty tree: re-import before detaching)
        - non-hidden file that is neither a concept file nor a reserved
          generated name -> `non_source_files` (originals the graph cannot
          reproduce: PDFs, images, ...)
        - baseline rows pointing at gone paths -> `already_sourceless`
          (info: that content already lives only in the graph)
        """
        candidates = sorted(
            fp for fp in root.rglob("*")
            if fp.is_file() and fp.suffix.lower() in SOURCE_EXTS
            and not in_skipped_dir(fp)
        )
        concept_files = [fp for fp in candidates if is_concept_file(fp)]
        report["file_count"] = report.get("file_count", 0) + len(concept_files)
        for fp in concept_files:
            rel = str(fp.relative_to(root))
            if alias:
                # Display namespaced so multi-root reports say which tree.
                rel = f"@{alias}/{rel}"
            try:
                concept, body, cid = self._parse_source_file(fp, root, alias)
            except Exception as e:
                report["untracked"].append(
                    {"path": rel, "reason": f"unparseable: {e}"}
                )
                continue
            rows = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN c.type AS type, "
                "c.title AS title, c.description AS description, "
                "c.resource AS resource, c.tags AS tags, c.body AS body",
                {"id": cid},
            ).rows_as_dict().get_all()
            if not rows:
                report["untracked"].append({"path": rel, "concept_id": cid})
                continue
            stored = rows[0]
            fields = []
            if (stored.get("type") or "") != (concept.type or ""):
                fields.append("type")
            if (stored.get("title") or "") != (concept.title or ""):
                fields.append("title")
            if (stored.get("description") or "") != (concept.description or ""):
                fields.append("description")
            if (stored.get("resource") or "") != (concept.resource or ""):
                fields.append("resource")
            if self._norm_tags(stored.get("tags")) != self._norm_tags(concept.tags):
                fields.append("tags")
            if (stored.get("body") or "") != (body or ""):
                fields.append("body")
            if fields:
                report["mismatched"].append(
                    {"path": rel, "concept_id": cid, "fields": fields}
                )
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            if in_skipped_dir(fp):
                continue
            if self._is_db_file(fp):
                continue
            rel = fp.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if fp.suffix.lower() in SOURCE_EXTS:
                if is_concept_file(fp):
                    continue
                if fp.name.lower() in RESERVED_FILENAMES:
                    continue  # generated files (index.md) - export reproduces them
            _ns = str(rel)
            if alias:
                _ns = f"@{alias}/{_ns}"
            report["non_source_files"].append(_ns)
        sourceless = 0
        for r in (
            self.conn.execute("MATCH (f:FileHash) RETURN f.path AS p")
            .rows_as_dict().get_all()
            or []
        ):
            # Only this root's namespace counts here (multi-root verify
            # runs once per root; another root's rows are not sourceless).
            from okfgraph.components.roots import strip_prefix as _sp
            _native = _sp(alias, r["p"])
            if _native is None:
                continue
            if not (root / _native).exists():
                sourceless += 1
        report["already_sourceless"] = report.get("already_sourceless", 0) + sourceless

    @staticmethod
    def _norm_tags(value: Any) -> List[str]:
        """Tags as a sorted string list regardless of storage shape."""
        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return [value]
        try:
            return sorted(str(x) for x in value)
        except TypeError:
            return [str(value)]

    def import_from_okf(
        self,
        file_path: Path,
        mode: "str | IngestMode" = IngestMode.TEXT,
        rebuild_indexes: bool = True,
        force: bool = False,
    ) -> str:
        """Parse an OKF .md/.txt file and create/update the concept in the graph.

        Args:
            file_path: Path to the source file (``.md``, ``.markdown`` or ``.txt``).
            mode: Image ingestion mode — ``text`` (alt-text / filename fallback,
                no omni model), ``optional`` (omni only for images without
                alt-text), or ``omni`` (omni for every image).

        Returns the concept ID (relative path without its extension).

        force: Bypass the detached-graph refusal (0.2.16). Rootless addressed
            write: never re-attaches.
        """
        mode = IngestMode.coerce(mode)
        self._require_attached(force)

        # 1-2. Parse frontmatter/body and build the Concept model.
        # Multi-root (§2.1/§2.6): a file inside a named root mints that
        # root's namespaced ID; outside every root keeps the legacy
        # bare-stem fallback.
        _alias = resolve_alias_for_path(file_path, self.roots) or ""
        _root = self.roots[_alias] if _alias else self.bundle_root
        concept, body, concept_id = self._parse_source_file(file_path, _root, _alias)

        # 2.5. Chunk the body (NEW)
        if self.enable_chunking:
            # Delete old chunks for this document (re-import) — by
            # parent_doc_id, not PART_OF edge (see _import_chunks_for_concept:
            # concept replacement orphans chunks first).
            self.conn.execute(
                "MATCH (ch:Chunk {parent_doc_id: $id}) DETACH DELETE ch",
                {"id": concept_id},
            )

            chunks = self.embed_engine._split_into_chunks(
                body, concept_id, count_tokens=self.token_counter)
            if chunks:
                # Compute overlap payloads for embedding
                payloads = self.embed_engine._compute_overlap_payloads(chunks)
                texts = [p["text"] for p in payloads]
                embeddings = self.embed_engine._encode_batch(texts, task="Document")

                self.conn.execute("BEGIN TRANSACTION")
                try:
                    for payload, emb in zip(payloads, embeddings):
                        chunk_id = payload["chunk_id"]
                        # Find the original chunk for metadata
                        orig_chunk = next(
                            c for c in chunks
                            if c["chunk_index"] == int(chunk_id.split(":")[-1])
                        )
                        self.conn.execute("""
                            CREATE (ch:Chunk {
                                id: $id, parent_doc_id: $doc_id,
                                chunk_index: $idx, chunk_text: $text,
                                block_type: $block_type,
                                start_offset: $start, end_offset: $end_offset,
                                embedding: $emb
                            })
                        """, {
                            "id": chunk_id,
                            "doc_id": concept_id,
                            "idx": orig_chunk["chunk_index"],
                            "text": orig_chunk["chunk_text"],
                            "block_type": orig_chunk["block_type"],
                            "start": orig_chunk["start_offset"],
                            "end_offset": orig_chunk["end_offset"],
                            "emb": emb,
                        })
                    self.conn.execute("COMMIT")
                except Exception:
                    try:
                        self.conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                self.schema_mgr._bump_write_epoch()

        # 3. Generate embedding (Document prefix)
        search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
        concept.embedding = self.embed_engine._encode(search_text, task="Document")

        # 4. Insert into graph (delegates to shared upsert logic)
        self._insert_concept(concept, body, concept_id)

        # 4.5. Create PART_OF relationships between Concept and its Chunks
        #      (must happen after _insert_concept so the Concept node exists)
        if self.enable_chunking:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                for chunk in chunks:
                    chunk_id = f"{concept_id}#chunk:{chunk['chunk_index']}"
                    self.conn.execute("""
                        MATCH (d:Concept {id: $doc})
                        MATCH (ch:Chunk {id: $cid})
                        MERGE (d)-[:PART_OF]->(ch)
                    """, {"doc": concept_id, "cid": chunk_id})
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        # 5. Ingest any embedded images under the requested mode
        self.image_mgr._ingest_concept_images(concept_id, body, file_path.parent, mode)

        # 6. Extract and create LINKS_TO relationships for this concept
        self._extract_links_for_concept(concept_id, body)

        # 7. Rebuild search indexes so this concept (and its images) are
        #    actually returned by vector/FTS search. Callers importing many
        #    files one-by-one can pass rebuild_indexes=False and call
        #    _build_search_indexes(rebuild=True) once at the end.
        if rebuild_indexes:
            self.schema_mgr._build_search_indexes(rebuild=True)

        return concept_id


    def list_broken_links(self) -> List[Dict[str, Any]]:
        """List all tracked broken links (references to concepts not yet imported)."""
        result = self.conn.execute("""
            MATCH (bl:BrokenLink)
            RETURN bl.source_id AS source, bl.target_id AS target, bl.timestamp AS timestamp
            ORDER BY bl.timestamp
        """)
        rows = result.rows_as_dict().get_all()
        return [
            {"source": r["source"], "target": r["target"], "timestamp": r["timestamp"]}
            for r in rows
        ]


    def repair_links(self, skip_sources: Optional[Set[str]] = None) -> int:
        """Attempt to repair broken links by re-checking if targets now exist.

        Scans all tracked broken links. A record repairs when its target is
        now an existing concept id — or when it is a ``[[wikiname]]`` that
        now resolves to exactly one concept via the name index (uid → alias
        → title → stem). Ambiguous names are left alone, never guessed.

        Returns:
            Number of links successfully repaired.
        """
        skip_sources = skip_sources or set()
        broken = self.list_broken_links()
        if not broken:
            return 0
        maps, known_ids = self._load_link_index()
        repaired = 0
        for link in broken:
            source_id = link["source"]
            raw_target = link["target"]
            if source_id in skip_sources:
                # reviewed:true concepts are never modified (doctor --fix).
                continue
            link_id = f"{source_id}→{raw_target}"

            if source_id not in known_ids:
                continue
            if raw_target in known_ids:
                target_id = raw_target
            else:
                # Path-style miss that is really a wikilink, or vice versa:
                # try name resolution before giving up.
                target_id = resolve_wiki(raw_target, maps, known_ids)
                if target_id is None:
                    continue

            self.conn.execute("""
                MATCH (source:Concept {id: $source})
                MATCH (target:Concept {id: $target})
                MERGE (source)-[:LINKS_TO]->(target)
            """, {"source": source_id, "target": target_id})
            # Remove the BrokenLink record
            self.conn.execute(
                "MATCH (bl:BrokenLink {id: $id}) DELETE bl",
                {"id": link_id},
            )
            repaired += 1

        return repaired

