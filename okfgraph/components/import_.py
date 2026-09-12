from __future__ import annotations

import base64
import hashlib
import heapq
import json
import logging
import math
import os
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

def parse_source_file(
    file_path: Path, root: Path
) -> Tuple["ConceptModel", str, str]:
    """Parse a .md/.txt source into ``(ConceptModel, body, concept_id)``.

    Module-level so the structural diff can parse bundle directories without
    a database-backed manager. ``ImportManager._parse_source_file`` delegates
    here; behaviour is identical (path-derived id, ``type``/``title``
    synthesis, frontmatter ``id:`` preserved as ``uid``).
    """
    post = frontmatter.load(file_path)
    body = post.content
    fm = dict(post.metadata)

    rel_path = file_path.relative_to(root) if file_path.is_relative_to(root) else None
    if rel_path is not None:
        # with_suffix("") strips only the final extension (.md/.txt/.markdown),
        # avoiding the old str.replace(".md","") which could corrupt paths.
        concept_id = str(rel_path.with_suffix("")).replace("\\", "/")
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

    concept = ConceptModel.model_validate({**fm, "id": concept_id, "body": body})
    return concept, body, concept_id


class ImportManager:
    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")
    def __init__(self, conn, bundle_root, _write_lock_ctx, enable_chunking,
                 schema_mgr, delta_mgr, embed_engine, image_mgr, purge_mgr,
                 token_counter, context_window=8192):
        self.conn = conn
        self.bundle_root = bundle_root
        self._write_lock_ctx = _write_lock_ctx
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
                self.conn.execute("""
                    MERGE (p:Directory {id: $parent})
                    MERGE (d:Directory {id: $child})
                    MERGE (p)-[:CONTAINS]->(d)
                """, {"parent": parent, "child": d})
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
                self.conn.execute("""
                    MERGE (d:Directory {id: $parent})
                    MERGE (c:Concept {id: $child})
                    MERGE (d)-[:CONTAINS]->(c)
                """, {"parent": parent_dir, "child": cid})


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
            out.append((source_id, resolve_wiki(raw, maps, known_ids), raw))
        return out

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


    def _import_bundle_inner(
        self,
        bundle_path: Optional[Path],
        batch_size: int,
        mode: "str | IngestMode",
        purge_deleted: bool,
    ) -> List[str]:
        """Inner implementation of import_bundle (called under write lock)."""
        mode = IngestMode.coerce(mode)
        root = bundle_path or self.bundle_root
        # Single walk, partitioned: reserved names (index.md, ...) are graph
        # noise, never knowledge — but counted in the log so silent loss is
        # impossible. Shared predicate with diff/delta/lint (is_concept_file).
        candidates = sorted(
            fp for fp in root.rglob("*")
            if fp.is_file() and fp.suffix.lower() in SOURCE_EXTS
        )
        source_files = [fp for fp in candidates if is_concept_file(fp)]
        if len(source_files) != len(candidates):
            logger.info(
                "skipping %d reserved file(s) (index.md, ...)",
                len(candidates) - len(source_files),
            )
        if not source_files:
            return []

        _t0 = time.monotonic()

        # Phase 0: Delta detection — directory-level hash aggregation.
        # Skips entire subtrees when directory hash is unchanged.
        _t1 = time.monotonic()
        changed, deleted = self.delta_mgr._changed_directories(source_files)
        logger.info("directory-delta: %d changed, %d deleted (%.1fs)", len(changed), len(deleted), time.monotonic() - _t1)

        # Purge deleted concepts if requested.
        if purge_deleted and deleted:
            cid_map = self.delta_mgr._load_file_hash_concept_ids()
            for path in deleted:
                cid = cid_map.get(path)
                if cid:
                    self.purge_mgr._purge_concept(cid)
            logger.info("purged %d deleted concept(s)", len(deleted))

        if not changed and not deleted:
            return []
        source_files = changed

        # Update FileHash table after purge (so deleted entries are removed).
        # This ensures the cid_map used by purge has the correct entries.
        file_hashes: Dict[str, str] = {}
        for fp in source_files:
            rel = str(fp.relative_to(self.bundle_root))
            h = self.delta_mgr._file_hash(fp)
            file_hashes[rel] = h
        self.delta_mgr._store_file_hashes(file_hashes)

        # Phase 1: Parse all files
        _t1 = time.monotonic()
        parsed: List[Dict[str, Any]] = []
        for fp in source_files:
            try:
                concept, body, cid = self._parse_source_file(fp, root)
                search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
                parsed.append({
                    "concept": concept,
                    "search_text": search_text,
                    "body": body,
                    "cid": cid,
                    "dir": fp.parent,
                })
            except Exception as e:
                logger.warning("parse failed %s: %s", fp.name, e)

        # No-op guard: if nothing parsed successfully, do no work (no encode, no
        # transaction, no index rebuild).
        if not parsed:
            return []

        logger.info("parsed %d concept(s)", len(parsed))

        # Phase 2: Batch encode (chunked by batch_size)
        _t1 = time.monotonic()
        all_search_texts = [p["search_text"] for p in parsed]
        all_embeddings: List[List[float]] = []
        for i in range(0, len(all_search_texts), batch_size):
            chunk = all_search_texts[i : i + batch_size]
            batch_embs = self.embed_engine._encode_batch(chunk, task="Document")
            all_embeddings.extend(batch_embs)
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

        # Delete old chunks for this document (re-import)
        self.conn.execute(
            "MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk) DETACH DELETE ch",
            {"id": cid},
        )

        chunks = self.embed_engine._split_into_chunks(body, cid)
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
    ) -> Dict[str, Any]:
        """Import a single concept using the full pipeline (encode, upsert, chunk, etc.).

        This is the core shared logic between ingest_md() and ingest_thoughts().
        """
        mode = IngestMode.coerce(mode)

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
            # Delete old chunks for this document (re-import)
            self.conn.execute(
                "MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk) DETACH DELETE ch",
                {"id": concept.id},
            )

            chunks = self.embed_engine._split_into_chunks(body, concept.id)
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
                    self.conn.execute("""
                        MERGE (d:Directory {id: $parent})
                        MERGE (c:Concept {id: $child})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, {"parent": parent, "child": child})
                else:
                    self.conn.execute("""
                        MERGE (p:Directory {id: $parent})
                        MERGE (d:Directory {id: $child})
                        MERGE (p)-[:CONTAINS]->(d)
                    """, {"parent": parent, "child": child})

        # NOTE: link extraction intentionally lives outside _insert_concept
        # (callers run _extract_links_for_concept after the node exists), so
        # path-links and [[wikilinks]] share one resolution path.

        return concept_id_val


    def _parse_source_file(
        self, file_path: Path, root: Path
    ) -> Tuple[ConceptModel, str, str]:
        """Parse a .md/.txt source into ``(ConceptModel, body, concept_id)``.

        Frontmatter is used when present. Plain-text files (and Markdown lacking
        frontmatter) get a synthesized ``type`` ('note') and a ``title`` derived
        from the filename, so the simplified text-only pipeline can ingest .txt
        alongside .md without every file needing OKF frontmatter.

        Delegates to :func:`parse_source_file` (shared with the diff tool).
        """
        return parse_source_file(file_path, root)


    def import_bundle(
        self,
        bundle_path: Optional[Path] = None,
        batch_size: int = 32,
        mode: "str | IngestMode" = IngestMode.TEXT,
        purge_deleted: bool = False,
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

        Returns:
            List of imported concept IDs.
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._import_bundle_inner(bundle_path, batch_size, mode, purge_deleted)


    def import_from_okf(
        self,
        file_path: Path,
        mode: "str | IngestMode" = IngestMode.TEXT,
        rebuild_indexes: bool = True,
    ) -> str:
        """Parse an OKF .md/.txt file and create/update the concept in the graph.

        Args:
            file_path: Path to the source file (``.md``, ``.markdown`` or ``.txt``).
            mode: Image ingestion mode — ``text`` (alt-text / filename fallback,
                no omni model), ``optional`` (omni only for images without
                alt-text), or ``omni`` (omni for every image).

        Returns the concept ID (relative path without its extension).
        """
        mode = IngestMode.coerce(mode)

        # 1-2. Parse frontmatter/body and build the Concept model.
        concept, body, concept_id = self._parse_source_file(file_path, self.bundle_root)

        # 2.5. Chunk the body (NEW)
        if self.enable_chunking:
            # Delete old chunks for this document (re-import)
            self.conn.execute(
                "MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk) DETACH DELETE ch",
                {"id": concept_id},
            )

            chunks = self.embed_engine._split_into_chunks(body, concept_id)
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

