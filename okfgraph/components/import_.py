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

logger = logging.getLogger(__name__)

class ImportManager:
    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")
    def __init__(self, conn, bundle_root, _write_lock_ctx, tokenizer, enable_chunking,
                 schema_mgr, delta_mgr, embed_engine, image_mgr, purge_mgr):
        self.conn = conn
        self.bundle_root = bundle_root
        self._write_lock_ctx = _write_lock_ctx
        self.tokenizer = tokenizer
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


    def _batch_extract_links(self, parsed: List[Dict[str, Any]]):
        """Extract and create LINKS_TO relationships for a batch of concepts.

        Collects all markdown links, checks which targets exist, and
        creates relationships or BrokenLink records in bulk.
        """
        # Collect all (source, target) pairs
        all_links: List[Tuple[str, str]] = []
        link_pattern = re.compile(r"\[.*?\]\((.*?\.md)\)")
        wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")
        for item in parsed:
            source_id = item["cid"]
            body = item["body"]
            for raw_link in link_pattern.findall(body):
                target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
                all_links.append((source_id, target_id))
            # Also handle wikilinks [[target]]
            for raw_link in wikilink_pattern.findall(body):
                target_id = raw_link.strip().lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
                all_links.append((source_id, target_id))

        if not all_links:
            return

        # Collect all unique target IDs
        all_targets = list(set(t for _, t in all_links))

        # Batch check which targets exist
        existing_targets = set()
        for target_id in all_targets:
            result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            cnt = result.rows_as_dict().get_all()[0]["cnt"]
            if cnt > 0:
                existing_targets.add(target_id)

        # Create LINKS_TO for existing targets
        for source_id, target_id in all_links:
            if target_id in existing_targets:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": source_id, "target": target_id})
            else:
                # Record broken link for later repair
                link_id = f"{source_id}\u2192{target_id}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": source_id, "target": target_id, "ts": now})


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

        Handles both markdown links [text](file.md) and wikilinks [[target]].
        """
        link_pattern = re.compile(r"\[.*?\]\((.*?\.md)\)")
        wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")

        all_links: List[Tuple[str, str]] = []
        for raw_link in link_pattern.findall(body):
            target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
            all_links.append((concept_id, target_id))
        for raw_link in wikilink_pattern.findall(body):
            target_id = raw_link.strip().lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
            all_links.append((concept_id, target_id))

        if not all_links:
            return

        # Collect all unique target IDs
        all_targets = list(set(t for _, t in all_links))

        # Batch check which targets exist
        existing_targets = set()
        for target_id in all_targets:
            result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            cnt = result.rows_as_dict().get_all()[0]["cnt"]
            if cnt > 0:
                existing_targets.add(target_id)

        # Create LINKS_TO for existing targets
        for source_id, target_id in all_links:
            if target_id in existing_targets:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": source_id, "target": target_id})
            else:
                # Record broken link for later repair
                link_id = f"{source_id}\u2192{target_id}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": source_id, "target": target_id, "ts": now})


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
        source_files = sorted(
            fp for fp in root.rglob("*")
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_SOURCE_EXTS
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
        ctx_window = self.tokenizer.model_max_length
        threshold = int(ctx_window * 0.9)
        for i, t in enumerate(texts):
            token_count = len(self.tokenizer.encode(t, add_special_tokens=False))
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

        # Extract Markdown links
        link_pattern = re.compile(r"\[.*?\]\((.*?\.md)\)")
        for raw_link in link_pattern.findall(body_text):
            target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
            # Check if target exists
            result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            cnt = result.rows_as_dict().get_all()[0]["cnt"]
            if cnt > 0:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": concept_id_val, "target": target_id})
            else:
                # Record broken link for later repair
                link_id = f"{concept_id_val}→{target_id}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": concept_id_val, "target": target_id, "ts": now})

        return concept_id_val


    def _parse_source_file(
        self, file_path: Path, root: Path
    ) -> Tuple[ConceptModel, str, str]:
        """Parse a .md/.txt source into ``(ConceptModel, body, concept_id)``.

        Frontmatter is used when present. Plain-text files (and Markdown lacking
        frontmatter) get a synthesized ``type`` ('note') and a ``title`` derived
        from the filename, so the simplified text-only pipeline can ingest .txt
        alongside .md without every file needing OKF frontmatter.
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

        concept = ConceptModel.model_validate({**fm, "id": concept_id, "body": body})
        return concept, body, concept_id


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


    def repair_links(self) -> int:
        """Attempt to repair broken links by re-checking if targets now exist.

        Scans all tracked broken links. For each one where the target concept
        now exists in the graph, creates the LINKS_TO relationship and removes
        the BrokenLink record.

        Returns:
            Number of links successfully repaired.
        """
        broken = self.list_broken_links()
        repaired = 0
        for link in broken:
            source_id = link["source"]
            target_id = link["target"]
            link_id = f"{source_id}→{target_id}"

            # Check if both source and target now exist
            source_result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": source_id},
            )
            target_result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            source_exists = source_result.rows_as_dict().get_all()[0]["cnt"] > 0
            target_exists = target_result.rows_as_dict().get_all()[0]["cnt"] > 0

            if source_exists and target_exists:
                # Create the LINKS_TO relationship
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

