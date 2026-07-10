"""SearchEngine — all read queries: hybrid search, chunk search, graph
traversal, hub-score reranking, and convenience lookups.

Query-vector encoding delegates to the injected EmbeddingEngine.
"""

from __future__ import annotations
import heapq
import json
import logging
import math
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from okfgraph.models import ChunkModel, ConceptModel

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(
        self,
        conn,
        tokenizer,
        embedding_dim: int,
        embed_engine,
    ):
        self.conn = conn
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embed_engine = embed_engine

    def search_chunks(
        self,
        query: str,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        limit: int = 10,
        max_chunks_per_doc: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search chunks using RRF-fused vector + FTS.

        Returns chunk-level results with optional parent concept metadata.
        Applies ``max_chunks_per_doc`` to limit how many chunks from the same
        document appear in results.
        """
        if not getattr(self, "_search_available", False):
            raise RuntimeError(
                "Search is unavailable: the 'vector'/'fts' extensions could not "
                "be loaded. Ensure the Ladybug extension repository is reachable, "
                "then reopen the router (ingestion and graph queries do not need "
                "these extensions)."
            )
        query_vec = self.embed_engine._encode(query, task="Query")

        # Stage 1: Vector search on chunks
        vec_results = self.conn.execute(
            "CALL QUERY_VECTOR_INDEX('Chunk', 'chunk_embedding', $vec, $k) RETURN node, distance",
            {"vec": query_vec, "k": limit * 3},
        )
        vec_rows = vec_results.rows_as_dict().get_all()
        vec_scores: Dict[str, float] = {}
        for row in vec_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                vec_scores[node_id] = 1 - row.get("distance", 0)

        # Stage 2: Full-text search on chunks
        fts_results = self.conn.execute(
            "CALL QUERY_FTS_INDEX('Chunk', 'chunk_fts', $query) RETURN node, score",
            {"query": query},
        )
        fts_rows = fts_results.rows_as_dict().get_all()
        fts_scores: Dict[str, float] = {}
        for row in fts_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                fts_scores[node_id] = row.get("score", 0)

        # Stage 3: RRF fusion (k=60)
        def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
            ordered = sorted(scores, key=scores.get, reverse=True)
            return {cid: i + 1 for i, cid in enumerate(ordered)}

        vec_rank = _rank_map(vec_scores)
        fts_rank = _rank_map(fts_scores)

        combined: List[Tuple[str, float]] = []
        for cid in set(vec_scores) | set(fts_scores):
            score = 0.0
            if cid in vec_rank:
                score += 1.0 / (60 + vec_rank[cid])
            if cid in fts_rank:
                score += 1.0 / (60 + fts_rank[cid])
            combined.append((cid, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Stage 4: Fetch chunk + parent metadata in bulk
        chunk_ids = [cid for cid, _ in combined]
        score_by_id = dict(combined)

        cypher = """
        MATCH (ch:Chunk)
        WHERE ch.id IN $ids
        OPTIONAL MATCH (parent:Concept)-[:PART_OF]->(ch)
        RETURN ch.id, ch.chunk_text, ch.block_type, ch.chunk_index,
               ch.parent_doc_id,
               parent.id AS parent_id, parent.title AS parent_title,
               parent.type AS parent_type, parent.tags AS parent_tags
        """
        rows = self.conn.execute(cypher, {"ids": chunk_ids}).rows_as_dict().get_all()
        meta_by_id = {row["ch.id"]: row for row in rows}

        # Assemble results, applying per-doc limit and graph filters
        results: List[Dict[str, Any]] = []
        doc_counts: Dict[str, int] = {}  # track chunks per doc

        for cid, _ in combined:
            row = meta_by_id.get(cid)
            if row is None:
                continue

            parent_id_val = row["ch.parent_doc_id"]

            # Apply graph filters on parent concept
            if concept_type:
                if row.get("parent_type") != concept_type:
                    continue
            if tags:
                parent_tags = row.get("parent_tags") or []
                if not any(t in parent_tags for t in tags):
                    continue
            if parent_id:
                if parent_id_val != parent_id:
                    continue

            # Apply per-doc limit
            if parent_id_val:
                doc_counts[parent_id_val] = doc_counts.get(parent_id_val, 0) + 1
                if doc_counts[parent_id_val] > max_chunks_per_doc:
                    continue

            chunk_text = row["ch.chunk_text"] or ""
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            result: Dict[str, Any] = {
                "chunk_id": cid,
                "chunk_text": chunk_text,
                "block_type": row["ch.block_type"],
                "chunk_index": row["ch.chunk_index"],
                "parent_doc_id": parent_id_val,
                "rrf_score": score_by_id[cid],
                "parent_title": row.get("parent_title"),
                "parent_type": row.get("parent_type"),
                "parent_tags": row.get("parent_tags"),
            }
            results.append(result)
            if len(results) >= limit:
                break
        return results


    def _compute_hub_scores(self, concept_ids: List[str]) -> Dict[str, float]:
        """Count incoming LINKS_TO edges for each concept.

        Higher hub score = more concepts point to this one = more authoritative.
        """
        if not concept_ids:
            return {}

        result = self.conn.execute("""
            MATCH (x:Concept)-[:LINKS_TO]->(c:Concept)
            WHERE c.id IN $ids
            RETURN c.id AS id, count(x) AS cnt
        """, {"ids": concept_ids})
        return {r["id"]: r["cnt"] for r in result.rows_as_dict().get_all()}


    def _get_ancestry(self, concept_id: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Return directory path from root to this concept."""
        # Directory nodes only have an id (path), no title.
        result = self.conn.execute("""
            MATCH (d:Directory)-[:CONTAINS*1..5]->(c:Concept {id: $cid})
            RETURN d.id AS dir_id
            LIMIT 1
        """, {"cid": concept_id})
        rows = result.rows_as_dict().get_all()
        if not rows:
            return []
        return [{"id": r["dir_id"]} for r in rows]


    def _get_siblings(self, concept_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return other concepts in the same parent directory."""
        parent_result = self.conn.execute("""
            MATCH (d:Directory)-[:CONTAINS]->(c:Concept {id: $id})
            RETURN d.id AS parent_id
        """, {"id": concept_id})
        rows = parent_result.rows_as_dict().get_all()
        if not rows:
            # Concept not in a directory — find siblings by root-level concepts
            root_result = self.conn.execute("""
                MATCH (c:Concept)
                WHERE c.id <> $cid
                AND NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(c) }
                RETURN c.id AS id, c.title AS title, c.type AS type
                LIMIT $limit
            """, {"cid": concept_id, "limit": limit})
            return [
                {"id": r["id"], "title": r["title"], "type": r["type"]}
                for r in root_result.rows_as_dict().get_all()
            ]
        parent_id = rows[0]["parent_id"]

        result = self.conn.execute("""
            MATCH (d:Directory {id: $pid})-[:CONTAINS]->(s:Concept)
            WHERE s.id <> $cid
            RETURN s.id AS id, s.title AS title, s.type AS type
            LIMIT $limit
        """, {"pid": parent_id, "cid": concept_id, "limit": limit})
        return [
            {"id": r["id"], "title": r["title"], "type": r["type"]}
            for r in result.rows_as_dict().get_all()
        ]


    def search_with_context(
        self,
        query: str,
        limit: int = 5,
        context_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """Search chunks + expand each result with graph neighborhood context.

        Returns chunks enriched with:
          - incoming_links: concepts that link TO this document
          - outgoing_links: concepts this document links TO
          - ancestry: directory path from root
          - siblings: other concepts in the same parent directory
        """
        chunks = self.search_chunks(query, limit=limit * 2)

        enriched: List[Dict[str, Any]] = []
        for chunk in chunks[:limit]:
            parent_id = chunk["parent_doc_id"]

            incoming = self.traverse(parent_id, "LINKS_TO", "INCOMING",
                                      depth=context_hops)
            outgoing = self.traverse(parent_id, "LINKS_TO", "OUTGOING",
                                      depth=context_hops)
            ancestry = self._get_ancestry(parent_id)
            siblings = self._get_siblings(parent_id)

            enriched.append({
                "chunk": chunk,
                "document": self.get_by_id(parent_id),
                "incoming_links": incoming[:5],
                "outgoing_links": outgoing[:5],
                "ancestry": ancestry,
                "siblings": siblings[:5],
            })

        return enriched


    def search_chunks_with_hub_score(
        self,
        query: str,
        limit: int = 10,
        hub_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search chunks + rerank by graph hub score.

        Final score = (1 - hub_weight) * rrf_score + hub_weight * normalized_hub_score.
        """
        raw_results = self.search_chunks(query, limit=limit * 2)

        # Collect unique parent doc IDs
        parent_ids = list({r["parent_doc_id"] for r in raw_results if r["parent_doc_id"]})
        hub_scores = self._compute_hub_scores(parent_ids)
        max_hub = max(hub_scores.values()) if hub_scores else 1

        for r in raw_results:
            pid = r["parent_doc_id"]
            hub = hub_scores.get(pid, 0)
            normalized_hub = hub / max_hub if max_hub > 0 else 0
            r["hub_score"] = normalized_hub
            r["final_score"] = (
                (1 - hub_weight) * r["rrf_score"] + hub_weight * normalized_hub
            )

        raw_results.sort(key=lambda x: x["final_score"], reverse=True)
        return raw_results[:limit]


    def expand_with_graph_context(
        self,
        chunk_ids: List[str],
        hops: int = 1,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand chunk search results with graph-context neighbours.

        For each seed chunk, find the parent document, then traverse
        LINKS_TO / CONTAINS to discover related concepts. Compute
        hub_score as the number of incoming LINKS_TO relationships.
        """
        if not chunk_ids:
            return []

        # Find parent documents for the seed chunks
        parents = self.conn.execute("""
            MATCH (p:Concept)-[:PART_OF]->(ch:Chunk)
            WHERE ch.id IN $ids
            RETURN p.id AS id, p.title AS title, p.type AS type, p.tags AS tags
        """, {"ids": chunk_ids})
        parent_rows = parents.rows_as_dict().get_all()
        parent_ids = [row["id"] for row in parent_rows]
        parent_meta = {row["id"]: row for row in parent_rows}

        if not parent_ids:
            return []

        # Expand via LINKS_TO (outgoing from parents)
        neighbours = self.conn.execute("""
            MATCH (p:Concept)-[:LINKS_TO]->(n:Concept)
            WHERE p.id IN $ids
            WITH DISTINCT n
            MATCH (other:Concept)-[:LINKS_TO]->(n)
            RETURN n.id AS id, n.title AS title, n.type AS type,
                   n.description AS description, n.tags AS tags,
                   count(other) AS hub_score
            ORDER BY hub_score DESC
            LIMIT $limit
        """, {"ids": parent_ids, "limit": max_results})
        neighbour_rows = neighbours.rows_as_dict().get_all()

        results: List[Dict[str, Any]] = []
        for row in neighbour_rows:
            results.append({
                "id": row["id"],
                "title": row["title"],
                "type": row["type"],
                "description": (row["description"] or "")[:200],
                "tags": row["tags"],
                "hub_score": row["hub_score"],
            })

        return results


    def rerank_with_hub_score(
        self,
        chunk_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank chunk results by combining RRF score with parent hub score.

        For each chunk result, look up the parent document's hub_score
        (incoming LINKS_TO count) and compute:
            final_score = rrf_score * (1 + 0.1 * hub_score)
        """
        if not chunk_results:
            return []

        parent_ids = list({r["parent_doc_id"] for r in chunk_results if r.get("parent_doc_id")})
        if not parent_ids:
            return chunk_results

        # Get hub scores for all parent documents
        hub_query = self.conn.execute("""
            MATCH (other:Concept)-[:LINKS_TO]->(p:Concept)
            WHERE p.id IN $ids
            RETURN p.id AS id, count(other) AS hub_score
        """, {"ids": parent_ids})
        hub_by_id = {row["id"]: row["hub_score"] for row in hub_query.rows_as_dict().get_all()}

        # Compute final scores
        for result in chunk_results:
            pid = result.get("parent_doc_id")
            hub = hub_by_id.get(pid, 0)
            rrf = result.get("rrf_score", 0)
            result["hub_score"] = hub
            result["final_score"] = rrf * (1 + 0.1 * hub)

        # Sort by final_score descending
        chunk_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return chunk_results


    def search_hybrid(
        self,
        query: str,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        exclude_reserved: bool = True,
        limit: int = 10,
        include_chunks: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: RRF fusion of vector + FTS with optional graph filters.

        If ``include_chunks=True``, each result also contains ``matched_chunks``
        — the top chunks from that document matching the query.
        """
        if not getattr(self, "_search_available", False):
            raise RuntimeError(
                "Search is unavailable: the 'vector'/'fts' extensions could not "
                "be loaded. Ensure the Ladybug extension repository is reachable, "
                "then reopen the router (ingestion and graph queries do not need "
                "these extensions)."
            )
        query_vec = self.embed_engine._encode(query, task="Query")

        # Stage 1: Vector search (ANN)
        vec_results = self.conn.execute(
            "CALL QUERY_VECTOR_INDEX('Concept', 'concept_embedding', $vec, $k) RETURN node, distance",
            {"vec": query_vec, "k": limit * 3},
        )
        vec_rows = vec_results.rows_as_dict().get_all()
        vec_scores: Dict[str, float] = {}
        for row in vec_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                vec_scores[node_id] = 1 - row.get("distance", 0)

        # Stage 2: Full-text search
        fts_results = self.conn.execute(
            "CALL QUERY_FTS_INDEX('Concept', 'concept_fts', $query) RETURN node, score",
            {"query": query},
        )
        fts_rows = fts_results.rows_as_dict().get_all()
        fts_scores: Dict[str, float] = {}
        for row in fts_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                fts_scores[node_id] = row.get("score", 0)

        # Stage 3: Reciprocal Rank Fusion (RRF, k=60). Precompute each source's
        # rank map once (O(n log n)) instead of re-sorting inside the loop.
        def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
            ordered = sorted(scores, key=scores.get, reverse=True)
            return {cid: i + 1 for i, cid in enumerate(ordered)}

        vec_rank = _rank_map(vec_scores)
        fts_rank = _rank_map(fts_scores)

        combined: List[Tuple[str, float]] = []
        for cid in set(vec_scores) | set(fts_scores):
            score = 0.0
            if cid in vec_rank:
                score += 1.0 / (60 + vec_rank[cid])
            if cid in fts_rank:
                score += 1.0 / (60 + fts_rank[cid])
            combined.append((cid, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Over-fetch candidates so post-filtering can still fill `limit`.
        candidate_ids = [cid for cid, _ in combined]
        score_by_id = dict(combined)

        # Stage 4: fetch metadata (and apply graph filters) in a SINGLE query.
        where_clauses: List[str] = ["c.id IN $ids"]
        params: Dict[str, Any] = {"ids": candidate_ids}
        if concept_type:
            where_clauses.append("c.type = $type")
            params["type"] = concept_type
        if tags:
            where_clauses.append("ANY(tag IN $tags WHERE tag IN c.tags)")
            params["tags"] = tags
        if parent_id:
            where_clauses.append(
                "EXISTS { MATCH (p:Directory {id: $parent})-[:CONTAINS*1..3]->(c) }"
            )
            params["parent"] = parent_id
        if exclude_reserved:
            where_clauses.append(
                "NOT c.id ENDS WITH 'index' AND NOT c.id ENDS WITH 'log'"
            )

        cypher = f"""
        MATCH (c:Concept)
        WHERE {" AND ".join(where_clauses)}
        RETURN c.id, c.title, c.type, c.description, c.tags
        """
        rows = self.conn.execute(cypher, params).rows_as_dict().get_all()
        meta_by_id = {row["c.id"]: row for row in rows}

        # Assemble in RRF order, keeping only rows that survived filtering.
        results: List[Dict[str, Any]] = []
        for cid, _ in combined:
            row = meta_by_id.get(cid)
            if row is None:
                continue
            desc = row["c.description"] or ""
            if len(desc) > 200:
                desc = desc[:200] + "..."
            result = {
                "id": cid,
                "title": row["c.title"],
                "type": row["c.type"],
                "description": desc,
                "tags": row["c.tags"],
                "relevance_score": score_by_id[cid],
            }

            # Attach chunks if requested
            if include_chunks:
                chunks = self.search_chunks(
                    query=query, limit=3, parent_id=cid
                )
                result["matched_chunks"] = chunks

            results.append(result)
            if len(results) >= limit:
                break
        return results


    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 6,
    ) -> List[Dict[str, Any]]:
        """Find the shortest path between two concepts.

        Uses BFS-style variable-length patterns across allowed edge types.
        Returns a list of nodes on the path (including start and end) with
        their id, title, and type.
        """
        max_length = max(1, min(int(max_length), 10))
        # Ladybug doesn't support MATCH path = ... or [*1..N] (any-rel).
        # Try increasing path lengths until we find a connection.
        # Note: Ladybug reserves $end as a parameter name, so use $sid/$eid.
        for length in range(1, max_length + 1):
            result = self.conn.execute(
                f"""
                MATCH (a:Concept {{id: $sid}})-[:CONTAINS|LINKS_TO|PART_OF|INCLUDES_ASSET*1..{length}]-(b:Concept {{id: $eid}})
                RETURN a.id AS id, a.title AS title, a.type AS type
                """,
                {"sid": start_id, "eid": end_id},
            )
            rows = result.rows_as_dict().get_all()
            if rows:
                # Found a path at this length — collect all nodes
                # Re-run to get full path nodes using path variable
                path_result = self.conn.execute(
                    f"""
                    MATCH path = (a:Concept {{id: $sid}})-[:CONTAINS|LINKS_TO|PART_OF|INCLUDES_ASSET*1..{length}]-(b:Concept {{id: $eid}})
                    WITH path AS p, length(path) AS len
                    ORDER BY len ASC
                    LIMIT 1
                    UNWIND nodes(p) AS node
                    RETURN node.id AS id, node.title AS title, node.type AS type
                    """,
                    {"sid": start_id, "eid": end_id},
                )
                return path_result.rows_as_dict().get_all()
        return []


    def traverse(
        self,
        start_id: str,
        relationship: str = "CONTAINS",
        direction: str = "OUTGOING",
        depth: int = 1,
        node_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Navigate graph relationships with whitelisted edges and depth cap."""
        ALLOWED_RELS = {"CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"}
        if relationship not in ALLOWED_RELS:
            raise ValueError(f"Invalid relationship. Must be one of {ALLOWED_RELS}")
        depth = max(1, min(int(depth), 5))

        # PART_OF targets Chunk nodes; INCLUDES_ASSET targets ImageAsset
        if relationship == "PART_OF":
            target_label = "Chunk"
        elif relationship == "INCLUDES_ASSET":
            target_label = "ImageAsset"
        else:
            target_label = "Concept"

        if direction == "OUTGOING":
            pattern = f"-[{relationship}*1..{depth}]->(target:{target_label})"
        elif direction == "INCOMING":
            pattern = f"<-[{relationship}*1..{depth}]-(target:{target_label})"
        else:  # BOTH
            pattern = f"-[{relationship}*1..{depth}]-(target:{target_label})"

        where_clause = ""
        query_params: Dict[str, Any] = {"start_id": start_id}
        if node_type:
            where_clause = "WHERE target.type = $node_type"
            query_params["node_type"] = node_type

        cypher = f"""
        MATCH (start {{id: $start_id}}){pattern}
        {where_clause}
        RETURN target.*
        LIMIT 100
        """
        results = self.conn.execute(cypher, query_params)
        rows = results.rows_as_dict().get_all()

        results_list: List[Dict[str, Any]] = []
        for row in rows:
            entry: Dict[str, Any] = {}
            for key, val in row.items():
                prop = key.split(".", 1)[-1]  # strip 'target.' prefix
                entry[prop] = val
            results_list.append(entry)
        return results_list


    def get_chunks(self, concept_id: str) -> List[ChunkModel]:
        """Get all chunks for a concept, ordered by chunk_index."""
        result = self.conn.execute("""
            MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk)
            RETURN ch.id AS id, ch.parent_doc_id AS parent_doc_id,
                   ch.chunk_index AS chunk_index, ch.chunk_text AS chunk_text,
                   ch.block_type AS block_type,
                   ch.start_offset AS start_offset, ch.end_offset AS end_offset
            ORDER BY ch.chunk_index
        """, {"id": concept_id})
        rows = result.rows_as_dict().get_all()
        return [
            ChunkModel(
                id=row["id"],
                parent_doc_id=row["parent_doc_id"],
                chunk_index=row["chunk_index"],
                chunk_text=row["chunk_text"],
                block_type=row["block_type"],
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
            )
            for row in rows
        ]


    def list_directory(self, directory_id: str) -> List[Dict[str, Any]]:
        """List immediate children of a directory (polymorphic: Directories + Concepts)."""
        results_directories: List[Dict[str, Any]] = []
        results_concepts: List[Dict[str, Any]] = []

        if not directory_id:
            # Root: find directories with no parent
            dir_result = self.conn.execute("""
                MATCH (d:Directory)
                WHERE NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(d) }
                RETURN d.id AS child_id, 'Directory' AS type, d.id AS title
            """)
            results_directories = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in dir_result.rows_as_dict().get_all()
            ]
            # Find concepts with no parent
            concept_result = self.conn.execute("""
                MATCH (c:Concept)
                WHERE NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(c) }
                RETURN c.id AS child_id, c.type AS type, c.title AS title
            """)
            results_concepts = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in concept_result.rows_as_dict().get_all()
            ]
        else:
            params = {"id": directory_id}
            dir_result = self.conn.execute("""
                MATCH (p:Directory {id: $id})-[:CONTAINS]->(d:Directory)
                RETURN d.id AS child_id, 'Directory' AS type, d.id AS title
            """, params)
            results_directories = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in dir_result.rows_as_dict().get_all()
            ]
            concept_result = self.conn.execute("""
                MATCH (p:Directory {id: $id})-[:CONTAINS]->(c:Concept)
                RETURN c.id AS child_id, c.type AS type, c.title AS title
            """, params)
            results_concepts = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in concept_result.rows_as_dict().get_all()
            ]

        return results_directories + results_concepts


    def get_by_id(self, concept_id: str) -> Optional[ConceptModel]:
        """Fetch a full concept by ID, merging extra MAP fields back into the model."""
        result = self.conn.execute(
            "MATCH (c:Concept {id: $id}) RETURN c.*", {"id": concept_id}
        )
        rows = result.rows_as_dict().get_all()
        if not rows:
            return None

        row = rows[0]
        data = {
            k.replace("c.", ""): v
            for k, v in row.items()
            if not k.startswith("c.extra")
        }
        extra = row.get("c.extra") or {}
        extra_decoded = {
            k: json.loads(v)
            if isinstance(v, str) and v.startswith(("{" , "["))
            else v
            for k, v in extra.items()
        }
        data.update(extra_decoded)
        data.pop("extra", None)
        return ConceptModel.model_validate(data)

