"""PurgeManager — hard purge, soft-delete, and recovery-window management.

Extracted from OKFRouter (purge + soft-delete-with-recovery sections).
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from okfgraph.models import ChunkModel, ConceptModel

logger = logging.getLogger(__name__)

class PurgeManager:
    def __init__(self, conn, write_lock_ctx: Callable):
        self.conn = conn
        self._write_lock_ctx = write_lock_ctx

    SOFT_DELETE_WINDOW = 24 * 60 * 60

    def _purge_concept(self, concept_id: str) -> bool:
        """Delete a concept and all its dependents from the graph.

        Removes:
        - The Concept node (and its embedding)
        - All Chunk nodes linked via PART_OF
        - All LINKS_TO relationships (incoming and outgoing)
        - All INCLUDES_ASSET relationships
        - ImageAsset nodes that have **zero remaining** INCLUDES_ASSET
          edges (orphan check — shared assets are preserved)
        - BrokenLink entries where this concept was source or target
        - FileHash entry for the concept's file path

        Args:
            concept_id: The ID of the concept to purge.

        Returns:
            True if a concept was found and purged, False if not found.
        """
        # Check if the concept exists
        rows = self.conn.execute(
            "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
            {"id": concept_id},
        ).rows_as_dict().get_all()
        if not rows or rows[0]["cnt"] == 0:
            logger.debug("purge: concept %s not found, skipping", concept_id)
            return False

        self.conn.execute("BEGIN TRANSACTION")
        try:
            # 1. Find and sever INCLUDES_ASSET edges, collecting asset IDs.
            asset_rows = self.conn.execute(
                """
                MATCH (c:Concept {id: $id})-[:INCLUDES_ASSET]->(i:ImageAsset)
                RETURN i.id AS aid
                """,
                {"id": concept_id},
            ).rows_as_dict().get_all()
            asset_ids = [r["aid"] for r in asset_rows]

            # 2. Delete all Chunk nodes for this concept (DETACH DELETE on
            #    Concept doesn't cascade to Chunk — they are separate nodes).
            self.conn.execute(
                """
                MATCH (ch:Chunk {parent_doc_id: $id})
                DETACH DELETE ch
                """,
                {"id": concept_id},
            )

            # 3. DETACH DELETE the Concept (cascades LINKS_TO, CONTAINS,
            #    INCLUDES_ASSET edges).
            self.conn.execute(
                "MATCH (c:Concept {id: $id}) DETACH DELETE c",
                {"id": concept_id},
            )

            # 4. Clean up orphaned ImageAssets — only delete if no other
            #    Concept still references them. This handles the case where
            #    two files share the same okf-asset://<id> URI.
            for aid in asset_ids:
                ref_count = self.conn.execute(
                    """
                    MATCH (i:ImageAsset {id: $aid})
                    OPTIONAL MATCH (i)<-[:INCLUDES_ASSET]-(other:Concept)
                    RETURN count(other) AS refs
                    """,
                    {"aid": aid},
                ).rows_as_dict().get_all()
                if ref_count and ref_count[0]["refs"] == 0:
                    self.conn.execute(
                        "MATCH (i:ImageAsset {id: $aid}) DELETE i",
                        {"aid": aid},
                    )
                    logger.debug("purge: deleted orphaned asset %s", aid)

            # 5. Clean up BrokenLink entries where this concept was source
            #    or target.
            self.conn.execute(
                """
                MATCH (bl:BrokenLink)
                WHERE bl.source_id = $id OR bl.target_id = $id
                DELETE bl
                """,
                {"id": concept_id},
            )

            # 6. Remove FileHash entry for this concept.
            self.conn.execute(
                """
                MATCH (f:FileHash {concept_id: $id})
                DELETE f
                """,
                {"id": concept_id},
            )

            self.conn.execute("COMMIT")
            logger.info("purge: deleted concept %s", concept_id)
            return True

        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise


    def _soft_delete_concept(self, concept_id: str) -> bool:
        """Soft-delete a concept by moving it to the DeletedConcept table.

        The concept is preserved with a timestamp so it can be recovered
        within the recovery window. After the window expires, the concept
        is permanently deleted.

        Args:
            concept_id: The ID of the concept to soft-delete.

        Returns:
            True if a concept was found and soft-deleted, False if not found.
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._soft_delete_concept_inner(concept_id)


    def _soft_delete_concept_inner(self, concept_id: str) -> bool:
        """Inner implementation (called under write lock)."""
        # Check if the concept exists
        rows = self.conn.execute(
            "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
            {"id": concept_id},
        ).rows_as_dict().get_all()
        if not rows or rows[0]["cnt"] == 0:
            logger.debug("soft_delete: concept %s not found, skipping", concept_id)
            return False

        # Fetch concept data for preservation
        concept_rows = self.conn.execute(
            """
            MATCH (c:Concept {id: $id})
            RETURN c.title AS title, c.body AS body, c.type AS ctype, c.tags AS tags
            """,
            {"id": concept_id},
        ).rows_as_dict().get_all()
        if not concept_rows:
            return False

        concept_data = concept_rows[0]
        deleted_at = datetime.now().isoformat()

        # Store in DeletedConcept table
        self.conn.execute(
            """
            INSERT INTO DeletedConcept (id, original_id, title, body, deleted_at, type, tags)
            VALUES ($id, $oid, $title, $body, $deleted_at, $type, $tags)
            """,
            {
                "id": f"deleted_{concept_id}_{int(time.time())}",
                "oid": concept_id,
                "title": concept_data.get("title", ""),
                "body": concept_data.get("body", ""),
                "deleted_at": deleted_at,
                "type": concept_data.get("ctype", ""),
                "tags": json.dumps(concept_data.get("tags", [])),
            },
        )

        # Now perform the hard delete
        self._purge_concept(concept_id)

        logger.info(
            "soft_delete: concept %s moved to DeletedConcept (recovery until %s)",
            concept_id,
            (datetime.now() + __import__("datetime").timedelta(seconds=self.SOFT_DELETE_WINDOW)).isoformat(),
        )
        return True


    def _recover_concept(self, concept_id: str) -> bool:
        """Recover a soft-deleted concept from the DeletedConcept table.

        Args:
            concept_id: The original ID of the concept to recover.

        Returns:
            True if the concept was recovered, False if not found.
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._recover_concept_inner(concept_id)


    def _recover_concept_inner(self, concept_id: str) -> bool:
        """Inner implementation (called under write lock)."""
        # Check if the concept exists in DeletedConcept
        rows = self.conn.execute(
            """
            MATCH (d:DeletedConcept {original_id: $id})
            RETURN d.title AS title, d.body AS body, d.type AS ctype, d.tags AS tags, d.deleted_at AS deleted_at
            """,
            {"id": concept_id},
        ).rows_as_dict().get_all()

        if not rows:
            logger.debug("recover: concept %s not found in DeletedConcept", concept_id)
            return False

        deleted_at = datetime.fromisoformat(rows[0]["deleted_at"])
        now = datetime.now()
        if (now - deleted_at).total_seconds() > self.SOFT_DELETE_WINDOW:
            logger.warning(
                "recover: concept %s deleted at %s is past the recovery window (%ds)",
                concept_id,
                deleted_at.isoformat(),
                self.SOFT_DELETE_WINDOW,
            )
            return False

        # Restore the concept
        concept_data = rows[0]
        tags = json.loads(concept_data["tags"]) if isinstance(concept_data["tags"], str) else concept_data["tags"]

        self.conn.execute(
            """
            INSERT INTO Concept (id, title, body, type, tags)
            VALUES ($id, $title, $body, $type, $tags)
            """,
            {
                "id": concept_id,
                "title": concept_data["title"],
                "body": concept_data["body"],
                "type": concept_data["ctype"],
                "tags": tags,
            },
        )

        # Remove from DeletedConcept
        self.conn.execute(
            "MATCH (d:DeletedConcept {original_id: $id}) DELETE d",
            {"id": concept_id},
        )

        logger.info("recover: concept %s restored from DeletedConcept", concept_id)
        return True


    def list_deleted_concepts(self) -> List[Dict[str, Any]]:
        """List all soft-deleted concepts with recovery status.

        Returns:
            List of dicts with concept_id, title, deleted_at, and recoverable status.
        """
        rows = self.conn.execute(
            """
            MATCH (d:DeletedConcept)
            RETURN d.original_id AS id, d.title AS title, d.deleted_at AS deleted_at, d.type AS type
            ORDER BY d.deleted_at DESC
            """
        ).rows_as_dict().get_all()

        now = datetime.now()
        results = []
        for row in rows:
            deleted_at = datetime.fromisoformat(row["deleted_at"])
            age_seconds = (now - deleted_at).total_seconds()
            results.append({
                "concept_id": row["id"],
                "title": row["title"],
                "type": row["type"],
                "deleted_at": row["deleted_at"],
                "age_seconds": age_seconds,
                "recoverable": age_seconds <= self.SOFT_DELETE_WINDOW,
            })
        return results


    def purge_deleted_concepts(self, older_than: Optional[int] = None) -> int:
        """Permanently delete soft-deleted concepts past the recovery window.

        Args:
            older_than: Optional override for the recovery window (seconds).
                        Defaults to SOFT_DELETE_WINDOW.

        Returns:
            Number of concepts permanently deleted.
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._purge_deleted_concepts_inner(older_than)


    def _purge_deleted_concepts_inner(self, older_than: Optional[int]) -> int:
        """Inner implementation (called under write lock)."""
        threshold = older_than if older_than is not None else self.SOFT_DELETE_WINDOW
        cutoff = datetime.now() - __import__("datetime").timedelta(seconds=threshold)

        # Find expired entries
        rows = self.conn.execute(
            """
            MATCH (d:DeletedConcept)
            WHERE d.deleted_at < $cutoff
            RETURN d.original_id AS id
            """,
            {"cutoff": cutoff.isoformat()},
        ).rows_as_dict().get_all()

        count = 0
        for row in rows:
            # Permanently delete (hard purge)
            if self._purge_concept(row["id"]):
                count += 1
            # Remove from DeletedConcept
            self.conn.execute(
                "MATCH (d:DeletedConcept {original_id: $id}) DELETE d",
                {"id": row["id"]},
            )

        logger.info("purge_deleted: permanently deleted %d expired concept(s)", count)
        return count

