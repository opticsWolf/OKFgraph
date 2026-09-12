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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse

import mordant
import numpy as np
import yaml
import frontmatter
from okfgraph.models import ChunkModel, ConceptModel

logger = logging.getLogger(__name__)

class ExportManager:
    def __init__(self, conn, search_engine):
        self.conn = conn
        self.search_engine = search_engine

    def _enrich_body_with_graph_links(
        self, concept_id: str, body: str, flavor: str = "okf",
        title_counts: Optional[Dict[str, int]] = None,
    ) -> str:
        """Enrich body with graph-derived links so the exported markdown
        faithfully reflects the LINKS_TO graph.

        Strategy (Option A — append, never replace):
          1. Query all outgoing LINKS_TO edges from this concept.
          2. For each target, check if a link to that target already exists
             in the body (by matching the target_id in link URLs).
          3. If not already linked, append a "See Also" bullet.
          4. Query all incoming LINKS_TO edges (concepts that link TO this one).
          5. If any exist, append a "Cited By" bullet list.

        This preserves the original body's links (which may have richer anchor
        text) while ensuring the graph structure is expressed in the export.

        ``flavor="obsidian"`` renders appended links as ``[[Title]]``
        (unique titles) or ``[[id|Title]]`` instead of ``[t](id.md)``; the
        original body is never rewritten in either flavor. The obsidian
        flavor omits the "Cited By" section: backlinks re-imported as
        forward wikilinks would reverse their direction, and Obsidian
        renders backlinks natively anyway.
        """
        import re

        wiki_refs = {
            m.split("|", 1)[0].strip()
            for m in re.findall(r"\[\[([^\]]+)\]\]", body or "")
        }
        wiki_refs_lower = {r.lower() for r in wiki_refs}

        def _link(target_id: str, title: str) -> str:
            label = title or target_id.split("/")[-1]
            if flavor == "obsidian":
                if label and (title_counts or {}).get(label.lower(), 0) == 1:
                    return f"- [[{label}]]"
                return f"- [[{target_id}|{label}]]"
            return f"- [{label}]({target_id}.md)"

        parts: List[str] = []

        # --- Outgoing links (See Also) ---
        result = self.conn.execute("""
            MATCH (s:Concept {id: $cid})-[:LINKS_TO]->(t:Concept)
            RETURN t.id AS target_id, t.title AS title, t.type AS type
            ORDER BY t.title
        """, {"cid": concept_id})
        outgoing_rows = result.rows_as_dict().get_all()

        if outgoing_rows:
            # Determine which targets are already linked in the body:
            # either a [t](...target...) URL or an equivalent [[ref]].
            existing_link_targets = set()
            for row in outgoing_rows:
                target_id = row["target_id"]
                title = row["title"] or ""
                if target_id in wiki_refs or title.lower() in wiki_refs_lower:
                    existing_link_targets.add(target_id)
                    continue
                # Check if target_id appears in any link URL in the body
                link_pattern = re.compile(
                    r"\]\(([^)]*?" + re.escape(target_id) + r"[^)]*)\)"
                )
                if link_pattern.search(body):
                    existing_link_targets.add(target_id)

            # Collect targets that need a link added
            new_links = []
            for row in outgoing_rows:
                target_id = row["target_id"]
                if target_id not in existing_link_targets:
                    title = row["title"] or target_id.split("/")[-1]
                    new_links.append(_link(target_id, title))

            if new_links:
                parts.append("\n## See Also\n" + "\n".join(new_links))

        # --- Incoming links (Cited By: okf flavor only) ---
        # Obsidian renders backlinks natively; emitting them as forward
        # [[wikilinks]] would reverse the edge on re-import (round-trip loss).
        result = self.conn.execute("""
            MATCH (s:Concept)-[:LINKS_TO]->(t:Concept {id: $cid})
            RETURN s.id AS source_id, s.title AS title, s.type AS type
            ORDER BY s.title
        """, {"cid": concept_id})
        incoming_rows = result.rows_as_dict().get_all()

        if incoming_rows and flavor == "okf":
            cited_lines = ["\n## Cited By\n"]
            for row in incoming_rows:
                source_id = row["source_id"]
                title = row["title"] or source_id.split("/")[-1]
                cited_lines.append(_link(source_id, title))
            parts.append("\n".join(cited_lines))

        return body + "".join(parts)


    def _fetch_concepts(
        self,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, ConceptModel]:
        """Fetch all concepts, optionally filtered by type and tags."""
        where_clauses: list[str] = []
        params: Dict[str, Any] = {}

        if concept_type:
            where_clauses.append("c.type = $type")
            params["type"] = concept_type
        if tags:
            where_clauses.append("ALL(tag IN $tags WHERE tag IN c.tags)")
            params["tags"] = tags

        where_str = " AND ".join(where_clauses) if where_clauses else "true"
        query = f"""
        MATCH (c:Concept)
        WHERE {where_str}
        RETURN c.id, c.type, c.title, c.description, c.resource,
               c.tags, c.timestamp, c.body, c.embedding, c.extra
        """
        results = self.conn.execute(query, params)
        rows = results.rows_as_dict().get_all()

        concepts: Dict[str, ConceptModel] = {}
        for row in rows:
            data: Dict[str, Any] = {}
            for key, val in row.items():
                col = key.split(".", 1)[-1]  # strip 'c.' prefix
                if col != "extra":
                    data[col] = val

            # Decode extra MAP fields
            extra = row.get("c.extra") or {}
            for k, v in extra.items():
                if isinstance(v, str) and v.startswith(("{", "[")):
                    try:
                        data[k] = json.loads(v)
                    except json.JSONDecodeError:
                        data[k] = v
                else:
                    data[k] = v

            try:
                concepts[data["id"]] = ConceptModel.model_validate(data)
            except Exception:
                pass  # Skip malformed concepts

        return concepts


    def _generate_index_files(
        self, output_dir: Path, concepts: Dict[str, ConceptModel]
    ) -> None:
        """Generate index.md files for every directory in the bundle.

        Each index.md lists the children (concepts and subdirectories) of that
        directory, enabling progressive disclosure for OKF consumers.
        """
        # Build a map of directory_id → list of (title, relative_path) children
        dir_children: Dict[str, List[Tuple[str, str]]] = {}

        for cid, concept in concepts.items():
            parts = cid.split("/")
            for i in range(1, len(parts)):
                dir_id = "/".join(parts[:i])
                dir_children.setdefault(dir_id, [])
                child_title = concept.title or parts[i]
                child_rel = cid.replace("/", os.sep) + ".md"
                dir_children[dir_id].append((child_title, child_rel))

        # Write index.md for each directory
        for dir_id, children in dir_children.items():
            # Sort children by title
            children.sort(key=lambda x: x[0])
            lines = [
                f"# {dir_id.split('/')[-1] or '(root)'}\n",
                "",
            ]
            for title, rel_path in children:
                lines.append(f"- [{title}]({rel_path})")
            lines.append("")

            # Create parent directories if needed
            dir_path = output_dir / dir_id.replace("/", os.sep)
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / "index.md").write_text("\n".join(lines), encoding="utf-8")


    def _is_under_directory(self, concept_id: str, directory_id: str) -> bool:
        """Check if a concept is under a given directory (via CONTAINS graph)."""
        result = self.conn.execute("""
            MATCH (d:Directory {id: $dir_id})-[:CONTAINS*1..5]->(c:Concept {id: $cid})
            RETURN count(c) AS cnt
        """, {"dir_id": directory_id, "cid": concept_id})
        rows = result.rows_as_dict().get_all()
        return rows[0]["cnt"] > 0 if rows else False


    def _title_counts(self) -> Dict[str, int]:
        """Count concepts per (lowercased) title.

        Lowercased to match the case-insensitive name index on import: two
        titles differing only by case are ambiguous as ``[[Title]]`` and
        must both export disambiguated.
        """
        counts: Dict[str, int] = {}
        for r in self.conn.execute(
            "MATCH (c:Concept) RETURN c.title"
        ).rows_as_dict().get_all():
            title = (r.get("c.title") or "").lower()
            counts[title] = counts.get(title, 0) + 1
        return counts

    def _write_okf(
        self, concept: ConceptModel, output_path: Path,
        flavor: str = "okf", title_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Internal: serialize a ConceptModel to an OKF .md file.

        Enriches the body with LINKS_TO relationships from the graph so that
        exported markdown faithfully reflects the graph structure.

        ``flavor="obsidian"`` renders appended graph links as ``[[Title]]``
        wikilinks; the default ``okf`` flavor uses ``[t](id.md)`` links.
        Either way the stored ``uid`` (frontmatter ``id:`` on import) is
        written back as ``id:`` so re-import is lossless.
        """
        data, body = concept.export_frontmatter()

        yaml_str = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        # ENRICH: add graph-derived links to the body
        body = self._enrich_body_with_graph_links(
            concept.id, body, flavor=flavor, title_counts=title_counts,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"---\n{yaml_str}---\n\n{body}", encoding="utf-8")


    def export_bundle(
        self,
        output_dir: Path,
        directory_id: Optional[str] = None,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        flavor: str = "okf",
    ) -> List[str]:
        """Export concepts from the graph back to an OKF bundle directory.

        Reconstructs the full directory hierarchy from CONTAINS relationships.
        Supports filtering by directory subtree, concept type, or tags.

        Args:
            output_dir: Root directory to write the bundle into.
            directory_id: If set, only export concepts under this directory.
            concept_type: If set, only export concepts of this type.
            tags: If set, only export concepts with ALL these tags.
            flavor: ``"okf"`` (``[t](id.md)`` graph links + index.md files)
                or ``"obsidian"`` (``[[Title]]`` wikilinks, no index files —
                a vault re-imports losslessly via the name index).

        Returns:
            List of exported concept IDs.
        """
        if flavor not in ("okf", "obsidian"):
            raise ValueError(f"flavor must be 'okf' or 'obsidian', got {flavor!r}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch all concepts (optionally filtered)
        concepts = self._fetch_concepts(
            concept_type=concept_type,
            tags=tags,
        )

        # If directory_id specified, filter to subtree
        if directory_id:
            concepts = {
                cid: c for cid, c in concepts.items()
                if self._is_under_directory(cid, directory_id)
            }

        if not concepts:
            return []

        title_counts = self._title_counts() if flavor == "obsidian" else None

        # Export each concept, reconstructing path from its ID
        exported: List[str] = []
        for cid, concept in sorted(concepts.items()):
            # Concept IDs use forward slashes; convert to OS path separator
            rel_path = cid.replace("/", os.sep)
            file_path = output_dir / (rel_path + ".md")
            try:
                self._write_okf(concept, file_path, flavor=flavor,
                                title_counts=title_counts)
                exported.append(cid)
            except Exception as e:
                print(f"  [WARN] Failed to export {cid}: {e}")

        if flavor == "okf":
            # Generate index.md files for progressive disclosure.
            # Obsidian vaults don't have them (they'd import as stray concepts).
            self._generate_index_files(output_dir, concepts)

        return exported

    def export_to_okf(
        self, concept_id: str, output_path: Path, flavor: str = "okf",
    ) -> None:
        """Export a concept back to an OKF .md file (see ``export_bundle``)."""
        concept = self.search_engine.get_by_id(concept_id)
        if not concept:
            raise FileNotFoundError(f"Concept {concept_id} not found")

        title_counts = self._title_counts() if flavor == "obsidian" else None
        self._write_okf(concept, output_path, flavor=flavor,
                        title_counts=title_counts)

