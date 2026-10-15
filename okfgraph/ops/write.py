"""Canonical write ops that edit the bundle, then import it (0.11.0, O11).

``update`` and ``supersede`` keep the single-writer rule: when a concept is
backed by a bundle file, the op rewrites that file's frontmatter and imports
the file — the graph is always what importing the bundle produced. A concept
with no file is materialized into the bundle first (when the router has a
bundle root); only a file-free router edits the graph directly.

Return shapes:

* ``update`` → ``{"concept_id", "path", "changed": [field, ...]}``
  (``path`` is None for a file-free router)
* ``supersede`` → ``{"old", "new", "old_path", "new_path"}``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import frontmatter

from okfgraph.errors import OKFError
from okfgraph.models import ConceptModel, normalize_tags

__all__ = ["WriteOps"]

#: Frontmatter keys ``update(extra=...)`` may not touch.
_RESERVED = frozenset({
    "id", "type", "title", "description", "tags", "timestamp", "body",
    "embedding", "uid", "source_path", "relations", "resource",
})


def _unknown(op: str, concept_id: str) -> OKFError:
    return OKFError(
        "UNKNOWN_CONCEPT", f"concept '{concept_id}' does not exist", op=op,
        fields={"concept_id": concept_id}, remedy="search first to find IDs")


class WriteOps:
    """Mixin for :class:`~okfgraph.OKFRouter`."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _source_file(self, concept_id: str) -> Optional[Path]:
        """The existing bundle file backing ``concept_id``, if any."""
        rows = self.conn.execute(
            "MATCH (f:FileHash {concept_id: $cid}) RETURN f.path AS p",
            {"cid": concept_id},
        ).rows_as_dict().get_all() or []
        for row in rows:
            for cand in self._resolve_file_paths(str(row.get("p") or "")):
                if cand.is_file():
                    return cand
        return None

    def _write_and_import(self, op: str, concept_id: str, edit) -> Dict[str, Any]:
        """Apply ``edit(meta: dict) -> changed fields`` and re-import.

        File-backed: edit the file's frontmatter in place, import it.
        Graph-only with a bundle root: materialize from the graph, edit,
        import. File-free router: edit the stored concept and re-run the
        single-concept pipeline (re-embeds; links into it are kept, O1).
        """
        concept = self.get_by_id(concept_id)
        if concept is None:
            raise _unknown(op, concept_id)
        path = self._source_file(concept_id)
        if path is not None:
            post = frontmatter.load(path)
            meta = dict(post.metadata)
            body = post.content
        else:
            meta, body = concept.export_frontmatter()
            meta.pop("source_path", None)
        changed = edit(meta)
        if not changed:
            return {"concept_id": concept_id, "path": str(path) if path else None,
                    "changed": []}
        if path is None and self.bundle_root is not None:
            path = self.ingest_mgr._bundle_file_for(concept_id)
            if self.identity == "frontmatter":
                meta["id"] = concept_id
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(frontmatter.dumps(frontmatter.Post(body, **meta)) + "\n",
                            encoding="utf-8")
            with self._write_lock_ctx():
                cid = self.import_mgr.import_from_okf(path)
            if cid != concept_id:
                raise OKFError(
                    "BAD_VALUE",
                    f"{op}: the file {path} now imports as '{cid}', not "
                    f"'{concept_id}' (its id moved?)",
                    op=op, fields={"concept_id": concept_id, "imported": cid})
            return {"concept_id": cid, "path": str(path), "changed": changed}
        # File-free: edit the graph through the single-concept pipeline.
        data = {**meta, "id": concept_id, "body": body}
        data.pop("embedding", None)
        if "id" in meta and meta["id"] != concept_id:
            data["uid"] = meta["id"]
        new = ConceptModel.model_validate(data)
        with self._write_lock_ctx():
            self.import_mgr._import_single_concept(new, body, "text", True)
        return {"concept_id": concept_id, "path": None, "changed": changed}

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(
        self,
        concept_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        type: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        add_tags: Optional[Iterable[str]] = None,
        remove_tags: Optional[Iterable[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        relations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Edit a concept's frontmatter through its bundle file (O11).

        ``tags`` replaces the tag list; ``add_tags``/``remove_tags`` edit it.
        ``extra`` sets plain frontmatter keys (``None`` value deletes the
        key); reserved keys have their own parameters. ``relations``
        replaces the whole ``relations:`` mapping (validated as import
        would, O6). The body is never touched.
        """
        if tags is not None and (add_tags or remove_tags):
            raise OKFError("BAD_VALUE", "pass tags or add_tags/remove_tags, not both",
                           op="update", fields={"concept_id": concept_id})
        bad = sorted(k for k in (extra or {}) if k in _RESERVED)
        if bad:
            raise OKFError(
                "BAD_VALUE", f"extra may not set {', '.join(bad)}", op="update",
                fields={"reserved": bad},
                remedy="use title/description/type/tags/relations")
        if relations is not None:
            from okfgraph.components.links import frontmatter_relations
            _pairs, problems = frontmatter_relations(relations)
            if problems:
                raise OKFError("BAD_VALUE", "; ".join(problems), op="update",
                               fields={"relations": problems})

        def edit(meta: Dict[str, Any]) -> List[str]:
            changed: List[str] = []
            for key, value in (("title", title), ("description", description),
                               ("type", type)):
                if value is not None and meta.get(key) != value:
                    meta[key] = value
                    changed.append(key)
            current = normalize_tags(meta.get("tags"))
            new_tags = list(current)
            if tags is not None:
                new_tags = normalize_tags(list(tags))
            for t in normalize_tags(list(add_tags or [])):
                if t not in new_tags:
                    new_tags.append(t)
            drop = set(normalize_tags(list(remove_tags or [])))
            new_tags = [t for t in new_tags if t not in drop]
            if new_tags != current:
                meta["tags"] = new_tags
                changed.append("tags")
            for key, value in (extra or {}).items():
                if value is None:
                    if key in meta:
                        del meta[key]
                        changed.append(key)
                elif meta.get(key) != value:
                    meta[key] = value
                    changed.append(key)
            if relations is not None and meta.get("relations") != relations:
                if relations:
                    meta["relations"] = relations
                else:
                    meta.pop("relations", None)
                changed.append("relations")
            return changed

        return self._write_and_import("update", concept_id, edit)

    # ------------------------------------------------------------------
    # supersede
    # ------------------------------------------------------------------

    def supersede(self, old_id: str, new_id: str) -> Dict[str, Any]:
        """Record that ``new_id`` supersedes ``old_id`` (O11).

        Adds ``old_id`` to the new concept's ``relations.supersedes`` (a
        typed ``LINKS_TO {kind: supersedes}`` edge, O6) and tags the old
        concept ``superseded`` with ``superseded_by: <new_id>`` — two file
        writes, each imported. OKFgraph stays a current-state store: the
        history of beliefs lives in Macrame; this records the latest one.
        """
        if old_id == new_id:
            raise OKFError("BAD_VALUE", "a concept cannot supersede itself",
                           op="supersede", fields={"old": old_id})
        for cid in (old_id, new_id):
            if self.get_by_id(cid) is None:
                raise _unknown("supersede", cid)

        def edit_new(meta: Dict[str, Any]) -> List[str]:
            rel = dict(meta.get("relations") or {})
            current = rel.get("supersedes") or []
            if isinstance(current, str):
                current = [current]
            if old_id in current:
                return []
            rel["supersedes"] = list(current) + [old_id]
            meta["relations"] = rel
            return ["relations"]

        def edit_old(meta: Dict[str, Any]) -> List[str]:
            changed: List[str] = []
            tags = normalize_tags(meta.get("tags"))
            if "superseded" not in tags:
                meta["tags"] = tags + ["superseded"]
                changed.append("tags")
            if meta.get("superseded_by") != new_id:
                meta["superseded_by"] = new_id
                changed.append("superseded_by")
            return changed

        new_res = self._write_and_import("supersede", new_id, edit_new)
        old_res = self._write_and_import("supersede", old_id, edit_old)
        return {"old": old_id, "new": new_id,
                "old_path": old_res["path"], "new_path": new_res["path"]}
