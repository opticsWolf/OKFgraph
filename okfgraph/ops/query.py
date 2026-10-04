"""Canonical graph query operations: ``search``, ``read``, ``traverse``.

One implementation behind the three surface adapters (CLI, MCP, Python).
Operations return plain data and raise :class:`~okfgraph.errors.OKFError`
on failure — see ``docs/surface-unification-plan.md`` §3.2–3.4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from okfgraph.errors import UsageError
from okfgraph.models import ConceptModel

__all__ = ["QueryOps"]

_WALK_DEFAULTS = {
    "relationship": "CONTAINS",
    "direction": "OUTGOING",
    "depth": 1,
    "node_type": None,
}


class QueryOps:
    """Mixin for :class:`~okfgraph.OKFRouter`; owns the target dispatch."""

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        target: str = "concepts",
        limit: int = 10,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        include_chunks: bool = False,
        max_chunks_per_doc: int = 3,
        expand: bool = False,
        context_hops: int = 1,
        hub_rerank: bool = False,
        hub_weight: float = 0.3,
        rank: str = "none",
    ) -> List[Dict[str, Any]]:
        """Unified search. ``target`` picks the path:

        * ``concepts`` (default) — hybrid vector+FTS on concepts via
          ``search_engine.search_hybrid``; ``include_chunks`` adds
          ``matched_chunks`` to each hit; ``rank`` = none|hub|ppr.
        * ``chunks`` — chunk-level RRF search, ``hub_rerank > expand >
          plain``; the plain path honours ``concept_type``/``tags``/
          ``parent_id`` and ``max_chunks_per_doc``.
        * ``images`` — text-query search over image assets.

        Params a chosen path ignores are refused (X3, rule): a filter
        with ``hub_rerank``/``expand``/``target='images'``, ``rank`` with
        ``target='chunks'``, ``include_chunks`` on a chunk path,
        ``max_chunks_per_doc``/``expand``/``context_hops``/``hub_rerank``
        on the concepts path, or both ``hub_rerank`` and ``expand``.
        Defaults count as not-passed.
        """
        if target not in ("concepts", "chunks", "images"):
            raise UsageError(
                "BAD_VALUE",
                f"target must be 'concepts', 'chunks' or 'images', got '{target}'",
                op="search",
                fields={"target": target},
            )
        if rank not in ("none", "hub", "ppr"):
            raise UsageError(
                "BAD_VALUE",
                f"rank must be 'none', 'hub' or 'ppr', got '{rank}'",
                op="search",
                fields={"rank": rank},
            )

        filt = {
            "concept_type": concept_type,
            "tags": tags,
            "parent_id": parent_id,
        }
        filt = {k: v for k, v in filt.items() if v is not None}

        ignored: List[str] = []
        if target == "images":
            # Everything except query/limit is out of scope for images.
            ignored += [*filt]
            for name, passed in (
                ("include_chunks", include_chunks),
                ("max_chunks_per_doc", max_chunks_per_doc != 3),
                ("expand", expand),
                ("context_hops", context_hops != 1),
                ("hub_rerank", hub_rerank),
                ("hub_weight", hub_weight != 0.3),
                ("rank", rank != "none"),
            ):
                if passed:
                    ignored.append(name)
        elif target == "chunks":
            if rank != "none":
                # Most specific refusal first: the historical message.
                raise UsageError(
                    "BAD_VALUE",
                    "rank is concepts-only; target='chunks' uses hub_rerank/expand",
                    op="search",
                    fields={"rank": rank, "target": "chunks"},
                )
            # Documented path precedence: hub_rerank > expand > plain.
            # When hub_rerank wins, expand is superseded, not ignored.
            if include_chunks:
                ignored.append("include_chunks")
            if hub_rerank:
                ignored += [*filt]
            elif expand:
                ignored += [*filt]
                if hub_weight != 0.3:
                    ignored.append("hub_weight")
            else:
                if hub_weight != 0.3:
                    ignored.append("hub_weight")
        else:  # concepts
            if max_chunks_per_doc != 3:
                ignored.append("max_chunks_per_doc")
            if expand:
                ignored.append("expand")
            if context_hops != 1:
                ignored.append("context_hops")
            if hub_rerank:
                ignored.append("hub_rerank")
        if ignored:
            raise UsageError(
                "BAD_VALUE",
                f"target='{target}' ignores: {', '.join(dict.fromkeys(ignored))}",
                op="search",
                fields={"target": target, "ignored": sorted(dict.fromkeys(ignored))},
                remedy="drop the ignored params or use a path that honours them",
            )

        if target == "images":
            return self.image_mgr.search_images_with_text(
                text_query=query, limit=limit,
            )
        if target == "chunks":
            if hub_rerank:
                return self.search_engine.search_chunks_with_hub_score(
                    query=query, limit=limit, hub_weight=hub_weight,
                )
            if expand:
                return self.search_engine.search_with_context(
                    query=query,
                    limit=min(limit, 20),
                    context_hops=context_hops,
                )
            return self.search_engine.search_chunks(
                query=query, limit=limit,
                max_chunks_per_doc=max_chunks_per_doc, **filt,
            )
        return self.search_engine.search_hybrid(
            query=query, limit=limit, include_chunks=include_chunks,
            rank=rank, hub_weight=hub_weight, **filt,
        )

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        concept_id: str,
        *,
        include: str = "body",
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Read a known concept. ``include`` picks the shape:

        * ``body`` (default) — full concept dict, body markdown included.
        * ``chunks`` — stored chunks ordered by index.
        * ``document`` — ``{"concept_id", "markdown"}`` rebuilt from chunks.
        * ``context`` — ``{incoming_links, outgoing_links, ancestry,
          siblings}``, capped at 10 entries per group. The op owns this
          assembler (the CLI/MCP copies are deleted).
        * ``max_tokens`` — budgeted reading across concept + link
          neighbours; overrides the plain shapes above.
        """
        if include not in ("body", "chunks", "document", "context"):
            raise UsageError(
                "BAD_VALUE",
                f"include must be 'body', 'chunks', 'document' or 'context', "
                f"got '{include}'",
                op="read",
                fields={"include": include},
            )
        concept = self.get_by_id(concept_id)
        if concept is None:
            raise UsageError(
                "UNKNOWN_CONCEPT",
                f"concept '{concept_id}' does not exist",
                op="read",
                fields={"concept_id": concept_id},
            )
        if max_tokens is not None:
            return self.search_engine.read_with_budget(
                concept_id, include=include, max_tokens=max_tokens,
            )
        if include == "chunks":
            return self.search_engine.get_chunks(concept_id)
        if include == "document":
            markdown = self.embed_engine.reconstruct_document(concept_id)
            if not markdown:
                raise UsageError(
                    "UNKNOWN_CONCEPT",
                    f"concept '{concept_id}' has no chunks to rebuild",
                    op="read",
                    fields={"concept_id": concept_id},
                )
            return {"concept_id": concept_id, "markdown": markdown}
        if include == "context":
            incoming = self.search_engine.traverse(
                concept_id, "LINKS_TO", "INCOMING", 1)
            outgoing = self.search_engine.traverse(
                concept_id, "LINKS_TO", "OUTGOING", 1)
            ctx = self.search_engine.get_context(concept_id, cap=10)
            return {
                "incoming_links": incoming[:10],
                "outgoing_links": outgoing[:10],
                "ancestry": ctx["ancestry"],
                "siblings": ctx["siblings"],
            }
        # Coercion boundary: whatever arrives (model or plain mapping) is
        # validated here so adapters always serialize a ConceptModel.
        if not hasattr(concept, "public_dict"):
            concept = ConceptModel.model_validate(concept)
        return concept.public_dict()

    # ------------------------------------------------------------------
    # traverse
    # ------------------------------------------------------------------

    def traverse(
        self,
        start_id: str = "",
        *,
        relationship: str = "CONTAINS",
        direction: str = "OUTGOING",
        depth: int = 1,
        node_type: Optional[str] = None,
        target: Optional[str] = None,
        max_path_length: int = 6,
    ) -> List[Dict[str, Any]]:
        """Three modes, dispatched once here:

        * empty ``start_id`` — root directory listing (walk params must
          be default, else ``BAD_VALUE``);
        * ``target`` set — shortest path from ``start_id`` to ``target``
          (walk params must be default, else ``BAD_VALUE``);
        * otherwise — relationship walk from ``start_id``; unknown
          ``start_id`` raises ``UNKNOWN_CONCEPT`` (was: silent empty).
        """
        walk_non_default = [
            k for k, v in (("relationship", relationship),
                           ("direction", direction), ("depth", depth),
                           ("node_type", node_type))
            if v != _WALK_DEFAULTS[k]
        ]

        if not start_id:
            if target is not None:
                raise UsageError(
                    "BAD_VALUE",
                    "target needs a non-empty start_id to build a path from",
                    op="traverse",
                    fields={"target": target},
                )
            if walk_non_default:
                raise UsageError(
                    "BAD_VALUE",
                    f"empty start_id lists the root directory; ignores: "
                    f"{', '.join(sorted(walk_non_default))}",
                    op="traverse",
                    fields={"ignored": sorted(walk_non_default)},
                    remedy="drop the walk params to list the root directory",
                )
            return self.list_directory("")

        if target is not None:
            if walk_non_default:
                raise UsageError(
                    "BAD_VALUE",
                    f"target mode ignores: {', '.join(sorted(walk_non_default))}",
                    op="traverse",
                    fields={"ignored": sorted(walk_non_default)},
                )
            return self.search_engine.find_path(
                start_id=start_id, target=target,
                max_path_length=max_path_length,
            )

        if not self.search_engine.node_exists(start_id):
            raise UsageError(
                "UNKNOWN_CONCEPT",
                f"start_id '{start_id}' does not exist",
                op="traverse",
                fields={"start_id": start_id},
            )
        return self.search_engine.traverse(
            start_id, relationship=relationship, direction=direction,
            depth=depth, node_type=node_type,
        )
