"""Canonical graph query operations: ``search``, ``read``, ``traverse``.

One implementation behind the three surface adapters (CLI, MCP, Python).
Operations return plain data and raise :class:`~okfgraph.errors.OKFError`
on failure — see ``docs/surface-unification-plan.md`` §3.2–3.4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from okfgraph.errors import OKFError
from okfgraph.models import ConceptModel

__all__ = ["QueryOps"]

_TARGETS = ("concepts", "chunks", "images")
_RANKS = ("none", "hub", "ppr")
_INCLUDES = ("body", "chunks", "document", "context")
_RELATIONSHIPS = ("CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET")
_DIRECTIONS = ("OUTGOING", "INCOMING", "BOTH")

#: Optional search params and their defaults. A param counts as passed
#: when it differs from its default; a passed param the chosen path does
#: not honour is refused (X3) instead of silently dropped.
_SEARCH_DEFAULTS: Dict[str, Any] = {
    "concept_type": None,
    "tags": None,
    "parent_id": None,
    "include_chunks": False,
    "max_chunks_per_doc": 3,
    "expand": False,
    "context_hops": 1,
    "hub_rerank": False,
    "hub_weight": 0.3,
    "rank": "none",
}
_FILTERS = frozenset({"concept_type", "tags", "parent_id"})

#: What each search path honours. Chunk precedence is hub_rerank > expand
#: > plain; when hub_rerank wins, expand (and its context_hops) are
#: superseded rather than refused.
_HONOURS = {
    "images": frozenset(),
    "concepts:none": _FILTERS | {"include_chunks", "rank"},
    "concepts:hub": _FILTERS | {"include_chunks", "rank", "hub_weight"},
    "concepts:ppr": _FILTERS | {"rank"},
    "chunks:hub": frozenset({"hub_rerank", "hub_weight", "expand", "context_hops"}),
    "chunks:expand": frozenset({"expand", "context_hops"}),
    "chunks:plain": _FILTERS | {"max_chunks_per_doc"},
}


def _bad_value(op: str, message: str, **fields) -> OKFError:
    return OKFError("BAD_VALUE", message, op=op, fields=fields or None)


def _check_choice(op: str, name: str, value: Any, choices) -> None:
    if value not in choices:
        allowed = ", ".join(f"'{c}'" for c in choices)
        raise _bad_value(op, f"{name} must be one of {allowed}, got '{value}'",
                         **{name: value})


def _unknown_concept(op: str, field: str, concept_id: str) -> OKFError:
    return OKFError(
        "UNKNOWN_CONCEPT",
        f"{field} '{concept_id}' does not exist",
        op=op,
        fields={field: concept_id},
        remedy="search first to find IDs",
    )


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

        * ``concepts`` (default) — hybrid vector+FTS via
          ``search_engine.search_hybrid``. Honours the filters
          (``concept_type``/``tags``/``parent_id``) and ``rank``
          (none|hub|ppr); ``include_chunks`` adds ``matched_chunks`` (not
          with ``ppr``); ``hub_weight`` only with ``rank='hub'``.
        * ``chunks`` — chunk-level RRF, path precedence ``hub_rerank`` >
          ``expand`` > plain. ``hub_rerank`` honours ``hub_weight``;
          ``expand`` honours ``context_hops``; plain honours the filters
          and ``max_chunks_per_doc``.
        * ``images`` — text query over image assets (``query``/``limit``
          only).

        A passed param (non-default) the chosen path ignores is refused
        with ``BAD_VALUE`` naming it — never silently dropped.
        """
        _check_choice("search", "target", target, _TARGETS)
        _check_choice("search", "rank", rank, _RANKS)
        if not isinstance(limit, int) or limit < 1:
            raise _bad_value("search", f"limit must be >= 1, got {limit!r}",
                             limit=limit)

        params = {
            "concept_type": concept_type, "tags": tags, "parent_id": parent_id,
            "include_chunks": include_chunks,
            "max_chunks_per_doc": max_chunks_per_doc, "expand": expand,
            "context_hops": context_hops, "hub_rerank": hub_rerank,
            "hub_weight": hub_weight, "rank": rank,
        }
        passed = [k for k, v in params.items() if v != _SEARCH_DEFAULTS[k]]

        if target == "chunks":
            if rank != "none":
                # Most specific refusal first: the historical message.
                raise _bad_value(
                    "search",
                    "rank is concepts-only; target='chunks' uses hub_rerank/expand",
                    rank=rank, target="chunks",
                )
            path = ("chunks:hub" if hub_rerank
                    else "chunks:expand" if expand else "chunks:plain")
        elif target == "concepts":
            path = f"concepts:{rank}"
        else:
            path = "images"

        ignored = [k for k in passed if k not in _HONOURS[path]]
        if ignored:
            raise OKFError(
                "BAD_VALUE",
                f"target='{target}' ({path.partition(':')[2] or 'default'} path) "
                f"ignores: {', '.join(ignored)}",
                op="search",
                fields={"target": target, "path": path, "ignored": ignored},
                remedy="drop the ignored params or use a path that honours them",
            )

        filt = {k: params[k] for k in _FILTERS if params[k] is not None}
        if path == "images":
            return self.image_mgr.search_images_with_text(
                text_query=query, limit=limit,
            )
        if path == "chunks:hub":
            return self.search_engine.search_chunks_with_hub_score(
                query=query, limit=limit, hub_weight=hub_weight,
            )
        if path == "chunks:expand":
            return self.search_engine.search_with_context(
                query=query,
                limit=min(limit, 20),
                context_hops=context_hops,
            )
        if path == "chunks:plain":
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
        * ``chunks`` — stored chunks (dicts) ordered by index.
        * ``document`` — ``{"concept_id", "markdown"}`` rebuilt from chunks.
        * ``context`` — ``{incoming_links, outgoing_links, ancestry,
          siblings}``, capped at 10 entries per group.
        * ``max_tokens`` — budgeted reading across concept + link
          neighbours: ``{sections, used, budget, truncated}``; overrides
          the plain shapes above.
        """
        _check_choice("read", "include", include, _INCLUDES)
        if max_tokens is not None and max_tokens < 1:
            raise _bad_value("read", f"max_tokens must be >= 1, got {max_tokens}",
                             max_tokens=max_tokens)
        concept = self.get_by_id(concept_id)
        if concept is None:
            raise _unknown_concept("read", "concept", concept_id)
        if max_tokens is not None:
            return self.search_engine.read_with_budget(
                concept_id, include=include, max_tokens=max_tokens,
            )
        if include == "chunks":
            return [
                chunk.model_dump(exclude={"embedding"})
                if hasattr(chunk, "model_dump") else dict(chunk)
                for chunk in self.search_engine.get_chunks(concept_id)
            ]
        if include == "document":
            markdown = self.embed_engine.reconstruct_document(concept_id)
            if not markdown:
                raise OKFError(
                    "UNKNOWN_CONCEPT",
                    f"concept '{concept_id}' has no chunks to rebuild",
                    op="read",
                    fields={"concept_id": concept_id},
                    remedy="use include='body' for unchunked concepts",
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

        * empty ``start_id`` — root directory listing;
        * ``target`` set — shortest path from ``start_id`` to ``target``
          (honours ``max_path_length``);
        * otherwise — relationship walk from ``start_id`` (honours
          ``relationship``/``direction``/``depth``/``node_type``); an
          unknown ``start_id`` raises ``UNKNOWN_CONCEPT``.

        Params the chosen mode ignores are refused with ``BAD_VALUE``.
        """
        _check_choice("traverse", "relationship", relationship, _RELATIONSHIPS)
        _check_choice("traverse", "direction", direction, _DIRECTIONS)
        if not isinstance(depth, int) or depth < 1:
            raise _bad_value("traverse", f"depth must be >= 1, got {depth!r}",
                             depth=depth)
        walk_passed = [
            k for k, v, d in (("relationship", relationship, "CONTAINS"),
                              ("direction", direction, "OUTGOING"),
                              ("depth", depth, 1),
                              ("node_type", node_type, None))
            if v != d
        ]
        path_passed = ["max_path_length"] if max_path_length != 6 else []

        if not start_id:
            if target is not None:
                raise _bad_value(
                    "traverse",
                    "target needs a non-empty start_id to build a path from",
                    target=target,
                )
            ignored = walk_passed + path_passed
            mode = "root listing"
        elif target is not None:
            ignored = walk_passed
            mode = "path"
        else:
            ignored = path_passed
            mode = "walk"
        if ignored:
            raise OKFError(
                "BAD_VALUE",
                f"traverse {mode} mode ignores: {', '.join(ignored)}",
                op="traverse",
                fields={"mode": mode, "ignored": ignored},
                remedy="drop the ignored params",
            )

        if not start_id:
            return self.list_directory("")
        if target is not None:
            return self.search_engine.find_path(
                start_id=start_id, target=target,
                max_path_length=max_path_length,
            )
        if not self.search_engine.node_exists(start_id):
            raise _unknown_concept("traverse", "start_id", start_id)
        return self.search_engine.traverse(
            start_id, relationship=relationship, direction=direction,
            depth=depth, node_type=node_type,
        )
