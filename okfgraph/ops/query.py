"""Canonical graph query operations: ``search``, ``read``, ``traverse``.

One implementation behind the three surface adapters (CLI, MCP, Python).
Operations return plain data and raise :class:`~okfgraph.errors.OKFError`
on failure — see ``docs/surface-unification-plan.md`` §3.2–3.4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from okfgraph.components.search import MAX_HOPS
from okfgraph.errors import OKFError
from okfgraph.models import ConceptModel

__all__ = ["QueryOps"]

_TARGETS = ("concepts", "chunks", "images")
_RANKS = ("none", "hub", "ppr", "cross")
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
    "where": None,
    "since": None,
    "until": None,
    "exclude_tags": None,
    "link_kinds": None,
    "near_id": None,
    "hops": 2,
    "offset": 0,
    "recency": None,
    "recency_weight": 0.3,
    "rerank": False,
}
_FILTERS = frozenset({"concept_type", "tags", "parent_id",
                      # 0.11.0 (O7)
                      "where", "since", "until", "exclude_tags",
                      # 0.11.0: graph neighbourhood
                      "near_id"})
#: Paging, the neighbourhood radius and the recency boost ride with the
#: filters.
_PAGED = _FILTERS | {"hops", "offset", "recency", "recency_weight"}

#: What each search path honours. Chunk precedence is hub_rerank > expand
#: > plain; when hub_rerank wins, expand (and its context_hops) are
#: superseded rather than refused.
_HONOURS = {
    "images": frozenset(),
    "concepts:none": _PAGED | {"include_chunks", "rank"},
    "concepts:hub": _PAGED | {"include_chunks", "rank", "hub_weight", "link_kinds"},
    "concepts:ppr": _PAGED | {"rank", "link_kinds"},
    # Cross-encoder rerank (0.11.0): no recency — logits have no scale to
    # blend with (refused with its own message below).
    "concepts:cross": _PAGED | {"include_chunks", "rank"},
    "chunks:hub": frozenset({"hub_rerank", "hub_weight", "expand", "context_hops",
                             "rerank"}),
    "chunks:expand": frozenset({"expand", "context_hops"}),
    "chunks:plain": _PAGED | {"max_chunks_per_doc", "rerank"},
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
        where: Optional[Dict[str, Any]] = None,
        since: Any = None,
        until: Any = None,
        exclude_tags: Optional[List[str]] = None,
        link_kinds: Optional[List[str]] = None,
        near_id: Optional[str] = None,
        hops: int = 2,
        offset: int = 0,
        recency: Optional[float] = None,
        recency_weight: float = 0.3,
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """Unified search. ``target`` picks the path:

        * ``concepts`` (default) — hybrid vector+FTS via
          ``search_engine.search_hybrid``. Honours the filters
          (``concept_type``/``tags``/``parent_id``) and ``rank``
          (none|hub|ppr|cross); ``include_chunks`` adds ``matched_chunks``
          (not with ``ppr``); ``hub_weight`` only with ``rank='hub'``.
        * ``chunks`` — chunk-level RRF, path precedence ``hub_rerank`` >
          ``expand`` > plain. ``hub_rerank`` honours ``hub_weight``;
          ``expand`` honours ``context_hops``; plain honours the filters
          and ``max_chunks_per_doc``. ``rerank`` works on plain and
          ``hub_rerank`` (after the hub blend).
        * ``images`` — text query over image assets (``query``/``limit``
          only).

        0.11.0 filters (O7, concepts and plain chunks): ``where`` (equality
        on ``extra`` keys, values compared as stored strings), ``since`` /
        ``until`` (on ``effective_at``: the authored date, else when the
        content last changed), ``exclude_tags``. ``exclude_tags=None``
        applies the router's configured default (``[search] exclude_tags``);
        pass ``[]`` to search everything. ``link_kinds`` (O6) limits the
        graph signal of ``rank='hub'``/``'ppr'`` to those link kinds.

        ``near_id`` (concepts and plain chunks) keeps results within
        ``hops`` (1–3, default 2) LINKS_TO edges of that concept, either
        direction, the concept itself included; the neighbourhood is
        scored exactly, so a small one is never starved by the index
        window. ``offset`` skips that many results (paging; ties break
        by id, so pages are stable across processes).

        ``recency`` (concepts and plain chunks; a half-life in days) blends
        recency into the order: ``(1 - recency_weight) * relevance +
        recency_weight * 0.5 ** (age / recency)``, age from
        ``effective_at``. Off by default; ``recency_weight`` (0–1, default
        0.3) needs it.

        Cross-encoder rerank (0.11.0, opt-in): ``rerank=True`` (chunks) or
        ``rank='cross'`` (concepts) rescores the filtered top
        ``rerank_top_n`` (router setting, default 30) with the configured
        cross-encoder and reorders them by it; each rescored row gains
        ``cross_score`` (a native logit, not a probability). The model
        loads on first use. Not combined with ``recency``.

        A passed param (non-default) the chosen path ignores is refused
        with ``BAD_VALUE`` naming it — never silently dropped.
        """
        _check_choice("search", "target", target, _TARGETS)
        if where is not None and not isinstance(where, dict):
            raise _bad_value("search", "where must be a mapping of extra key -> value",
                             where=str(where))
        _check_choice("search", "rank", rank, _RANKS)
        if not isinstance(limit, int) or limit < 1:
            raise _bad_value("search", f"limit must be >= 1, got {limit!r}",
                             limit=limit)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise _bad_value("search", f"offset must be >= 0, got {offset!r}",
                             offset=offset)
        if (not isinstance(hops, int) or isinstance(hops, bool)
                or not 1 <= hops <= MAX_HOPS):
            raise _bad_value("search", f"hops must be 1..{MAX_HOPS}, got {hops!r}",
                             hops=hops)
        if hops != _SEARCH_DEFAULTS["hops"] and not near_id:
            raise _bad_value("search", "hops needs near_id", hops=hops)
        if recency is not None and (
                isinstance(recency, bool) or not isinstance(recency, (int, float))
                or not recency > 0):
            raise _bad_value("search", f"recency must be a half-life in days > 0, "
                             f"got {recency!r}", recency=recency)
        if (isinstance(recency_weight, bool)
                or not isinstance(recency_weight, (int, float))
                or not 0 <= recency_weight <= 1):
            raise _bad_value("search", f"recency_weight must be 0..1, got "
                             f"{recency_weight!r}", recency_weight=recency_weight)
        if recency_weight != _SEARCH_DEFAULTS["recency_weight"] and recency is None:
            raise _bad_value("search", "recency_weight needs recency",
                             recency_weight=recency_weight)
        if not isinstance(rerank, bool):
            raise _bad_value("search", f"rerank must be true or false, got {rerank!r}",
                             rerank=rerank)
        if rerank and target == "concepts":
            raise _bad_value(
                "search", "rerank is for target='chunks'; concepts use rank='cross'",
                rerank=True, target="concepts")
        if (rerank or rank == "cross") and recency is not None:
            # Cross-encoder logits have no scale to blend recency into.
            raise _bad_value(
                "search", "recency does not combine with the cross-encoder rerank "
                "(rerank / rank='cross'); use one or the other",
                recency=recency, rerank=rerank, rank=rank)

        params = {
            "concept_type": concept_type, "tags": tags, "parent_id": parent_id,
            "include_chunks": include_chunks,
            "max_chunks_per_doc": max_chunks_per_doc, "expand": expand,
            "context_hops": context_hops, "hub_rerank": hub_rerank,
            "hub_weight": hub_weight, "rank": rank,
            "where": where, "since": since, "until": until,
            "exclude_tags": exclude_tags, "link_kinds": link_kinds,
            "near_id": near_id, "hops": hops, "offset": offset,
            "recency": recency, "recency_weight": recency_weight,
            "rerank": rerank,
        }
        passed = [k for k, v in params.items() if v != _SEARCH_DEFAULTS[k]]

        if target == "chunks":
            if rank != "none":
                # Most specific refusal first: the historical message.
                raise _bad_value(
                    "search",
                    "rank is concepts-only; target='chunks' uses "
                    "hub_rerank/expand/rerank",
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

        if near_id and self.get_by_id(near_id) is None:
            raise _unknown_concept("search", "near_id", near_id)
        filt = {k: params[k] for k in _FILTERS if params[k] is not None}
        if near_id:
            filt["hops"] = hops
        if offset:
            filt["offset"] = offset
        if recency is not None:
            filt["recency"] = float(recency)
            filt["recency_weight"] = float(recency_weight)
        if exclude_tags is None:
            # Configured default (O7): e.g. superseded findings drop out
            # unless the caller asks for them.
            default = list(getattr(self, "exclude_tags", None) or [])
            if default and path != "images":
                filt["exclude_tags"] = default
        if path == "images":
            return self.image_mgr.search_images_with_text(
                text_query=query, limit=limit,
            )
        cross = {"rerank": True} if rerank else {}
        if path == "chunks:hub":
            return self.search_engine.search_chunks_with_hub_score(
                query=query, limit=limit, hub_weight=hub_weight, **cross,
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
                max_chunks_per_doc=max_chunks_per_doc, **cross, **filt,
            )
        if link_kinds:
            filt["link_kinds"] = list(link_kinds)
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
        link_kinds: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Three modes, dispatched once here:

        * empty ``start_id`` — root directory listing;
        * ``target`` set — shortest path from ``start_id`` to ``target``
          (honours ``max_path_length``);
        * otherwise — relationship walk from ``start_id`` (honours
          ``relationship``/``direction``/``depth``/``node_type``); an
          unknown ``start_id`` raises ``UNKNOWN_CONCEPT``.

        ``link_kinds`` (0.11.0, O6; walk mode with ``LINKS_TO``) follows
        only links of those kinds, e.g. ``["supersedes"]``; ``mention`` is
        the kind of every body link.

        Params the chosen mode ignores are refused with ``BAD_VALUE``.
        """
        if link_kinds is not None and relationship != "LINKS_TO":
            raise _bad_value("traverse", "link_kinds needs relationship='LINKS_TO'",
                             link_kinds=list(link_kinds), relationship=relationship)
        _check_choice("traverse", "relationship", relationship, _RELATIONSHIPS)
        _check_choice("traverse", "direction", direction, _DIRECTIONS)
        if not isinstance(depth, int) or depth < 1:
            raise _bad_value("traverse", f"depth must be >= 1, got {depth!r}",
                             depth=depth)
        walk_passed = [
            k for k, v, d in (("relationship", relationship, "CONTAINS"),
                              ("direction", direction, "OUTGOING"),
                              ("depth", depth, 1),
                              ("node_type", node_type, None),
                              ("link_kinds", link_kinds, None))
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
            # An unknown end is an error, not "no path".
            for name, cid in (("start_id", start_id), ("target", target)):
                if not self.search_engine.node_exists(cid):
                    raise _unknown_concept("traverse", name, cid)
            return self.search_engine.find_path(
                start_id=start_id, target=target,
                max_path_length=max_path_length,
            )
        if not self.search_engine.node_exists(start_id):
            raise _unknown_concept("traverse", "start_id", start_id)
        kinds = {"link_kinds": list(link_kinds)} if link_kinds else {}
        return self.search_engine.traverse(
            start_id, relationship=relationship, direction=direction,
            depth=depth, node_type=node_type, **kinds,
        )
