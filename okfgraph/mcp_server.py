"""MCP server for the OKF knowledge graph.

Exposes the graph ops as eleven Model Context Protocol tools so any
MCP-compatible client (Claude Desktop, Cursor, Continue, …) can search,
read, traverse, ingest and export directly. Every tool is a thin adapter
over the canonical op on :class:`~okfgraph.router.OKFRouter`; results are
the envelope ``{ok, op, data, warnings, error}`` and failures raise so
the wire result carries ``isError: true``.

Usage:
    # Entry point (configured in pyproject.toml)
    okf-mcp --db-path ./my_graph.db

    # Or as a module
    python -m okfgraph.mcp_server --db-path ./my_graph.db

    # Or programmatically
    from okfgraph.mcp_server import create_mcp_server
    from okfgraph.settings import Settings
    create_mcp_server(Settings(db_path="./my_graph.db")).run()

    # 0.11.0 (O4): a stdio proxy to one `okf serve` owner
    okf-mcp --connect unix:///run/okf.sock

Boot flags are the CLI's global flags (one settings table); the server
also reads ``okfgraph.toml`` and ``OKFGRAPH_*`` env vars. ``db_path`` must
come from one of them — the client's working directory is no anchor.
"""

import argparse
import json
import logging
import sys
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from okfgraph.errors import OKFError, dumps_envelope, envelope, internal_error
from okfgraph.router import OKFRouter
from okfgraph.settings import MCP_REQUIRED, Settings, cli_signal, set_cli_flags

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "OKF knowledge graph with ONNX + Jina v5 embeddings: persistent project "
    "knowledge as markdown concepts with hybrid (vector + full-text) search "
    "and graph traversal. Start with search; read a concept once you have "
    "its id; traverse to follow links or list directories; ingest to "
    "persist new knowledge (kind='thoughts' for reasoning, 'md' for a "
    "markdown file, 'pdf' for PDF/Office documents). Every result is an "
    "envelope {ok, op, data, warnings, error}; failures set isError with "
    "a typed error code (e.g. UNKNOWN_CONCEPT means search for the id "
    "first, BAD_VALUE names the parameter to drop)."
)


@dataclass
class GraphContext:
    """Shared context for the MCP server lifespan."""
    router: OKFRouter
    settings: Optional[Settings] = None


def make_lifespan(settings: Settings, given_router=None):
    """Factory that returns a lifespan async-context-manager for MCPServer.

    With ``router`` given (0.11.0: ``okf serve``'s locked owner router, or an
    ``OKFClient`` for ``--connect``), the server uses it and does not close
    it — its owner does.
    """

    @asynccontextmanager
    async def _lifespan(mcp: MCPServer):
        if given_router is not None:
            yield GraphContext(router=given_router, settings=settings)
            return
        db_path = settings.db_path
        # Default bundle root: the database's parent directory.
        root = settings.bundle_root or str(Path(db_path).parent)
        router = OKFRouter(
            db_path=db_path,
            bundle_root=root,
            **settings.router_kwargs(),
        )
        logger.info(
            "OKFgraph MCP server started: db=%s model=%s device=%s precision=%s",
            db_path, settings.model_id, settings.device, settings.precision,
        )
        try:
            yield GraphContext(router=router, settings=settings)
        finally:
            router.close()
            logger.info("OKFgraph MCP server shutdown complete")

    return _lifespan


def _graph(ctx: Context) -> GraphContext:
    gc = ctx.request_context.lifespan_context
    if isinstance(gc, GraphContext):
        return gc
    if isinstance(gc, dict) and "router" in gc:
        return GraphContext(router=gc["router"], settings=gc.get("settings"))
    raise RuntimeError("No OKFRouter found in lifespan context")


def _get_router(ctx: Context) -> OKFRouter:
    """Extract the OKFRouter from the MCP context."""
    return _graph(ctx).router


class _ToolFailure(ToolError):
    """Raised by a tool so the wire result carries ``isError: true`` (D7)
    with the full error envelope as the result text (after the SDK's
    ``Error executing tool <name>:`` prefix)."""

    def __init__(self, payload):
        super().__init__(json.dumps(payload, default=str, indent=2))
        self.payload = payload


def _tool_result(op: str, produce):
    """Envelope adapter (§4): success → success envelope; failure → raise.

    Non-OKFError exceptions never cross as bare tracebacks: they are
    wrapped as ``INTERNAL`` with the type name. Warnings recorded during
    the call ride on the envelope.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            data = produce()
        except OKFError as exc:
            raise _ToolFailure(envelope(op, error=exc, warnings=caught)) from None
        except Exception as exc:  # noqa: BLE001 — catch-all is the point (§4)
            raise _ToolFailure(envelope(
                op, error=internal_error(exc, op=op), warnings=caught)) from None
    return dumps_envelope(envelope(op, data=data, warnings=caught))


_RO = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                      idempotentHint=True, openWorldHint=False)
# Exports rewrite files in output_dir: repeatable, nothing else touched.
_EXPORT = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                          idempotentHint=True, openWorldHint=False)
# Ingest adds graph content; thoughts mint a new concept per call.
_INGEST = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                          idempotentHint=False, openWorldHint=False)
# 0.11.0 (O11): update/supersede rewrite frontmatter (repeatable);
# delete moves the file to the trash and soft-deletes (24h recovery).
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                         idempotentHint=True, openWorldHint=False)
_DELETE = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                          idempotentHint=False, openWorldHint=False)

_FLAVOR = Literal["okf", "obsidian"]


def create_mcp_server(settings: Optional[Settings], router=None) -> MCPServer:
    """Create an MCP server instance connected to an OKFgraph database.

    Args:
        settings: Merged canonical settings (``Settings.load`` or direct
            construction in tests).
        router: Use this router-shaped object instead of opening one
            (0.11.0, O4: ``okf serve``'s owner, or an ``OKFClient``).

    Returns:
        Configured MCPServer instance.
    """
    settings = settings or Settings()
    mcp = MCPServer(
        name="OKFgraph MCP Server",
        instructions=INSTRUCTIONS,
        lifespan=make_lifespan(settings, router),
    )

    @mcp.tool(annotations=_RO)
    def search(
        query: Annotated[str, Field(description="The search query text.")],
        target: Annotated[
            Literal["concepts", "chunks", "images"],
            Field(
                description=(
                    "'concepts' = hybrid vector+FTS over concepts (open-ended questions). "
                    "'chunks' = hybrid search over document chunks (exact passages). "
                    "'images' = text query against image assets (query/limit only)."
                ),
            ),
        ] = "concepts",
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum results.")] = 10,
        concept_type: Annotated[Optional[str], Field(description="Concept type filter (concepts; plain chunks path).")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Tag filter, ALL must match (concepts; plain chunks path).")] = None,
        parent_id: Annotated[Optional[str], Field(description="Directory ID to constrain search (concepts; plain chunks path).")] = None,
        include_chunks: Annotated[bool, Field(description="Concepts (rank none|hub): attach matched chunks per hit. For chunk-level hits use target='chunks'.")] = False,
        max_chunks_per_doc: Annotated[int, Field(ge=1, le=10, description="Plain chunks path: cap hits per source document.")] = 3,
        expand: Annotated[bool, Field(description="Chunks: attach graph neighborhood (links, ancestry, siblings) to each hit.")] = False,
        context_hops: Annotated[int, Field(ge=1, le=3, description="Neighborhood depth with expand=true.")] = 1,
        hub_rerank: Annotated[bool, Field(description="Chunks: rerank by graph hub score (incoming links). Wins over expand.")] = False,
        hub_weight: Annotated[float, Field(ge=0, le=1, description="Hub weight for hub_rerank (chunks) or rank='hub' (concepts).")] = 0.3,
        rank: Annotated[
            Literal["none", "hub", "ppr"],
            Field(
                description=(
                    "Concepts only: 'none' = RRF order. 'hub' = blend RRF with "
                    "incoming-link authority. 'ppr' = model-free lexical-seed "
                    "PPR: no ONNX load, deterministic — use when cold, when "
                    "the model is unavailable, or when the query names topics."
                ),
            ),
        ] = "none",
        where: Annotated[Optional[dict[str, str]], Field(description="Equality filter on extra frontmatter keys, e.g. {\"jev_category\": \"evidence\"} (concepts; plain chunks path).")] = None,
        since: Annotated[Optional[str], Field(description="Only concepts dated >= this ISO date/time (authored date, else last content change).")] = None,
        until: Annotated[Optional[str], Field(description="Only concepts dated < this ISO date/time.")] = None,
        exclude_tags: Annotated[Optional[list[str]], Field(description="Drop concepts with any of these tags. Omit for the server default (e.g. superseded); [] searches everything.")] = None,
        link_kinds: Annotated[Optional[list[str]], Field(description="rank='hub'|'ppr': count only these link kinds (mention, supersedes, supports, ...).")] = None,
        near_id: Annotated[Optional[str], Field(description="Only results within `hops` links of this concept (either direction, itself included); concepts and plain chunks. Use it to search one neighbourhood, e.g. the notes around a design doc.")] = None,
        hops: Annotated[Optional[int], Field(description="Radius for near_id, 1-3 (default 2).")] = None,
        offset: Annotated[int, Field(description="Skip this many results (paging: offset=limit gets page 2).")] = 0,
        recency: Annotated[Optional[float], Field(description="Favour recent concepts: half-life in days (e.g. 30). Concepts and plain chunks.")] = None,
        recency_weight: Annotated[Optional[float], Field(description="Share of the recency term, 0-1 (default 0.3); needs recency.")] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Search the knowledge graph — start here for any question over stored knowledge.

        Routing: target='concepts' for open-ended questions; 'chunks' for exact
        passages (expand=true adds graph neighborhood, hub_rerank=true ranks by
        importance); 'images' for image assets. rank='ppr' answers from the
        link graph alone when the embedder is cold or unavailable. Params the
        chosen path ignores are refused with BAD_VALUE naming them. If you
        already have a concept ID, use read; for related concepts, traverse."""
        router = _get_router(ctx)
        return _tool_result("search", lambda: router.search(
            query,
            target=target,
            limit=limit,
            concept_type=concept_type,
            tags=tags,
            parent_id=parent_id,
            include_chunks=include_chunks,
            max_chunks_per_doc=max_chunks_per_doc,
            expand=expand,
            context_hops=context_hops,
            hub_rerank=hub_rerank,
            hub_weight=hub_weight,
            rank=rank,
            **{k: v for k, v in (("where", where), ("since", since),
                                 ("until", until), ("exclude_tags", exclude_tags),
                                 ("link_kinds", link_kinds), ("near_id", near_id),
                                 ("hops", hops), ("recency", recency),
                                 ("recency_weight", recency_weight)) if v is not None},
            **({"offset": offset} if offset else {}),
        ))

    @mcp.tool(annotations=_RO)
    def read(
        concept_id: Annotated[str, Field(description="ID of the concept to read.")],
        include: Annotated[
            Literal["body", "chunks", "document", "context"],
            Field(
                description=(
                    "'body' = the concept with its full markdown. "
                    "'chunks' = stored chunks ordered by index. "
                    "'document' = original markdown rebuilt from chunks. "
                    "'context' = incoming/outgoing links, ancestry, siblings."
                ),
            ),
        ] = "body",
        max_tokens: Annotated[
            Optional[int],
            Field(
                ge=100,
                description=(
                    "Token budget: assemble the concept plus PPR-ranked link "
                    "neighbours (index-first for context), truncating the last "
                    "section to fit. Omit for the full uncapped shapes."
                ),
            ),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Read a known concept — search first to find IDs.

        Routing: include='body' for full text; 'chunks' for passages;
        'document' to rebuild the original markdown; 'context' for links,
        ancestry, and siblings. max_tokens caps the reading to a budgeted
        section list. For open questions use search; to walk
        relationships use traverse."""
        router = _get_router(ctx)
        return _tool_result("read", lambda: router.read(
            concept_id, include=include, max_tokens=max_tokens,
        ))

    @mcp.tool(annotations=_RO)
    def traverse(
        start_id: Annotated[str, Field(description="ID of the starting concept or directory. Empty string (default) lists the root directory.")] = "",
        relationship: Annotated[
            Literal["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"],
            Field(description="Relationship type (walk mode). CONTAINS depth 1 = directory listing."),
        ] = "CONTAINS",
        direction: Annotated[
            Literal["OUTGOING", "INCOMING", "BOTH"],
            Field(description="Traversal direction (walk mode)."),
        ] = "OUTGOING",
        depth: Annotated[int, Field(ge=1, le=5, description="Maximum traversal depth (walk mode).")] = 1,
        node_type: Annotated[
            Optional[str],
            Field(description="Filter walk results by node type (e.g. 'Concept', 'Directory'). Walk mode only."),
        ] = None,
        target: Annotated[
            Optional[str],
            Field(description="If set, find the shortest path from start_id to this concept ID instead of walking."),
        ] = None,
        max_path_length: Annotated[int, Field(ge=1, le=10, description="Maximum path length (path mode, with target).")] = 6,
        link_kinds: Annotated[
            Optional[list[str]],
            Field(description="LINKS_TO walk only: follow only these link kinds, e.g. ['supersedes']. Body links are 'mention'."),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Navigate graph relationships, browse directories, or connect two concepts.

        Routing: empty start_id lists the root; CONTAINS depth 1 lists a
        directory; LINKS_TO follows references; target=<id> finds the
        shortest path instead. Params the chosen mode ignores are refused.
        Search first to find IDs."""
        router = _get_router(ctx)
        return _tool_result("traverse", lambda: router.traverse(
            start_id,
            relationship=relationship,
            direction=direction,
            depth=depth,
            node_type=node_type,
            target=target,
            max_path_length=max_path_length,
            **({"link_kinds": link_kinds} if link_kinds is not None else {}),
        ))

    @mcp.tool(annotations=_INGEST)
    def ingest(
        kind: Annotated[
            Literal["md", "pdf", "thoughts"],
            Field(description="'md' = import a markdown file. 'pdf' = convert a PDF/Office file (bobine) and import. 'thoughts' = persist LLM reasoning as a searchable concept."),
        ],
        md_path: Annotated[Optional[str], Field(description="Markdown file to import (kind='md').")] = None,
        pdf_path: Annotated[Optional[str], Field(description="File to convert and import (kind='pdf'): PDF or Office (docx/xlsx/pptx, legacy doc/xls/ppt). Bobine dispatches on extension.")] = None,
        thoughts: Annotated[Optional[str], Field(description="Raw reasoning text (kind='thoughts').")] = None,
        topic: Annotated[Optional[str], Field(description="Topic for kind='thoughts' (required then). Keep a consistent per-project scheme.")] = None,
        concept_id: Annotated[Optional[str], Field(description="Explicit concept ID (md/thoughts). Generated if omitted (thoughts: thoughts/<topic>/<ts>_<id>, a virtual namespace — no file needed). Slashes are namespaces.")] = None,
        title: Annotated[Optional[str], Field(description="Title override (md/thoughts). A thought title must be unique within its topic; omit it to get the first sentence of the text.")] = None,
        description: Annotated[Optional[str], Field(description="Description override (md/thoughts).")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Tags to apply (md/thoughts).")] = None,
        mode: Annotated[
            Optional[Literal["text", "optional", "omni"]],
            Field(description="Image ingestion mode (md/pdf): text (captions), optional/omni (ONNX vision content, needs a text-nano graph). Omit for the server's configured default."),
        ] = None,
        routing_mode: Annotated[
            Literal["auto", "surgical", "always", "never"],
            Field(description="PDF converter routing (kind='pdf'). 'never' = fast path, no ONNX."),
        ] = "auto",
        extract_images: Annotated[bool, Field(description="Extract embedded images (kind='pdf').")] = True,
        type: Annotated[Optional[str], Field(description="Concept type (md/thoughts), e.g. 'decision'. thoughts default to 'thought'.")] = None,
        extra: Annotated[Optional[dict], Field(description="Extra frontmatter fields (md/thoughts), e.g. {'jev_category': 'evidence', 'relations': {'supersedes': ['finding/x']}}.")] = None,
        materialize: Annotated[Optional[bool], Field(description="md/thoughts: write the markdown into the bundle at <concept_id>.md, then import that file (the bundle stays the source of truth).")] = None,
        related: Annotated[Optional[list[str]], Field(description="thoughts: ids (or paths/titles) of the concepts this thought is about; stored as typed 'related' links so traverse and rank='hub'|'ppr' see the thought. Unresolved ones come back in unresolved_related.")] = None,
        folder: Annotated[Optional[str], Field(description="thoughts: also write the thought as a file, <folder>/<title>.md in the bundle, and import it; '@alias/dir' for another root. Omit it (the default): thoughts live in the graph only. Replaces concept_id + materialize.")] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Add content to the knowledge graph. Returns the new concept ID(s).

        Thoughts are stored in the graph only (no file) unless folder or
        materialize asks for one.

        Results: md → {concept_id, chunk_count}; thoughts → {concept_id, title}
        (+ unresolved_related when related was passed);
        an explicit concept_id upserts (the same call twice is one concept);
        pdf → {concept_ids, page_count, ...} (its md_path is a temp file —
        verify with search, not by reading the path). Markdown is
        mordant-linted before import. Missing required params raise
        MISSING_PARAM; absent files FILE_NOT_FOUND."""
        gc = _graph(ctx)
        cfg = gc.settings or Settings()
        return _tool_result("ingest", lambda: gc.router.ingest(
            kind,
            md_path=md_path,
            pdf_path=pdf_path,
            thoughts=thoughts,
            topic=topic,
            concept_id=concept_id,
            title=title,
            description=description,
            tags=tags,
            mode=mode or cfg.mode,
            routing_mode=routing_mode,
            extract_images=extract_images,
            batch_size=cfg.batch_size,
            **{k: v for k, v in (("type", type), ("extra", extra),
                                 ("materialize", materialize),
                                 ("related", related), ("folder", folder))
               if v is not None},
        ))

    @mcp.tool(annotations=_EXPORT)
    def export_bundle(
        output_dir: Annotated[str, Field(description="Output directory for the bundle.")],
        directory_id: Annotated[Optional[str], Field(description="Only export concepts under this directory.")] = None,
        concept_type: Annotated[Optional[str], Field(description="Only export concepts of this type.")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Only export concepts with ALL these tags.")] = None,
        flavor: Annotated[
            _FLAVOR,
            Field(description="'okf' = [t](id.md) links + index files. 'obsidian' = [[Title]] wikilinks, no index files, re-imports losslessly."),
        ] = "okf",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Export concepts to an OKF bundle directory. Returns {output_dir, concept_ids, flavor}.

        To add content back to the graph, use ingest."""
        router = _get_router(ctx)
        return _tool_result("export_bundle", lambda: router.export_bundle(
            output_dir,
            directory_id=directory_id,
            concept_type=concept_type,
            tags=tags,
            flavor=flavor,
        ))

    @mcp.tool(annotations=_EXPORT)
    def export_concept(
        concept_id: Annotated[str, Field(description="ID of the concept to export.")],
        output_dir: Annotated[str, Field(description="Output directory; writes <output_dir>/<concept_id>.md.")],
        flavor: Annotated[
            _FLAVOR,
            Field(description="'okf' = [t](id.md) links. 'obsidian' = [[Title]] wikilinks."),
        ] = "okf",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Export a single concept to a .md file. Returns {concept_id, path}.

        Unknown IDs raise UNKNOWN_CONCEPT (search first). For a whole
        bundle use export_bundle."""
        router = _get_router(ctx)
        return _tool_result("export_concept", lambda: router.export_concept(
            concept_id, output_dir=output_dir, flavor=flavor,
        ))

    @mcp.tool(annotations=_RO)
    def list_images(
        concept_id: Annotated[str, Field(description="ID of the concept whose images to list.")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """List image assets attached to a concept (metadata, no bytes).

        Unknown concepts raise UNKNOWN_CONCEPT — search first."""
        router = _get_router(ctx)
        return _tool_result("list_images", lambda: router.list_images(concept_id))

    @mcp.tool(annotations=_RO)
    def get_image(
        asset_id: Annotated[str, Field(description="ID of the image asset (from list_images or an images search).")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Fetch one image asset: metadata plus base64-encoded ``data``.

        Unknown assets raise UNKNOWN_ASSET."""
        router = _get_router(ctx)
        return _tool_result("get_image", lambda: router.get_image(asset_id))

    @mcp.tool(annotations=_WRITE)
    def update(
        concept_id: Annotated[str, Field(description="ID of the concept to edit.")],
        title: Annotated[Optional[str], Field(description="New title.")] = None,
        description: Annotated[Optional[str], Field(description="New description.")] = None,
        type: Annotated[Optional[str], Field(description="New concept type.")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Replace the tag list (not with add_tags/remove_tags).")] = None,
        add_tags: Annotated[Optional[list[str]], Field(description="Tags to add.")] = None,
        remove_tags: Annotated[Optional[list[str]], Field(description="Tags to remove.")] = None,
        extra: Annotated[Optional[dict], Field(description="Frontmatter keys to set (a null value deletes the key).")] = None,
        relations: Annotated[Optional[dict], Field(description="Replace the relations mapping, e.g. {'supports': ['docs/ledger']}.")] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Edit a concept's frontmatter (never its body). Writes the bundle file, then imports it.

        Returns {concept_id, path, changed}. Unknown IDs raise UNKNOWN_CONCEPT."""
        router = _get_router(ctx)
        return _tool_result("update", lambda: router.update(
            concept_id, title=title, description=description, type=type,
            tags=tags, add_tags=add_tags, remove_tags=remove_tags,
            extra=extra, relations=relations,
        ))

    @mcp.tool(annotations=_WRITE)
    def supersede(
        old_id: Annotated[str, Field(description="The concept that is no longer current.")],
        new_id: Annotated[str, Field(description="The concept that replaces it.")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Record that new_id supersedes old_id: a typed 'supersedes' link on the new
        concept and the 'superseded' tag on the old one (both files rewritten and imported).

        Superseded concepts drop out of search when the server excludes that tag."""
        router = _get_router(ctx)
        return _tool_result("supersede", lambda: router.supersede(old_id, new_id))

    @mcp.tool(annotations=_DELETE)
    def delete(
        concept_id: Annotated[str, Field(description="ID of the concept to delete.")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Soft-delete a concept: its bundle file moves to .okf-trash and the concept is
        recoverable for 24h (deleted-recover). Returns {concept_id, recoverable_until}."""
        router = _get_router(ctx)
        return _tool_result("delete", lambda: router.delete(concept_id, remove_file=True))

    return mcp


def main():
    """Entry point for ``okf-mcp``."""
    parser = argparse.ArgumentParser(
        description=(
            "OKFgraph MCP Server — expose knowledge graph tools "
            "via Model Context Protocol (stdio)"
        ),
        allow_abbrev=False,
    )
    # Boot flags derive from the same settings table as the CLI's globals.
    set_cli_flags(parser, mcp=True)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--connect",
        metavar="URL",
        default=None,
        help="Proxy to a running 'okf serve' (http://HOST:PORT or "
             "unix:///path) instead of opening the database (0.11.0).",
    )
    args = parser.parse_args()

    # stderr only: stdout is the stdio transport.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.connect:
        from okfgraph.client import OKFClient
        client = OKFClient(args.connect)
        try:
            client.health()
        except Exception as e:
            print(f"[ERROR] CONFIG_INVALID: cannot reach okf serve at "
                  f"{args.connect} ({e}); start it with 'okf serve' or fix --connect",
                  file=sys.stderr)
            raise SystemExit(2)
        try:
            settings = Settings.load(cli_args=cli_signal(args))
        except ValueError as e:
            print(f"[ERROR] CONFIG_INVALID: {e}", file=sys.stderr)
            raise SystemExit(2)
        create_mcp_server(settings, router=client).run(transport="stdio")
        return

    try:
        signal = cli_signal(args)
        settings = Settings.load(cli_args=signal, require=MCP_REQUIRED)
        if settings.bundle_root:
            # Second pass: a --bundle-root'd okfgraph.toml is a lookup
            # candidate too (the CWD candidate already had its chance).
            settings = Settings.load(bundle_root=settings.bundle_root,
                                     cli_args=signal, require=MCP_REQUIRED)
    except ValueError as e:
        print(f"[ERROR] CONFIG_INVALID: {e}", file=sys.stderr)
        raise SystemExit(2)

    create_mcp_server(settings).run(transport="stdio")


if __name__ == "__main__":
    main()
