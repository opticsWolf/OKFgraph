"""MCP server for the OKF knowledge graph.

Exposes all OKFgraph tools via the Model Context Protocol so any
MCP-compatible client (Claude Desktop, Cursor, Continue, etc.) can
call search, traverse, ingest, and export operations directly.

Usage:
    # CLI entry point (configured in pyproject.toml)
    okf-mcp --db-path ./my_graph.db

    # Or as a Python module
    python -m okfgraph.mcp_server --db-path ./my_graph.db

    # Or programmatically
    from okfgraph.mcp_server import create_mcp_server
    mcp = create_mcp_server(db_path="./my_graph.db")
    mcp.run()
"""

import argparse
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter
from okfgraph.settings import Settings, cli_signal, set_cli_flags

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Shared context for the MCP server lifespan."""
    router: OKFRouter


def make_lifespan(settings: "Settings"):
    """Factory that returns a lifespan async-context-manager for MCPServer."""

    @asynccontextmanager
    async def _lifespan(mcp: MCPServer):
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
            db_path,
            settings.model_id,
            settings.device,
            settings.precision,
        )

        try:
            yield GraphContext(router=router)
        finally:
            router.close()
            logger.info("OKFgraph MCP server shutdown complete")

    return _lifespan


def _get_router(ctx: Context) -> OKFRouter:
    """Extract the OKFRouter from the MCP context."""
    gc = ctx.request_context.lifespan_context
    if isinstance(gc, GraphContext):
        return gc.router
    # Fallback: if lifespan_context is a dict (default MCP behavior)
    if isinstance(gc, dict):
        return gc["router"]
    raise RuntimeError("No OKFRouter found in lifespan context")


def create_mcp_server(settings: Settings) -> MCPServer:
    """Create an MCP server instance connected to an OKFgraph database.

    Args:
        settings: Merged canonical settings (``Settings.load`` or direct
            construction in tests; ``db_path`` required).

    Returns:
        Configured MCPServer server instance.
    """
    lifespan_fn = make_lifespan(settings)

    mcp = MCPServer(
        name="OKFgraph MCP Server",
        instructions=(
            "OKF knowledge graph with ONNX + Jina v5 embeddings. "
            "Provides semantic search, graph traversal, document ingestion, "
            "and thought persistence capabilities."
        ),
        lifespan=lifespan_fn,
    )

    _RO = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    _WR = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @mcp.tool(annotations=_RO)
    def search(
        query: Annotated[str, Field(description="The search query text.")],
        target: Annotated[
            Literal["concepts", "chunks", "images"],
            Field(
                description=(
                    "'concepts' = RRF hybrid search over concepts (open-ended questions). "
                    "'chunks' = RRF vector+FTS over document chunks (exact passages). "
                    "'images' = text query against image assets."
                ),
            ),
        ] = "concepts",
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum results.")] = 10,
        concept_type: Annotated[Optional[str], Field(description="Concept type filter (concepts/chunks only; plain chunks path).")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Tag filter, ALL must match (concepts/chunks only).")] = None,
        parent_id: Annotated[Optional[str], Field(description="Directory ID to constrain search (concepts/chunks only).")] = None,
        include_chunks: Annotated[bool, Field(description="Concepts only: attach matched chunks per concept result. For chunk-level hits, use target='chunks' instead.")] = False,
        max_chunks_per_doc: Annotated[int, Field(ge=1, le=10, description="Chunks only: cap results per source document. Must stay default on other paths.")] = 3,
        expand: Annotated[bool, Field(description="Chunks only: attach graph neighborhood (incoming/outgoing links, ancestry, siblings) to each hit.")] = False,
        context_hops: Annotated[int, Field(ge=1, le=3, description="Expansion depth when expand=True.")] = 1,
        hub_rerank: Annotated[bool, Field(description="Chunks only: rerank by graph hub score (incoming link count). Wins over expand.")] = False,
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
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Search the knowledge graph — start here for any question over stored knowledge.

        Routing: target='concepts' for open-ended questions; 'chunks' for exact
        passages (expand=true adds graph neighborhood, hub_rerank=true ranks by
        importance); 'images' for image assets. rank='ppr' answers from the
        link graph alone when the embedder is cold or unavailable. If you
        already have a concept ID, use read instead of searching. If you want
        related concepts, use traverse. To add content, use ingest."""
        router = _get_router(ctx)
        try:
            results = router.search(
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
            )
        except OKFError as err:
            return f"error: {err.code}: {err.message}"
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(annotations=_RO)
    def read(
        concept_id: Annotated[str, Field(description="ID of the concept to read.")],
        include: Annotated[
            Literal["body", "chunks", "document", "context"],
            Field(
                description=(
                    "'body' = full markdown of the concept. "
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
        try:
            return json.dumps(
                router.read(concept_id, include=include, max_tokens=max_tokens),
                default=str, indent=2,
            )
        except OKFError as err:
            return f"error: {err.code}: {err.message}"

    @mcp.tool(annotations=_RO)
    def traverse(
        start_id: Annotated[str, Field(description="ID of the starting concept or directory. Empty string lists the root directory.")],
        relationship: Annotated[
            Literal["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"],
            Field(description="Relationship type. CONTAINS depth 1 = directory listing."),
        ] = "CONTAINS",
        direction: Annotated[
            Literal["OUTGOING", "INCOMING", "BOTH"],
            Field(description="Traversal direction."),
        ] = "OUTGOING",
        depth: Annotated[int, Field(ge=1, le=5, description="Maximum traversal depth.")] = 1,
        node_type: Annotated[
            Optional[str],
            Field(description="Filter walk results by node type (e.g. 'Concept', 'Directory'). Walk mode only."),
        ] = None,
        target: Annotated[
            Optional[str],
            Field(description="If set, find the shortest path from start_id to this concept ID instead of traversing (uses max_path_length)."),
        ] = None,
        max_path_length: Annotated[int, Field(ge=1, le=10, description="Maximum path length, only used with target.")] = 6,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Navigate graph relationships, browse directories, or connect two concepts.

        Routing: default CONTAINS depth 1 lists a directory (empty id = root);
        LINKS_TO follows references; target=<id> finds the shortest path
        instead. Search first to find IDs. To add content use ingest."""
        router = _get_router(ctx)
        try:
            return json.dumps(
                router.traverse(
                    start_id,
                    relationship=relationship,
                    direction=direction,
                    depth=depth,
                    node_type=node_type,
                    target=target,
                    max_path_length=max_path_length,
                ),
                default=str, indent=2,
            )
        except OKFError as err:
            return f"error: {err.code}: {err.message}"

    @mcp.tool(annotations=_WR)
    def ingest(
        kind: Annotated[
            Literal["md", "pdf", "thoughts"],
            Field(description="'md' = import a markdown file. 'pdf' = convert a PDF (bobine) and import. 'thoughts' = persist LLM reasoning as a searchable concept."),
        ],
        md_path: Annotated[Optional[str], Field(description="Markdown file to import (kind='md').")] = None,
        pdf_path: Annotated[Optional[str], Field(description="File to convert and import (kind='pdf'): PDF or Office (docx/xlsx/pptx, legacy doc/xls/ppt). Bobine dispatches on extension.")] = None,
        thoughts: Annotated[Optional[str], Field(description="Raw reasoning text (kind='thoughts').")] = None,
        topic: Annotated[Optional[str], Field(description="Topic for kind='thoughts' (required then).")] = None,
        concept_id: Annotated[Optional[str], Field(description="Explicit concept ID (md/thoughts). Generated if omitted (thoughts: thoughts/<topic>/<ts>_<id>, a virtual namespace — no file needed, exports there later). Slashes are namespaces.")] = None,
        title: Annotated[Optional[str], Field(description="Title override (md only).")] = None,
        description: Annotated[Optional[str], Field(description="Description override (md only).")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Tags to apply (md/thoughts).")] = None,
        mode: Annotated[Literal["text", "optional", "omni"], Field(description="Image ingestion mode: text (captions), optional/omni (ONNX vision content, needs a text-nano graph).", json_schema_extra={"enum": ["text", "optional", "omni"]})] = "text",
        routing_mode: Annotated[
            Literal["auto", "surgical", "always", "never"],
            Field(description="PDF converter routing (kind='pdf'). 'never' = fast path, no ONNX."),
        ] = "auto",
        extract_images: Annotated[bool, Field(description="Extract embedded images (kind='pdf').")] = True,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Add content to the knowledge graph. Returns concept ID(s) for search/traverse.

        Routing: kind='md' imports a markdown file; 'pdf' converts a PDF
        ('never' routing = fast, no ONNX); 'thoughts' persists reasoning.
        Markdown is mordant-linted before import."""
        router = _get_router(ctx)
        try:
            result = router.ingest(
                kind,
                md_path=md_path,
                pdf_path=pdf_path,
                thoughts=thoughts,
                topic=topic,
                concept_id=concept_id,
                title=title,
                description=description,
                tags=tags,
                mode=mode,
                routing_mode=routing_mode,
                extract_images=extract_images,
                auto_import=True,
            )
        except OKFError as err:
            return f"error: {err.code}: {err.message}"
        return json.dumps(result, default=str, indent=2)

    @mcp.tool(annotations=_WR)
    def export_bundle(
        output_dir: Annotated[str, Field(description="Output directory for the bundle.")],
        directory_id: Annotated[Optional[str], Field(description="Only export concepts under this directory.")] = None,
        concept_type: Annotated[Optional[str], Field(description="Only export concepts of this type.")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Only export concepts with ALL these tags.")] = None,
        flavor: Annotated[
            Literal["okf", "obsidian"],
            Field(description="'okf' = [t](id.md) links + index files. 'obsidian' = [[Title]] wikilinks, no index files, re-imports losslessly."),
        ] = "okf",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Export concepts from the graph to an OKF-compliant bundle directory. To add content back to the graph, use ingest."""
        router = _get_router(ctx)
        try:
            result = router.export_bundle(
                output_dir,
                directory_id=directory_id,
                concept_type=concept_type,
                tags=tags,
                flavor=flavor,
            )
        except OKFError as err:
            return f"error: {err.code}: {err.message}"
        return json.dumps(result, default=str, indent=2)

    @mcp.tool(annotations=_WR)
    def export_concept(
        concept_id: Annotated[str, Field(description="ID of the concept to export.")],
        output_dir: Annotated[str, Field(description="Output directory; writes <output_dir>/<concept_id>.md.")],
        flavor: Annotated[
            Literal["okf", "obsidian"],
            Field(description="'okf' = [t](id.md) links. 'obsidian' = [[Title]] wikilinks."),
        ] = "okf",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Export a single concept to a .md file. Returns {concept_id, path}.

        Unknown IDs raise UNKNOWN_CONCEPT (search first). For a whole
        bundle use export_bundle."""
        router = _get_router(ctx)
        try:
            result = router.export_concept(concept_id, output_dir=output_dir, flavor=flavor)
        except OKFError as err:
            return f"error: {err.code}: {err.message}"
        return json.dumps(result, default=str, indent=2)


    @mcp.tool(annotations=_RO)
    def list_images(
        concept_id: Annotated[str, Field(description="ID of the concept whose images to list.")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """List image assets attached to a concept (metadata, no bytes).

        Unknown concepts raise UNKNOWN_CONCEPT — search first."""
        router = _get_router(ctx)
        try:
            return json.dumps(router.list_images(concept_id), default=str, indent=2)
        except OKFError as err:
            return f"error: {err.code}: {err.message}"

    @mcp.tool(annotations=_RO)
    def get_image(
        asset_id: Annotated[str, Field(description="ID of the image asset (from list_images).")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Fetch one image asset: metadata plus base64-encoded ``data``.

        Unknown assets raise UNKNOWN_ASSET."""
        router = _get_router(ctx)
        try:
            return json.dumps(router.get_image(asset_id), default=str, indent=2)
        except OKFError as err:
            return f"error: {err.code}: {err.message}"

    return mcp


def main():
    """CLI entry point for the MCP server."""
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "OKFgraph MCP Server — expose knowledge graph tools "
            "via Model Context Protocol"
        ),
    )
    # Boot flags derive from the same settings table as the CLI's globals;
    # db_path is required here. The server reads okfgraph.toml and
    # OKFGRAPH_* exactly like the CLI does.
    set_cli_flags(parser, mcp=True)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        signal = cli_signal(args)
        settings = Settings.load(cli_args=signal)
        if settings.bundle_root:
            # Second pass: a --bundle-root'd okfgraph.toml is a lookup
            # candidate too (the CWD candidate already had its chance).
            settings = Settings.load(
                bundle_root=settings.bundle_root, cli_args=signal)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(2)

    logger.info(
        "starting OKFgraph MCP server: db=%s model=%s device=%s precision=%s",
        settings.db_path,
        settings.model_id,
        settings.device,
        settings.precision,
    )

    mcp = create_mcp_server(settings)

    # Run with stdio transport (default for MCP servers)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
