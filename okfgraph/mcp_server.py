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

from okfgraph.models import ConceptModel
from okfgraph.router import OKFRouter

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Shared context for the MCP server lifespan."""
    router: OKFRouter


def make_lifespan(
    db_path: str,
    bundle_root: Optional[str],
    device: str,
    embedding_dim: int,
    enable_chunking: bool,
    max_length: Optional[int] = None,
    roots: Optional[Dict[str, str]] = None,
):
    """Factory that returns a lifespan async-context-manager for MCPServer."""

    @asynccontextmanager
    async def _lifespan(mcp: MCPServer):
        root = bundle_root if bundle_root is not None else str(Path(db_path).parent)

        router = OKFRouter(
            db_path=db_path,
            bundle_root=root,
            device=device,
            embedding_dim=embedding_dim,
            max_length=max_length,
            enable_chunking=enable_chunking,
            roots=roots,
        )
        logger.info(
            "OKFgraph MCP server started: db=%s device=%s",
            db_path,
            device,
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


def create_mcp_server(
    db_path: str,
    bundle_root: Optional[str] = None,
    device: str = "cpu",
    embedding_dim: int = 512,
    enable_chunking: bool = True,
    max_length: Optional[int] = None,
    roots: Optional[Dict[str, str]] = None,
) -> MCPServer:
    """Create an MCP server instance connected to an OKFgraph database.

    Args:
        db_path: Path to the Ladybug database file.
        bundle_root: Optional root directory for the OKF bundle.
        device: Device for ONNX inference ("cpu" or "cuda").
        embedding_dim: Dimension of the embedding vectors.
        enable_chunking: Whether to enable document chunking.
        max_length: Token truncation ceiling 1..=32768 (default: 8192).
        roots: Optional additional {alias: path} bundle roots (§2.6).

    Returns:
        Configured MCPServer server instance.
    """
    lifespan_fn = make_lifespan(
        db_path=db_path,
        bundle_root=bundle_root,
        device=device,
        embedding_dim=embedding_dim,
        enable_chunking=enable_chunking,
        max_length=max_length,
        roots=roots,
    )

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
        type_filter: Annotated[Optional[str], Field(description="Concept type filter (concepts/chunks only).")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Tag filter, ALL must match (concepts/chunks only).")] = None,
        parent_id: Annotated[Optional[str], Field(description="Directory ID to constrain search (concepts/chunks only).")] = None,
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
        if target == "images":
            return json.dumps(
                router.image_mgr.search_images_with_text(text_query=query, limit=limit),
                default=str, indent=2,
            )
        filt: dict = {}
        if type_filter is not None:
            filt["concept_type"] = type_filter
        if tags is not None:
            filt["tags"] = tags
        if parent_id is not None:
            filt["parent_id"] = parent_id
        if target == "chunks":
            if rank != "none":
                return "error: rank is concepts-only; use hub_rerank/expand for chunks"
            if hub_rerank:
                results = router.search_engine.search_chunks_with_hub_score(
                    query, limit=limit, hub_weight=hub_weight
                )
            elif expand:
                results = router.search_engine.search_with_context(
                    query, limit=min(limit, 20), context_hops=context_hops
                )
            else:
                results = router.search_engine.search_chunks(query, limit=limit, **filt)
            return json.dumps(results, default=str, indent=2)
        results = router.search_hybrid(query, limit=limit, rank=rank, hub_weight=hub_weight, **filt)
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
        if max_tokens is not None:
            try:
                reading = router.search_engine.read_with_budget(
                    concept_id, include=include, max_tokens=max_tokens,
                )
            except KeyError:
                return f"Concept not found: {concept_id}"
            return json.dumps(reading, default=str, indent=2)
        if include == "chunks":
            return json.dumps(router.search_engine.get_chunks(concept_id), default=str, indent=2)
        if include == "document":
            return json.dumps(router.embed_engine.reconstruct_document(concept_id), default=str, indent=2)
        if include == "context":
            return json.dumps({
                "incoming_links": router.traverse(concept_id, "LINKS_TO", "INCOMING", 1)[:10],
                "outgoing_links": router.traverse(concept_id, "LINKS_TO", "OUTGOING", 1)[:10],
                "ancestry": router.search_engine._get_ancestry(concept_id),
                "siblings": router.search_engine._get_siblings(concept_id)[:10],
            }, default=str, indent=2)
        concept = router.get_by_id(concept_id)
        if concept is None:
            return f"Concept not found: {concept_id}"
        # Coercion boundary: validate whatever arrives (model or plain
        # mapping) so the serialize surface below is always a ConceptModel.
        concept = ConceptModel.model_validate(concept)
        return json.dumps(concept.public_dict(), default=str, indent=2)

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
        if not start_id:
            return json.dumps(router.list_directory(""), default=str, indent=2)
        if target is not None:
            return json.dumps(
                router.search_engine.find_path(start_id, target, max_length=max_path_length),
                default=str, indent=2,
            )
        return json.dumps(router.traverse(start_id, relationship, direction, depth), default=str, indent=2)

    @mcp.tool(annotations=_WR)
    def ingest(
        kind: Annotated[
            Literal["md", "pdf", "thoughts"],
            Field(description="'md' = import a markdown file. 'pdf' = convert a PDF (bobine) and import. 'thoughts' = persist LLM reasoning as a searchable concept."),
        ],
        md_path: Annotated[Optional[str], Field(description="Markdown file to import (kind='md').")] = None,
        pdf_path: Annotated[Optional[str], Field(description="PDF file to convert and import (kind='pdf').")] = None,
        thoughts: Annotated[Optional[str], Field(description="Raw reasoning text (kind='thoughts').")] = None,
        topic: Annotated[Optional[str], Field(description="Topic for kind='thoughts' (required then).")] = None,
        concept_id: Annotated[Optional[str], Field(description="Explicit concept ID (md/thoughts). Generated if omitted.")] = None,
        title: Annotated[Optional[str], Field(description="Title override (md only).")] = None,
        description: Annotated[Optional[str], Field(description="Description override (md only).")] = None,
        tags: Annotated[Optional[list[str]], Field(description="Tags to apply (md/thoughts).")] = None,
        mode: Annotated[Literal["text", "optional", "omni"], Field(description="Image ingestion mode.")] = "text",
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
        from okfgraph.components.converters import BobineConverter
        router = _get_router(ctx)
        if kind == "md":
            if not md_path:
                return "error: md_path is required for kind='md'"
            kwargs: dict = {"md_path": md_path, "mode": mode}
            if concept_id is not None:
                kwargs["concept_id"] = concept_id
            if title is not None:
                kwargs["title"] = title
            if description is not None:
                kwargs["description"] = description
            if tags is not None:
                kwargs["tags"] = tags
            result = router.ingest_mgr.ingest_md(**kwargs)
        elif kind == "pdf":
            if not pdf_path:
                return "error: pdf_path is required for kind='pdf'"
            result = router.ingest_mgr.ingest_pdf(
                pdf_path=pdf_path, auto_import=True, mode=mode,
                converter=BobineConverter(routing_mode=routing_mode, extract_images=extract_images),
            )
        elif kind == "thoughts":
            if not thoughts or not topic:
                return "error: thoughts and topic are required for kind='thoughts'"
            result = router.ingest_mgr.ingest_thoughts(
                thoughts, topic=topic, concept_id=concept_id, tags=tags,
            )
        else:  # pragma: no cover - Literal constrains this
            return f"error: unknown kind '{kind}'"
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
        kwargs: dict = {"output_dir": output_dir, "flavor": flavor}
        if directory_id is not None:
            kwargs["directory_id"] = directory_id
        if concept_type is not None:
            kwargs["concept_type"] = concept_type
        if tags is not None:
            kwargs["tags"] = tags
        result = router.export_mgr.export_bundle(**kwargs)
        return json.dumps(result, default=str, indent=2)


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
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the Ladybug database file.",
    )
    parser.add_argument(
        "--bundle-root",
        type=str,
        default=None,
        help="Root directory for the OKF bundle (defaults to db parent).",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        metavar="ALIAS=PATH",
        help="Additional named bundle root (repeatable; named roots mint "
        "@alias/rel IDs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for ONNX inference (default: cpu).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Dimension of the embedding vectors (default: 512; Matryoshka ladder: 32, 64, 128, 256, 512, 768, 1024).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Token truncation ceiling 1..=32768 (default: 8192).",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Disable document chunking.",
    )
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

    logger.info(
        "starting OKFgraph MCP server: db=%s device=%s",
        args.db_path,
        args.device,
    )

    roots: Optional[Dict[str, str]] = None
    if args.root:
        roots = {}
        for item in args.root:
            alias, sep, path = str(item).partition("=")
            if not sep or not alias or not path:
                parser.error(f"--root must be ALIAS=PATH, got {item!r}")
            roots[alias] = path

    mcp = create_mcp_server(
        db_path=args.db_path,
        bundle_root=args.bundle_root,
        device=args.device,
        embedding_dim=args.embedding_dim,
        enable_chunking=not args.no_chunking,
        max_length=args.max_length,
        roots=roots,
    )

    # Run with stdio transport (default for MCP servers)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
