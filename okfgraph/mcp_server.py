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
from typing import Annotated, Any, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

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
):
    """Factory that returns a lifespan async-context-manager for FastMCP."""

    @asynccontextmanager
    async def _lifespan(mcp: FastMCP):
        if bundle_root is None:
            bundle_root = str(Path(db_path).parent)

        router = OKFRouter(
            db_path=db_path,
            bundle_root=bundle_root,
            device=device,
            embedding_dim=embedding_dim,
            enable_chunking=enable_chunking,
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
    embedding_dim: int = 1024,
    enable_chunking: bool = True,
) -> FastMCP:
    """Create an MCP server instance connected to an OKFgraph database.

    Args:
        db_path: Path to the Ladybug database file.
        bundle_root: Optional root directory for the OKF bundle.
        device: Device for ONNX inference ("cpu" or "cuda").
        embedding_dim: Dimension of the embedding vectors.
        enable_chunking: Whether to enable document chunking.

    Returns:
        Configured FastMCP server instance.
    """
    lifespan_fn = make_lifespan(
        db_path=db_path,
        bundle_root=bundle_root,
        device=device,
        embedding_dim=embedding_dim,
        enable_chunking=enable_chunking,
    )

    mcp = FastMCP(
        name="OKFgraph MCP Server",
        instructions=(
            "OKF knowledge graph with ONNX + Jina v5 embeddings. "
            "Provides semantic search, graph traversal, document ingestion, "
            "and thought persistence capabilities."
        ),
        lifespan=lifespan_fn,
    )

    # ------------------------------------------------------------------
    # Read Tools
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def search_hybrid(
        query: Annotated[str, Field(description="The search query text.")],
        type_filter: Annotated[
            Optional[str],
            Field(description="Optional concept type filter."),
        ] = None,
        tags: Annotated[
            Optional[list[str]],
            Field(description="Optional tag filters."),
        ] = None,
        parent_id: Annotated[
            Optional[str],
            Field(description="Optional directory ID to constrain search."),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Semantic + keyword search over concepts. Use for open-ended questions. To add new content to the graph, use ingest_md or ingest_thoughts instead."""
        router = _get_router(ctx)
        kwargs: dict = {"query": query}
        if type_filter is not None:
            kwargs["type"] = type_filter
        if tags is not None:
            kwargs["tags"] = tags
        if parent_id is not None:
            kwargs["parent_id"] = parent_id
        results = router.search_hybrid(**kwargs)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def traverse(
        start_id: Annotated[str, Field(description="ID of the starting concept or directory.")],
        relationship: Annotated[
            Literal["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"],
            Field(description="Relationship type to traverse."),
        ] = "CONTAINS",
        direction: Annotated[
            Literal["OUTGOING", "INCOMING", "BOTH"],
            Field(description="Traversal direction."),
        ] = "OUTGOING",
        depth: Annotated[
            int,
            Field(ge=1, le=5, description="Maximum traversal depth."),
        ] = 1,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Navigate relationships (CONTAINS or LINKS_TO) from a concept. To expand the graph, use ingest_md to add documents or ingest_thoughts to persist reasoning."""
        router = _get_router(ctx)
        results = router.traverse(start_id, relationship, direction, depth)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def get_by_id(
        concept_id: Annotated[str, Field(description="ID of the concept to fetch.")],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Fetch the full markdown body of a specific concept."""
        router = _get_router(ctx)
        concept = router.get_by_id(concept_id)
        if concept is None:
            return f"Concept not found: {concept_id}"
        return json.dumps(
            concept.model_dump() if hasattr(concept, "model_dump") else dict(concept),
            default=str,
            indent=2,
        )

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def list_directory(
        directory_id: Annotated[
            str,
            Field(description="Directory ID (e.g., 'tables'). Empty string for root."),
        ] = "",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """List contents of a directory for progressive disclosure."""
        router = _get_router(ctx)
        results = router.list_directory(directory_id)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def search_images(
        query: Annotated[str, Field(description="Text describing the image(s) to find.")],
        limit: Annotated[
            int,
            Field(ge=1, le=50, description="Maximum number of images to return."),
        ] = 10,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Find image assets by a text description via the unified vector index."""
        router = _get_router(ctx)
        results = router.search_images(query, limit=limit)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def search_chunks(
        query: Annotated[str, Field(description="The search query text.")],
        limit: Annotated[
            int,
            Field(ge=1, le=50, description="Maximum number of chunks to return."),
        ] = 10,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Search document chunks using RRF-fused vector + FTS. Returns chunk-level results with parent document metadata."""
        router = _get_router(ctx)
        results = router.search_chunks(query, limit=limit)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def search_with_context(
        query: Annotated[str, Field(description="The search query text.")],
        limit: Annotated[
            int,
            Field(ge=1, le=20, description="Maximum number of results."),
        ] = 5,
        context_hops: Annotated[
            int,
            Field(ge=1, le=3, description="How many hops to expand for context."),
        ] = 1,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Search chunks + expand results with graph neighborhood context."""
        router = _get_router(ctx)
        results = router.search_with_context(
            query, limit=limit, context_hops=context_hops
        )
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def search_chunks_with_hub_score(
        query: Annotated[str, Field(description="The search query text.")],
        limit: Annotated[
            int,
            Field(ge=1, le=50, description="Maximum number of results."),
        ] = 10,
        hub_weight: Annotated[
            float,
            Field(ge=0, le=1, description="Weight for hub score in final ranking."),
        ] = 0.3,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Search chunks and rerank by graph hub score (incoming link count)."""
        router = _get_router(ctx)
        results = router.search_chunks_with_hub_score(
            query, limit=limit, hub_weight=hub_weight
        )
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def expand_with_graph_context(
        chunk_ids: Annotated[
            list[str],
            Field(description="List of chunk IDs to expand from."),
        ],
        hops: Annotated[
            int,
            Field(ge=1, le=3, description="Number of graph hops."),
        ] = 1,
        max_results: Annotated[
            int,
            Field(ge=1, le=100, description="Maximum neighbours to return."),
        ] = 50,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Expand chunk search results with graph-context neighbours."""
        router = _get_router(ctx)
        results = router.expand_with_graph_context(
            chunk_ids, hops=hops, max_results=max_results
        )
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def get_chunks(
        concept_id: Annotated[
            str, Field(description="ID of the concept whose chunks to fetch.")
        ],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Get all chunks for a concept, ordered by chunk index."""
        router = _get_router(ctx)
        results = router.get_chunks(concept_id)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def reconstruct_document(
        concept_id: Annotated[
            str, Field(description="ID of the concept to reconstruct.")
        ],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Reconstruct the original markdown document from its stored chunks."""
        router = _get_router(ctx)
        result = router.reconstruct_document(concept_id)
        return json.dumps(result, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def find_path(
        start_id: Annotated[str, Field(description="ID of the starting concept.")],
        end_id: Annotated[str, Field(description="ID of the ending concept.")],
        max_length: Annotated[
            int,
            Field(ge=1, le=10, description="Maximum path length to search."),
        ] = 6,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Find the shortest path between two concepts in the knowledge graph."""
        router = _get_router(ctx)
        results = router.find_path(start_id, end_id, max_length=max_length)
        return json.dumps(results, default=str, indent=2)

    # ------------------------------------------------------------------
    # Write Tools
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations={
            "read_only_hint": False,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def export_bundle(
        output_dir: Annotated[
            str, Field(description="Output directory for the bundle.")
        ],
        directory_id: Annotated[
            Optional[str],
            Field(description="Optional: only export concepts under this directory."),
        ] = None,
        concept_type: Annotated[
            Optional[str],
            Field(description="Optional: only export concepts of this type."),
        ] = None,
        tags: Annotated[
            Optional[list[str]],
            Field(description="Optional: only export concepts with ALL these tags."),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Export concepts from the graph to an OKF-compliant bundle directory. To add content back to the graph, use ingest_md."""
        router = _get_router(ctx)
        kwargs: dict = {"output_dir": output_dir}
        if directory_id is not None:
            kwargs["directory_id"] = directory_id
        if concept_type is not None:
            kwargs["concept_type"] = concept_type
        if tags is not None:
            kwargs["tags"] = tags
        result = router.export_bundle(**kwargs)
        return json.dumps(result, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": False,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def ingest_md(
        md_path: Annotated[
            str, Field(description="Path to the markdown file to import.")
        ],
        concept_id: Annotated[
            Optional[str],
            Field(
                description=(
                    "Optional explicit concept ID. "
                    "If not provided, generated from filename."
                )
            ),
        ] = None,
        title: Annotated[
            Optional[str],
            Field(description="Optional title override."),
        ] = None,
        description: Annotated[
            Optional[str],
            Field(description="Optional description override."),
        ] = None,
        tags: Annotated[
            Optional[list[str]],
            Field(description="Optional tags to apply."),
        ] = None,
        mode: Annotated[
            Literal["text", "optional", "omni"],
            Field(description="Image ingestion mode."),
        ] = "text",
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Import a single markdown file into the knowledge graph. The file is linted with mordant before import — fixable formatting issues (MD009, MD012, MD047) are auto-corrected. Returns the concept ID so the content can be searched or traversed. For storing LLM reasoning, use ingest_thoughts instead."""
        router = _get_router(ctx)
        kwargs: dict = {"md_path": md_path, "mode": mode}
        if concept_id is not None:
            kwargs["concept_id"] = concept_id
        if title is not None:
            kwargs["title"] = title
        if description is not None:
            kwargs["description"] = description
        if tags is not None:
            kwargs["tags"] = tags
        result = router.ingest_md(**kwargs)
        return json.dumps(result, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": False,
            "destructive_hint": False,
            "idempotent_hint": False,
            "open_world_hint": False,
        },
    )
    def ingest_thoughts(
        thoughts: Annotated[
            str, Field(description="The raw reasoning text from the LLM.")
        ],
        topic: Annotated[
            str,
            Field(description="High-level topic or domain for the reasoning."),
        ],
        concept_id: Annotated[
            Optional[str],
            Field(
                description=(
                    "Optional explicit concept ID. "
                    "If not provided, generated from topic + timestamp."
                )
            ),
        ] = None,
        tags: Annotated[
            Optional[list[str]],
            Field(description="Optional additional tags."),
        ] = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Store LLM reasoning or thinking as a searchable concept in the knowledge graph. Wraps the text in OKF-compliant markdown with metadata (type=thought, thought_type=reasoning) so it can be filtered and searched. For importing existing files, use ingest_md instead."""
        router = _get_router(ctx)
        kwargs: dict = {"thoughts": thoughts, "topic": topic}
        if concept_id is not None:
            kwargs["concept_id"] = concept_id
        if tags is not None:
            kwargs["tags"] = tags
        result = router.ingest_thoughts(**kwargs)
        return json.dumps(result, default=str, indent=2)

    @mcp.tool(
        annotations={
            "read_only_hint": False,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    def ingest_pdf(
        pdf_path: Annotated[
            str, Field(description="Path to the PDF file to convert and import.")
        ],
        routing_mode: Annotated[
            Literal["auto", "surgical", "always", "never"],
            Field(
                default="auto",
                description=(
                    "ONNX routing mode. 'auto' = use ONNX only when needed, "
                    "'surgical' = target specific elements, 'always' = force ONNX, "
                    "'never' = pdf_oxide fast path only."
                ),
            ),
        ] = "auto",
        mode: Annotated[
            Literal["text", "optional", "omni"],
            Field(
                default="text",
                description=(
                    "Image ingestion mode for the converted content. "
                    "'text' = embed alt-text only, 'optional' = rich for missing alt-text, "
                    "'omni' = rich for all images."
                ),
            ),
        ] = "text",
        extract_images: Annotated[
            bool,
            Field(default=True, description="Whether to extract embedded images from the PDF."),
        ] = True,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> str:
        """Convert a PDF to markdown and import into the knowledge graph. Uses the HybridConverter pipeline (pdf_oxide fast path + ONNX/Rapid heavy passes). The resulting markdown is linted with mordant before import — fixable formatting issues are auto-corrected. Returns the concept ID(s) so the content can be searched or traversed. For importing existing markdown files, use ingest_md instead."""
        router = _get_router(ctx)
        result = router.ingest_pdf(
            pdf_path=pdf_path,
            auto_import=True,
            routing_mode=routing_mode,
            mode=mode,
            extract_images=extract_images,
        )
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for ONNX inference (default: cpu).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1024,
        help="Dimension of the embedding vectors (default: 1024).",
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

    mcp = create_mcp_server(
        db_path=args.db_path,
        bundle_root=args.bundle_root,
        device=args.device,
        embedding_dim=args.embedding_dim,
        enable_chunking=not args.no_chunking,
    )

    # Run with stdio transport (default for MCP servers)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
