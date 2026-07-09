"""LLM tool definitions for the OKF knowledge graph."""

TOOLS = [
    {
        "name": "search_hybrid",
        "description": "Semantic + keyword search over concepts. Use for open-ended questions. To add new content to the graph, use ``ingest_md``, ``ingest_thoughts``, or ``ingest_pdf`` instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "concept_type": {
                    "type": "string",
                    "description": "Optional concept type filter.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filters.",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Optional directory ID to constrain search.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "traverse",
        "description": "Navigate relationships (CONTAINS or LINKS_TO) from a concept. To expand the graph, use ``ingest_md`` to add documents, ``ingest_thoughts`` to persist reasoning, or ``ingest_pdf`` to convert PDFs.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_id": {
                    "type": "string",
                    "description": "ID of the starting concept or directory.",
                },
                "relationship": {
                    "type": "string",
                    "enum": ["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"],
                    "default": "CONTAINS",
                    "description": "Relationship type to traverse.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["OUTGOING", "INCOMING", "BOTH"],
                    "default": "OUTGOING",
                    "description": "Traversal direction.",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1,
                    "description": "Maximum traversal depth.",
                },
            },
            "required": ["start_id"],
        },
    },
    {
        "name": "get_by_id",
        "description": "Fetch the full markdown body of a specific concept.",
        "parameters": {
            "type": "object",
            "properties": {
                "concept_id": {
                    "type": "string",
                    "description": "ID of the concept to fetch.",
                },
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "list_directory",
        "description": "List contents of a directory for progressive disclosure.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_id": {
                    "type": "string",
                    "description": "Directory ID (e.g., 'tables'). Empty string for root.",
                },
            },
            "required": ["directory_id"],
        },
    },
    {
        "name": "search_images",
        "description": (
            "Find image assets by a text description via the unified vector "
            "index. Works whether images were embedded from alt-text or by the "
            "multimodal model, since both share one vector space."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text describing the image(s) to find.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of images to return (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_chunks",
        "description": (
            "Search document chunks using RRF-fused vector + FTS. Returns "
            "chunk-level results with parent document metadata. Use for "
            "fine-grained retrieval within specific sections of documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of chunks to return (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_with_context",
        "description": (
            "Search chunks + expand results with graph neighborhood context. "
            "Returns chunks enriched with incoming/outgoing links, "
            "directory ancestry, and sibling concepts. Use when you need "
            "relational context, not just the matched text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of results.",
                },
                "context_hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 1,
                    "description": "How many hops to expand for context.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_chunks_with_hub_score",
        "description": (
            "Search chunks and rerank by graph hub score (incoming link count). "
            "Favours chunks from authoritative documents that many other "
            "concepts link to. Use when authority matters more than raw relevance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of results.",
                },
                "hub_weight": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.3,
                    "description": "Weight for hub score in final ranking (0-1, default 0.3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "expand_with_graph_context",
        "description": (
            "Expand chunk search results with graph-context neighbours. "
            "For each seed chunk, discovers related concepts via LINKS_TO "
            "and reranks by hub score (incoming link count)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of chunk IDs to expand from.",
                },
                "hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Number of graph hops (default 1).",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum neighbours to return (default 50).",
                },
            },
            "required": ["chunk_ids"],
        },
    },
    {
        "name": "get_chunks",
        "description": "Get all chunks for a concept, ordered by chunk index.",
        "parameters": {
            "type": "object",
            "properties": {
                "concept_id": {
                    "type": "string",
                    "description": "ID of the concept whose chunks to fetch.",
                },
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "reconstruct_document",
        "description": (
            "Reconstruct the original markdown document from its stored chunks. "
            "Uses block_type to determine correct delimiters. ~98% fidelity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "concept_id": {
                    "type": "string",
                    "description": "ID of the concept to reconstruct.",
                },
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "find_path",
        "description": (
            "Find the shortest path between two concepts in the knowledge "
            "graph. Returns the sequence of nodes connecting them. Use to "
            "understand how two topics are related."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_id": {
                    "type": "string",
                    "description": "ID of the starting concept.",
                },
                "end_id": {
                    "type": "string",
                    "description": "ID of the ending concept.",
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 6,
                    "description": "Maximum path length to search (default 6).",
                },
            },
            "required": ["start_id", "end_id"],
        },
    },
    {
        "name": "export_bundle",
        "description": (
            "Export concepts from the graph to an OKF-compliant bundle directory. "
            "Each concept is written as a markdown file with YAML frontmatter. "
            "The body is enriched with graph-derived LINKS_TO links (See Also + Cited By). "
            "index.md files are generated for every directory. "
            "To add content back to the graph, use ``ingest_md``."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Output directory for the bundle.",
                },
                "directory_id": {
                    "type": "string",
                    "description": "Optional: only export concepts under this directory.",
                },
                "concept_type": {
                    "type": "string",
                    "description": "Optional: only export concepts of this type.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: only export concepts with ALL these tags.",
                },
            },
            "required": ["output_dir"],
        },
    },
    {
        "name": "ingest_md",
        "description": (
            "Import a single markdown file into the knowledge graph. "
            "Use when a user provides a document path or asks to add content from a file. "
            "The file is linted with mordant before import — fixable formatting "
            "issues (MD009, MD012, MD047) are auto-corrected. "
            "Returns the concept ID so the content can be searched or traversed. "
            "For storing LLM reasoning, use ``ingest_thoughts`` instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "md_path": {
                    "type": "string",
                    "description": "Path to the markdown file to import.",
                },
                "concept_id": {
                    "type": "string",
                    "description": (
                        "Optional explicit concept ID. If not provided, "
                        "generated from filename."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional title override.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description override.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags to apply.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "optional", "omni"],
                    "default": "text",
                    "description": "Image ingestion mode.",
                },
            },
            "required": ["md_path"],
        },
    },
    {
        "name": "ingest_thoughts",
        "description": (
            "Store LLM reasoning or thinking as a searchable concept in the "
            "knowledge graph. Use when the agent wants to persist its reasoning "
            "process, decision logs, or intermediate conclusions. The stored "
            "thoughts can later be searched, traversed, and used as context "
            "for other queries. Wraps the text in OKF-compliant markdown with "
            "metadata (type=thought, thought_type=reasoning) so it can be "
            "filtered and searched. For importing existing files, use ``ingest_md`` instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "string",
                    "description": "The raw reasoning text from the LLM.",
                },
                "topic": {
                    "type": "string",
                    "description": "High-level topic or domain for the reasoning.",
                },
                "concept_id": {
                    "type": "string",
                    "description": (
                        "Optional explicit concept ID. If not provided, "
                        "generated from topic + timestamp."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional additional tags.",
                },
            },
            "required": ["thoughts", "topic"],
        },
    },
    {
        "name": "ingest_pdf",
        "description": (
            "Convert a PDF to markdown and import into the knowledge graph. "
            "Uses the HybridConverter pipeline (pdf_oxide fast path + ONNX/Rapid "
            "heavy passes). The resulting markdown is linted with mordant before "
            "import — fixable formatting issues are auto-corrected. "
            "Returns the concept ID(s) so the content can be searched or traversed. "
            "For importing existing markdown files, use ``ingest_md`` instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the PDF file to convert and import.",
                },
                "routing_mode": {
                    "type": "string",
                    "enum": ["auto", "surgical", "always", "never"],
                    "default": "auto",
                    "description": (
                        "ONNX routing mode. 'auto' = use ONNX only when needed, "
                        "'surgical' = target specific elements, 'always' = force ONNX, "
                        "'never' = pdf_oxide fast path only."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "optional", "omni"],
                    "default": "text",
                    "description": (
                        "Image ingestion mode for the converted content. "
                        "'text' = embed alt-text only, 'optional' = rich for missing alt-text, "
                        "'omni' = rich for all images."
                    ),
                },
                "extract_images": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to extract embedded images from the PDF.",
                },
            },
            "required": ["pdf_path"],
        },
    },
]
