"""LLM tool definitions for the OKF knowledge graph."""

TOOLS = [
    {
        "name": "search_hybrid",
        "description": "Semantic + keyword search over concepts. Use for open-ended questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "type": {
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
        "description": "Navigate relationships (CONTAINS or LINKS_TO) from a concept.",
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
                    "description": "Relationship type to traverse.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["OUTGOING", "INCOMING", "BOTH"],
                    "description": "Traversal direction.",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
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
]
