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
                    "enum": ["CONTAINS", "LINKS_TO"],
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
]
