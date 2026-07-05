"""Pydantic models for OKF concepts."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ConceptModel(BaseModel):
    """OKF concept with support for arbitrary frontmatter keys."""

    id: str
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    body: str = ""
    embedding: Optional[List[float]] = None  # internal use only

    # Capture arbitrary OKF frontmatter keys
    model_config = {"extra": "allow"}

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> Any:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class ChunkModel(BaseModel):
    """Represents a chunked section of a document."""

    id: str
    parent_doc_id: str
    chunk_index: int
    chunk_text: str
    block_type: str
    start_offset: int = 0
    end_offset: int = 0
    rrf_score: Optional[float] = None
    hub_score: Optional[float] = None
    final_score: Optional[float] = None
    parent_title: Optional[str] = None
    parent_type: Optional[str] = None
    parent_tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None  # internal use only


class ImageAssetModel(BaseModel):
    """Metadata for an image asset stored in the unified vector index.

    The raw bytes (``data``) and ``embedding`` are kept out of the default
    serialisation surface; this model is mainly for typed return values and
    listing, not for shuttling BLOBs around.
    """

    id: str
    file_name: str = ""
    mime_type: str = "application/octet-stream"
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    embed_route: Optional[str] = None  # "text" | "omni"
    content_hash: Optional[str] = None
    embedding: Optional[List[float]] = None  # internal use only

    model_config = {"extra": "allow"}

