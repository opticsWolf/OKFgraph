"""Public Pydantic models for the Rust-backed facade. No graph logic."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "normalize_tags",
    "GRAPH_TIMES",
    "ConceptModel",
    "ChunkModel",
    "ImageAssetModel",
]


def normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    return [str(value)]


GRAPH_TIMES = ("created_at", "updated_at", "effective_at", "content_sig")


class ConceptModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    body: str = ""
    embedding: Optional[List[float]] = None

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> List[str]:
        return normalize_tags(v)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> Any:
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    def public_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude={"embedding"})

    def export_frontmatter(self) -> Tuple[Dict[str, Any], str]:
        data = self.model_dump()
        body = data.pop("body", "")
        data.pop("id", None)
        data.pop("embedding", None)
        data.pop("source_path", None)
        for key in GRAPH_TIMES:
            data.pop(key, None)
        if "uid" in data:
            data["id"] = data.pop("uid")
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data, body


class ChunkModel(BaseModel):
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
    embedding: Optional[List[float]] = None

    @field_validator("parent_tags", mode="before")
    @classmethod
    def coerce_parent_tags(cls, v: Any) -> List[str]:
        return normalize_tags(v)


class ImageAssetModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    file_name: str = ""
    mime_type: str = "application/octet-stream"
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    embed_route: Optional[str] = None
    content_hash: Optional[str] = None
    embedding: Optional[List[float]] = None
