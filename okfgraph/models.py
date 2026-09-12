"""Pydantic models for OKF concepts."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_tags(value: Any) -> List[str]:
    """Coerce frontmatter ``tags`` into a clean ``List[str]``.

    Single choke point for every path that touches tags (model validation,
    ingest tag-merging): ``None`` → ``[]``, a bare string → ``[string]``
    (YAML authors write ``tags: foo`` more often than you'd hope), other
    sequences → stringified list. Non-sequence scalars → ``[str(v)]``
    rather than a validation crash mid-import.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    return [str(value)]


class ConceptModel(BaseModel):
    """OKF concept with support for arbitrary frontmatter keys."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    body: str = ""
    embedding: Optional[List[float]] = None  # internal use only

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> List[str]:
        return normalize_tags(v)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> Any:
        # datetime.fromisoformat handles "Z", offsets, and date-only
        # strings on Python ≥ 3.11; anything else raises a proper
        # ValidationError (surfacing as "parse failed …", skip file).
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    def public_dict(self) -> Dict[str, Any]:
        """Agent/CLI-safe dump: everything except the embedding vector.

        Replaces the scattered ``model_dump()`` + ``pop("embedding")`` at
        every output surface (CLI ``get``, MCP ``read``). The 1024-float
        vector is index-internal — it must never reach an LLM context or
        terminal again.
        """
        return self.model_dump(exclude={"embedding"})

    def export_frontmatter(self) -> Tuple[Dict[str, Any], str]:
        """Split into ``(frontmatter_dict, body)`` for OKF serialization.

        Mirror image of the parse-side ``id:`` → ``uid`` preservation in
        ``parse_source_file``: the stored ``uid`` is written back as
        ``id:`` so export → re-import is lossless. Timestamps are ISO
        strings (YAML-safe); ``id``/``body``/``embedding`` never leak
        into frontmatter.
        """
        data = self.model_dump()
        body = data.pop("body", "")
        data.pop("id", None)
        data.pop("embedding", None)
        if "uid" in data:
            data["id"] = data.pop("uid")
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data, body


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

    @field_validator("parent_tags", mode="before")
    @classmethod
    def coerce_parent_tags(cls, v: Any) -> List[str]:
        return normalize_tags(v)


class ImageAssetModel(BaseModel):
    """Metadata for an image asset stored in the unified vector index.

    The raw bytes (``data``) and ``embedding`` are kept out of the default
    serialisation surface; this model is mainly for typed return values and
    listing, not for shuttling BLOBs around.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    file_name: str = ""
    mime_type: str = "application/octet-stream"
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    embed_route: Optional[str] = None  # "text" | "omni"
    content_hash: Optional[str] = None
    embedding: Optional[List[float]] = None  # internal use only
