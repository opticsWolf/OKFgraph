"""Pydantic model contract: coercion, config, serialization surfaces."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from okfgraph.models import (
    ChunkModel,
    ConceptModel,
    ImageAssetModel,
    normalize_tags,
)


class TestNormalizeTags:
    def test_none(self):
        assert normalize_tags(None) == []

    def test_bare_string(self):
        # YAML authors write `tags: foo` — must not crash validation.
        assert normalize_tags("foo") == ["foo"]

    def test_empty_string(self):
        assert normalize_tags("") == []

    def test_list_passthrough(self):
        assert normalize_tags(["a", "b"]) == ["a", "b"]

    def test_tuple_set_stringified(self):
        assert sorted(normalize_tags(("a", 1))) == ["1", "a"]

    def test_scalar(self):
        assert normalize_tags(7) == ["7"]


class TestConceptTags:
    def test_string_tags_coerced(self):
        c = ConceptModel.model_validate({"id": "x", "type": "note", "tags": "solo"})
        assert c.tags == ["solo"]

    def test_missing_tags_default(self):
        c = ConceptModel.model_validate({"id": "x", "type": "note"})
        assert c.tags == []

    def test_null_tags(self):
        c = ConceptModel.model_validate({"id": "x", "type": "note", "tags": None})
        assert c.tags == []


class TestTimestamp:
    def test_zulu(self):
        c = ConceptModel.model_validate(
            {"id": "x", "type": "note", "timestamp": "2024-01-02T03:04:05Z"}
        )
        assert c.timestamp == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    def test_date_only(self):
        c = ConceptModel.model_validate(
            {"id": "x", "type": "note", "timestamp": "2024-01-02"}
        )
        assert c.timestamp == datetime(2024, 1, 2)

    def test_datetime_passthrough(self):
        now = datetime.now(timezone.utc)
        c = ConceptModel.model_validate({"id": "x", "type": "note", "timestamp": now})
        assert c.timestamp == now

    def test_garbage_rejected(self):
        with pytest.raises(ValidationError):
            ConceptModel.model_validate(
                {"id": "x", "type": "note", "timestamp": "not-a-date"}
            )


class TestExtraAllow:
    def test_arbitrary_frontmatter_survives(self):
        c = ConceptModel.model_validate(
            {"id": "x", "type": "note", "uid": "stable-1", "aliases": ["X"], "custom": 5}
        )
        assert c.model_extra["uid"] == "stable-1"
        d = c.model_dump()
        assert d["custom"] == 5 and d["aliases"] == ["X"]


class TestPublicDict:
    def test_embedding_excluded_extras_kept(self):
        c = ConceptModel.model_validate({
            "id": "x", "type": "note", "body": "b",
            "embedding": [0.1, 0.2], "uid": "u1",
        })
        d = c.public_dict()
        assert "embedding" not in d
        assert d["body"] == "b" and d["uid"] == "u1" and d["id"] == "x"


class TestExportFrontmatter:
    def test_split_and_uid_rewrite(self):
        c = ConceptModel.model_validate({
            "id": "path/x", "type": "note", "body": "# Hi",
            "embedding": [0.1], "uid": "stable-1",
            "timestamp": "2024-05-06T07:08:09",
        })
        fm, body = c.export_frontmatter()
        assert body == "# Hi"
        assert fm["id"] == "stable-1"  # uid written back as id:
        assert "uid" not in fm and "embedding" not in fm and "body" not in fm
        assert fm["timestamp"] == "2024-05-06T07:08:09"

    def test_no_uid_no_id_key(self):
        c = ConceptModel.model_validate({"id": "x", "type": "note", "body": "b"})
        fm, body = c.export_frontmatter()
        assert "id" not in fm and body == "b"

    def test_round_trip_through_validate(self):
        # export output must re-validate (import side accepts it back).
        c = ConceptModel.model_validate({
            "id": "x", "type": "note", "body": "b", "uid": "u9",
        })
        fm, body = c.export_frontmatter()
        c2 = ConceptModel.model_validate({**fm, "id": "x", "body": body})
        assert c2.title == c.title and c2.type == "note"


class TestChunkParentTags:
    def test_string_coerced(self):
        c = ChunkModel(
            id="c", parent_doc_id="p", chunk_index=0,
            chunk_text="t", block_type="para", parent_tags="solo",
        )
        assert c.parent_tags == ["solo"]


class TestImageAssetExtra:
    def test_extra_allowed(self):
        m = ImageAssetModel(id="a", custom="v")
        assert m.model_extra["custom"] == "v"
