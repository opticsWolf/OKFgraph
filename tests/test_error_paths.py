"""Dedicated error-path tests: links / schema / purge / embedding.

Happy paths are covered by router-level integration tests; this file pins
the failure modes that bite on user machines — ambiguous names, stale DBs,
double purges, missing runtimes, corrupt model bytes. Everything here is
model-free (stubbed encoder, mini tokenizer, no downloads), so it runs in
CI alongside the fast list.
"""
import json
from pathlib import Path

import pytest

from okfgraph.components.links import build_name_index, resolve_wiki
from okfgraph.router import OKFRouter


def _concepts():
    return [
        {"id": "docs/alpha", "title": "Shared Title", "uid": "alpha-uid",
         "aliases": ["alpha-alias"]},
        {"id": "docs/beta", "title": "Shared Title", "uid": "beta-uid",
         "aliases": []},
        {"id": "gamma", "title": "Gamma Doc", "aliases": []},
    ]


class TestLinksErrors:
    def test_ambiguous_title_never_resolves(self):
        maps, ambiguous = build_name_index(_concepts())
        assert "title:shared title" in ambiguous
        assert resolve_wiki("Shared Title", maps, set()) is None

    def test_uid_wins_over_colliding_title(self):
        maps, _ = build_name_index(_concepts())
        # uid lookup precedes title: exact uid hits even though the title
        # is ambiguous.
        assert resolve_wiki("alpha-uid", maps, set()) == "docs/alpha"

    def test_reference_normalization(self):
        maps, _ = build_name_index(_concepts())
        known = {"docs/alpha", "gamma"}
        assert resolve_wiki("  GAMMA.MD#section ", maps, known) == "gamma"
        assert resolve_wiki("docs/alpha", maps, known) == "docs/alpha"

    def test_empty_and_unknown_miss(self):
        maps, _ = build_name_index(_concepts())
        assert resolve_wiki("", maps, set()) is None
        assert resolve_wiki("   ", maps, set()) is None
        assert resolve_wiki("no-such-doc", maps, set()) is None


def _router(tmp_path, dim=512, **kw):
    return OKFRouter(
        db_path=str(tmp_path / "err.db"),
        bundle_root=str(tmp_path),
        embedding_dim=dim,
        enable_chunking=False,
        device="cpu",
        **kw,
    )


class TestSchemaErrors:
    def test_reinit_is_idempotent(self, tmp_path):
        r1 = _router(tmp_path)
        r1.close()
        r2 = _router(tmp_path)
        assert r2.embedding_dim == 512
        r2.close()

    def test_existing_dim_wins_over_requested(self, tmp_path):
        r1 = _router(tmp_path, dim=64)
        r1.close()
        r2 = _router(tmp_path, dim=512)
        assert r2.embedding_dim == 64
        r2.close()

    def test_fresh_db_indexes_not_dirty(self, tmp_path):
        r = _router(tmp_path)
        assert r.schema_mgr._indexes_dirty() is False
        r.close()

    def test_v5_db_without_table_migrates(self, tmp_path):
        """A 0.2.x-era DB (version 5, no DeletedConcept table) gains the
        full v6 table on open."""
        import ladybug as lb

        db_path = str(tmp_path / "v5.db")
        conn = lb.Connection(lb.Database(db_path))
        conn.execute(
            "CREATE NODE TABLE Meta (key STRING PRIMARY KEY, value INT64)"
        )
        conn.execute(
            "CREATE (m:Meta {key: 'schema_version', value: 5})"
        )
        del conn
        r = OKFRouter(db_path=db_path, bundle_root=str(tmp_path),
                       embedding_dim=512, enable_chunking=False, device="cpu")
        assert r.schema_mgr._get_meta("schema_version") == 6
        # The migrated table supports the full lifecycle.
        (tmp_path / "m.md").write_text("# M\n\nBody.\n", encoding="utf-8")
        r.embed_engine._encode = lambda text, task="Document": [0.0] * 512
        r.import_mgr.import_bundle(tmp_path)
        assert r.purge_mgr._soft_delete_concept("m") is True
        assert r.purge_mgr._recover_concept("m") is True
        r.close()

    def test_v5_table_gains_snapshot_column(self, tmp_path):
        """A pre-0.2.x DB whose v5 DeletedConcept lacks `snapshot` is
        ALTERed on open — the lifecycle then works against it."""
        import ladybug as lb

        db_path = str(tmp_path / "v5t.db")
        conn = lb.Connection(lb.Database(db_path))
        conn.execute(
            "CREATE NODE TABLE Meta (key STRING PRIMARY KEY, value INT64)"
        )
        conn.execute(
            "CREATE (m:Meta {key: 'schema_version', value: 5})"
        )
        conn.execute("""
            CREATE NODE TABLE DeletedConcept (
                id STRING PRIMARY KEY, original_id STRING, title STRING,
                body STRING, deleted_at STRING, type STRING, tags STRING
            )
        """)
        del conn
        r = OKFRouter(db_path=db_path, bundle_root=str(tmp_path),
                       embedding_dim=512, enable_chunking=False, device="cpu")
        assert r.schema_mgr._get_meta("schema_version") == 6
        (tmp_path / "m.md").write_text("# M\n\nBody.\n", encoding="utf-8")
        r.embed_engine._encode = lambda text, task="Document": [0.0] * 512
        r.import_mgr.import_bundle(tmp_path)
        assert r.purge_mgr._soft_delete_concept("m") is True
        assert r.purge_mgr._recover_concept("m") is True
        assert r.get_by_id("m").model_dump()["body"].strip() == "# M\n\nBody."
        r.close()


def _stubbed_router(tmp_path, monkeypatch):
    r = _router(tmp_path)
    monkeypatch.setattr(
        r.embed_engine, "_encode", lambda text, task="Document": [0.0] * 512
    )
    monkeypatch.setattr(
        r.embed_engine, "_encode_batch",
        lambda texts, task="Document": [[0.0] * 512 for _ in texts],
    )
    monkeypatch.setattr(r.embed_engine, "count_tokens", lambda text: len(text) // 4)
    return r


def _two_docs(tmp_path):
    (tmp_path / "one.md").write_text(
        "# One\n\nFirst doc body.\n", encoding="utf-8"
    )
    (tmp_path / "two.md").write_text(
        "# Two\n\nSecond doc body.\n", encoding="utf-8"
    )


class TestPurgeErrors:
    def test_soft_delete_missing_id_is_false(self, tmp_path, monkeypatch):
        r = _stubbed_router(tmp_path, monkeypatch)
        _two_docs(tmp_path)
        assert len(r.import_mgr.import_bundle(tmp_path)) == 2
        assert r.purge_mgr._soft_delete_concept("no-such-id") is False
        r.close()

    def test_recover_lifecycle(self, tmp_path, monkeypatch):
        r = _stubbed_router(tmp_path, monkeypatch)
        _two_docs(tmp_path)
        r.import_mgr.import_bundle(tmp_path)
        before = r.get_by_id("one").model_dump()
        assert r.purge_mgr._soft_delete_concept("one") is True
        assert {d["concept_id"] for d in r.purge_mgr.list_deleted_concepts()} == {"one"}
        # Active or missing ids cannot be recovered.
        assert r.purge_mgr._recover_concept("two") is False
        assert r.purge_mgr._recover_concept("no-such-id") is False
        assert r.purge_mgr._recover_concept("one") is True
        assert r.purge_mgr.list_deleted_concepts() == []
        # Full fidelity: body, tags, and the vector survive the round-trip.
        after = r.get_by_id("one").model_dump()
        assert after["body"] == before["body"]
        assert after["tags"] == before["tags"]
        assert after["embedding"] == before["embedding"]
        r.close()

    def test_purge_is_idempotent(self, tmp_path, monkeypatch):
        r = _stubbed_router(tmp_path, monkeypatch)
        _two_docs(tmp_path)
        r.import_mgr.import_bundle(tmp_path)
        assert r.purge_mgr._soft_delete_concept("one") is True
        assert r.purge_mgr.purge_deleted_concepts(older_than=0) == 1
        assert r.purge_mgr.purge_deleted_concepts(older_than=0) == 0
        assert r.purge_mgr.list_deleted_concepts() == []
        r.close()


MINI_TOKENIZER = {
    "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None, "decoder": None,
    "model": {"type": "WordLevel",
              "vocab": {"hello": 0, "world": 1, "[UNK]": 2},
              "unk_token": "[UNK]"},
}


class TestEmbeddingErrors:
    def test_bogus_dylib_path_passes_through(self, monkeypatch):
        """resolve_ort_dylib never raises: a bogus explicit path is returned
        verbatim; session creation is the fail-fast boundary."""
        from okfgraph.components.embedding import resolve_ort_dylib

        monkeypatch.setenv("ORT_DYLIB_PATH", "/nonexistent/ort.dll")
        assert resolve_ort_dylib() == "/nonexistent/ort.dll"

    def test_corrupt_onnx_fails_cleanly(self, tmp_path):
        """Garbage ONNX bytes must raise a Python exception — never abort
        the process (the stale-System32 class of failure)."""
        embroider = pytest.importorskip("embroider")
        from okfgraph.components.embedding import resolve_ort_dylib

        if resolve_ort_dylib() is None:
            pytest.skip("no ONNX Runtime dylib resolvable")
        bad_onnx = tmp_path / "model.onnx"
        bad_onnx.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096)
        tok_path = tmp_path / "tokenizer.json"
        tok_path.write_text(json.dumps(MINI_TOKENIZER), encoding="utf-8")
        with pytest.raises(Exception):
            embroider.JinaV5.open_files(
                str(bad_onnx), str(tok_path), truncate_dim=64, device="cpu"
            )
