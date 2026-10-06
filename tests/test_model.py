"""Model registry surface tests (0.6.0) — no model, no session.

Config defaults/validation/merge (TOML/env/CLI) for `embedding.model_id`,
router eager validation (stays cold), and the MetaText model pin
(set / match / mismatch-refuses / empty-graph re-pins) against a real
Ladybug connection. The default id is frozen: adding registry entries
never moves existing graphs.
"""

import pytest

from okfgraph.components.embedding import (
    MODEL_META_KEY,
    enforce_model_pin,
)
from okfgraph.settings import DEFAULT_MODEL_ID, Settings

NANO = "jinaai/jina-embeddings-v5-text-nano-retrieval"


# ---- config ---------------------------------------------------------------

def test_embedding_model_default_frozen():
    assert DEFAULT_MODEL_ID == "jinaai/jina-embeddings-v5-text-small-retrieval"
    s = Settings()
    assert s.model_id == DEFAULT_MODEL_ID
    assert s.checks() == ([], [])


def test_embedding_rejects_bad_model_id():
    for bad in ["", "no-slash", "a\"/b", "a'/b", "a;/b", "a\\/b"]:
        errors, _ = Settings(model_id=bad).checks()
        assert any("embedding.model_id" in e for e in errors), bad


def test_embedding_accepts_registry_and_custom_ids():
    assert Settings(model_id=NANO).checks() == ([], [])
    assert Settings(model_id="someone/custom-embed").checks() == ([], [])


def test_toml_parses_model_id(tmp_path):
    toml = tmp_path / "okfgraph.toml"
    toml.write_text(f'[embedding]\nmodel_id = "{NANO}"\n', encoding="utf-8")
    s = Settings.load(bundle_root=str(tmp_path))
    assert s.model_id == NANO


def test_env_overrides_model_id(monkeypatch):
    monkeypatch.setenv("OKFGRAPH_MODEL_ID", NANO)
    s = Settings.load()
    assert s.model_id == NANO


def test_cli_overrides_model_id():
    s = Settings.load(cli_args={"model_id": NANO})
    assert s.model_id == NANO


# ---- router (cold) ---------------------------------------------------------

def test_router_stores_model_id_cold(tmp_path):
    from okfgraph.router import OKFRouter

    r = OKFRouter(
        db_path=str(tmp_path / "m.db"), bundle_root=str(tmp_path),
        model_id=NANO)
    try:
        assert r.model_id == NANO
    finally:
        r.close()


def test_router_rejects_bad_model_id_cold(tmp_path):
    from okfgraph.router import OKFRouter

    with pytest.raises(ValueError, match="model_id must be"):
        OKFRouter(
            db_path=str(tmp_path / "m.db"), bundle_root=str(tmp_path),
            model_id="no-slash")
    with pytest.raises(ValueError, match="model_id must be"):
        OKFRouter(
            db_path=str(tmp_path / "m.db"), bundle_root=str(tmp_path),
            model_id="a\"/b")


def test_cli_model_reaches_router(tmp_path):
    from okfgraph.cli import _router, build_parser

    args = build_parser().parse_args(
        ["doctor", "--db-path", str(tmp_path / "m.db"),
         "--bundle-root", str(tmp_path), "--model-id", NANO])
    router = _router(args)
    try:
        assert router.model_id == NANO
    finally:
        router.close()


# ---- model pin --------------------------------------------------------------

def _router(tmp_path, name):
    from okfgraph.router import OKFRouter

    return OKFRouter(
        db_path=str(tmp_path / name), bundle_root=str(tmp_path))


def _pin_value(router):
    # MetaText (unlike Meta) is not part of the base schema — ensure it
    # before reading so the pre-pin assertion sees empty, not an error.
    router.conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS MetaText (key STRING PRIMARY KEY, value STRING)"
    )
    rows = router.conn.execute(
        f"MATCH (m:MetaText {{key: '{MODEL_META_KEY}'}}) RETURN m.value AS v"
    ).rows_as_dict().get_all()
    return rows[0]["v"] if rows else None


def test_pin_sets_on_first_open(tmp_path):
    r = _router(tmp_path, "pin1.db")
    try:
        assert _pin_value(r) is None
        assert enforce_model_pin(r.conn, NANO) == NANO
        assert _pin_value(r) == NANO
    finally:
        r.close()


def test_pin_match_passes(tmp_path):
    r = _router(tmp_path, "pin2.db")
    try:
        enforce_model_pin(r.conn, DEFAULT_MODEL_ID)
        assert enforce_model_pin(r.conn, DEFAULT_MODEL_ID) == DEFAULT_MODEL_ID
    finally:
        r.close()


def test_pin_mismatch_refuses_on_nonempty_graph(tmp_path):
    r = _router(tmp_path, "pin3.db")
    try:
        enforce_model_pin(r.conn, DEFAULT_MODEL_ID)
        r.conn.execute(
            "CREATE (c:Concept {id: 't', embedding: $vec})",
            {"vec": [0.0] * 512},
        )
        with pytest.raises(RuntimeError, match="pinned to model="):
            enforce_model_pin(r.conn, NANO)
        assert _pin_value(r) == DEFAULT_MODEL_ID  # refusal leaves the pin
    finally:
        r.close()


def test_pin_empty_graph_repins(tmp_path):
    r = _router(tmp_path, "pin4.db")
    try:
        enforce_model_pin(r.conn, DEFAULT_MODEL_ID)
        assert enforce_model_pin(r.conn, NANO) == NANO
        assert _pin_value(r) == NANO
    finally:
        r.close()
