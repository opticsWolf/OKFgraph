"""Precision + CPU-arena surface tests (0.5.0) — no model, no session.

Config defaults/validation/merge (TOML/env/CLI), router eager validation
(stays cold), and the Meta precision pin (set / match / mismatch-refuses /
empty-graph re-pins) against a real Ladybug connection.
"""

import logging

import pytest

from okfgraph.components.embedding import (
    PRECISION_META_KEY,
    enforce_precision_pin,
)
from okfgraph.config import EmbeddingConfig, OKFConfig


# ---- config ---------------------------------------------------------------

def test_embedding_defaults_follow_device():
    cfg = EmbeddingConfig()
    assert cfg.device == "auto"
    assert cfg.precision == "auto"
    assert cfg.cpu_arena is False
    assert cfg.validate() == []


def test_embedding_rejects_bad_precision():
    cfg = EmbeddingConfig(precision="int8")
    errors = cfg.validate()
    assert any("embedding.precision" in e for e in errors)


def test_toml_parses_precision_and_arena(tmp_path):
    toml = tmp_path / "okfgraph.toml"
    toml.write_text(
        "[embedding]\nprecision = \"fp16\"\ncpu_arena = true\n",
        encoding="utf-8",
    )
    cfg = OKFConfig.load(bundle_root=str(tmp_path))
    assert cfg.embedding.precision == "fp16"
    assert cfg.embedding.cpu_arena is True


def test_env_overrides_precision_and_arena(monkeypatch):
    monkeypatch.setenv("OKFGRAPH_PRECISION", "fp32")
    monkeypatch.setenv("OKFGRAPH_CPU_ARENA", "yes")
    cfg = OKFConfig.load()
    assert cfg.embedding.precision == "fp32"
    assert cfg.embedding.cpu_arena is True


def test_cli_overrides_precision_and_arena():
    cfg = OKFConfig()
    OKFConfig._apply_cli(cfg, {"precision": "fp16", "cpu_arena": True})
    assert cfg.embedding.precision == "fp16"
    assert cfg.embedding.cpu_arena is True


# ---- router (cold) ---------------------------------------------------------

def test_router_rejects_bad_precision_cold(tmp_path):
    from okfgraph.router import OKFRouter

    with pytest.raises(ValueError, match="precision must be"):
        OKFRouter(
            db_path=str(tmp_path / "p.db"),
            bundle_root=str(tmp_path),
            precision="int8",
        )


def test_router_defaults_and_stores_surface(tmp_path):
    from okfgraph.router import OKFRouter

    r = OKFRouter(
        db_path=str(tmp_path / "d.db"), bundle_root=str(tmp_path))
    try:
        assert r.device == "auto"
        assert r.precision == "auto"
        assert r.cpu_arena is False
        assert r.encoder.is_loaded is False
    finally:
        r.close()


def test_router_warns_fp16_on_cpu(tmp_path, caplog):
    from okfgraph.router import OKFRouter

    with caplog.at_level(logging.WARNING, logger="okfgraph.router"):
        r = OKFRouter(
            db_path=str(tmp_path / "w.db"),
            bundle_root=str(tmp_path),
            device="cpu",
            precision="fp16",
        )
    try:
        assert any("FP16 on CPU" in m for m in caplog.messages)
    finally:
        r.close()


# ---- precision pin ----------------------------------------------------------

def _router(tmp_path, name):
    from okfgraph.router import OKFRouter

    return OKFRouter(
        db_path=str(tmp_path / name), bundle_root=str(tmp_path))


def _pin_value(router):
    rows = router.conn.execute(
        f"MATCH (m:Meta {{key: '{PRECISION_META_KEY}'}}) RETURN m.value AS v"
    ).rows_as_dict().get_all()
    return rows[0]["v"] if rows else None


def test_pin_sets_on_first_open(tmp_path):
    r = _router(tmp_path, "pin1.db")
    try:
        assert _pin_value(r) is None
        assert enforce_precision_pin(r.conn, "fp16") == "fp16"
        assert _pin_value(r) == 16
    finally:
        r.close()


def test_pin_match_passes(tmp_path):
    r = _router(tmp_path, "pin2.db")
    try:
        enforce_precision_pin(r.conn, "fp32")
        assert enforce_precision_pin(r.conn, "fp32") == "fp32"
        assert _pin_value(r) == 32
    finally:
        r.close()


def test_pin_mismatch_refuses_on_nonempty_graph(tmp_path):
    r = _router(tmp_path, "pin3.db")
    try:
        enforce_precision_pin(r.conn, "fp32")
        r.conn.execute(
            "CREATE (c:Concept {id: 't', embedding: $vec})",
            {"vec": [0.0] * 512},
        )
        with pytest.raises(RuntimeError, match="pinned to precision=fp32"):
            enforce_precision_pin(r.conn, "fp16")
        assert _pin_value(r) == 32  # refusal leaves the pin untouched
    finally:
        r.close()


def test_pin_empty_graph_repins(tmp_path):
    r = _router(tmp_path, "pin4.db")
    try:
        enforce_precision_pin(r.conn, "fp32")
        assert enforce_precision_pin(r.conn, "fp16") == "fp16"
        assert _pin_value(r) == 16
    finally:
        r.close()
