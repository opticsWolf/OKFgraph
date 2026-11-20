#!/usr/bin/env python3
"""Smoke-test the documented Python facade against the Rust wheel."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import okfgraph
from okfgraph import (
    DEFAULT_MODEL_ID, SETTINGS, OKFClient, OKFError, OKFRouter, ConceptModel,
    discover, serve_file_for,
)
from okfgraph.client import OPS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    assert okfgraph.version().startswith("0.16.")
    assert len(OPS) == 30
    assert len(SETTINGS) == 52
    assert DEFAULT_MODEL_ID == "google/embeddinggemma-2"
    assert callable(OKFClient) and callable(discover) and callable(serve_file_for)
    parameters = inspect.signature(OKFRouter.__init__).parameters
    assert "db_path" in parameters and "bundle_root" in parameters
    missing = [name for name in OPS if not hasattr(OKFRouter, name)]
    assert not missing, f"facade is missing op methods: {missing}"
    doctor_params = inspect.signature(OKFRouter.doctor).parameters
    assert {"stale_days", "fix", "strict", "cache_dir"} <= set(doctor_params)
    router = OKFRouter(
        db_path=str(Path(args.db)),
        embedding_dim=512,
        model_id="google/embeddinggemma-2",
    )
    try:
        concept = router.get_by_id("alpha")
        assert isinstance(concept, ConceptModel)
        assert concept.id == "alpha"
        assert router.get_by_id("missing") is None
        assert router.list_directory("kb")[0]["id"] == "alpha"
        try:
            router.read("missing")
        except OKFError as exc:
            assert exc.code == "UNKNOWN_CONCEPT"
        else:
            raise AssertionError("read('missing') should raise")
        report = router.doctor()
        assert report["report"]["concepts"] == 2
        print("OK: Python facade delegates read ops to Rust")
    finally:
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
