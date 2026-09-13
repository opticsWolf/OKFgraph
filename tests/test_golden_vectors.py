"""Golden vector-space contract: live embroider output vs vendored fixture.

The fixture is generated in the embroider repo
(`fixtures/golden_jina_v5_text_small.json` — see its README "Conformance"
and `COMPAT.md`) and vendored here byte-identically. This test uses the
cached model snapshot only (no network): it proves the wheel in this env
honors the frozen Jina contract (prefixes, last-token pooling, Matryoshka
truncation) for the exact embroider+ORT stack under test.

Tolerance `abs=1e-6` catches wrong model / pooling / prefix / truncation
(O(1) differences) while staying immune to cross-CPU kernel noise.
"""
import json
from pathlib import Path

import pytest

embroider = pytest.importorskip("embroider")

from okfgraph.components.embedding import EmbeddingEngine, resolve_ort_dylib

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "golden_jina_v5_text_small.json"
)
MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"


@pytest.fixture(scope="module")
def golden():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["contract"] == "jina-v5-text", data.get("contract")
    assert data["model"] == MODEL, data.get("model")
    return data


@pytest.fixture(scope="module")
def snap_paths():
    resolve_ort_dylib()
    hub = Path(EmbeddingEngine.default_cache_dir()) / "hub"
    cached = hub / ("models--" + MODEL.replace("/", "--")) / "snapshots"
    assert cached.is_dir(), f"model not cached locally: {MODEL}"
    snap = sorted(cached.iterdir())[-1]
    onnx_path = snap / "onnx" / "model.onnx"
    tok_path = snap / "tokenizer.json"
    assert onnx_path.is_file() and tok_path.is_file()
    return onnx_path, tok_path


def test_golden_token_counts_exact(golden, snap_paths):
    _, tok_path = snap_paths
    tok = embroider.JinaTokenizer.open_files(str(tok_path))
    for key, expected in golden["token_counts"].items():
        assert tok.count_tokens(golden["texts"][key]) == expected


def test_golden_vectors_match(golden, snap_paths):
    onnx_path, tok_path = snap_paths
    sessions = {}
    for case in golden["cases"]:
        dim = case["truncate_dim"]
        if dim not in sessions:
            sessions[dim] = embroider.JinaV5.open_files(
                str(onnx_path), str(tok_path), truncate_dim=dim, device="cpu"
            )
        got = sessions[dim].encode(
            golden["texts"][case["text_key"]], task=case["task"]
        )
        assert got == pytest.approx(case["vector"], abs=1e-6)
