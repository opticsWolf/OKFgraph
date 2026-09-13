"""Parity: embroider (Rust/ort) vs torch-free baseline (transformers + onnxruntime).

Replicates EmbeddingEngine._encode exactly in numpy and pins the Rust port
at max abs diff <= 1e-5, cosine >= 0.999999, across dims x tasks x texts.

Slow: needs the Jina v5 download (~2GB first run) + onnxruntime/transformers.
Run:  .venv/Scripts/python.exe -m pytest tests/test_parity.py -m slow
"""
import math
import os
from pathlib import Path

import pytest

onnxruntime = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")
embroider = pytest.importorskip("embroider")

MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
DIMS = [32, 64, 128, 256, 512, 1024]
TEXTS = [
    "Knowledge graphs link concepts with typed relationships.",
    "Hybrid search fuses vector and keyword retrieval scores.",
    "What is a knowledge graph?",
    "",
    "x",
    "Query: already-prefixed text",
    "Document: already-prefixed text",
    "Unicode: Übergröße naïve façade — 数学",
    "Long: " + "retrieval-augmented generation over bitemporal ledgers. " * 60,
]

pytestmark = pytest.mark.slow


def _ort_dylib():
    if "ORT_DYLIB_PATH" not in os.environ:
        dll = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
        if dll.exists():
            os.environ["ORT_DYLIB_PATH"] = str(dll)


def truncate_normalize(vec, dim):
    v = list(vec[:dim])
    if len(v) < dim:
        v = v + [0.0] * (dim - len(v))
    n = math.sqrt(sum(x * x for x in v))
    if n > 0:
        v = [x / n for x in v]
    return v


def baseline_encode(sess, tok, text, task, dim):
    if not text.startswith(("Query:", "Document:")):
        text = f"{task}: {text}"
    inp = tok(text, return_tensors="np", truncation=True, max_length=8192, padding=True)
    out = sess.run(
        ["last_hidden_state"],
        {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]},
    )[0]
    last_idx = max(int(inp["attention_mask"].sum(axis=1)[0] - 1), 0)
    pooled = out[0, last_idx]
    pooled = pooled / np.linalg.norm(pooled)
    return truncate_normalize(pooled.tolist(), dim)


@pytest.fixture(scope="module")
def baseline():
    from huggingface_hub import snapshot_download

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    snap = snapshot_download(MODEL, allow_patterns=["onnx/*"])
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    sess = onnxruntime.InferenceSession(
        str(Path(snap) / "onnx" / "model.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    return sess, tok


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("task", ["Document", "Query"])
def test_parity(baseline, dim, task):
    _ort_dylib()
    sess, tok = baseline
    enc = embroider.JinaV5.open(MODEL, truncate_dim=dim, device="cpu")
    assert enc.used_cuda is False
    assert enc.dim == dim
    for text in TEXTS:
        a = np.array(baseline_encode(sess, tok, text, task, dim), dtype=np.float64)
        b = np.array(enc.encode(text, task=task), dtype=np.float64)
        assert len(b) == dim
        assert abs(np.linalg.norm(b) - 1.0) < 1e-5 or np.linalg.norm(a) == 0
        assert float(np.max(np.abs(a - b))) <= 1e-5, f"maxdiff exceeded for {text[:40]!r}"
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        assert cos >= 0.999999, f"cosine {cos} for {text[:40]!r}"


def test_batch_shape():
    _ort_dylib()
    enc = embroider.JinaV5.open(MODEL, truncate_dim=64, device="cpu")
    vecs = enc.encode_batch(["doc %d" % i for i in range(5)])
    assert len(vecs) == 5 and all(len(v) == 64 for v in vecs)


def test_bad_dim_rejected():
    _ort_dylib()

    with pytest.raises(RuntimeError):
        embroider.JinaV5.open(MODEL, truncate_dim=2048, device="cpu")
    with pytest.raises(RuntimeError):
        embroider.JinaV5.open(MODEL, truncate_dim=16, device="cpu")


def test_cuda_parity_and_flag():
    """GPU path: same vectors within fp-reduction drift, flag honest.

    Skipped where no CUDA EP is registered. Tolerance is looser than the
    CPU bar (1e-5) — GPU reduction order drifts ~1e-4 abs while cosine
    stays >= 0.999999, which cannot flip ANN neighbours.
    """
    _ort_dylib()
    try:
        gpu = embroider.JinaV5.open(MODEL, truncate_dim=512, device="cuda")
    except RuntimeError:
        pytest.skip("no ONNX Runtime with CUDA EP")
    if not gpu.used_cuda:
        pytest.skip("CUDA requested but CPU fallback engaged")
    cpu = embroider.JinaV5.open(MODEL, truncate_dim=512, device="cpu")
    worst, worstcos = 0.0, 1.0
    for text in TEXTS:
        a = np.array(cpu.encode(text), dtype=np.float64)
        b = np.array(gpu.encode(text), dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(a - b))))
        worstcos = min(worstcos, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    assert worst <= 5e-4, f"cpu/gpu drift {worst:.2e}"
    assert worstcos >= 0.999999, f"cpu/gpu cosine {worstcos:.9f}"
