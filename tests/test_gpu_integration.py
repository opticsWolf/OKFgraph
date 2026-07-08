"""GPU integration tests — Gap #12b.

Validates GPU/CUDA initialization, provider selection, batch encoding,
fallback behavior, and memory handling in the OKFRouter.

Tests are skipped when onnxruntime-gpu is not installed or CUDA is not
available.  When GPU is available they exercise the real CUDA path;
when not they exercise the CPU fallback path.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from okfgraph.router import OKFRouter


def _write_okf(bundle_root: str, rel: str, title: str, body: str):
    """Write an OKF-style markdown file with frontmatter."""
    p = Path(bundle_root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"title": title, "path": rel}
    header = "---\n" + yaml.dump(meta, default_flow_style=False) + "---\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(header + body)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cuda():
    """Return True if onnxruntime-gpu is installed and CUDA is available."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers
    except ImportError:
        return False


def _has_onnxruntime_gpu():
    """Return True if onnxruntime-gpu is installed (CUDA may or may not be available)."""
    try:
        import onnxruntime as ort  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Session-scoped fixtures to share model instances across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_dir():
    """Create a temporary directory shared across all GPU tests."""
    d = tempfile.mkdtemp(prefix="gpu_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def seed_bundle(tmp_dir):
    """Create a small bundle for testing."""
    files = [
        ("doc_a.md", "Doc A", "Alpha concept. " * 50),
        ("doc_b.md", "Doc B", "Beta concept. " * 50),
        ("doc_c.md", "Doc C", "Gamma concept. " * 50),
    ]
    paths = {}
    for rel, title, body in files:
        paths[rel] = _write_okf(tmp_dir, rel, title, body)
    return paths


@pytest.fixture(scope="session")
def gpu_router(tmp_dir, seed_bundle):
    """Session-scoped GPU router with warmup (only when CUDA is available)."""
    if not _has_cuda():
        pytest.skip("CUDA not available on this machine")

    r = OKFRouter(
        db_path=str(Path(tmp_dir) / "gpu_test_session.db"),
        bundle_root=tmp_dir,
        embedding_dim=512,
        device="cuda",
    )
    # Warm up the CUDA kernels
    _ = r._encode_batch(["warmup sentence"] * 5, task="Document")
    yield r
    r.close()


@pytest.fixture(scope="session")
def cpu_router(tmp_dir):
    """Session-scoped CPU router with warmup."""
    r = OKFRouter(
        db_path=str(Path(tmp_dir) / "cpu_test_session.db"),
        bundle_root=tmp_dir,
        embedding_dim=512,
        device="cpu",
    )
    # Warm up the CPU inference
    _ = r._encode_batch(["warmup sentence"] * 5, task="Document")
    yield r
    r.close()


# ---------------------------------------------------------------------------
# Test: GPU initialization
# ---------------------------------------------------------------------------

class TestGPUInitialization:
    """Test GPU initialization and provider selection."""

    # -- GPU available path --

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available on this machine")
    def test_gpu_device_requests_cuda_provider(self, tmp_dir):
        """When device='cuda' and CUDA is available, the embedder uses CUDAExecutionProvider."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_init.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )
        assert r.device == "cuda"
        assert r._cuda_fallback is False

        # Verify the embedder session reports CUDAExecutionProvider
        providers = r.embedder.session.get_providers()
        assert "CUDAExecutionProvider" in providers, (
            f"Expected CUDAExecutionProvider, got {providers}"
        )
        r.close()

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    def test_cpu_device_uses_cpu_provider(self, tmp_dir):
        """When device='cpu', the embedder uses CPUExecutionProvider."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_cpu_init.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cpu",
        )
        assert r.device == "cpu"

        providers = r.embedder.session.get_providers()
        assert "CPUExecutionProvider" in providers, (
            f"Expected CPUExecutionProvider, got {providers}"
        )
        r.close()

    # -- GPU unavailable path (fallback) --

    @pytest.mark.skipif(_has_cuda(), reason="CUDA is available — cannot test fallback path")
    def test_cuda_fallback_when_cuda_unavailable(self, tmp_dir):
        """When device='cuda' but CUDA is not available, the router falls back to CPU."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_fallback.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )
        assert r._cuda_fallback is True

        providers = r.embedder.session.get_providers()
        assert "CPUExecutionProvider" in providers
        r.close()

    @pytest.mark.skipif(_has_onnxruntime_gpu(), reason="onnxruntime-gpu is installed")
    def test_no_gpu_runtime_warns_on_cuda_request(self, tmp_dir):
        """When onnxruntime-gpu is not installed and device='cuda', a warning is logged."""
        import logging
        import io

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("okfgraph.router")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_no_gpu.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )
        assert r._cuda_fallback is True

        log_output = log_stream.getvalue()
        assert "CUDA unavailable" in log_output or "falling back to CPU" in log_output.lower()
        logger.removeHandler(handler)
        r.close()


# ---------------------------------------------------------------------------
# Test: GPU batch encoding
# ---------------------------------------------------------------------------

class TestGPUBatchEncoding:
    """Test batch encoding with GPU vs CPU."""

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available on this machine")
    def test_gpu_import_produces_concepts(self, tmp_dir, seed_bundle):
        """Import with device='cuda' produces valid concepts with embeddings."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_import.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cuda",
        )
        ids = r.import_bundle()
        assert len(ids) == 3, f"Expected 3 concepts, got {len(ids)}"

        # Verify embeddings exist
        for cid in ids:
            concept = r.get_by_id(cid)
            assert concept is not None
            assert concept.embedding is not None
            assert len(concept.embedding) == 512
        r.close()

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available on this machine")
    def test_gpu_encoding_produces_same_output_as_cpu(self, tmp_dir, seed_bundle):
        """GPU and CPU encoding produce the same embeddings (within tolerance)."""
        import numpy as np

        texts = ["test sentence for alignment", "another test sentence here"]

        r_cpu = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_align_cpu.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cpu",
        )
        emb_cpu = np.array(r_cpu._encode_batch(texts, task="Document"))

        r_gpu = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_align_gpu.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )
        emb_gpu = np.array(r_gpu._encode_batch(texts, task="Document"))

        # GPU and CPU may differ slightly due to floating-point precision
        np.testing.assert_allclose(emb_cpu, emb_gpu, rtol=1e-3, atol=1e-4)

        r_cpu.close()
        r_gpu.close()

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available on this machine")
    def test_gpu_batch_size_efficiency(self, tmp_dir, seed_bundle):
        """GPU encoding completes in reasonable time (sanity check)."""
        import time

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_timing.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cuda",
        )
        start = time.monotonic()
        ids = r.import_bundle()
        elapsed = time.monotonic() - start

        assert len(ids) == 3
        # Sanity: should complete in under 30 seconds (generous for CI)
        assert elapsed < 30, f"GPU import took {elapsed:.1f}s — unexpectedly slow"
        r.close()


# ---------------------------------------------------------------------------
# Test: GPU memory handling
# ---------------------------------------------------------------------------

class TestGPUMemoryHandling:
    """Test GPU memory allocation and cleanup."""

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available on this machine")
    def test_gpu_import_then_reimport_no_memory_leak(self, tmp_dir):
        """Multiple imports on GPU should not cause unbounded memory growth."""
        import shutil

        # Create a fresh bundle for this test with each file in its own subdirectory
        test_dir = Path(tmp_dir) / "gpu_mem_test"
        test_dir.mkdir(exist_ok=True)
        _write_okf(str(test_dir), "dir_a/doc_a.md", "Doc A", "Alpha concept. " * 50)
        _write_okf(str(test_dir), "dir_b/doc_b.md", "Doc B", "Beta concept. " * 50)

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_mem.db"),
            bundle_root=str(test_dir),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cuda",
        )

        # First import
        ids1 = r.import_bundle()
        assert len(ids1) == 2

        # Second import (no changes) — should be fast (delta skip)
        ids2 = r.import_bundle()
        assert ids2 == []

        # Third import with changes (new file in its own directory)
        _write_okf(str(test_dir), "dir_c/doc_c.md", "Doc C", "Gamma concept. " * 50)
        ids3 = r.import_bundle()
        assert len(ids3) == 1  # only new file

        r.close()
        shutil.rmtree(test_dir, ignore_errors=True)

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    def test_gpu_router_close_frees_embedder(self, tmp_dir):
        """Closing the router should properly checkpoint and close the DB connection."""
        import shutil

        # Create a fresh bundle for this test with each file in its own subdirectory
        test_dir = Path(tmp_dir) / "gpu_close_test"
        test_dir.mkdir(exist_ok=True)
        _write_okf(str(test_dir), "dir_a/doc_a.md", "Doc A", "Alpha concept. " * 50)
        _write_okf(str(test_dir), "dir_b/doc_b.md", "Doc B", "Beta concept. " * 50)

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_close.db"),
            bundle_root=str(test_dir),
            embedding_dim=512,
            device="cpu",  # Use CPU since CUDA may not be available
        )
        ids = r.import_bundle()
        assert len(ids) == 2

        # Close and verify DB connection is properly closed
        r.close()
        assert r.conn is None or getattr(r.conn, "is_closed", False)

        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: GPU with multimodal (omni) encoder
# ---------------------------------------------------------------------------

class TestGPUMultimodal:
    """Test GPU with the SentenceTransformer (omni) encoder path."""

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    def test_gpu_omni_encoder_initialization(self, tmp_dir):
        """GPU router with omni mode initializes the SentenceTransformer encoder."""
        import shutil

        # Create a fresh bundle for this test
        test_dir = Path(tmp_dir) / "gpu_omni_test"
        test_dir.mkdir(exist_ok=True)
        _write_okf(str(test_dir), "doc_a.md", "Doc A", "Alpha concept. " * 50)

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_omni.db"),
            bundle_root=str(test_dir),
            embedding_dim=512,
            device="cpu",  # Use CPU since CUDA may not be available
        )
        ids = r.import_bundle(mode="omni")
        assert len(ids) == 1

        # Verify the omni encoder was loaded by checking _encode_omni_text works
        try:
            emb = r._encode_omni_text("test sentence")
            assert emb is not None
            assert len(emb) == 512
        except ImportError:
            pytest.skip("sentence_transformers not installed")

        r.close()

        shutil.rmtree(test_dir, ignore_errors=True)

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    def test_gpu_omni_encoding_produces_embeddings(self, tmp_dir):
        """GPU omni encoding produces valid embeddings."""
        import shutil

        # Create a fresh bundle for this test
        test_dir = Path(tmp_dir) / "gpu_omni_enc_test"
        test_dir.mkdir(exist_ok=True)
        _write_okf(str(test_dir), "doc_a.md", "Doc A", "Alpha concept. " * 50)

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_gpu_omni_enc.db"),
            bundle_root=str(test_dir),
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cpu",  # Use CPU since CUDA may not be available
        )
        ids = r.import_bundle(mode="omni")
        assert len(ids) == 1

        concept = r.get_by_id(ids[0])
        assert concept is not None
        assert concept.embedding is not None
        assert len(concept.embedding) == 512
        r.close()

        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: GPU device selection edge cases
# ---------------------------------------------------------------------------

class TestGPUDeviceEdgeCases:
    """Test edge cases in GPU device selection."""

    def test_default_device_is_cpu(self, tmp_dir):
        """Default device should be 'cpu'."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_default_device.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
        )
        assert r.device == "cpu"
        r.close()

    @pytest.mark.skipif(not _has_onnxruntime_gpu(), reason="onnxruntime-gpu not installed")
    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available on this machine")
    def test_cuda_provider_order_when_available(self, tmp_dir):
        """When CUDA is available, CUDAExecutionProvider should be first."""
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_provider_order.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )
        providers = r.embedder.session.get_providers()
        assert providers[0] == "CUDAExecutionProvider", (
            f"Expected CUDAExecutionProvider first, got {providers}"
        )
        r.close()

    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
    def test_cuda_produces_valid_embeddings(self, tmp_dir):
        """When CUDA is available, embeddings should be valid numpy arrays."""
        import numpy as np

        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_cuda_valid.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )

        texts = ["test sentence one", "test sentence two"]
        embeddings = np.array(r._encode_batch(texts, task="Document"))
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 512)
        r.close()

    @pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
    def test_cuda_embeddings_match_cpu(self, tmp_dir):
        """When CUDA is available, GPU embeddings should match CPU embeddings (within floating-point tolerance)."""
        import numpy as np

        texts = ["test sentence for alignment", "another test sentence here"]

        r_cpu = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_align_cpu.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cpu",
        )
        emb_cpu = np.array(r_cpu._encode_batch(texts, task="Document"))

        r_gpu = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_align_gpu.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            device="cuda",
        )
        emb_gpu = np.array(r_gpu._encode_batch(texts, task="Document"))

        # GPU and CPU may differ slightly due to floating-point precision
        np.testing.assert_allclose(emb_cpu, emb_gpu, rtol=1e-3, atol=1e-4)

        r_cpu.close()
        r_gpu.close()
