"""End-to-end: OKFRouter with the Rust embedding stack (no torch/optimum/transformers).

Needs ladybug + okf_embed wheel + mordant + model download (cached after first
run). Slow-marked like test_parity.py.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("ladybug")
okf_embed = pytest.importorskip("okf_embed")
pytest.importorskip("mordant")

from okfgraph import OKFRouter

pytestmark = pytest.mark.slow

INTRO = """---
title: Intro
type: chapter
tags: [graph, basics]
---

Knowledge graphs link concepts with typed relationships.
"""
ADVANCED = """---
title: Advanced
type: section
tags: [graph, search]
---

Hybrid search fuses vector and keyword retrieval scores. See [Intro](intro.md).
"""


@pytest.fixture()
def router():
    tmp = Path(tempfile.mkdtemp(prefix="okf_rust_e2e_"))
    bundle = tmp / "bundle"
    bundle.mkdir()
    (bundle / "intro.md").write_text(INTRO, encoding="utf-8")
    (bundle / "advanced.md").write_text(ADVANCED, encoding="utf-8")
    r = OKFRouter(
        db_path=str(tmp / "e2e.db"),
        bundle_root=str(bundle),
        embedding_dim=64,
        device="cpu",
    )
    yield r, bundle
    r.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_backend_selection(router):
    r, _ = router
    assert isinstance(r.encoder, okf_embed.JinaV5)


def test_no_heavy_imports():
    """Importing okfgraph must not pull torch/optimum/transformers (hermetic)."""
    import subprocess
    from pathlib import Path as _Path

    code = (
        "import sys, okfgraph; "
        "banned=[m for m in ('torch','optimum','transformers','sentence_transformers') "
        "if m in sys.modules]; "
        "print('banned:', banned); raise SystemExit(1 if banned else 0)"
    )
    root = _Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_import_lookup_search(router):
    r, bundle = router
    ids = r.import_mgr.import_bundle(bundle, mode="text")
    assert sorted(ids) == ["advanced", "intro"]

    got = r.search_engine.get_by_id("advanced")
    title = got["title"] if isinstance(got, dict) else got.title
    assert title == "Advanced"

    res = r.search_engine.search_hybrid("hybrid search fusion", limit=2)
    assert res
    top = res[0].get("id") if isinstance(res[0], dict) else res[0].id
    assert top == "advanced"

    chunks = r.search_engine.search_chunks("typed relationships", limit=2)
    assert len(chunks) >= 1
