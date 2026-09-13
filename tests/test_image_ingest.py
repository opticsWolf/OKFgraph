"""0.2.18 — image ingest hardening (router-level, real encoder).

- Phantom refs (dangling local paths, `<...>` placeholders in prose) grow
  no asset rows.
- DB-only assets (source file deleted after import) are preserved, not pruned.
- Replacing an image (caption change) exercises delete-then-create on a
  real row without crashing (ladybug 0.2.3 segfaults on
  WHERE NOT EXISTS + DETACH DELETE run after an earlier image transaction).

Chunking is off: images run in Phase 6 either way, and tiny bodies stay
clear of the short-doc chunk quirk.
"""

import base64
import shutil
import tempfile
from pathlib import Path

import pytest

from okfgraph.router import OKFRouter

PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def env():
    d = tempfile.mkdtemp()
    r = OKFRouter(
        db_path=str(Path(d) / "img.db"),
        bundle_root=d,
        embedding_dim=512,
        enable_chunking=False,
        device="cpu",
    )
    yield {"tmp": d, "router": r}
    r.close()
    shutil.rmtree(d, ignore_errors=True)


def _assets(router):
    return router.conn.execute(
        "MATCH (i:ImageAsset) RETURN i.id AS id, i.caption AS caption"
    ).rows_as_dict().get_all()


def _edges(router):
    return router.conn.execute(
        "MATCH ()-[r:INCLUDES_ASSET]->() RETURN count(r) AS n"
    ).rows_as_dict().get_all()[0]["n"]


def test_phantom_refs_skipped(env):
    tmp, router = env["tmp"], env["router"]
    (Path(tmp) / "doc.md").write_text(
        "---\ntitle: Doc\n---\nProse about syntax: link `![alt](rel)` here.\n"
        "Placeholder example: ![screenshot](<id>) there.\n",
        encoding="utf-8",
    )
    assert router.import_mgr.import_bundle() == ["doc"]
    assert _assets(router) == []
    assert _edges(router) == 0


def test_db_only_asset_preserved(env):
    tmp, router = env["tmp"], env["router"]
    md = Path(tmp) / "pic.md"
    png = Path(tmp) / "a.png"
    md.write_text(
        "---\ntitle: Pic\n---\nA real photo: ![photo](a.png) here.\n",
        encoding="utf-8",
    )
    png.write_bytes(PNG_1x1)
    assert router.import_mgr.import_bundle() == ["pic"]
    assert len(_assets(router)) == 1
    assert _edges(router) == 1

    # Source image deleted, doc touched so the delta reruns images.
    png.unlink()
    md.write_text(
        "---\ntitle: Pic\n---\nA real photo: ![photo](a.png) here, plus words.\n",
        encoding="utf-8",
    )
    assert router.import_mgr.import_bundle() == ["pic"]
    assert len(_assets(router)) == 1
    assert _edges(router) == 1


def test_replace_image_safe_delete(env):
    tmp, router = env["tmp"], env["router"]
    md = Path(tmp) / "pic.md"
    (Path(tmp) / "a.png").write_bytes(PNG_1x1)
    md.write_text(
        "---\ntitle: Pic\n---\nFirst caption: ![old](a.png) here.\n",
        encoding="utf-8",
    )
    assert router.import_mgr.import_bundle() == ["pic"]
    before = _assets(router)
    assert len(before) == 1

    # New caption, same file: content hash differs -> delete-then-create.
    md.write_text(
        "---\ntitle: Pic\n---\nSecond caption: ![new](a.png) here.\n",
        encoding="utf-8",
    )
    assert router.import_mgr.import_bundle() == ["pic"]
    after = _assets(router)
    assert len(after) == 1
    assert _edges(router) == 1
    assert "new" in (after[0]["caption"] or "")
