"""WS5.3: every search result row carries its owning concept under ``id``.

Rows about a sub-part keep their own key (``chunk_id``, ``asset_id``) and also
carry ``concept_id``. One parametrized test covers every target and rank.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from okfgraph.router import OKFRouter

FIX = Path(__file__).parent / "fixtures" / "ppr_bundle"
DIM = 512
CONCEPTS = {"honey-badger", "lion", "orphan", "savanna"}
OWNER = "honey-badger"

CASES = [
    pytest.param({"target": "concepts", "rank": "none"}, id="concepts-none"),
    pytest.param({"target": "concepts", "rank": "hub"}, id="concepts-hub"),
    pytest.param({"target": "concepts", "rank": "ppr"}, id="concepts-ppr"),
    pytest.param({"target": "concepts", "rank": "none", "include_chunks": True},
                 id="concepts-none-chunks"),
    pytest.param({"target": "chunks"}, id="chunks-plain"),
    pytest.param({"target": "chunks", "hub_rerank": True}, id="chunks-hub"),
    pytest.param({"target": "chunks", "expand": True}, id="chunks-expand"),
    pytest.param({"target": "images"}, id="images"),
]


@pytest.fixture(scope="module")
def router(tmp_path_factory):
    r = OKFRouter(
        db_path=str(tmp_path_factory.mktemp("stable_keys") / "keys.db"),
        bundle_root=str(FIX),
        embedding_dim=DIM,
        chunk_size=50,
        chunk_overlap=10,
        enable_chunking=True,
        device="cpu",
    )
    r.import_mgr.import_bundle(FIX)
    vec = [0.0] * DIM
    vec[0] = 1.0
    img = SimpleNamespace(asset_id="img_cat", filename="cat.png",
                          mime_type="image/png", alt_text="a cat", data=b"x")
    r.image_mgr._upsert_image_asset(OWNER, {
        "img": img, "caption": "a cat", "route": "text",
        "content_hash": "h-cat", "embedding": vec,
    })
    yield r
    r.close()


def _sub_rows(rows, case):
    """The rows about a sub-part (chunk or image), wherever the key sits."""
    if case.get("expand"):
        return [r["chunk"] for r in rows]
    if case.get("include_chunks"):
        return [c for r in rows for c in r.get("matched_chunks", [])]
    if case["target"] in ("chunks", "images"):
        return rows
    return []


@pytest.mark.parametrize("case", CASES)
def test_every_row_carries_owning_concept_id(router, case):
    rows = router.search(query="honey badger", limit=5, **case)
    assert rows, f"no rows for {case}"
    for row in rows:
        assert row["id"] in CONCEPTS, row
    for sub in _sub_rows(rows, case):
        assert sub["concept_id"] == sub["id"] in CONCEPTS, sub
        if case["target"] == "images":
            assert sub["asset_id"] == "img_cat", sub
        else:
            assert sub["chunk_id"], sub
            assert sub["chunk_id"] != sub["id"], sub


def test_image_row_keeps_asset_id_under_its_own_key(router):
    rows = router.search(query="cat", target="images", limit=5)
    assert rows[0]["id"] == OWNER
    assert rows[0]["concept_id"] == OWNER
    assert rows[0]["asset_id"] == "img_cat"


def test_chunk_row_keeps_chunk_id_and_owner(router):
    rows = router.search(query="honey badger", target="chunks", limit=5)
    assert rows[0]["chunk_id"] != rows[0]["id"]
    assert rows[0]["parent_doc_id"] == rows[0]["id"] == rows[0]["concept_id"]
