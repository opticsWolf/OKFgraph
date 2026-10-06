"""cache_hygiene(): legacy snapshot-only repo detection (read-only).

All tests build fake hub roots under tmp_path — the real
~/.cache/huggingface/hub is never walked here. A final doctor test
proves the wiring: the hygiene message appears as info, never as a
finding, and the score stays 100.
"""

import os

from okfgraph.components.embedding import EmbeddingEngine as EE


def _modern(root, owner="o", name="n", payload=b"x" * 100):
    base = os.path.join(root, f"models--{owner}--{name}")
    os.makedirs(os.path.join(base, "blobs"))
    snap = os.path.join(base, "snapshots", "abc123")
    os.makedirs(snap)
    with open(os.path.join(base, "blobs", "deadbeef"), "wb") as f:
        f.write(payload)
    with open(os.path.join(snap, "model.onnx"), "wb") as f:
        f.write(payload)
    os.makedirs(os.path.join(base, "refs"))
    with open(os.path.join(base, "refs", "main"), "w") as f:
        f.write("abc123")
    return base


def _legacy(root, owner="o", name="old", payload=b"y" * 64):
    base = os.path.join(root, f"models--{owner}--{name}")
    snap = os.path.join(base, "snapshots", "def456")
    os.makedirs(snap)
    with open(os.path.join(snap, "model.onnx"), "wb") as f:
        f.write(payload)
    return base, len(payload)


def test_modern_layout_not_flagged(tmp_path):
    hub = str(tmp_path)
    _modern(hub)
    rep = EE.cache_hygiene(cache_dir=hub)
    assert rep == {"cache_dir": hub, "scanned": 1, "legacy": [],
                   "reclaimable_bytes": 0}


def test_legacy_snapshot_only_flagged_with_bytes(tmp_path):
    hub = str(tmp_path)
    _, n = _legacy(hub)
    rep = EE.cache_hygiene(cache_dir=hub)
    assert rep["scanned"] == 1
    assert rep["legacy"] == [{"repo": "o/old", "bytes": n}]
    assert rep["reclaimable_bytes"] == n


def test_empty_blobs_dir_counts_as_legacy(tmp_path):
    hub = str(tmp_path)
    base, n = _legacy(hub, name="emptyblobs")
    os.makedirs(os.path.join(base, "blobs"))
    rep = EE.cache_hygiene(cache_dir=hub)
    assert [r["repo"] for r in rep["legacy"]] == ["o/emptyblobs"]


def test_snapshots_without_files_not_flagged(tmp_path):
    hub = str(tmp_path)
    base = os.path.join(hub, "models--o--partial")
    os.makedirs(os.path.join(base, "snapshots", "abc"))  # dirs only, no files
    rep = EE.cache_hygiene(cache_dir=hub)
    assert rep["legacy"] == []


def test_non_model_entries_ignored_and_missing_root_empty(tmp_path):
    hub = str(tmp_path)
    open(os.path.join(hub, "stray.txt"), "w").write("x")
    os.makedirs(os.path.join(hub, "tmp_download"))
    rep = EE.cache_hygiene(cache_dir=hub)
    assert rep == {"cache_dir": hub, "scanned": 0, "legacy": [],
                   "reclaimable_bytes": 0}
    rep = EE.cache_hygiene(cache_dir=os.path.join(hub, "nope"))
    assert rep["legacy"] == [] and rep["scanned"] == 0


def test_walk_is_read_only(tmp_path):
    hub = str(tmp_path)
    base, _ = _legacy(hub)
    target = os.path.join(base, "snapshots", "def456", "model.onnx")
    before = (os.path.getmtime(target), open(target, "rb").read())
    EE.cache_hygiene(cache_dir=hub)
    assert (os.path.getmtime(target), open(target, "rb").read()) == before


class _StubResult:
    def rows_as_dict(self):
        return self

    def get_all(self):
        return []


class _StubConn:
    def execute(self, *a, **k):
        return _StubResult()


def test_doctor_hygiene_message_and_score(tmp_path):
    from okfgraph.components.doctor import DoctorManager

    hub = str(tmp_path)
    _, n = _legacy(hub, owner="acme", name="big-old")
    _modern(hub, owner="acme", name="current")
    rep = DoctorManager(_StubConn(), object()).diagnose(cache_dir=hub)
    by_rule = {i["rule"]: i["message"] for i in rep["info"]}
    msg = by_rule["cache_hygiene"]
    assert "look unused by cache-mode" in msg
    assert "acme/big-old" in msg
    assert "acme/current" not in msg
    assert "safe to delete" in msg
    assert "re-downloads once" in msg
    assert rep["findings"] == []
    assert rep["score"] == 100


def test_doctor_hygiene_clean_bill(tmp_path):
    from okfgraph.components.doctor import DoctorManager

    hub = str(tmp_path)
    _modern(hub)
    rep = DoctorManager(_StubConn(), object()).diagnose(cache_dir=hub)
    by_rule = {i["rule"]: i["message"] for i in rep["info"]}
    assert "none look unused by cache-mode" in by_rule["cache_hygiene"]
