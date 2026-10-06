"""converter_status(): guarded aggregation of bobine.model_status().

Live case runs against the real venv (bobine 0.6.0 installed); guard
cases simulate missing/old bobine via sys.modules patching. Doctor
wiring is covered with a stub-conn diagnose on an empty fake hub root
(hermetic: no real-cache walk, offline probe only).
"""

import sys
import types
from unittest.mock import patch

import pytest

from okfgraph.components.converters import converter_status

bobine = pytest.importorskip("bobine")

REQUIRED_KEYS = {"repo", "cached", "snapshot_path", "disk_usage_bytes"}


def test_live_available_shape():
    if not hasattr(bobine, "model_status"):
        pytest.skip("bobine <0.6.0 in this venv")
    st = converter_status()
    assert st["available"] is True
    assert st["note"] is None
    assert len(st["reports"]) >= 4
    for r in st["reports"]:
        assert REQUIRED_KEYS <= set(r), r
        assert "/" in r["repo"]
        if r["cached"]:
            assert r["snapshot_path"], r
            assert r["disk_usage_bytes"] > 0, r


def test_missing_bobine_is_data_not_error():
    with patch.dict(sys.modules, {"bobine": None}):
        st = converter_status()
    assert st == {"available": False, "reports": [],
                  "note": st["note"]}
    assert "not installed" in st["note"]


def test_old_bobine_names_the_floor():
    stub = types.ModuleType("bobine")  # no model_status attr
    with patch.dict(sys.modules, {"bobine": stub}):
        st = converter_status()
    assert st["available"] is False
    assert st["reports"] == []
    assert ">=0.6.0" in st["note"]


class _StubResult:
    def rows_as_dict(self):
        return self

    def get_all(self):
        return []


class _StubConn:
    def execute(self, *a, **k):
        return _StubResult()


def test_doctor_lists_converters_as_info(tmp_path):
    from okfgraph.components.doctor import DoctorManager

    empty_hub = tmp_path / "hub"
    empty_hub.mkdir()
    rep = DoctorManager(_StubConn(), object()).diagnose(cache_dir=str(empty_hub))
    by_rule = {i["rule"]: i["message"] for i in rep["info"]}
    assert "converter_cache" in by_rule
    # Empty hub: offline probe finds nothing cached, downloads nothing.
    assert "0/4 families cached" in by_rule["converter_cache"]
    assert rep["findings"] == []
    assert rep["score"] == 100
