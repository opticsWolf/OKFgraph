"""0.11.0 acceptance: one owner process, many clients (O4) and offline
search extensions (O5).

O4: with ``okf serve`` running, an ``okf-mcp --connect`` client and the CLI
both succeed against the same database, and concurrent writes serialise.
O5: with ``extension_dir`` set, hybrid search works without any download.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from okfgraph import OKFRouter
from okfgraph.client import OKFClient, discover, serve_file_for
from okfgraph.errors import OKFError
from okfgraph.serve import parse_listen, serve
from okfgraph.settings import Settings

from tests._v011 import note, stub_encoder, write  # noqa: F401  (fixture import)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# O4 — okf serve
# ---------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path, stub_encoder):
    """``okf serve`` on a Unix socket, run on a thread in this process."""
    kb = tmp_path / "kb"
    write(kb, "a.md", note("Alpha", "alpha zebra"))
    db = tmp_path / "kb.db"
    sock = tmp_path / "okf.sock"
    settings = Settings(db_path=str(db), bundle_root=str(kb))
    ready, stop = threading.Event(), threading.Event()
    t = threading.Thread(target=serve, args=(settings, f"unix://{sock}"),
                         kwargs={"ready": ready, "stop": stop}, daemon=True)
    t.start()
    assert ready.wait(60), "okf serve did not start"
    try:
        yield {"url": f"unix://{sock}", "db": db, "kb": kb, "tmp": tmp_path}
    finally:
        stop.set()
        t.join(60)


def test_o4_client_mirrors_the_router_and_errors_cross_typed(server):
    c = OKFClient(server["url"])
    assert c.health()["db_path"] == str(server["db"].resolve())
    assert c.import_bundle()["concept_ids"] == ["a"]
    assert [h["id"] for h in c.search("zebra")] == ["a"]
    assert [h["id"] for h in c.search("zebra", rank="ppr")] == ["a"]
    with pytest.raises(OKFError) as e:
        c.read("nope")
    assert e.value.code == "UNKNOWN_CONCEPT"
    assert e.value.fields == {"concept": "nope"}
    with pytest.raises(OKFError) as e:
        c.call("init")  # closing the owner's router is not an op
    assert e.value.code == "BAD_VALUE"


def test_o4_concurrent_writes_serialise(server):
    c = OKFClient(server["url"])
    errors = []

    def write_one(i):
        try:
            c.ingest(kind="thoughts", thoughts=f"thought {i} about quokkas",
                     topic="q", concept_id=f"thoughts/q/{i}")
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == []
    hits = c.search("quokkas", limit=20, rank="ppr")
    assert sorted(h["id"] for h in hits) == [f"thoughts/q/{i}" for i in range(8)]


def test_o4_discovery_and_cli_go_through_the_owner(server):
    db = server["db"]
    assert discover(db) == server["url"]
    OKFClient(server["url"]).import_bundle()
    env = dict(os.environ, PYTHONPATH=str(REPO))
    cmd = [sys.executable, "-m", "okfgraph.cli", "search", "alpha", "--rank", "ppr",
           "--db-path", str(db), "--bundle-root", str(server["kb"]), "--json"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180,
                         cwd=str(server["tmp"]))
    assert out.returncode == 0, out.stderr[-800:]
    assert [h["id"] for h in json.loads(out.stdout)["data"]] == ["a"]
    # Opting out opens the file directly — and meets the owner's lock.
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                         env=dict(env, OKFGRAPH_NO_SERVER="1"), cwd=str(server["tmp"]))
    assert out.returncode != 0 and "DB_LOCKED" in out.stderr


def test_o4_mcp_connect_proxies_to_the_owner(server):
    from tests.test_mcp_stdio import StdioClient
    from mcp.types import LATEST_PROTOCOL_VERSION

    OKFClient(server["url"]).import_bundle()
    proxy = StdioClient([sys.executable, "-m", "okfgraph.mcp_server",
                         "--connect", server["url"]], cwd=REPO)
    try:
        proxy.call("initialize", {"protocolVersion": LATEST_PROTOCOL_VERSION,
                                  "capabilities": {},
                                  "clientInfo": {"name": "t", "version": "0"}})
        proxy.notify("notifications/initialized")
        names = [t["name"] for t in proxy.call("tools/list")["tools"]]
        assert {"search", "update", "supersede", "delete"} <= set(names)
        res = proxy.call("tools/call", {"name": "search", "arguments":
                                        {"query": "alpha", "rank": "ppr"}})
        payload = json.loads(res["content"][0]["text"])
        assert payload["ok"] is True and [h["id"] for h in payload["data"]] == ["a"]
    finally:
        proxy.close()


def test_o4_serve_file_lifecycle_and_stale_files_are_ignored(tmp_path, stub_encoder):
    db = tmp_path / "x.db"
    sf = serve_file_for(db)
    sf.write_text(json.dumps({"url": "unix:///nonexistent.sock", "pid": 999999,
                              "db_path": str(db)}))
    assert discover(db) is None  # dead pid: never trusted
    sf.write_text(json.dumps({"url": "unix:///nonexistent.sock", "pid": os.getpid(),
                              "db_path": str(db)}))
    assert discover(db) is None  # live pid but no /health answer
    assert parse_listen("127.0.0.1:8765")[3] == "http://127.0.0.1:8765"
    assert parse_listen("unix:///tmp/okf.sock")[2] == "/tmp/okf.sock"
    with pytest.raises(OKFError):
        parse_listen("localhost")


def test_o4_serve_removes_its_discovery_file_on_shutdown(tmp_path, stub_encoder):
    kb = tmp_path / "kb"
    kb.mkdir()
    db = tmp_path / "kb.db"
    ready, stop = threading.Event(), threading.Event()
    t = threading.Thread(target=serve, args=(Settings(db_path=str(db), bundle_root=str(kb)),
                                             f"unix://{tmp_path / 's.sock'}"),
                         kwargs={"ready": ready, "stop": stop}, daemon=True)
    t.start()
    assert ready.wait(60)
    assert serve_file_for(db).is_file()
    stop.set()
    t.join(60)
    assert not serve_file_for(db).exists()
    # The owner released the file: a direct router opens it again.
    OKFRouter(db_path=str(db), bundle_root=str(kb)).close()


# ---------------------------------------------------------------------------
# O5 — offline extensions
# ---------------------------------------------------------------------------

def _cached_extensions():
    from okfgraph.components.extensions import find_extension, ladybug_cache_dir
    cache = ladybug_cache_dir()
    if all(find_extension(cache, e) for e in ("vector", "fts")):
        return cache
    return None


def test_o5_extension_dir_gives_hybrid_search_without_download(tmp_path, stub_encoder):
    cache = _cached_extensions()
    if cache is None:
        pytest.skip("Ladybug extensions not in the local cache")
    target = tmp_path / "ext"
    res = OKFRouter.setup_extensions(target=target, source=cache, download=False)
    assert res["ok"] is True and res["verified"] == {"vector": "ok", "fts": "ok"}
    assert f'extension_dir = "{target.as_posix()}"' in res["toml"]

    kb = tmp_path / "kb"
    write(kb, "a.md", note("Alpha", "alpha zebra"))
    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  extension_dir=str(target))
    try:
        r.import_bundle()
        assert [h["id"] for h in r.search("zebra")] == ["a"]
        assert not any(f["rule"] == "search_unavailable"
                       for f in r.doctor()["report"]["findings"])
    finally:
        r.close()


def test_o5_missing_extension_is_a_doctor_error_not_a_download(tmp_path, stub_encoder):
    empty = tmp_path / "empty-ext"
    empty.mkdir()
    kb = tmp_path / "kb"
    write(kb, "a.md", note("Alpha", "alpha zebra"))
    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  extension_dir=str(empty))
    try:
        r.import_bundle()
        assert "extension_dir" in r.schema_mgr.extension_errors["vector"]
        with pytest.raises(OKFError) as e:
            r.search("zebra")
        assert e.value.code == "SEARCH_UNAVAILABLE"
        assert [h["id"] for h in r.search("zebra", rank="ppr")] == ["a"]
        report = r.doctor()["report"]
        found = [f for f in report["findings"] if f["rule"] == "search_unavailable"]
        assert found and found[0]["severity"] == "error"
        assert report["score"] <= 80
    finally:
        r.close()


def test_o5_setup_reports_what_is_missing(tmp_path):
    with pytest.raises(OKFError) as e:
        OKFRouter.setup_extensions(target=tmp_path / "t", source=tmp_path, download=False)
    assert e.value.code == "FILE_NOT_FOUND"
    assert "vector" in str(e.value) and "fts" in str(e.value)


def test_o5_settings_refuse_a_missing_extension_dir(tmp_path):
    errors, _warnings = Settings(db_path="x.db",
                                 extension_dir=str(tmp_path / "nope")).checks()
    assert any("extension_dir" in e for e in errors)
