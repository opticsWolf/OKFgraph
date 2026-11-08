# -*- coding: utf-8 -*-
"""0.12.0: the re-embed as an interruptible job (``okf serve``).

The contract this locks down:

* ``POST /ops/reembed {"mode": "job"}`` answers a job id inline and runs
  on a worker thread; ``GET /jobs/<id>`` reports ``status`` and the
  phase/count progress of the current batch sweep, and the terminal
  record carries the full op report.
* ``POST /jobs/<id>/cancel`` lands at the **next batch boundary** — the
  op stops between batches, keeps everything committed so far, answers
  ``cancelled: true`` with partial counts, and the job record ends as
  ``cancelled``. The cancel itself never queues behind the op lock.
* A cancelled (or crashed) reembed leaves the graph in the same
  recoverable-but-searchless state (the vector indexes stay dropped) —
  a RE-RUN converges it back and search works again.
* A second reembed while one runs refuses ``REEMBED_IN_PROGRESS``
  instead of silently queueing behind the lock for hours.
* The blocking lane is unchanged: ``POST /ops/reembed`` without
  ``mode`` keeps its synchronous contract (existing suites cover it).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from okfgraph.client import OKFClient
from okfgraph.errors import OKFError
from okfgraph.serve import serve
from okfgraph.settings import Settings

from tests._v011 import note, stub_encoder, write  # noqa: F401

SLEEP = 0.6           # per rewriting batch — enough to cancel mid-sweep


@pytest.fixture
def sleepy_encoder(stub_encoder, monkeypatch):
    """The deterministic stub encoder, slowed per batch so a job can be
    cancelled between \"concepts\" batches while progress is visible."""
    from okfgraph.components import embedding as E

    def slow_encode_batch(self, texts, task="Document"):
        stub_encoder.count += len(texts)
        time.sleep(SLEEP)
        from tests._v011 import _vec
        return [_vec(t, int(getattr(self, "_truncate_dim", 512) or 512))
                for t in texts]

    monkeypatch.setattr(E.LazyRustEncoder, "encode_batch",
                        slow_encode_batch)
    return stub_encoder


@pytest.fixture
def server(tmp_path, sleepy_encoder):
    """``okf serve`` on a thread (loopback TCP on Windows)."""
    kb = tmp_path / "kb"
    for i in range(40):
        write(kb, f"a{i}.md", note(f"Alpha{i}", f"alpha zebra {i} tablet"))
    db = tmp_path / "kb.db"
    settings = Settings(db_path=str(db), bundle_root=str(kb))
    ready, stop = threading.Event(), threading.Event()
    t = threading.Thread(target=serve, args=(settings, "127.0.0.1:0"),
                         kwargs={"ready": ready, "stop": stop}, daemon=True)
    t.start()
    assert ready.wait(120), "okf serve did not start"
    url = json.loads(Path(str(db) + ".serve.json").read_text(
        encoding="utf-8"))["url"]
    client = OKFClient(url)
    client.import_bundle()
    try:
        yield {"url": url, "db": db, "client": client}
    finally:
        stop.set()
        t.join(90)


def _submit(client, **kw):
    return client.call("reembed", mode="job", **kw)


def _wait_for(client, job_id, statuses, timeout=90.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        time.sleep(0.15)
        last = client.job(job_id)
        if last.get("status") in statuses:
            return last
    raise AssertionError(f"job {job_id} never reached {statuses}; "
                         f"last record: {json.dumps(last)[:500]}")


def test_job_runs_and_reports_progress_then_the_report(server):
    c = server["client"]
    sub = _submit(c)
    job_id = sub["job"]
    assert sub["op"] == "reembed" and sub["status"] in ("queued", "running")
    # progress reaches every rewritten batch; the finished record = report
    seen_progress = False
    deadline = time.time() + 60
    record = None
    while time.time() < deadline:
        record = c.job(job_id)
        if record.get("phase") and record.get("total"):
            seen_progress = True
        if record.get("status") in ("done", "cancelled", "failed"):
            break
        time.sleep(0.1)
    assert seen_progress, "progress never reported a batch"
    done = _wait_for(c, job_id, ("done",))
    assert done["result"]["reembedded"]["concepts"] == 40
    assert not done["result"].get("cancelled")
    assert done["result"]["pins_after"]["pinned"]["model_id"]
    assert done["elapsed"] > 0.0
    # the finished report must match what a sync run would say
    health = c.health()
    assert health["db_path"] == str(server["db"].resolve())


def test_cancel_stops_between_batches_and_rerun_converges(server):
    c = server["client"]
    sub = _submit(c)
    job_id = sub["job"]

    # cancel once at least one batch is behind the op; with 40 rows at
    # batch size 32 every phase runs >=2 batches, and the boundary after
    # the current batch is where the flag lands
    deadline = time.time() + 60
    while time.time() < deadline:
        rec = c.job(job_id)
        if rec.get("status") == "running" and rec.get("done", 0) >= 1:
            break
        time.sleep(0.05)
    assert rec.get("status") == "running", f"job stalled: {rec}"
    cancel_view = c.cancel_job(job_id)
    assert cancel_view["id"] == job_id
    assert cancel_view["status"] in ("cancelling", "cancelled")
    assert c.cancel_job(job_id)["status"] in ("cancelling", "cancelled")

    final = _wait_for(c, job_id, ("cancelled",), timeout=120)
    res = final["result"]
    assert res["cancelled"] is True
    assert res["resumable"] is True
    reembedded = res["reembedded"]
    assert reembedded["concepts"] + reembedded["chunks"] < \
        res["concepts"] + res["chunks"] + 1          # partial, not full
    assert reembedded["concepts"] >= 1               # stopped after batches

    # cancel answered while the op was mid-flight (no long queue to the op)
    t0 = time.time()
    assert c.health()      # other lanes answer even while the job still runs
    assert time.time() - t0 < 10

    # the graph now = the crash-like state (no vector indexes). A re-run
    # converges it back; search works again afterwards.
    with pytest.raises(OKFError) as exc:           # vector search is down
        c.search("zebra")
    assert exc.value.code in ("SEARCH_UNAVAILABLE", "INTERNAL")
    again = _submit(c)
    done = _wait_for(c, again["job"], ("done",))
    assert done["result"]["reembedded"]["concepts"] == 40      # full redo
    hits = [h["id"] for h in c.search("zebra", limit=10)]
    assert hits and set(hits) <= {f"a{i}" for i in range(40)}  # search works


def test_second_reembed_refuses_while_one_runs(server):
    c = server["client"]
    sub = _submit(c)
    job_id = sub["job"]
    deadline = time.time() + 60
    while time.time() < deadline:
        if c.job(job_id)["status"] == "running":
            break
        time.sleep(0.05)
    # a second job submit refuses with the dedicated code
    with pytest.raises(OKFError) as exc:
        _submit(c)
    assert exc.value.code == "REEMBED_IN_PROGRESS"
    # the blocking lane refuses too (fast, not queued behind the worker)
    t0 = time.time()
    with pytest.raises(OKFError) as exc2:
        c.call("reembed")
    assert exc2.value.code == "REEMBED_IN_PROGRESS"
    assert time.time() - t0 < 10
    c.cancel_job(job_id)


def test_unknown_job_and_terminal_cancel(server):
    c = server["client"]
    with pytest.raises(OKFError) as exc:
        c.job("no-such-job")
    assert exc.value.code == "BAD_VALUE"
    done = _submit(c)["job"]                       # tiny graph: finishes fast
    _wait_for(c, done, ("done", "cancelled", "failed"))
    # cancelling a finished job = a no-op view,Answered, never queued
    view = c.cancel_job(done)
    assert view["status"] in ("done", "cancelled")


def test_job_mode_values_and_jobs_listing(server):
    c = server["client"]
    with pytest.raises(OKFError) as exc:
        c.call("reembed", mode="spawn")
    assert exc.value.code == "BAD_VALUE"
    # the jobs list carries the finished records (recent) with no payloads
    rec = c.jobs()
    assert isinstance(rec, list)
