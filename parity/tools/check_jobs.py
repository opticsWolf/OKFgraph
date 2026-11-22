#!/usr/bin/env python3
"""HTTP job lane smoke against the Rust `okf serve` owner."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def wait_terminal(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = client.job(job_id)
        if row["status"] in ("done", "failed", "cancelled"):
            return row
        time.sleep(0.1)
    raise SystemExit(f"job {job_id} did not finish: {row}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="target/debug/okf.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10-import-minimal.db")
    ap.add_argument("--listen", default="127.0.0.1:8787")
    args = ap.parse_args()
    from okfgraph.client import OKFClient, serve_file_for
    from okfgraph.errors import OKFError

    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = ROOT / exe
    db = ROOT / "parity/out/jobs-smoke.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / args.fixture, db)
    disc = serve_file_for(db)
    if disc.exists():
        disc.unlink()
    env = dict(os.environ, OKFGRAPH_JOB_TEST_DELAY_MS="400")
    proc = subprocess.Popen(
        [str(exe), "--db-path", str(db), "serve", "--listen", args.listen],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = None
        for _ in range(100):
            if disc.exists():
                url = json.loads(disc.read_text(encoding="utf-8"))["url"]
                break
            if proc.poll() is not None:
                raise SystemExit(f"serve exited: {proc.stderr.read()}")
            time.sleep(0.1)
        assert url, "serve did not write its discovery file"
        client = OKFClient(url)

        bad_mode = client._request("POST", "/ops/reembed", {"mode": "bogus"})
        assert bad_mode["ok"] is False, bad_mode
        assert bad_mode["error"]["code"] == "BAD_VALUE", bad_mode

        not_a_job = client._request("POST", "/jobs/search", {"query": "alpha"})
        assert not_a_job["ok"] is False, not_a_job
        assert not_a_job["error"]["code"] == "BAD_VALUE", not_a_job
        assert not_a_job["error"]["fields"]["job_ops"] == ["reembed"], not_a_job

        job_id = client.reembed_job(dry_run=True)
        queued = client.job(job_id)
        assert queued["elapsed"] == 0.0, queued
        try:
            client.reembed_job(dry_run=True)
        except OKFError as err:
            assert err.code == "REEMBED_IN_PROGRESS", err.code
        else:
            raise SystemExit("a second reembed job was accepted")

        assert len(job_id) == 12 and all(c in "0123456789abcdef" for c in job_id), job_id
        listed = [row for row in client.jobs() if row["id"] == job_id]
        assert listed and listed[0]["kwargs"] == {"dry_run": True}, listed
        assert "mode" not in listed[0]["kwargs"], listed
        row = wait_terminal(client, job_id)
        assert row["status"] == "done", row
        assert row["error"] is None, row
        assert row["result"]["dry_run"] is True, row
        assert row["finished"] is not None and row["elapsed"] >= 0, row

        # A terminal job is not restarted by a late cancel.
        assert client.cancel_job(job_id)["status"] == "done"

        second = client._request("POST", "/jobs/reembed", {"dry_run": True})
        second_id = second["data"]["job"]
        cancelling = client.cancel_job(second_id)
        assert cancelling["status"] == "cancelling", cancelling
        # A dry_run reembed has no batch boundary, so the op cannot observe
        # the cancel; the terminal status follows its own report like Python.
        finished = wait_terminal(client, second_id)
        assert finished["status"] == "done", finished
        assert finished["result"]["dry_run"] is True, finished

        try:
            client.job("does-not-exist")
        except OKFError as err:
            assert err.code == "BAD_VALUE", err.code
            assert err.fields["job"] == "does-not-exist"
        else:
            raise SystemExit("unknown job did not raise")
        print("OK: HTTP jobs submit, validate mode, dedupe, complete and cancel")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
