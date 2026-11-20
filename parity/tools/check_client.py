#!/usr/bin/env python3
"""Public OKFClient smoke against the Rust `okf serve` owner."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="target/debug/okf.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10.db")
    args = ap.parse_args()
    from okfgraph.client import OPS, OKFClient, discover, serve_file_for
    from okfgraph.errors import OKFError

    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = ROOT / exe
    db = ROOT / "parity/out/client-smoke.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(ROOT / args.fixture, db)
    disc = serve_file_for(db)
    if disc.exists():
        disc.unlink()
    proc = subprocess.Popen(
        [str(exe), "--db-path", str(db), "serve"],
        cwd=ROOT,
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
        health = client.health()
        assert health["pid"] == proc.pid, health
        assert len(health["ops"]) == 30
        assert discover(db) == url
        concept = client.get_by_id("alpha")
        assert concept["id"] == "alpha"
        assert client.call("get_by_id", concept_id="alpha")["id"] == "alpha"
        assert client.jobs() == []
        try:
            client.read(concept_id="missing")
        except OKFError as err:
            assert err.code == "UNKNOWN_CONCEPT", err.code
        else:
            raise SystemExit("unknown concept did not raise")
        assert len(OPS) == 30

        # The Rust CLI must borrow the live owner instead of opening the DB.
        cli = subprocess.run(
            [str(exe), "--json", "--db-path", str(db), "read", "alpha"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert cli.returncode == 0, cli.stderr
        assert json.loads(cli.stdout)["data"]["id"] == "alpha", cli.stdout

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda _: client.get_by_id("alpha"), range(4)))
        assert [row["id"] for row in rows] == ["alpha"] * 4
        print("OK: OKFClient talks to the Rust serve owner")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # A crashed owner leaves a discovery file behind; discover() must
        # reject it rather than handing clients a dead URL.
        if url:
            disc.write_text(json.dumps({
                "url": url, "pid": 999999999, "db_path": str(db),
            }), encoding="utf-8")
            try:
                assert discover(db) is None
            finally:
                disc.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
