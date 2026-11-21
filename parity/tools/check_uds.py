#!/usr/bin/env python3
"""Public OKFClient over a Unix-domain `okf serve` owner (POSIX only)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--serve-exe",
        default="target/debug/okf.exe" if os.name == "nt" else "target/debug/okf",
    )
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10.db")
    args = ap.parse_args()
    if args.python and Path(args.python).resolve() != Path(sys.executable).resolve():
        os.execv(args.python, [args.python, str(Path(__file__).resolve()), *sys.argv[1:]])
    if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
        print("SKIP: UDS transport is POSIX-only")
        return 0

    exe = ROOT / args.serve_exe
    db = ROOT / "parity/out/uds-smoke.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / args.fixture, db)
    sock = ROOT / "parity/out/uds-smoke.sock"
    for path in (sock, Path(str(db) + ".serve.json")):
        if path.exists():
            path.unlink()

    owner = subprocess.Popen(
        [str(exe), "--db-path", str(db), "serve", "--listen", f"unix://{sock}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = None
        discovery = Path(str(db) + ".serve.json")
        for _ in range(100):
            if discovery.exists():
                info = json.loads(discovery.read_text(encoding="utf-8"))
                url = info["url"]
                break
            if owner.poll() is not None:
                raise SystemExit(f"serve exited: {owner.stderr.read()}")
            time.sleep(0.1)
        assert url and url.startswith("unix://"), url
        assert sock.exists(), sock

        import okfgraph
        from okfgraph import OKFClient
        from okfgraph.errors import OKFError

        assert hasattr(okfgraph, "OKFClient")
        client = OKFClient(url, timeout=10)
        health = client.health()
        assert health["pid"] == owner.pid, health
        assert health["ops"] and len(health["ops"]) == 30, health
        concept = client.get_by_id("alpha")
        assert concept["id"] == "alpha", concept
        try:
            client.read("missing")
        except OKFError as err:
            assert err.code == "UNKNOWN_CONCEPT", err
        else:
            raise AssertionError("missing concept did not raise OKFError")
        print("OK: OKFClient talks to okf serve over a Unix socket")
        return 0
    finally:
        owner.terminate()
        try:
            owner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owner.kill()
        for path in (sock, Path(str(db) + ".serve.json")):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
