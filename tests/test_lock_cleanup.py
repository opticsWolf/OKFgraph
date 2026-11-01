"""WS4: the write-lock file ``<db>.lock`` is removed when nothing holds it.

fasteners releases the OS lock on exit but leaves the file behind. The router
now unlinks it on close, unless another process is holding the lock.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from okfgraph import OKFRouter


def _lock_of(db: Path) -> Path:
    return db.with_suffix(db.suffix + ".lock")


def _open(db: Path, kb: Path) -> OKFRouter:
    return OKFRouter(db_path=str(db), bundle_root=str(kb))


def test_open_and_close_leave_no_lock_file(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    db = tmp_path / "kb.db"
    r = _open(db, kb)
    with r._write_lock_ctx():  # a write creates the file (fasteners makes it on acquire)
        pass
    assert _lock_of(db).exists()
    r.close()
    assert not _lock_of(db).exists()


def test_close_leaves_the_file_when_another_process_holds_the_lock(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    db = tmp_path / "kb.db"
    r = _open(db, kb)
    holder_src = (
        "import sys, time\n"
        "from fasteners import InterProcessLock\n"
        "lock = InterProcessLock(sys.argv[1])\n"
        "assert lock.acquire(blocking=True)\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    holder = subprocess.Popen([sys.executable, "-c", holder_src, str(_lock_of(db))],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        r.close()
        assert _lock_of(db).exists()
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_doctor_still_reports_a_planted_stale_lock(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    db = tmp_path / "kb.db"
    _lock_of(db).write_text("")  # leftover from a crash: nothing holds it
    r = _open(db, kb)
    try:
        info = r.doctor()["report"]["info"]
        assert any(f["rule"] == "stale_lock" for f in info)
    finally:
        r.close()
    assert not _lock_of(db).exists()
