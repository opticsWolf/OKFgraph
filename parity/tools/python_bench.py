#!/usr/bin/env python3
"""Paired performance benchmark body for the Python 0.15 oracle."""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import oracle_root  # noqa: E402

sys.path.insert(0, str(oracle_root()))


def peak_rss_kb() -> int:
    """Peak working set / high-water mark in KiB, without psutil."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize // 1024) if ok else 0

    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def main() -> int:
    from okfgraph import OKFRouter

    corpus = sys.argv[1] if len(sys.argv) > 1 else "parity/corpus/filters"
    with tempfile.TemporaryDirectory(prefix="okf-perf-") as tmp:
        db = Path(tmp) / "perf-bench.db"
        router = OKFRouter(
            db_path=str(db),
            bundle_root=corpus,
            model_id="google/embeddinggemma-2",
            device="cuda",
            precision="auto",
            embedding_dim=512,
            enable_chunking=True,
            chunk_size=32,
            chunk_overlap=4,
        )
        try:
            started = time.perf_counter()
            first = router.import_bundle(bundle_path=corpus)
            first_seconds = time.perf_counter() - started

            started = time.perf_counter()
            second = router.import_bundle(bundle_path=corpus)
            second_seconds = time.perf_counter() - started

            router.embed(texts=["alpha links"], task="Query")
            embed_samples = []
            for _ in range(5):
                started = time.perf_counter()
                router.embed(texts=["alpha links"], task="Query")
                embed_samples.append(time.perf_counter() - started)

            warm = router.search("alpha links", target="concepts", limit=3)
            result_count = len(warm)
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                router.search("alpha links", target="concepts", limit=3)
                samples.append(time.perf_counter() - started)
        finally:
            router.close()

    delta = second.get("delta") or {}
    report = {
        "runner": "python",
        "corpus": corpus,
        "concepts": len(first.get("concept_ids") or []),
        "first_seconds": first_seconds,
        "second_seconds": second_seconds,
        "second_queued": int(delta.get("queued") or 0),
        "search_results": result_count,
        "embed_median_seconds": statistics.median(embed_samples),
        "embed_samples_seconds": embed_samples,
        "search_median_seconds": statistics.median(samples),
        "search_samples_seconds": samples,
        "peak_rss_kb": peak_rss_kb(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
