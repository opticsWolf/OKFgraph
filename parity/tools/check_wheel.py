#!/usr/bin/env python3
"""Smoke-test the built wheel from a clean environment."""
from __future__ import annotations

import threading
import time

import okfgraph
from okfgraph import _native


def main() -> int:
    print("module:", okfgraph.__file__)
    print("version:", okfgraph.version())
    print("schema_version:", okfgraph.schema_version())
    names = okfgraph.op_names()
    print("op_names:", len(names), names[:3])

    # Two 150 ms native sleeps should overlap if allow_threads releases the GIL.
    start = time.perf_counter()
    threads = [threading.Thread(target=_native.sleep_ms, args=(150,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    print(f"two native sleeps: {elapsed:.3f}s")
    if elapsed >= 0.27:
        raise SystemExit("FAIL: native call did not release the GIL")
    print("OK: wheel imports and native calls release the GIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
