#!/usr/bin/env python3
"""Smoke-test the built wheel from a clean environment."""
from __future__ import annotations

import sys
import threading
import time

import okfgraph
from okfgraph import _native


def main() -> int:
    print("module:", okfgraph.__file__)
    print("version:", okfgraph.version())
    print("schema_version:", okfgraph.schema_version())
    names = tuple(okfgraph.op_names())
    expected = (
        "search", "read", "traverse", "embed", "ingest", "export_bundle",
        "export_concept", "import_bundle", "import_file", "import_files",
        "sync", "update", "supersede", "delete", "doctor", "diff", "reindex",
        "reembed", "pin_info", "lanes", "list_broken_links", "repair_links",
        "list_deleted", "recover_deleted", "purge_deleted", "detach",
        "list_images", "get_image", "get_by_id", "list_directory",
    )
    print("op_names:", len(names), names[:3])
    assert callable(_native.run_cli) and callable(_native.run_mcp)
    forbidden = {"ladybug", "numpy", "mcp", "fasteners", "yaml", "PIL"}
    assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)
    from importlib.util import find_spec

    assert find_spec("okfgraph.components") is None
    assert find_spec("okfgraph.ops") is None
    from okfgraph import mcp_server

    assert callable(mcp_server.main)
    if names != expected:
        raise SystemExit(f"FAIL: op_names drift: {names!r} != {expected!r}")

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
