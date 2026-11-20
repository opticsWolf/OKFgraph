#!/usr/bin/env python3
"""Run a parity scenario against the frozen Python oracle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_error_payload(exc: BaseException, op: str) -> dict[str, Any]:
    """Wrap an oracle exception in the same envelope the Rust runner emits."""
    from okfgraph.errors import envelope

    if not hasattr(exc, "envelope"):
        # Python 0.15 lets some raw RuntimeError/FileNotFoundError escape
        # (e.g. a missing PDF converter); record it as INTERNAL instead of
        # crashing the harness.
        return {
            "ok": False,
            "op": op,
            "data": None,
            "warnings": [],
            "error": {
                "code": "INTERNAL",
                "message": str(exc),
                "fields": None,
                "remedy": None,
            },
        }
    return envelope(op, error=exc)


def _expand_runner(value: Any, runner: str) -> Any:
    if isinstance(value, str):
        return value.replace("@RUNNER@", runner)
    if isinstance(value, list):
        return [_expand_runner(item, runner) for item in value]
    if isinstance(value, dict):
        return {key: _expand_runner(item, runner) for key, item in value.items()}
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    from okfgraph import OKFRouter  # oracle import, on purpose

    settings = _expand_runner(dict(scenario.get("settings") or {}), "python")
    setup = scenario.get("setup") or {}
    copied = setup.get("copy")
    if copied:
        import shutil
        source = Path(_expand_runner(copied["from"], "python"))
        target = Path(_expand_runner(copied["to"], "python"))
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    copied_file = setup.get("copy_file")
    if copied_file:
        import shutil
        source = Path(_expand_runner(copied_file["from"], "python"))
        target = Path(_expand_runner(copied_file["to"], "python"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    imported = setup.get("import")
    if imported:
        corpus = Path(_expand_runner(imported["corpus"], "python"))
        db = Path(settings["db_path"])
        db.unlink(missing_ok=True)
        builder = dict(settings)
        builder.pop("db_path", None)
        builder.pop("bundle_root", None)
        builder.pop("mode", None)
        build_router = OKFRouter(
            db_path=str(db), bundle_root=str(corpus), **builder)
        try:
            import_params = {
                key: value
                for key, value in imported.items()
                if key != "corpus"
            }
            build_router.import_bundle(**import_params)
        finally:
            build_router.close()
    for remove in setup.get("remove") or []:
        target = Path(_expand_runner(remove, "python"))
        if target.is_dir():
            import shutil
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    router = OKFRouter(**settings)
    results: list[dict[str, Any]] = []
    try:
        for call in scenario.get("ops") or []:
            op = call["op"]
            params = call.get("params") or {}
            params = _expand_runner(params, "python")
            try:
                data = getattr(router, op)(**params)
                # Public Pydantic models are normalized so JSON comparison sees
                # their documented public shape, not repr(default=str).
                if hasattr(data, "public_dict"):
                    data = data.public_dict()
                elif hasattr(data, "model_dump"):
                    data = data.model_dump()
                results.append({
                    "ok": True,
                    "op": op,
                    "data": data,
                    "warnings": [],
                    "error": None,
                })
            except Exception as exc:  # oracle errors are part of the golden output
                results.append(load_error_payload(exc, op))
    finally:
        router.close()

    out = {"runner": "python", "scenario": scenario.get("name"), "ops": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
