#!/usr/bin/env python3
"""Generate Python oracle goldens for frontmatter and chunking."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"__type__": "float", "value": str(value)}
    if isinstance(value, (datetime, date)):
        return {"__type__": "datetime" if isinstance(value, datetime) else "date", "value": value.isoformat()}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def main() -> int:
    import frontmatter
    import yaml
    spec = importlib.util.spec_from_file_location("oracle_embedding", ROOT / "okfgraph" / "components" / "embedding.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bodies = {
        "simple": "# Title\n\nAlpha paragraph with [[beta]].\n\n- one\n- two\n\n> quoted\n",
        "long": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen\n\n" + "x " * 40,
        "code": "Intro.\n\n```rust\nfn main() {}\n```\n\nEnd.\n",
        "unicode": "# Ünïcødé\n\nEmoji 🧪 and Cyrillic Привет.\n",
    }
    chunks = {}
    for name, body in bodies.items():
        engine = mod.EmbeddingEngine.__new__(mod.EmbeddingEngine)
        engine.chunk_size = 5
        chunks[name] = {
            "body": body,
            "chunk_size": 5,
            "chunks": engine._split_into_chunks(body, name, count_tokens=None),
        }

    yaml_cases = {
        "yes_no": "a: yes\nb: no\nc: on\nd: off\n",
        "nulls": "a: null\nb: ~\nc:\n",
        "numbers": "a: 0x10\nb: 0o17\nc: 1_000\nd: 1e3\ne: .inf\n",
        "dates": "a: 2024-01-02\nb: 2024-01-02T03:04:05Z\n",
        "quoted": "a: 'yes'\nb: \"no\"\nc: 'a: b'\n",
        "nested": "a: [1, two, false]\nb: {x: 1, y: yes}\n",
        "duplicate": "a: 1\na: 2\n",
    }
    frontmatter_cases = {
        "basic": "---\nid: alpha\ntitle: Alpha\ntags: [parity, minimal]\n---\n\nBody.\n",
        "typed": "---\nenabled: yes\ncount: 0x10\nwhen: 2024-01-02\n---\n\nBody.\n",
    }
    goldens = {
        "chunks": chunks,
        "yaml": {name: {"input": text, "value": encode(yaml.safe_load(text))} for name, text in yaml_cases.items()},
        "frontmatter": {
            name: {
                "input": text,
                "metadata": encode(frontmatter.loads(text).metadata),
                "content": frontmatter.loads(text).content,
            }
            for name, text in frontmatter_cases.items()
        },
    }
    out = ROOT / "parity" / "golden" / "markdown.json"
    out.write_text(json.dumps(goldens, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
