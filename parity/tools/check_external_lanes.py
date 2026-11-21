#!/usr/bin/env python3
"""Differential check for the external embed/rerank/vision HTTP lanes.

Starts three local mock hosts, runs the same scenario through the Python 0.15
oracle and the Rust parity runner, and compares envelopes plus the resulting
DB dump. The mocks also assert the wire contracts (identity probe, bearer key,
task prefix, tokenize, rerank payload, image bytes) that BUG-013 requires.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import oracle_root  # noqa: E402
from run_matrix import (  # noqa: E402
    DB_IGNORE_KEYS,
    INFO_IGNORE_RULES,
    OPS_IGNORE_KEYS,
    expand,
    run,
)

API_KEY_ENV = "OKFGRAPH_EXTERNAL_TEST_KEY"
API_KEY = "local-test-key"
TEXT_SPACE = (
    "embroider/google/embeddinggemma-2@"
    "296b2a67735a025ed90e2967d58aba6a41233535:"
    "fp16:512:8192:text_embed:v1"
)
IMAGE_SPACE = (
    "embroider/google/embeddinggemma-2-vision@"
    "oracle-revision:fp16:512:0:vision:v1"
)

REQUESTS: list[dict[str, Any]] = []


def vector(seed: str, dim: int) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values = [(byte - 127.5) / 127.5 for byte in digest]
    return [round(values[index % len(values)], 6) for index in range(dim)]


class Handler(BaseHTTPRequestHandler):
    lane = "text"
    tei_only = False

    def log_message(self, *_args: Any) -> None:
        pass

    def _headers_ok(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {API_KEY}"

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        REQUESTS.append({
            "lane": self.lane,
            "method": "GET",
            "path": self.path,
            "auth": self._headers_ok(),
        })
        if self.path == "/v1/models":
            if self.lane == "text":
                row = {
                    "engine": "embroider", "kind": "embed",
                    "id": "google/embeddinggemma-2", "family": "gemma-2",
                    "dim": 512, "max_length": 8192, "precision": "fp16",
                    "space": TEXT_SPACE, "revision":
                    "296b2a67735a025ed90e2967d58aba6a41233535",
                    "contract": "text_embed", "device": "cuda",
                    "engine_version": "0.15.1",
                }
            elif self.lane.startswith("rerank"):
                row = {
                    "engine": "embroider", "kind": "rerank",
                    "id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                    "family": "mmarco", "dim": 384, "max_length": 512,
                    "precision": "fp32", "device": "cpu",
                    "engine_version": "0.15.1",
                }
            else:
                row = {
                    "engine": "embroider", "kind": "vision",
                    "id": "google/embeddinggemma-2-vision",
                    "dim": 512, "max_length": 8192, "precision": "fp16",
                    "space": IMAGE_SPACE,
                    "text_partner": "google/embeddinggemma-2",
                    "text_partner_space": TEXT_SPACE,
                    "revision": "oracle-revision", "contract": "vision",
                    "device": "cuda", "engine_version": "0.15.1",
                }
            self._send({"data": [row]})
        else:
            self._send({"error": "no route"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._body()
        REQUESTS.append({
            "lane": self.lane,
            "method": "POST",
            "path": self.path,
            "auth": self._headers_ok(),
            "payload": payload,
        })
        if self.path == "/tokenize":
            content = str(payload.get("content") or "")
            self._send({"count": len(content.split()), "tokens": content.split()})
        elif self.path == "/v1/embeddings":
            inputs = payload.get("input") or []
            dim = int(payload.get("dimensions") or 512)
            rows = [
                {"embedding": vector(f"{payload.get('task', 'Document')}|{text}", dim)}
                for text in inputs
            ]
            self._send({"data": rows})
        elif self.path == "/v1/rerank" and self.tei_only:
            self._send({"error": "use TEI /rerank"}, 404)
        elif self.path == "/v1/rerank":
            docs = payload.get("documents") or []
            rows = [
                {
                    "index": index,
                    "relevance_score": round(1.0 / (index + 1) + len(doc) / 10000.0, 6),
                }
                for index, doc in enumerate(docs)
            ]
            self._send({"results": rows})
        elif self.path == "/rerank" and self.lane.startswith("rerank"):
            docs = payload.get("texts") or []
            self._send([
                {
                    "index": index,
                    "score": round(1.0 / (index + 1) + len(doc) / 10000.0, 6),
                }
                for index, doc in enumerate(docs)
            ])
        elif self.path == "/v1/images/embeddings":
            images = payload.get("image") or []
            rows = [
                {"embedding": vector(base64.b64decode(value).decode("utf-8", "replace"), 512)}
                for value in images
            ]
            self._send({"data": rows})
        else:
            self._send({"error": "no route"}, 404)


def serve(lane: str, tei_only: bool = False) -> ThreadingHTTPServer:
    handler = type(
        f"{lane.title()}Handler", (Handler,),
        {"lane": lane, "tei_only": tei_only},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def make_gguf_fixture(path: Path) -> None:
    """Write a small valid GGUF header with a 1000-token vocabulary."""
    tokens = [chr(code) for code in range(32, 127)]
    words = [
        "retrieval", "augmented", "generation", "grounds", "answer",
        "evidence", "knowledge", "graph", "concept", "title", "chunk",
        "vector", "space", "cosine", "similarity", "nearest", "neighbours",
        "matryoshka", "truncation", "declared", "contract", "prefixes",
        "search_query:", "search_document:", "mean", "pooling", "batch",
        "budget", "memory", "tokens", "2024", "768", "1024", "4096",
    ]
    tokens.extend(words)
    tokens.extend(f"piece{index} " for index in range(len(tokens), 1000))

    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    key = "tokenizer.ggml.tokens"
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    body += string(key)
    body += struct.pack("<IIQ", 9, 8, len(tokens))  # array of strings
    body += b"".join(string(token) for token in tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def run_scenario(
    name: str,
    scenario: dict[str, Any],
    env: dict[str, str],
    oracle: str,
    *,
    compare_db: bool,
    out: Path,
) -> bool:
    """Run one generated scenario through both runners and compare envelopes."""
    scenario_path = ROOT / f"parity/tmp/{name}.json"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    py_out = out / f"{name}.python.json"
    rust_out = out / f"{name}.rust.json"

    py = run([
        oracle, "parity/tools/run_python.py",
        "--scenario", str(scenario_path.relative_to(ROOT)),
        "--out", str(py_out.relative_to(ROOT)),
    ], env, capture=True)
    if py.returncode:
        print(name, "python runner:", py.stderr[-4000:])
        return False
    rust = run([
        "cargo", "run", "-q", "-p", "okfgraph", "--bin", "parity-run", "--",
        str(scenario_path.relative_to(ROOT)),
    ], env, capture=True)
    if rust.returncode:
        print(name, "rust runner:", rust.stderr[-4000:])
        return False
    rust_out.write_text(rust.stdout, encoding="utf-8")

    cmd = [
        sys.executable, "parity/tools/compare.py",
        "--python", str(py_out), "--rust", str(rust_out),
    ]
    for key in OPS_IGNORE_KEYS:
        cmd += ["--ignore-key", key]
    for rule in INFO_IGNORE_RULES:
        cmd += ["--ignore-info-rule", rule]
    result = run(cmd, env, capture=True)
    if result.returncode:
        print(name, "envelope diff:", result.stdout)
        return False

    if compare_db:
        db_path = (scenario.get("settings") or {}).get("db_path")
        py_db = ROOT / expand(db_path, "python")
        rust_db = ROOT / expand(db_path, "rust")
        dumps: list[Path] = []
        for db, suffix, base in [
            (py_db, "python", [oracle, "parity/tools/dump_db.py"]),
            (rust_db, "rust", ["cargo", "run", "-q", "-p", "okfgraph", "--example", "dump_db", "--"]),
        ]:
            target = out / f"{name}.{suffix}-dump.json"
            if suffix == "python":
                dumped = run(base + ["--db", str(db), "--out", str(target)], env, capture=True)
            else:
                dumped = run(base + [str(db)], env, capture=True)
                if dumped.returncode == 0:
                    target.write_text(dumped.stdout, encoding="utf-8")
            if dumped.returncode:
                print(name, "dump:", dumped.stderr[-4000:])
                return False
            dumps.append(target)
        db_cmd = [
            sys.executable, "parity/tools/compare_db.py",
            "--python", str(dumps[0]), "--rust", str(dumps[1]),
        ]
        for key in DB_IGNORE_KEYS:
            db_cmd += ["--ignore-key", key]
        result = run(db_cmd, env, capture=True)
        if result.returncode:
            print(name, "DB diff:", result.stdout)
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-python", default="parity/.venv-oracle/Scripts/python.exe")
    args = ap.parse_args()

    servers = {
        "text": serve("text"),
        "rerank": serve("rerank"),
        "vision": serve("vision"),
        "rerank_tei": serve("rerank_tei", tei_only=True),
    }
    try:
        text_url = f"http://127.0.0.1:{servers['text'].server_port}"
        rerank_url = f"http://127.0.0.1:{servers['rerank'].server_port}"
        vision_url = f"http://127.0.0.1:{servers['vision'].server_port}"
        tei_url = f"http://127.0.0.1:{servers['rerank_tei'].server_port}"
        env = os.environ.copy()
        env.update({
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1",
            "OKFGRAPH_ORACLE_ROOT": str(oracle_root()),
            API_KEY_ENV: API_KEY,
        })
        out = ROOT / "parity/out"
        out.mkdir(parents=True, exist_ok=True)
        oracle = str(ROOT / args.oracle_python)

        base_settings: dict[str, Any] = {
            "db_path": "parity/out/external-lanes-@RUNNER@.db",
            "bundle_root": "parity/corpus/images",
            "model_id": "google/embeddinggemma-2",
            "embedding_dim": 512,
            "precision": "auto",
            "embed_backend": "external",
            "endpoint_url": text_url,
            "api_key_env": API_KEY_ENV,
            "embed_identity": "auto",
            "enable_chunking": False,
        }
        setup = {
            "copy_file": {
                "from": "parity/fixtures/oracle-v10-import-minimal.db",
                "to": "parity/out/external-lanes-@RUNNER@.db",
            }
        }
        scenario: dict[str, Any] = {
            "name": "external-lanes-mock",
            "settings": {
                **base_settings,
                "rerank_backend": "external",
                "rerank_endpoint_url": rerank_url,
                "rerank_api_key_env": API_KEY_ENV,
                "rerank_model_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                "rerank_identity": "auto",
                "image_backend": "external",
                "image_endpoint_url": vision_url,
                "image_precision": "auto",
            },
            "setup": setup,
            "ops": [
                {"op": "embed", "params": {
                    "texts": ["red fox", "city lights"], "task": "Document", "dim": 64,
                }},
                {"op": "import_bundle", "params": {
                    "bundle_path": "parity/corpus/images", "mode": "omni",
                }},
                {"op": "list_images", "params": {"concept_id": "red-fox"}},
                {"op": "search", "params": {
                    "query": "red fox", "target": "chunks", "rerank": True, "limit": 3,
                }},
                {"op": "pin_info", "params": {}},
            ],
        }
        if not run_scenario(
            "external-lanes-mock", scenario, env, oracle, compare_db=True, out=out
        ):
            return 1

        # Wire assertions for the primary scenario: both runners contacted
        # all three hosts with the declared bearer key and the expected body.
        assert all(row["auth"] for row in REQUESTS), REQUESTS
        lanes = {row["lane"] for row in REQUESTS}
        assert {"text", "rerank", "vision"} <= lanes, lanes
        embeds = [
            row for row in REQUESTS
            if row["method"] == "POST" and row["path"] == "/v1/embeddings"
        ]
        assert len(embeds) >= 4, len(embeds)  # two runners x two calls
        assert all(row["payload"].get("task") in ("Document", "Query") for row in embeds)
        # The identity-backed host declares only native width, so dim=64 is
        # refused before any wire call; the declared-ladder route is covered
        # by Rust unit tests.
        assert all("dimensions" not in row["payload"] for row in embeds)
        assert any(row["path"] == "/tokenize" for row in REQUESTS)
        reranks = [row for row in REQUESTS if row["path"] == "/v1/rerank"]
        assert len(reranks) >= 2 and all(
            set(row["payload"]) >= {"model", "query", "documents", "top_n"}
            for row in reranks
        )
        images = [
            row for row in REQUESTS if row["path"] == "/v1/images/embeddings"
        ]
        assert len(images) >= 2 and all(row["payload"].get("image") for row in images)

        # TEI fallback: the standard route is refused, then the host-root
        # bare-list wire answers. Both runners must latch the same dialect.
        tei_before = len(REQUESTS)
        tei_scenario = {
            "name": "external-rerank-tei",
            "settings": {
                **base_settings,
                "rerank_backend": "external",
                "rerank_endpoint_url": tei_url,
                "rerank_api_key_env": API_KEY_ENV,
                "rerank_model_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                "rerank_identity": "auto",
            },
            "setup": setup,
            "ops": [
                {"op": "search", "params": {
                    "query": "alpha links", "target": "chunks",
                    "rerank": True, "limit": 3,
                }},
            ],
        }
        if not run_scenario(
            "external-rerank-tei", tei_scenario, env, oracle,
            compare_db=False, out=out,
        ):
            return 1
        tei_requests = REQUESTS[tei_before:]
        assert all(row["auth"] for row in tei_requests), tei_requests
        standard = [row for row in tei_requests if row["path"] == "/v1/rerank"]
        fallback = [row for row in tei_requests if row["path"] == "/rerank"]
        assert len(standard) >= 2 and len(fallback) >= 2, (standard, fallback)
        assert all(
            set(row["payload"]) >= {"query", "texts", "truncate"}
            and row["payload"]["truncate"] is True
            for row in fallback
        )

        # GGUF counter: a valid header with a 1000-token vocabulary must be
        # accepted by both runners and report the same effective counter.
        gguf = ROOT / "parity/tmp/external-tokens.gguf"
        make_gguf_fixture(gguf)
        gguf_before = len(REQUESTS)
        gguf_scenario = {
            "name": "external-gguf-counter",
            "settings": {
                **base_settings,
                "embed_identity": "declared",
                "embed_family": "gguf-test",
                "embed_counter": "gguf-meta",
                "embed_counter_gguf": str(gguf.relative_to(ROOT)),
            },
            "setup": setup,
            "ops": [
                {"op": "embed", "params": {
                    "texts": ["alpha links"], "task": "Document",
                }},
            ],
        }
        if not run_scenario(
            "external-gguf-counter", gguf_scenario, env, oracle,
            compare_db=False, out=out,
        ):
            return 1
        gguf_requests = REQUESTS[gguf_before:]
        assert all(row["auth"] for row in gguf_requests), gguf_requests
        assert not any(row["path"] == "/tokenize" for row in gguf_requests)
        gguf_embeds = [
            row for row in gguf_requests if row["path"] == "/v1/embeddings"
        ]
        assert len(gguf_embeds) == 2, gguf_embeds
        assert all("dimensions" not in row["payload"] for row in gguf_embeds)
        for suffix in ("python", "rust"):
            body = json.loads((out / f"external-gguf-counter.{suffix}.json").read_text(encoding="utf-8"))
            assert body["ops"][0]["data"]["counter"] == "gguf-meta", body

        print(
            "OK: external lanes match the oracle over mock hosts, "
            "including TEI fallback and a real GGUF counter fixture"
        )
        return 0
    finally:
        for server in servers.values():
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
