"""Wire-level MCP stdio test: the real `okf-mcp` entry point as a child process.

`tests/test_mcp_server.py` covers the registry in-process; this file proves
the transport agents actually use — raw JSON-RPC over stdio pipes. Any stray
stdout print, lifespan mis-ordering, or argv-plumbing bug fails here and
nowhere else.

Model-free by construction: `initialize`, `tools/list`, and a root-listing
`traverse` call touch no encoder, so this runs in CI (added to the fast
list) with no model cache and no network.
"""
import json
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mcp.types import LATEST_PROTOCOL_VERSION


class StdioClient:
    """Minimal JSON-RPC-over-stdio client (newline-delimited, like the SDK)."""

    def __init__(self, argv, cwd):
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(cwd),
        )
        self._incoming: queue.Queue[str] = queue.Queue()
        self._stderr: list[str] = []
        self._id = 0
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self):
        try:
            for line in self.proc.stdout:
                # Every stdout line must be a JSON-RPC message — a stray
                # print here corrupts the stream for real clients, so fail
                # loudly instead of skipping.
                self._incoming.put(line)
        finally:
            self._incoming.put(None)

    def _pump_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr.append(line)
        except ValueError:
            pass

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout=120):
        try:
            line = self._incoming.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                "no JSON-RPC reply within "
                f"{timeout}s; server stderr tail:\n" + "".join(self._stderr[-20:])
            )
        assert line is not None, (
            "server closed stdout; stderr tail:\n" + "".join(self._stderr[-20:])
        )
        return json.loads(line)

    def call(self, method, params=None, timeout=120):
        self._id += 1
        rid = self._id
        message = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        reply = self.recv(timeout)
        assert reply.get("id") == rid, reply
        assert "error" not in reply, reply["error"]
        return reply["result"]

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def close(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def wire(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mcp_stdio")
    client = StdioClient(
        [
            sys.executable,
            "-m",
            "okfgraph.mcp_server",
            "--db-path",
            str(tmp / "wire.db"),
            "--bundle-root",
            str(tmp),
        ],
        cwd=Path(__file__).resolve().parent.parent,
    )
    # The server answers with the highest protocol version IT supports,
    # which may lag the newest SDK constant — offer latest, accept the
    # negotiation, don't pin it.
    try:
        result = client.call(
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "okfgraph-wire-test", "version": "0"},
            },
        )
    except Exception:
        client.close()
        raise
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", result["protocolVersion"]), result
    assert result["serverInfo"]["name"] == "OKFgraph MCP Server", result
    client.notify("notifications/initialized")
    yield client
    client.close()


def test_wire_lists_five_tools(wire):
    result = wire.call("tools/list")
    names = [t["name"] for t in result["tools"]]
    assert names == ["search", "read", "traverse", "ingest", "export_bundle"]


def test_wire_traverse_root_listing(wire):
    """Root listing over the wire — model-free, proves tools/call works."""
    result = wire.call("tools/call", {"name": "traverse", "arguments": {"start_id": ""}})
    assert not result.get("isError", False), result
    payload = json.loads(result["content"][0]["text"])
    assert isinstance(payload, list)


def test_wire_unknown_tool_errors_cleanly(wire):
    # Unknown tools surface as a result with isError (not a JSON-RPC
    # error) — either shape must be explicit, never a hang or garbage.
    result = wire.call(
        "tools/call", {"name": "nope", "arguments": {}}
    )
    assert result.get("isError", False) is True, result
