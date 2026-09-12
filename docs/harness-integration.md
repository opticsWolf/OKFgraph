# Harness integration (Codex / Claude Code / pi)

OKFgraph exposes its graph two ways. Prefer **MCP** (typed tools, no
prompt engineering); fall back to the **CLI** for scripts and debugging.

## MCP server

```bash
uv run --project . okf-mcp --db-path ./kb.db --bundle .
```

- Transport: stdio (default). Entry point: `okf-mcp` (`okfgraph.mcp_server:main`).
- Dependencies resolve from `pyproject.toml` (`uv sync`); the Rust
  `okf-embed` wheel builds from `rust/okf-embed` via `[tool.uv.sources]`.
- The `pdf` extra (`bobine`) is needed only for `ingest_pdf` with the
  default converter; the `omni` extra only for image embeddings.
- Optional: `ORT_DYLIB_PATH` to pin the ONNX Runtime binary shared by
  bobine + okf-embed (defaults to the pip-installed `onnxruntime==1.29.0`).

### Claude Code

Project-local `.mcp.json` is checked in — Claude Code picks it up
automatically (runs with the project root as cwd):

```json
{
  "mcpServers": {
    "okfgraph": {
      "command": "uv",
      "args": ["run", "--project", ".", "okf-mcp",
               "--db-path", "./kb.db", "--bundle", "."]
    }
  }
}
```

### Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.okfgraph]
command = "uv"
args = ["run", "--project", "D:/path/to/OKFgraph", "okf-mcp",
        "--db-path", "D:/path/to/OKFgraph/kb.db",
        "--bundle", "D:/path/to/OKFgraph"]
```

### pi / generic harnesses

Any harness that speaks MCP over stdio: command `uv`, same args as above
with absolute paths. No auth, no headers, no sidecars.

### Skills

- `skills/okfgraph-mcp/SKILL.md` (`okfgraph-mcp`) — the MCP skill: when to reach
  for the graph and which tool to use. Install when `okf-mcp` is wired.
- `skills/okfgraph-cli/SKILL.md` (`okfgraph-cli`) — the CLI skill: same
  routing table as shell commands, plus cold-boot batching conventions.
  Install for shell-only environments. The two descriptions cross-reference
  so only the applicable one triggers.

## CLI fallback

`uv run --project . okf --help` — the same five verbs (`search`, `read`,
`traverse`, `ingest`, `export`) plus maintenance commands (`init`, `import`,
`shell`, `reindex`, `broken-links`, ...). Useful when MCP is unavailable
or for shell scripting.

## Tool surface (5 MCP tools)

- `search` (concepts/chunks/images, expand, hub_rerank),
  `read` (body/chunks/document/context),
  `traverse` (relationships, directory listing, shortest path),
  `ingest` (md/pdf/thoughts), `export_bundle`.
All tools carry descriptions, JSON schemas, and read-only/destructive
annotations — harnesses can gate writes on those.
