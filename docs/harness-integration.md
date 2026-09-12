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

### Skill

`skills/okfgraph/SKILL.md` tells an agent *when* to reach for the graph
(search vs traverse vs ingest). Copy/symlink it into the harness's skill
dir (Claude Code: `.claude/skills/`).

## CLI fallback

`uv run --project . okf --help` — 25 commands mirroring the MCP tools
(`search`, `search-chunks`, `traverse`, `get`, `ingest`, `context`,
`hub-search`, `path`, `reconstruct`, ...). Useful when MCP is unavailable
or for shell scripting.

## Tool surface (16 MCP tools)

Read: `search_hybrid`, `search_chunks`, `search_with_context`,
`search_chunks_with_hub_score`, `search_images`, `traverse`, `find_path`,
`get_by_id`, `get_chunks`, `reconstruct_document`, `list_directory`,
`expand_with_graph_context`, `export_bundle`.
Write: `ingest_md`, `ingest_thoughts`, `ingest_pdf`.
All tools carry descriptions, JSON schemas, and read-only/destructive
annotations — harnesses can gate writes on those.
