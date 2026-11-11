# Harness integration (Codex / Claude Code / pi)

OKFgraph exposes its graph two ways. Prefer **MCP** (typed tools, no
prompt engineering); fall back to the **CLI** for scripts and debugging.

## MCP server

```bash
uv run --project . okf-mcp            # defaults: the .okfgraph/ home
uv run --project . okf-mcp --db-path ./kb.db --bundle-root .   # explicit paths win
```

- Transport: stdio (default). Entry point: `okf-mcp` (`okfgraph.mcp_server:main`).
- The database must be set explicitly: `--db-path`, `OKFGRAPH_DB_PATH`, or
  `[database] db_path` in an `okfgraph.toml` in the server's CWD (or its
  `--bundle-root`). Without one the server exits 2 with
  `[ERROR] CONFIG_INVALID` on stderr instead of silently opening a fresh
  graph. Boot flags are the CLI's global flags (same table, same env names).
- Dependencies resolve from `pyproject.toml` (`uv sync`); the embedding
  engine is the external `embroider` package (PyPI wheels, also used by
  bobine — no Rust toolchain needed to run okfgraph). The 0.12.0 default
  engine is EmbeddingGemma 2; jina-v5 stays registered for pinned graphs.
- The `pdf` extra (`bobine`) is needed only for `ingest` with `kind='pdf'`
  and the default converter. Image embeddings (`mode=optional`/`omni`) run
  through the text model's registry vision pairing (`GemmaVision` on
  embeddinggemma-2 graphs, `JinaV5Vision` on text-nano graphs) on the
  shared ONNX Runtime — install `okfgraph[cpu]` or `[gpu]` for that;
  there is no separate vision extra.
- All three lanes can run out of process (0.14): `backend=external`
  (`/v1/embeddings`), `rerank_backend=external` (`/v1/rerank`) and
  `image_backend=external` (`/v1/images/embeddings`, an
  `embroider.serve --kind vision` host). With all three set, no ONNX
  runtime and no `embroider` wheel need to exist in the harness
  process at all — that is the configuration Weave's Silk plugin runs.
- Optional: `ORT_DYLIB_PATH` to pin the ONNX Runtime binary shared by
  bobine + embroider (defaults to the pip-installed `onnxruntime==1.31.1`;
  since embroider 0.14 the crate also self-serves the CUDA loader dirs —
  venv `nvidia*` dists + `onnxruntime*` `capi` + the cuDNN SDK layouts —
  so a fresh server process needs no machine PATH edits).

### Claude Code

Project-local `.mcp.json` is checked in — Claude Code picks it up
automatically (runs with the project root as cwd):

```json
{
  "mcpServers": {
    "okfgraph": {
      "command": "uv",
      "args": ["run", "--project", ".", "okf-mcp",
               "--db-path", "./kb.db", "--bundle-root", "."]
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
        "--bundle-root", "D:/path/to/OKFgraph"]
```

### pi / generic harnesses

Any harness that speaks MCP over stdio: command `uv`, same args as above
with absolute paths. No auth, no headers, no sidecars.

### Skills

The skills ship in the wheel (`okfgraph/skills/`, 0.13.1). `okf setup
skills` copies them to `.claude/skills/<name>/SKILL.md`; `--to DIR` puts
them in another harness's folder and `--skill NAME` (repeatable) picks
some. Rerun after an upgrade to refresh changed files.

- `okfgraph/skills/okfgraph-mcp/SKILL.md` (`okfgraph-mcp`) — the MCP skill: when to reach
  for the graph and which tool to use. Install when `okf-mcp` is wired.
- `okfgraph/skills/okfgraph-cli/SKILL.md` (`okfgraph-cli`) — the CLI skill: same
  routing table as shell commands, plus cold-boot batching conventions.
  Install for shell-only environments. The two descriptions cross-reference
  so only the applicable one triggers.
- `okfgraph/skills/okfgraph-ingest/SKILL.md` (`okfgraph-ingest`) — the feeding skill:
  what to store, kind selection, converter modes, topic discipline.
  Install alongside either of the above when the agent should persist
  knowledge, not just read it.

## CLI fallback

`uv run --project . okf --help` — the same retrieval verbs (`search`, `read`,
`traverse`, `ingest`, `export`) plus image ops (`images`, `image`) and
maintenance commands (`init`, `model-info`, `import`, `diff`, `doctor`, `lint`,
`produce`, `shell`, `reindex`, `broken-links`, `repair-links`, `deleted-*`,
`delete`, `detach`). Useful when MCP is unavailable or for shell scripting. Every
command takes `--json`: the human renderer by default, the result envelope
(`{ok, op, data, warnings, error}`) on stdout under `--json`; results go to
stdout and logs to stderr (with `--json`, logs default to WARNING so
merged streams still parse; `-q` silences them entirely). Failure paths print
`[ERROR] CODE: message (remedy)` on stderr and exit 2 (usage) / 1 (state,
diff/doctor/lint outcomes — those still print their report first).
Concurrency: one process per database — when `okf-mcp` is running, query
through it; a second direct `okf` on the same file refuses with
`DB_LOCKED`. Only commands that embed text (`search` except `--rank ppr`,
`ingest`) load the model, so `read`/`traverse`/`doctor` stay fast.

## Cold search (no model load)

`search` with `rank=ppr` (MCP) / `--rank ppr` (CLI) answers from the link
graph alone — lexical seeds into exact Personalized PageRank, no ONNX load,
deterministic across runs. Route topic-naming queries there when the
embedder is cold or unavailable; keep hybrid ranking for phrasing-sensitive
questions. `read --max-tokens N` / `max_tokens` caps any reading to a
budgeted section list (self first, then PPR-ranked neighbours).

## Tool surface (11 MCP tools)

- `search` (concepts/chunks/images, expand, hub_rerank, rank),
  `read` (body/chunks/document/context, max_tokens),
  `traverse` (relationships, directory listing, shortest path),
  `ingest` (md/pdf/thoughts/bib), `export_bundle` (okf/obsidian flavors),
  `export_concept` (single concept), `list_images` (assets on a concept),
  `get_image` (metadata + base64 bytes) — and since 0.11 the write tools
  `update` (edit a concept/bump dim), `supersede` (fold into a successor),
  `delete` (remove a node or nested tree).
- Wire contract (D7): a success result is the envelope string
  (`{ok: true, op, data, warnings}`); a failure raises, so the result
  carries `isError: true` and the text is the error envelope — typed code,
  message, fields, remedy; untyped component exceptions arrive wrapped as
  `INTERNAL` with the type name (no bare tracebacks).
- Params a tool path ignores are refused with `BAD_VALUE` naming them in
  `fields.ignored` (e.g. `context_hops` without `expand`, filters on
  `target="images"`, `title` on `kind="thoughts"`), so an agent learns the
  contract from the first mistake instead of getting silently wrong results.
All tools carry descriptions, JSON schemas, and read-only/destructive
annotations — harnesses can gate writes on those.
