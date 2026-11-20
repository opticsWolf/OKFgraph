//! Thin MCP binary shim: all tool logic lives in `okfgraph::mcp`.

fn main() {
    // clap expects argv[0], exactly like `cli::run` and the PyO3 shim.
    let argv = std::env::args().collect();
    std::process::exit(okfgraph::mcp::run(argv));
}
