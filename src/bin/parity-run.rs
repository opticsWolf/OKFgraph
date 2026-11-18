//! Skeleton Rust runner for the differential harness.
use std::{env, fs, process::ExitCode};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: parity-run <scenario.json>");
            return ExitCode::from(2);
        }
    };
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let scenario: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad scenario {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let out = serde_json::json!({
        "ok": true,
        "runner": "rust",
        "scenario": scenario.get("name").cloned().unwrap_or_default(),
        "core_version": okfgraph::version(),
        "schema_version": okfgraph::schema_version(),
        "ops": [],
        "note": "M0 skeleton; op execution lands with M1",
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    ExitCode::SUCCESS
}
