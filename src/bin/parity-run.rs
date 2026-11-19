//! Rust runner for the differential harness.
use okfgraph::error::Envelope;
use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use serde_json::{Map, Value};
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
    let scenario: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad scenario {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let settings_map: Map<String, Value> = scenario
        .get("settings")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let settings = match Settings::from_map(&settings_map) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("bad settings: {}", err.titles());
            return ExitCode::from(2);
        }
    };
    let router = match Router::open(&settings) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("cannot open router: {}", err.titles());
            return ExitCode::from(1);
        }
    };

    let mut results = Vec::new();
    for call in scenario
        .get("ops")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let op = call.get("op").and_then(Value::as_str).unwrap_or_default();
        let params = call
            .get("params")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        match router.call(op, &params) {
            Ok(data) => results.push(Envelope::success(op, data)),
            Err(err) => results.push(Envelope::failure(op, &err)),
        }
    }
    let out = serde_json::json!({
        "runner": "rust",
        "scenario": scenario.get("name").cloned().unwrap_or_default(),
        "ops": results
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    ExitCode::SUCCESS
}
