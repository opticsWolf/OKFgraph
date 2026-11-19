//! Rust runner for the differential harness.
use okfgraph::error::Envelope;
use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use serde_json::{Map, Value};
use std::path::PathBuf;
use std::{env, fs, process::ExitCode};

fn expand_runner(value: Value, runner: &str) -> Value {
    match value {
        Value::String(text) => Value::String(text.replace("@RUNNER@", runner)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| expand_runner(item, runner))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, item)| (key, expand_runner(item, runner)))
                .collect(),
        ),
        other => other,
    }
}

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
    let settings_map: Map<String, Value> = expand_runner(
        scenario
            .get("settings")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
            .into(),
        "rust",
    )
    .as_object()
    .cloned()
    .unwrap_or_default();
    let settings = match Settings::from_map(&settings_map) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("bad settings: {}", err.titles());
            return ExitCode::from(2);
        }
    };
    if let Some(imported) = scenario.get("setup").and_then(|value| value.get("import")) {
        let corpus = imported
            .get("corpus")
            .and_then(Value::as_str)
            .map(|value| expand_runner(Value::String(value.to_string()), "rust"))
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_default();
        let db = PathBuf::from(&settings.db_path);
        let _ = fs::remove_file(&db);
        if let Err(err) = Router::init(&db, settings.embedding_dim) {
            eprintln!("cannot init import db: {}", err.titles());
            return ExitCode::from(1);
        }
        let mut builder = settings.clone();
        builder.bundle_root = Some(corpus.to_string());
        match Router::open_writable(&builder) {
            Ok(writer) => {
                let params = imported.as_object().cloned().unwrap_or_default();
                let params = serde_json::Value::Object(params);
                if let Err(err) = writer.call("import_bundle", params.as_object().unwrap()) {
                    eprintln!("import failed: {}", err.titles());
                    return ExitCode::from(1);
                }
            }
            Err(err) => {
                eprintln!("cannot open import db: {}", err.titles());
                return ExitCode::from(1);
            }
        }
    }
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
        let params = expand_runner(call.get("params").cloned().unwrap_or(Value::Null), "rust");
        let params = params.as_object().cloned().unwrap_or_default();
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
