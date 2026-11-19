//! Rust runner for the differential harness.
use okfgraph::error::Envelope;
use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::{env, fs, process::ExitCode};

fn collect_markdown(dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_markdown(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("md") {
            out.push(path.to_string_lossy().into_owned());
        }
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
    if let Some(imported) = scenario.get("setup").and_then(|value| value.get("import")) {
        let corpus = imported
            .get("corpus")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let db = PathBuf::from(&settings.db_path);
        let _ = fs::remove_file(&db);
        if let Err(err) = Router::init(&db, settings.embedding_dim) {
            eprintln!("cannot init import db: {}", err.titles());
            return ExitCode::from(1);
        }
        let mut builder = settings.clone();
        builder.bundle_root = Some(corpus.to_string());
        let mut paths = Vec::new();
        collect_markdown(Path::new(corpus), &mut paths);
        paths.sort();
        if paths.is_empty() {
            eprintln!("no markdown files under {corpus}");
            return ExitCode::from(1);
        }
        let files = paths;
        match Router::open_writable(&builder) {
            Ok(writer) => {
                let params = serde_json::json!({"paths": files});
                if let Err(err) = writer.call("import_files", params.as_object().unwrap()) {
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
