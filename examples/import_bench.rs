//! Timing smoke for the filters-corpus import (BUG-040).
//!
//! Measures a cold import and an unchanged re-import against a temporary v10
//! DB. The second pass must be a delta no-op; the checker uses these numbers
//! as a regression baseline, not as a strict cross-machine SLA.

use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use serde_json::{json, Map, Value};
use std::time::Instant;

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "parity/corpus/filters".into());
    let dir = tempfile::tempdir().expect("temp dir");
    let db = dir.path().join("import-bench.db");
    Router::init(&db, 512).expect("init");

    let mut params = Map::new();
    params.insert("db_path".into(), json!(db.to_string_lossy()));
    params.insert("bundle_root".into(), json!(corpus));
    params.insert("model_id".into(), json!("google/embeddinggemma-2"));
    params.insert("device".into(), json!("cuda"));
    params.insert("precision".into(), json!("auto"));
    params.insert("enable_chunking".into(), json!(true));
    params.insert("chunk_size".into(), json!(32));
    params.insert("chunk_overlap".into(), json!(4));
    let settings = Settings::from_map(&params).expect("settings");
    let router = Router::open_writable(&settings).expect("open");
    let import_params =
        Map::from_iter([("bundle_path".to_string(), Value::String(corpus.clone()))]);

    let started = Instant::now();
    let first = router
        .call("import_bundle", &import_params)
        .expect("first import");
    let first_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let second = router
        .call("import_bundle", &import_params)
        .expect("second import");
    let second_seconds = started.elapsed().as_secs_f64();

    let concepts = first
        .get("concept_ids")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let delta = second.get("delta").cloned().unwrap_or(Value::Null);
    let queued = delta
        .get("queued")
        .and_then(Value::as_i64)
        .unwrap_or_default();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "corpus": corpus,
            "concepts": concepts,
            "first_seconds": first_seconds,
            "second_seconds": second_seconds,
            "second_queued": queued,
        }))
        .expect("json")
    );
    router.cleanup_write_lock_file();
}
