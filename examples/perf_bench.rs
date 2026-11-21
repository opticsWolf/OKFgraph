//! Paired performance benchmark body for the Rust core.
//!
//! Measures cold import, unchanged re-import, warm search latency and peak
//! RSS against the same corpus/settings used by `parity/tools/python_bench.py`.
//! Timing thresholds are smoke gates, not cross-machine SLAs.

use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use serde_json::{json, Map, Value};
use std::time::Instant;
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

fn search_params_for_query_vector(vector: &[f32]) -> Map<String, Value> {
    Map::from_iter([
        ("query".to_string(), Value::String("alpha links".into())),
        ("target".to_string(), Value::String("concepts".into())),
        ("limit".to_string(), Value::from(3)),
        (
            "query_vector".to_string(),
            Value::Array(vector.iter().map(|value| json!(*value)).collect()),
        ),
    ])
}

fn peak_rss_kb() -> u64 {
    let Ok(pid) = get_current_pid() else {
        return 0;
    };
    let mut processes = System::new();
    processes.refresh_processes(ProcessesToUpdate::Some(&[pid]));
    processes
        .process(pid)
        .map(|process| process.memory() / 1024)
        .unwrap_or(0)
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "parity/corpus/filters".into());
    let dir = tempfile::tempdir().expect("temp dir");
    let db = dir.path().join("perf-bench.db");
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

    let embed_params = Map::from_iter([
        (
            "texts".to_string(),
            Value::Array(vec![Value::String("alpha links".into())]),
        ),
        ("task".to_string(), Value::String("Query".into())),
    ]);
    let _ = router.call("embed", &embed_params).expect("warm embed");
    let mut embed_samples = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let _ = router.call("embed", &embed_params).expect("embed");
        embed_samples.push(started.elapsed().as_secs_f64());
    }
    let embed_median_seconds = median(&embed_samples);

    let embed_result = router.call("embed", &embed_params).expect("profile embed");
    let query_vector = embed_result
        .get("vectors")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_f64)
                .map(|value| value as f32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let search_vec_params = search_params_for_query_vector(&query_vector);
    let mut search_vec_samples = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let _ = router
            .call("search", &search_vec_params)
            .expect("search with vector");
        search_vec_samples.push(started.elapsed().as_secs_f64());
    }
    let search_vec_median_seconds = median(&search_vec_samples);

    let search_params = Map::from_iter([
        ("query".to_string(), Value::String("alpha links".into())),
        ("target".to_string(), Value::String("concepts".into())),
        ("limit".to_string(), Value::from(3)),
    ]);
    let warm = router.call("search", &search_params).expect("warm search");
    let result_count = warm.as_array().map(Vec::len).unwrap_or(0);

    let mut samples = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let _ = router.call("search", &search_params).expect("search");
        samples.push(started.elapsed().as_secs_f64());
    }
    let search_median_seconds = median(&samples);
    let peak_rss_kb = peak_rss_kb();

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
            "runner": "rust",
            "corpus": corpus,
            "concepts": concepts,
            "first_seconds": first_seconds,
            "second_seconds": second_seconds,
            "second_queued": queued,
            "search_results": result_count,
            "search_vec_median_seconds": search_vec_median_seconds,
            "search_vec_samples_seconds": search_vec_samples,
            "embed_median_seconds": embed_median_seconds,
            "embed_samples_seconds": embed_samples,
            "search_median_seconds": search_median_seconds,
            "search_samples_seconds": samples,
            "peak_rss_kb": peak_rss_kb,
        }))
        .expect("json")
    );
    router.cleanup_write_lock_file();
}
