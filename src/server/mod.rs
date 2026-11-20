//! axum HTTP surface for `okf serve`: `/ops/*`, `/health`, discovery file.
//!
//! Stage C starts with the blocking op plane and the discovery contract the
//! public `OKFClient` depends on. MCP over `/mcp` and the job lane are added
//! on top of the same owning router in the following M4 slice.

use crate::error::{Envelope, OkfError, Result};
use crate::ops::{Router, OP_NAMES};
use crate::settings::Settings;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use serde_json::{json, Map, Value};
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    router: Arc<Router>,
    settings: Settings,
}

fn envelope_json(
    op: &str,
    result: std::result::Result<Value, OkfError>,
) -> (StatusCode, Json<Value>) {
    match result {
        Ok(data) => (
            StatusCode::OK,
            Json(serde_json::to_value(Envelope::success(op, data)).expect("envelope")),
        ),
        Err(err) => {
            let status = match err.code.as_str() {
                "DB_LOCKED" | "WRITE_LOCK_TIMEOUT" | "REEMBED_IN_PROGRESS" => StatusCode::CONFLICT,
                "BAD_VALUE" | "MISSING_PARAM" | "FILE_NOT_FOUND" | "CONFIG_INVALID" => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::OK,
            };
            (
                status,
                Json(serde_json::to_value(Envelope::failure(op, &err)).expect("envelope")),
            )
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let db_path = std::path::absolute(&state.settings.db_path)
        .unwrap_or_else(|_| PathBuf::from(&state.settings.db_path));
    let data = json!({
        "db_path": db_path.to_string_lossy(),
        "created": false,
        "pid": std::process::id(),
        "ops": OP_NAMES,
        "lanes": state.router.lanes().unwrap_or(Value::Null),
    });
    Json(serde_json::to_value(Envelope::success("health", data)).expect("envelope"))
}

async fn list_ops() -> Json<Value> {
    Json(json!(OP_NAMES))
}

async fn list_jobs() -> Json<Value> {
    Json(serde_json::to_value(Envelope::success("jobs", json!([]))).expect("envelope"))
}

async fn unknown_job(Path(job_id): Path<String>) -> (StatusCode, Json<Value>) {
    let err = OkfError::new("BAD_VALUE", format!("unknown job '{job_id}'"))
        .field("job_id", json!(job_id));
    envelope_json("job", Err(err))
}

async fn call_op(
    State(state): State<Arc<AppState>>,
    Path(op): Path<String>,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    if !OP_NAMES.contains(&op.as_str()) {
        let err = OkfError::new("BAD_VALUE", format!("unknown op '{op}'"))
            .field("op", json!(op))
            .field("ops", json!(OP_NAMES));
        return envelope_json(&op, Err(err));
    }
    let params: Map<String, Value> = if body.is_empty() {
        Map::new()
    } else {
        match serde_json::from_slice::<Map<String, Value>>(&body) {
            Ok(map) => map,
            Err(err) => {
                return envelope_json(
                    &op,
                    Err(OkfError::new(
                        "BAD_VALUE",
                        format!("bad request body: {err}"),
                    )),
                )
            }
        }
    };
    let router = Arc::clone(&state.router);
    let op_name = op.clone();
    let result = tokio::task::spawn_blocking(move || router.call(&op_name, &params))
        .await
        .unwrap_or_else(|err| Err(OkfError::internal(err.to_string(), "join")));
    envelope_json(&op, result)
}

pub fn http_router(router: Arc<Router>, settings: Settings) -> AxumRouter {
    let state = Arc::new(AppState { router, settings });
    AxumRouter::new()
        .route("/health", get(health))
        .route("/ops", get(list_ops))
        .route("/ops/{op}", post(call_op))
        .route("/jobs", get(list_jobs))
        .route("/jobs/{id}", get(unknown_job))
        .route("/jobs/{id}/cancel", post(unknown_job))
        .with_state(state)
}

pub fn discovery_file(db_path: &str) -> PathBuf {
    let path = FsPath::new(db_path);
    let name = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{name}.serve.json"))
}

/// Own the database and serve the HTTP op plane until Ctrl-C. Returns the
/// discovery file path so the CLI can print it.
pub fn serve(settings: &Settings, listen: &str) -> Result<PathBuf> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| OkfError::internal(err.to_string(), "tokio"))?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(listen)
            .await
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
        let url = listener
            .local_addr()
            .map(|addr| format!("http://{}", addr))
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
        let router = Arc::new(Router::open(settings)?);
        let app = http_router(router, settings.clone());
        let discovery = discovery_file(&settings.db_path);
        let db_path = std::path::absolute(&settings.db_path)
            .unwrap_or_else(|_| PathBuf::from(&settings.db_path));
        let record = json!({
            "url": url,
            "pid": std::process::id(),
            "db_path": db_path.to_string_lossy(),
        });
        if let Some(parent) = discovery.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&discovery, serde_json::to_string_pretty(&record).unwrap())
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
        tracing::info!("okf serve: {url} owns {}", settings.db_path);
        let result = axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await;
        let _ = std::fs::remove_file(&discovery);
        result.map_err(|err| OkfError::internal(err.to_string(), "axum"))?;
        Ok(discovery)
    })
}
