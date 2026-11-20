//! axum HTTP surface for `okf serve`: `/ops/*`, `/health`, discovery file.
//!
//! Stage C starts with the blocking op plane and the discovery contract the
//! public `OKFClient` depends on. Long ops can be submitted with
//! `mode="job"` (currently `reembed`) or `POST /jobs/{op}`; the owner keeps
//! job state in memory and runs the blocking router op on the same
//! `spawn_blocking` pool, serialised by one owner lock. The MCP mount and
//! TCP/UDS transports are live M4 surfaces.

pub mod client;

use crate::error::{Envelope, OkfError, Result};
use crate::ops::{Router, OP_NAMES};
use crate::settings::Settings;
use axum::body::Bytes;
#[cfg(feature = "mcp")]
use axum::extract::Request;
use axum::extract::{Path, State};
use axum::http::StatusCode;
#[cfg(feature = "mcp")]
use axum::response::Response;
#[cfg(feature = "mcp")]
use axum::routing::any;
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct AppState {
    router: Arc<Router>,
    settings: Settings,
    jobs: JobManager,
    /// One lock per owner: Ladybug connections are not concurrent. Python's
    /// `LockedRouter` serialises every op; HTTP jobs take the same lock.
    op_lock: Arc<Mutex<()>>,
    #[cfg(feature = "mcp")]
    mcp: crate::mcp::McpHttpService,
}

/// Public job view returned by `/jobs` and `/jobs/{id}`.
///
/// The field names and terminal statuses mirror Python 0.15's
/// `JobRegistry.view()`: clients can poll `phase`/`done`/`total` and read a
/// cancelled reembed's partial counts from `result`.
#[derive(Clone, Serialize)]
struct JobView {
    id: String,
    op: String,
    status: String,
    kwargs: Map<String, Value>,
    phase: Option<String>,
    done: usize,
    total: usize,
    submitted: u64,
    started: Option<u64>,
    finished: Option<u64>,
    result: Value,
    error: Option<Value>,
    elapsed: f64,
}

struct Job {
    view: JobView,
    cancel: Arc<AtomicBool>,
}

#[derive(Clone, Default)]
struct JobManager {
    jobs: Arc<Mutex<BTreeMap<String, Job>>>,
    next: Arc<AtomicU64>,
}

const KEEP_TERMINAL: usize = 64;
const RETIRE_AFTER_SECS: u64 = 3600;

impl JobManager {
    fn new_id(&self) -> String {
        let n = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        format!("{:08x}{:04x}", now_secs() as u32, n as u16)
    }

    #[cfg(test)]
    fn submit_with<F>(&self, op: &str, run: F) -> String
    where
        F: FnOnce() -> Result<Value> + Send + 'static,
    {
        let id = self.new_id();
        let cancel = Arc::new(AtomicBool::new(false));
        let view = JobView {
            id: id.clone(),
            op: op.to_string(),
            status: "queued".into(),
            kwargs: Map::new(),
            phase: None,
            done: 0,
            total: 0,
            submitted: now_secs(),
            started: None,
            finished: None,
            result: Value::Null,
            error: None,
            elapsed: 0.0,
        };
        self.jobs.lock().expect("job mutex").insert(
            id.clone(),
            Job {
                view,
                cancel: cancel.clone(),
            },
        );
        let jobs = self.clone();
        let job_id = id.clone();
        let worker_cancel = cancel.clone();
        tokio::task::spawn_blocking(move || {
            // Test-only hook: lets the parity smoke cancel a job while it is
            // still running without loading a large model on every CI box.
            if let Ok(delay) = std::env::var("OKFGRAPH_JOB_TEST_DELAY_MS") {
                if let Ok(millis) = delay.parse::<u64>() {
                    std::thread::sleep(std::time::Duration::from_millis(millis));
                }
            }
            if jobs.is_cancelled(&job_id) || worker_cancel.load(Ordering::Acquire) {
                jobs.finish(&job_id, "cancelled", Value::Null, None);
                return;
            }
            // Only the op's cooperative result decides cancellation: a
            // cancel that arrived after the op finished must not rewrite its
            // successful result, exactly like Python's JobRegistry.
            jobs.set_running(&job_id);
            match run() {
                Ok(value) => {
                    let status = if value.get("cancelled") == Some(&Value::Bool(true)) {
                        "cancelled"
                    } else {
                        "done"
                    };
                    jobs.finish(&job_id, status, value, None);
                }
                Err(err) => jobs.finish(
                    &job_id,
                    "failed",
                    Value::Null,
                    Some(serde_json::to_value(err.envelope_body()).unwrap_or(Value::Null)),
                ),
            }
        });
        id
    }

    fn submit(
        &self,
        op: &str,
        params: Map<String, Value>,
        router: Arc<Router>,
        op_lock: Arc<Mutex<()>>,
    ) -> Result<String> {
        if op == "reembed" && self.has_running("reembed") {
            return Err(OkfError::new(
                "REEMBED_IN_PROGRESS",
                "another reembed is already running (or queued) on this graph",
            )
            .op("reembed")
            .remedy("watch GET /jobs/<id>, or cancel it with POST /jobs/<id>/cancel"));
        }
        let id = self.new_id();
        let cancel = Arc::new(AtomicBool::new(false));
        let view = JobView {
            id: id.clone(),
            op: op.to_string(),
            status: "queued".into(),
            kwargs: params.clone(),
            phase: None,
            done: 0,
            total: 0,
            submitted: now_secs(),
            started: None,
            finished: None,
            result: Value::Null,
            error: None,
            elapsed: 0.0,
        };
        self.jobs.lock().expect("job mutex").insert(
            id.clone(),
            Job {
                view,
                cancel: cancel.clone(),
            },
        );
        let jobs = self.clone();
        let job_id = id.clone();
        let run: Box<dyn FnOnce() -> Result<Value> + Send> = if op == "reembed" {
            let progress_jobs = jobs.clone();
            let progress_id = job_id.clone();
            let progress = move |phase: &str, done: usize, total: usize| {
                progress_jobs.set_progress(&progress_id, phase, done, total);
            };
            let cancelled = {
                let flag = cancel.clone();
                move || flag.load(Ordering::Acquire)
            };
            let params = params.clone();
            let router = Arc::clone(&router);
            let reembed_lock = Arc::clone(&op_lock);
            Box::new(move || {
                let _guard = reembed_lock.lock().expect("op lock");
                router.reembed_with_control(
                    &params,
                    crate::ops::ReembedControl {
                        progress: Some(Arc::new(progress)),
                        cancelled: Some(Arc::new(cancelled)),
                    },
                )
            })
        } else {
            let op_name = op.to_string();
            let call_lock = Arc::clone(&op_lock);
            Box::new(move || {
                let _guard = call_lock.lock().expect("op lock");
                router.call(&op_name, &params)
            })
        };
        tokio::task::spawn_blocking(move || {
            if let Ok(delay) = std::env::var("OKFGRAPH_JOB_TEST_DELAY_MS") {
                if let Ok(millis) = delay.parse::<u64>() {
                    std::thread::sleep(std::time::Duration::from_millis(millis));
                }
            }
            // Do not short-circuit here: a queued reembed must enter the op so
            // its cooperative hook can return the same cancelled-with-counts
            // report Python produces, even when the cancel arrived first.
            // Only the op's own `cancelled` flag decides the terminal status;
            // a cancel after completion must not rewrite a successful result.
            jobs.set_running(&job_id);
            match run() {
                Ok(value) => {
                    let status = if value.get("cancelled") == Some(&Value::Bool(true)) {
                        "cancelled"
                    } else {
                        "done"
                    };
                    jobs.finish(&job_id, status, value, None);
                }
                Err(err) => jobs.finish(
                    &job_id,
                    "failed",
                    Value::Null,
                    Some(serde_json::to_value(err.envelope_body()).unwrap_or(Value::Null)),
                ),
            }
        });
        Ok(id)
    }

    fn has_running(&self, op: &str) -> bool {
        self.jobs.lock().expect("job mutex").values().any(|job| {
            job.view.op == op
                && matches!(
                    job.view.status.as_str(),
                    "queued" | "running" | "cancelling"
                )
        })
    }

    fn retire_locked(jobs: &mut BTreeMap<String, Job>) {
        let now = now_secs();
        jobs.retain(|_, job| {
            job.view
                .finished
                .is_none_or(|finished| now.saturating_sub(finished) <= RETIRE_AFTER_SECS)
        });
    }

    fn view_with_elapsed(view: &JobView) -> JobView {
        let mut view = view.clone();
        let end = view.finished.unwrap_or_else(now_secs);
        let start = view.started.unwrap_or(view.submitted);
        view.elapsed = (end.saturating_sub(start) as f64 * 10.0).round() / 10.0;
        view
    }

    fn list(&self) -> Vec<JobView> {
        let mut jobs = self.jobs.lock().expect("job mutex");
        Self::retire_locked(&mut jobs);
        let mut rows: Vec<JobView> = jobs
            .values()
            .map(|job| Self::view_with_elapsed(&job.view))
            .collect();
        rows.sort_by(|a, b| b.submitted.cmp(&a.submitted).then_with(|| b.id.cmp(&a.id)));
        for row in &mut rows {
            row.result = Value::Null;
            row.error = None;
        }
        rows.truncate(KEEP_TERMINAL);
        rows
    }

    fn get(&self, id: &str) -> Option<JobView> {
        self.jobs
            .lock()
            .expect("job mutex")
            .get(id)
            .map(|job| Self::view_with_elapsed(&job.view))
    }

    fn cancel(&self, id: &str) -> Option<JobView> {
        let mut jobs = self.jobs.lock().expect("job mutex");
        let job = jobs.get_mut(id)?;
        job.cancel.store(true, Ordering::Release);
        if matches!(job.view.status.as_str(), "queued" | "running") {
            job.view.status = "cancelling".into();
        }
        Some(Self::view_with_elapsed(&job.view))
    }

    #[cfg(test)]
    fn is_cancelled(&self, id: &str) -> bool {
        self.jobs
            .lock()
            .expect("job mutex")
            .get(id)
            .map(|job| job.cancel.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    fn set_running(&self, id: &str) {
        if let Some(job) = self.jobs.lock().expect("job mutex").get_mut(id) {
            if job.view.status == "queued" {
                job.view.status = "running".into();
                job.view.started = Some(now_secs());
            }
        }
    }

    fn set_progress(&self, id: &str, phase: &str, done: usize, total: usize) {
        if let Some(job) = self.jobs.lock().expect("job mutex").get_mut(id) {
            job.view.phase = Some(phase.to_string());
            job.view.done = done;
            job.view.total = total;
        }
    }

    fn finish(&self, id: &str, status: &str, result: Value, error: Option<Value>) {
        if let Some(job) = self.jobs.lock().expect("job mutex").get_mut(id) {
            job.view.status = status.to_string();
            job.view.finished = Some(now_secs());
            job.view.result = result;
            job.view.error = error;
            job.view.phase = None;
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default()
}

fn envelope_json(
    op: &str,
    result: std::result::Result<Value, OkfError>,
) -> (StatusCode, Json<Value>) {
    let status = match &result {
        Err(err) => match err.code.as_str() {
            "DB_LOCKED" | "WRITE_LOCK_TIMEOUT" | "REEMBED_IN_PROGRESS" => StatusCode::CONFLICT,
            "BAD_VALUE" | "MISSING_PARAM" | "FILE_NOT_FOUND" | "CONFIG_INVALID" => {
                StatusCode::BAD_REQUEST
            }
            _ => StatusCode::OK,
        },
        Ok(_) => StatusCode::OK,
    };
    envelope_json_status(op, result, status)
}

fn envelope_json_status(
    op: &str,
    result: std::result::Result<Value, OkfError>,
    status: StatusCode,
) -> (StatusCode, Json<Value>) {
    let body = match result {
        Ok(data) => Envelope::success(op, data),
        Err(err) => Envelope::failure(op, &err),
    };
    (status, Json(serde_json::to_value(body).expect("envelope")))
}

fn unknown_op(op: &str) -> (StatusCode, Json<Value>) {
    let err = OkfError::new("BAD_VALUE", format!("unknown op '{op}'"))
        .field("op", json!(op))
        .field("ops", json!(OP_NAMES));
    envelope_json_status(op, Err(err), StatusCode::NOT_FOUND)
}

fn submit_result(
    state: &Arc<AppState>,
    op: &str,
    params: Map<String, Value>,
) -> (StatusCode, Json<Value>) {
    match state.jobs.submit(
        op,
        params,
        Arc::clone(&state.router),
        Arc::clone(&state.op_lock),
    ) {
        Ok(id) => envelope_json(
            op,
            Ok(json!({
                "job": id,
                "status": "queued",
                "op": op,
            })),
        ),
        Err(err) => {
            let status = if err.code == "REEMBED_IN_PROGRESS" {
                StatusCode::CONFLICT
            } else {
                StatusCode::BAD_REQUEST
            };
            envelope_json_status(op, Err(err), status)
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let db_path = std::path::absolute(&state.settings.db_path)
        .unwrap_or_else(|_| PathBuf::from(&state.settings.db_path));
    let data = json!({
        "db_path": db_path.to_string_lossy(),
        "created": state.router.db_created(),
        "pid": std::process::id(),
        "ops": OP_NAMES,
        "lanes": state.router.lanes().unwrap_or(Value::Null),
    });
    Json(serde_json::to_value(Envelope::success("health", data)).expect("envelope"))
}

#[cfg(feature = "mcp")]
async fn mcp_http(State(state): State<Arc<AppState>>, request: Request) -> Response {
    state.mcp.handle(request).await
}

async fn list_ops() -> Json<Value> {
    Json(json!(OP_NAMES))
}

async fn list_jobs(State(state): State<Arc<AppState>>) -> Json<Value> {
    let jobs = serde_json::to_value(state.jobs.list()).expect("jobs");
    Json(serde_json::to_value(Envelope::success("jobs", jobs)).expect("envelope"))
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> (StatusCode, Json<Value>) {
    match state.jobs.get(&id) {
        Some(view) => envelope_json("job", Ok(serde_json::to_value(view).expect("job"))),
        None => unknown_job(&id),
    }
}

async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> (StatusCode, Json<Value>) {
    match state.jobs.cancel(&id) {
        Some(view) => envelope_json("jobs.cancel", Ok(serde_json::to_value(view).expect("job"))),
        None => unknown_job(&id),
    }
}

async fn submit_job_op(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    let op = id;
    if !OP_NAMES.contains(&op.as_str()) {
        return unknown_op(&op);
    }
    let params = parse_params(&body, &op);
    let params = match params {
        Ok(params) => params,
        Err(status) => return status,
    };
    submit_result(&state, &op, params)
}

async fn submit_job_body(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    let request: Map<String, Value> = if body.is_empty() {
        Map::new()
    } else {
        match serde_json::from_slice(&body) {
            Ok(map) => map,
            Err(err) => {
                return envelope_json(
                    "job",
                    Err(OkfError::new(
                        "BAD_VALUE",
                        format!("bad request body: {err}"),
                    )),
                )
            }
        }
    };
    let op = match request.get("op").and_then(Value::as_str) {
        Some(op) if OP_NAMES.contains(&op) => op.to_string(),
        Some(op) => {
            let err = OkfError::new("BAD_VALUE", format!("unknown op '{op}'"))
                .field("op", json!(op))
                .field("ops", json!(OP_NAMES));
            return envelope_json_status("job", Err(err), StatusCode::NOT_FOUND);
        }
        None => {
            return envelope_json(
                "job",
                Err(OkfError::new("MISSING_PARAM", "job submission needs op")
                    .field("op", Value::Null)),
            )
        }
    };
    let mut params = request
        .get("params")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if params.is_empty() {
        for (key, value) in &request {
            if key != "op" && key != "params" {
                params.insert(key.clone(), value.clone());
            }
        }
    }
    submit_result(&state, &op, params)
}

fn parse_params(
    body: &[u8],
    op: &str,
) -> std::result::Result<Map<String, Value>, (StatusCode, Json<Value>)> {
    if body.is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_slice(body).map_err(|err| {
        envelope_json(
            op,
            Err(OkfError::new(
                "BAD_VALUE",
                format!("bad request body: {err}"),
            )),
        )
    })
}

fn unknown_job(job_id: &str) -> (StatusCode, Json<Value>) {
    let err =
        OkfError::new("BAD_VALUE", format!("unknown job {job_id:?}")).field("job", json!(job_id));
    envelope_json_status("job", Err(err), StatusCode::NOT_FOUND)
}

async fn call_op(
    State(state): State<Arc<AppState>>,
    Path(op): Path<String>,
    body: Bytes,
) -> (StatusCode, Json<Value>) {
    if !OP_NAMES.contains(&op.as_str()) {
        return unknown_op(&op);
    }
    let mut params = match parse_params(&body, &op) {
        Ok(params) => params,
        Err(response) => return response,
    };
    // `OKFClient.reembed_job` uses this documented switch. Python pops the
    // mode before calling the op and refuses values other than `sync`/`job`.
    if op == "reembed" {
        if let Some(mode) = params.remove("mode") {
            let mode = match mode.as_str() {
                Some(value) => value,
                None => {
                    let shown = serde_json::to_string(&mode).unwrap_or_default();
                    return envelope_json(
                        &op,
                        Err(OkfError::new(
                            "BAD_VALUE",
                            format!("mode must be 'sync' or 'job', got {shown}"),
                        )
                        .op(&op)),
                    );
                }
            };
            if !matches!(mode, "sync" | "job") {
                return envelope_json(
                    &op,
                    Err(OkfError::new(
                        "BAD_VALUE",
                        format!("mode must be 'sync' or 'job', got '{mode}'"),
                    )
                    .op(&op)),
                );
            }
            if mode == "job" {
                return submit_result(&state, &op, params);
            }
        }
    }
    let router = Arc::clone(&state.router);
    let op_lock = Arc::clone(&state.op_lock);
    let op_name = op.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _guard = op_lock.lock().expect("op lock");
        router.call(&op_name, &params)
    })
    .await
    .unwrap_or_else(|err| Err(OkfError::internal(err.to_string(), "join")));
    envelope_json(&op, result)
}

pub fn http_router(router: Arc<Router>, settings: Settings) -> AxumRouter {
    #[cfg(feature = "mcp")]
    let mcp = crate::mcp::McpHttpService::new(
        Arc::clone(&router) as Arc<dyn crate::ops::OpClient>,
        settings.clone(),
    );
    let state = Arc::new(AppState {
        router,
        settings,
        jobs: JobManager::default(),
        op_lock: Arc::new(Mutex::new(())),
        #[cfg(feature = "mcp")]
        mcp,
    });
    #[allow(unused_mut)]
    let mut app = AxumRouter::new()
        .route("/health", get(health))
        .route("/ops", get(list_ops))
        .route("/ops/{op}", post(call_op))
        .route("/jobs", get(list_jobs).post(submit_job_body))
        .route("/jobs/{id}", post(submit_job_op).get(get_job))
        .route("/jobs/{id}/cancel", post(cancel_job));
    #[cfg(feature = "mcp")]
    {
        app = app.route("/mcp", any(mcp_http));
    }
    app.with_state(state)
}

pub fn discovery_file(db_path: &str) -> PathBuf {
    let path = FsPath::new(db_path);
    let name = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{name}.serve.json"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ListenTarget {
    Tcp(String),
    Unix(PathBuf),
}

fn parse_listen(listen: &str) -> Result<ListenTarget> {
    if let Some(path) = listen.strip_prefix("unix://") {
        if path.is_empty() {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                "unix:// listen needs a socket path",
            ));
        }
        let path = std::path::absolute(path).unwrap_or_else(|_| PathBuf::from(path));
        Ok(ListenTarget::Unix(path))
    } else if !listen.is_empty() && listen.contains(':') {
        Ok(ListenTarget::Tcp(listen.to_string()))
    } else {
        Err(OkfError::new(
            "CONFIG_INVALID",
            format!("invalid --listen {listen:?}; use host:port or unix:///path"),
        ))
    }
}

/// Write the discovery record, serve until Ctrl-C, then remove both the
/// discovery file and a Unix socket (if this owner created one).
async fn serve_bound<L>(
    listener: L,
    url: String,
    settings: &Settings,
    socket: Option<PathBuf>,
) -> Result<PathBuf>
where
    L: axum::serve::Listener + Send + 'static,
    L::Addr: std::fmt::Debug,
{
    let router = Arc::new(Router::open(settings)?);
    let app = http_router(router, settings.clone());
    let discovery = discovery_file(&settings.db_path);
    let db_path =
        std::path::absolute(&settings.db_path).unwrap_or_else(|_| PathBuf::from(&settings.db_path));
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
    if let Some(path) = socket {
        let _ = std::fs::remove_file(path);
    }
    result.map_err(|err| OkfError::internal(err.to_string(), "axum"))?;
    Ok(discovery)
}

#[cfg(unix)]
fn bind_unix(path: &FsPath) -> Result<tokio::net::UnixListener> {
    match tokio::net::UnixListener::bind(path) {
        Ok(listener) => Ok(listener),
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
            // A stale socket file is common after SIGKILL. Refuse to displace
            // a live listener; otherwise clear the file and retry once.
            if std::os::unix::net::UnixStream::connect(path).is_ok() {
                return Err(OkfError::new(
                    "DB_LOCKED",
                    format!("unix socket already in use: {}", path.display()),
                )
                .field("path", json!(path.to_string_lossy())));
            }
            std::fs::remove_file(path)
                .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
            tokio::net::UnixListener::bind(path)
                .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))
        }
        Err(err) => Err(OkfError::internal(err.to_string(), "std::io::Error")),
    }
}

#[cfg(unix)]
async fn serve_unix(settings: &Settings, path: PathBuf) -> Result<PathBuf> {
    let listener = bind_unix(&path)?;
    let url = format!("unix://{}", path.to_string_lossy());
    serve_bound(listener, url, settings, Some(path)).await
}

#[cfg(not(unix))]
async fn serve_unix(_settings: &Settings, path: PathBuf) -> Result<PathBuf> {
    Err(OkfError::new(
        "CONFIG_INVALID",
        format!(
            "unix sockets are unavailable on this platform: {}",
            path.display()
        ),
    ))
}

/// Own the database and serve the HTTP op plane until Ctrl-C. Returns the
/// discovery file path so the CLI can print it. Accepts `host:port` and, on
/// Unix hosts, `unix:///absolute/path`.
pub fn serve(settings: &Settings, listen: &str) -> Result<PathBuf> {
    let target = parse_listen(listen)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| OkfError::internal(err.to_string(), "tokio"))?;
    runtime.block_on(async move {
        match target {
            ListenTarget::Tcp(addr) => {
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
                let url = listener
                    .local_addr()
                    .map(|addr| format!("http://{}", addr))
                    .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
                serve_bound(listener, url, settings, None).await
            }
            ListenTarget::Unix(path) => serve_unix(settings, path).await,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    async fn terminal(jobs: &JobManager, id: &str) -> JobView {
        for _ in 0..200 {
            if let Some(view) = jobs.get(id) {
                if matches!(view.status.as_str(), "done" | "failed" | "cancelled") {
                    return view;
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("job {id} never reached a terminal state");
    }

    #[test]
    fn listen_target_supports_tcp_and_unix() {
        assert_eq!(
            parse_listen("127.0.0.1:8765").unwrap(),
            ListenTarget::Tcp("127.0.0.1:8765".into())
        );
        let unix = parse_listen("unix:///tmp/okf.sock").unwrap();
        match unix {
            ListenTarget::Unix(path) => assert!(path.is_absolute()),
            other => panic!("unexpected {other:?}"),
        }
        assert!(parse_listen("nonsense").is_err());
        assert!(parse_listen("unix://").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn stale_unix_socket_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("okf.sock");
        std::fs::write(&path, b"stale").unwrap();
        let listener = bind_unix(&path).unwrap();
        assert!(path.exists());
        drop(listener);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_listener_serves_the_health_envelope() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("okf.sock");
        let db = dir.path().join("graph.db");
        std::fs::copy("parity/fixtures/oracle-v10.db", &db).unwrap();
        let mut settings = Settings::default();
        settings.db_path = db.to_string_lossy().into_owned();

        let listener = bind_unix(&path).unwrap();
        let app = http_router(Arc::new(Router::open(&settings).unwrap()), settings);
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut stream = tokio::net::UnixStream::connect(&path).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("\"op\":\"health\""), "{response}");
        server.abort();
    }

    #[tokio::test]
    async fn job_runs_to_completion_with_a_result() {
        let jobs = JobManager::default();
        let id = jobs.submit_with("reembed", || Ok(json!({"dry_run": true})));
        let view = terminal(&jobs, &id).await;
        assert_eq!(view.status, "done");
        assert_eq!(view.result, json!({"dry_run": true}));
        assert!(view.finished.is_some());
        assert_eq!(jobs.list().len(), 1);
        assert_eq!(jobs.list()[0].result, Value::Null);
        assert!(jobs.cancel(&id).unwrap().error.is_none());
    }

    #[tokio::test]
    async fn late_cancel_does_not_rewrite_a_finished_result() {
        let jobs = JobManager::default();
        let ran = Arc::new(AtomicUsize::new(0));
        let worker = ran.clone();
        let id = jobs.submit_with("reembed", move || {
            worker.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(250));
            Ok(json!({"too": "late"}))
        });
        for _ in 0..50 {
            if jobs.get(&id).is_some_and(|view| view.status == "running") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let cancelling = jobs.cancel(&id).unwrap();
        assert_eq!(cancelling.status, "cancelling");
        assert!(cancelling.kwargs.is_empty());
        let view = terminal(&jobs, &id).await;
        // Python's job registry lets the op decide: only a cooperative
        // `cancelled: true` report becomes a cancelled job.
        assert_eq!(view.status, "done");
        assert_eq!(view.result, json!({"too": "late"}));
    }

    #[test]
    fn unknown_job_error_uses_the_typed_envelope() {
        let (status, body) = unknown_job("nope");
        assert_eq!(status, StatusCode::NOT_FOUND);
        let value = body.0;
        assert_eq!(value["error"]["code"], "BAD_VALUE");
        assert_eq!(value["error"]["fields"]["job"], "nope");
    }
}
