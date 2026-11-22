//! MCP server: 11 rmcp tools over one owning router or an `okf serve` client.
//!
//! Tool functions never contain business logic: they translate a typed request
//! into the same `params` map the HTTP/Python surfaces use and return the
//! canonical `{ok, op, data, warnings, error}` envelope as pretty JSON, exactly
//! like Python 0.15's `_tool_result`.

use crate::error::{Envelope, OkfError};
use crate::ops::OpClient;
use crate::settings::Settings;
use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;

use crate::server::client::OpHttpClient as HttpOpClient;

#[derive(Clone)]
pub struct OkfMcp {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    client: Arc<dyn OpClient>,
    settings: Settings,
}

impl OkfMcp {
    pub fn new(client: Arc<dyn OpClient>, settings: Settings) -> Self {
        Self {
            tool_router: Self::tool_router(),
            client,
            settings,
        }
    }

    fn params<T: Serialize>(value: T) -> Map<String, Value> {
        serde_json::to_value(value)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()
    }

    fn settings(&self) -> &Settings {
        &self.settings
    }
}

/// HTTP-mounted MCP service used by `okf serve /mcp`.
///
/// One service instance is shared by the axum router; rmcp creates a fresh
/// `OkfMcp` per client session, all backed by the same owning `OpClient`.
#[derive(Clone)]
pub struct McpHttpService {
    inner: StreamableHttpService<OkfMcp, LocalSessionManager>,
}

impl McpHttpService {
    pub fn new(client: Arc<dyn OpClient>, settings: Settings) -> Self {
        let config = StreamableHttpServerConfig::default()
            .with_json_response(true)
            .with_allowed_hosts(["localhost", "127.0.0.1", "::1"]);
        let inner = StreamableHttpService::new(
            move || Ok(OkfMcp::new(Arc::clone(&client), settings.clone())),
            Arc::new(LocalSessionManager::default()),
            config,
        );
        Self { inner }
    }

    /// Translate an axum request into rmcp's own `http` request/response types.
    pub async fn handle(&self, request: Request) -> Response {
        self.inner.handle(request).await.map(Body::new)
    }
}

async fn invoke(client: Arc<dyn OpClient>, op: &'static str, params: Map<String, Value>) -> String {
    let result = tokio::task::spawn_blocking(move || client.call(op, &params))
        .await
        .unwrap_or_else(|err| Err(OkfError::internal(err.to_string(), "join")));
    match result {
        Ok(data) => Envelope::success(op, data).to_json_pretty(),
        Err(err) => Envelope::failure(op, &err).to_json_pretty(),
    }
    .unwrap_or_else(|_| "{\"ok\":false}".to_string())
}

// `rmcp`'s attribute accepts a literal token list, so each tool repeats the
// four annotation literals below (matching Python's _RO/_WRITE/_DELETE/etc.).

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct SearchRequest {
    query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    limit: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    concept_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    include_chunks: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_chunks_per_doc: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expand: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_hops: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hub_rerank: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hub_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rank: Option<String>,
    #[serde(rename = "where", default, skip_serializing_if = "Option::is_none")]
    where_: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    since: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    until: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exclude_tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    link_kinds: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    near_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hops: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offset: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recency: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recency_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    efs: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dedupe: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rerank: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct ReadRequest {
    concept_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    include: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct TraverseRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    start_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relationship: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    direction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    depth: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    node_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_path_length: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    link_kinds: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct IngestRequest {
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    md_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pdf_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thoughts: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    topic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    concept_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    routing_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extract_images: Option<bool>,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    concept_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extra: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    materialize: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    related: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    folder: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct ExportBundleRequest {
    output_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    directory_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    concept_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flavor: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct ExportConceptRequest {
    concept_id: String,
    output_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flavor: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct IdRequest {
    concept_id: String,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct AssetRequest {
    asset_id: String,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct UpdateRequest {
    concept_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    concept_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    add_tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remove_tags: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extra: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relations: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct SupersedeRequest {
    old_id: String,
    new_id: String,
}

#[tool_router]
impl OkfMcp {
    #[tool(
        description = "Search the knowledge graph — concepts, chunks, or images. Start here for questions over stored knowledge.",
        annotations(
            title = "read-only",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn search(&self, Parameters(req): Parameters<SearchRequest>) -> String {
        invoke(self.client.clone(), "search", Self::params(req)).await
    }

    #[tool(
        description = "Read a known concept: body, chunks, document, or context. Search first to find IDs.",
        annotations(
            title = "read-only",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn read(&self, Parameters(req): Parameters<ReadRequest>) -> String {
        invoke(self.client.clone(), "read", Self::params(req)).await
    }

    #[tool(
        description = "Traverse relationships from a concept or directory, or find a shortest path.",
        annotations(
            title = "read-only",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn traverse(&self, Parameters(req): Parameters<TraverseRequest>) -> String {
        invoke(self.client.clone(), "traverse", Self::params(req)).await
    }

    #[tool(
        description = "Add content: a markdown/PDF/Office file or a thought. Returns the new concept ID(s).",
        annotations(
            title = "ingest",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn ingest(&self, Parameters(mut req): Parameters<IngestRequest>) -> String {
        if req.mode.is_none() {
            req.mode = Some(self.settings().mode.clone());
        }
        invoke(self.client.clone(), "ingest", Self::params(req)).await
    }

    #[tool(
        description = "Export concepts to an OKF bundle directory ({output_dir, concept_ids, flavor}).",
        annotations(
            title = "export",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn export_bundle(&self, Parameters(req): Parameters<ExportBundleRequest>) -> String {
        invoke(self.client.clone(), "export_bundle", Self::params(req)).await
    }

    #[tool(
        description = "Export one concept to <output_dir>/<concept_id>.md.",
        annotations(
            title = "export",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn export_concept(&self, Parameters(req): Parameters<ExportConceptRequest>) -> String {
        invoke(self.client.clone(), "export_concept", Self::params(req)).await
    }

    #[tool(
        description = "List image assets attached to a concept (metadata, no bytes).",
        annotations(
            title = "read-only",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_images(&self, Parameters(req): Parameters<IdRequest>) -> String {
        invoke(self.client.clone(), "list_images", Self::params(req)).await
    }

    #[tool(
        description = "Fetch one image asset: metadata plus base64-encoded data.",
        annotations(
            title = "read-only",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_image(&self, Parameters(req): Parameters<AssetRequest>) -> String {
        invoke(self.client.clone(), "get_image", Self::params(req)).await
    }

    #[tool(
        description = "Edit a concept's frontmatter (never its body) by rewriting its bundle file, then importing it.",
        annotations(
            title = "write",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn update(&self, Parameters(req): Parameters<UpdateRequest>) -> String {
        invoke(self.client.clone(), "update", Self::params(req)).await
    }

    #[tool(
        description = "Record that new_id supersedes old_id (typed link + superseded tag on both files).",
        annotations(
            title = "write",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn supersede(&self, Parameters(req): Parameters<SupersedeRequest>) -> String {
        invoke(self.client.clone(), "supersede", Self::params(req)).await
    }

    #[tool(
        description = "Soft-delete a concept: its bundle file moves to .okf-trash and it is recoverable for 24h.",
        annotations(
            title = "delete",
            read_only_hint = false,
            destructive_hint = true,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn delete(&self, Parameters(req): Parameters<IdRequest>) -> String {
        let mut params = Self::params(req);
        params.insert("remove_file".into(), Value::Bool(true));
        invoke(self.client.clone(), "delete", params).await
    }
}

#[tool_handler(
    name = "OKFgraph MCP Server",
    instructions = "OKFgraph: search and read the knowledge graph, ingest and export bundle content, edit concepts, and manage image assets. All 11 tools return the canonical JSON envelope."
)]
impl ServerHandler for OkfMcp {}

fn command() -> clap::Command {
    crate::cli::with_global_args(
        clap::Command::new("okf-mcp")
            .version(crate::version())
            .about("OKFgraph MCP server (stdio, or proxy to okf serve)")
            .arg(
                clap::Arg::new("log-level")
                    .long("log-level")
                    .default_value("INFO")
                    .value_parser(["DEBUG", "INFO", "WARNING", "ERROR"]),
            )
            .arg(
                clap::Arg::new("connect")
                    .long("connect")
                    .value_name("URL")
                    .help("Proxy to a running okf serve instead of opening the DB"),
            ),
    )
}

/// Blocking entry point used by both `src/bin/okf-mcp.rs` and the PyO3 shim.
pub fn run(argv: Vec<String>) -> i32 {
    let matches = match command().try_get_matches_from(argv) {
        Ok(value) => value,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            let _ = err.print();
            return code;
        }
    };
    let settings = match crate::cli::settings_from_mcp(&matches) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("[ERROR] CONFIG_INVALID: {}", err.message);
            return 2;
        }
    };
    let client: Arc<dyn OpClient> = if let Some(url) = matches.get_one::<String>("connect") {
        match HttpOpClient::new(url).and_then(|client| {
            let health = client.health()?;
            let served = health
                .get("db_path")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !served.is_empty() {
                let expected = std::path::absolute(&settings.db_path)
                    .unwrap_or_else(|_| std::path::PathBuf::from(&settings.db_path));
                if std::path::Path::new(served) != expected {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!("serve owns {served}, not {}", settings.db_path),
                    ));
                }
            }
            Ok(client)
        }) {
            Ok(client) => Arc::new(client),
            Err(err) => {
                eprintln!(
                    "[ERROR] CONFIG_INVALID: cannot reach okf serve at {url} ({})",
                    err.message
                );
                return 2;
            }
        }
    } else {
        match crate::ops::Router::open(&settings) {
            Ok(router) => Arc::new(router),
            Err(err) => {
                eprintln!("[ERROR] {}", err.titles());
                return err.exit_code();
            }
        }
    };
    let server = OkfMcp::new(client, settings);
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(value) => value,
        Err(err) => {
            eprintln!("[ERROR] INTERNAL: tokio: {err}");
            return 1;
        }
    };
    runtime.block_on(async move {
        match server.serve(rmcp::transport::stdio()).await {
            Ok(connection) => match connection.waiting().await {
                Ok(_) => 0,
                Err(err) => {
                    eprintln!("[ERROR] INTERNAL: MCP transport: {err}");
                    1
                }
            },
            Err(err) => {
                eprintln!("[ERROR] INTERNAL: MCP server: {err}");
                1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_shape_is_stable() {
        let env = Envelope::success("read", Value::String("ok".into()));
        assert_eq!(
            env.to_json().unwrap(),
            "{\"ok\":true,\"op\":\"read\",\"data\":\"ok\",\"warnings\":[],\"error\":null}"
        );
    }

    #[test]
    fn connect_url_accepts_http_and_unix() {
        assert!(HttpOpClient::new("unix:///tmp/okf.sock").is_ok() || cfg!(not(unix)));
        assert!(HttpOpClient::new("http://127.0.0.1:8765").is_ok());
        assert!(HttpOpClient::new("nonsense").is_err());
    }
}
