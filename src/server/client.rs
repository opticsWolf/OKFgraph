//! Thin Rust client for an `okf serve` owner.
//!
//! This is the transport half of the M4 plan's `server::client`: discovery,
//! typed-envelope translation and the same `OpClient` trait the MCP proxy
//! uses. Python's public `OKFClient` stays a stdlib shim; the Rust CLI needs
//! this so a second process never opens a database already owned by a
//! server.

use crate::error::{OkfError, Result};
use crate::ops::{OpClient, OP_NAMES};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Location of the rendezvous file a live owner writes next to its database.
pub fn discovery_file(db_path: &str) -> PathBuf {
    super::discovery_file(db_path)
}

fn resolve(path: &Path) -> PathBuf {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// URL of a live `okf serve` owning `db_path`, or `None` for a stale file.
///
/// A discovery record is only trusted when its `/health` answers for the
/// same database. This also handles a stale pid whose process id was reused:
/// either the old URL is dead, or it names a different graph.
pub fn discover(db_path: &str, timeout: Duration) -> Option<String> {
    let record = discovery_file(db_path);
    let raw = std::fs::read_to_string(record).ok()?;
    let data: Value = serde_json::from_str(&raw).ok()?;
    let url = data.get("url")?.as_str()?.to_string();
    // `pid` is provenance; `/health` is the liveness check. Keep the field
    // required so a half-written record is ignored.
    data.get("pid")?.as_u64()?;
    let client = OpHttpClient::with_timeout(&url, timeout).ok()?;
    let health = client.health().ok()?;
    let served = health.get("db_path")?.as_str()?;
    if resolve(Path::new(served)) == resolve(Path::new(db_path)) {
        Some(url)
    } else {
        None
    }
}

/// HTTP/Unix client that returns an op's `data` or a typed `OKFError`.
pub struct OpHttpClient {
    base: String,
    #[allow(dead_code)] // read by the POSIX Unix-socket branch
    timeout: Duration,
    http: Option<reqwest::blocking::Client>,
    #[cfg(unix)]
    socket: Option<PathBuf>,
}

impl OpHttpClient {
    pub fn new(url: &str) -> Result<Self> {
        Self::with_timeout(url, DEFAULT_TIMEOUT)
    }

    pub fn with_timeout(url: &str, timeout: Duration) -> Result<Self> {
        let base = url.trim_end_matches('/').to_string();
        if let Some(path) = base.strip_prefix("unix://") {
            #[cfg(unix)]
            {
                if path.is_empty() {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        "unix:// client URL needs a socket path",
                    ));
                }
                return Ok(Self {
                    base,
                    timeout,
                    http: None,
                    socket: Some(PathBuf::from(path)),
                });
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!("unix sockets are unavailable on this platform: {url}"),
                ));
            }
        }
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!("--connect must be an http:// or unix:// URL, got {url:?}"),
            ));
        }
        let http = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| OkfError::internal(err.to_string(), "reqwest"))?;
        Ok(Self {
            base,
            timeout,
            http: Some(http),
            #[cfg(unix)]
            socket: None,
        })
    }

    pub fn health(&self) -> Result<Value> {
        self.request("GET", "/health", None)
    }

    pub fn request(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value> {
        let envelope = if let Some(client) = &self.http {
            let mut request = client
                .request(
                    method.parse().unwrap_or(reqwest::Method::GET),
                    format!("{}{}", self.base, path),
                )
                .header("Content-Type", "application/json");
            if let Some(body) = &body {
                request = request.json(body);
            }
            let response = request
                .send()
                .map_err(|err| OkfError::internal(err.to_string(), "reqwest"))?;
            response
                .json::<Value>()
                .map_err(|err| OkfError::internal(err.to_string(), "serde_json"))?
        } else {
            #[cfg(unix)]
            {
                self.unix_request(method, path, body)?
            }
            #[cfg(not(unix))]
            {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    "unix sockets are unavailable",
                ));
            }
        };
        unwrap_envelope(&envelope)
    }

    #[cfg(unix)]
    fn unix_request(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value> {
        use std::io::{Read, Write};

        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| OkfError::internal("missing unix socket path", "server::client"))?;
        let mut stream = std::os::unix::net::UnixStream::connect(socket)
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
        stream.set_read_timeout(Some(self.timeout)).ok();
        stream.set_write_timeout(Some(self.timeout)).ok();
        let payload = body.map_or_else(String::new, |value| value.to_string());
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?;
        let split = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or_else(|| {
                OkfError::internal("server answered without HTTP headers", "okf serve")
            })?;
        let body_start = split + 4;
        serde_json::from_slice(&raw[body_start..])
            .map_err(|err| OkfError::internal(err.to_string(), "serde_json"))
    }
}

fn unwrap_envelope(envelope: &Value) -> Result<Value> {
    if envelope.get("ok") == Some(&Value::Bool(true)) {
        return Ok(envelope.get("data").cloned().unwrap_or(Value::Null));
    }
    if let Some(error) = envelope.get("error") {
        let mut err = OkfError::from_error_value(error);
        // Outcome errors carry their report in the envelope's data slot.
        err.data = envelope.get("data").cloned();
        err.op = envelope
            .get("op")
            .and_then(Value::as_str)
            .map(str::to_string);
        return Err(err);
    }
    Err(OkfError::internal(
        "server answered without an error envelope",
        "okf serve",
    ))
}

impl OpClient for OpHttpClient {
    fn call(&self, op: &str, params: &Map<String, Value>) -> Result<Value> {
        if !OP_NAMES.contains(&op) && op != "lint" && op != "produce" {
            return Err(OkfError::new("BAD_VALUE", format!("unknown op {op:?}"))
                .field("op", Value::String(op.into())));
        }
        self.request(
            "POST",
            &format!("/ops/{op}"),
            Some(Value::Object(params.clone())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_urls_are_validated() {
        assert!(OpHttpClient::new("http://127.0.0.1:8765").is_ok());
        assert!(OpHttpClient::new("unix:///tmp/okf.sock").is_ok() || cfg!(not(unix)));
        assert!(OpHttpClient::new("nonsense").is_err());
    }

    #[test]
    fn stale_discovery_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("missing.db");
        let record = discovery_file(db.to_str().unwrap());
        std::fs::write(
            &record,
            r#"{"url":"http://127.0.0.1:1","pid":999999999,"db_path":"x"}"#,
        )
        .unwrap();
        assert!(discover(db.to_str().unwrap(), Duration::from_millis(50)).is_none());
    }
}
