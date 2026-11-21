//! HTTP clients for the external embedding, rerank and vision lanes.
//!
//! This module is a transport adapter: the graph still owns every pin and
//! every vector-space decision. It mirrors the Python 0.15 contracts in
//! `okfgraph/components/external_*.py` — declared identities, OpenAI-shaped
//! embed/rerank wires, embroider identity probes, token counters, and the
//! distinction between a per-request refusal and a lane outage.

use super::{Embedder, Reranker, Task, VisionEncoder};
use crate::error::{OkfError, Result};
use crate::settings::Settings;
use base64::Engine as _;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Map, Value as Json};
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;
use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const EMBED_TOKENIZE_PROBE: &str = "okfgraph tokenize probe for exact batch budgets";
const RERANK_TOKENIZE_PROBE: &str =
    "The quick brown fox jumps over the lazy dog while retrieval-augmented generation grounds an answer in the stored evidence.";

/// A per-request refusal. The host is alive; this one input was rejected.
#[derive(Debug, Clone)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn outage(url: &str, err: &reqwest::Error, remedy: &str) -> OkfError {
    OkfError::new(
        "EXTERNAL_UNAVAILABLE",
        format!("EXTERNAL_UNAVAILABLE|{url} is unreachable: {err}"),
    )
    .field("endpoint", Json::String(url.into()))
    .remedy(remedy)
}

fn external_error(message: String) -> OkfError {
    match message.strip_prefix("EXTERNAL_UNAVAILABLE|") {
        Some(detail) => OkfError::new("EXTERNAL_UNAVAILABLE", detail)
            .field("endpoint", Json::Null)
            .remedy("start the configured external host, or drop this external lane"),
        None => OkfError::new("INTERNAL", message),
    }
}

pub fn normalize_endpoint(url: &str) -> String {
    let mut base = url.trim().trim_end_matches('/').to_string();
    for suffix in ["/images/embeddings", "/embeddings", "/rerank"] {
        if base.ends_with(suffix) {
            base.truncate(base.len() - suffix.len());
        }
    }
    if !base.ends_with("/v1") {
        base.push_str("/v1");
    }
    base
}

pub fn tokenize_url(normalized: &str) -> String {
    let root = normalized.strip_suffix("/v1").unwrap_or(normalized);
    format!("{}/tokenize", root.trim_end_matches('/'))
}

fn auth_headers(api_key_env: &str, lane: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if !api_key_env.is_empty() {
        let value = std::env::var(api_key_env).map_err(|_| {
            OkfError::new(
                "CONFIG_INVALID",
                format!(
                    "[{lane}] api_key_env={api_key_env} is set but the environment variable is empty or missing"
                ),
            )
        })?;
        let value = HeaderValue::from_str(&format!("Bearer {value}"))
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        headers.insert(AUTHORIZATION, value);
    }
    Ok(headers)
}

fn post_json(
    client: &Client,
    url: &str,
    payload: &Json,
    headers: &HeaderMap,
    remedy: &str,
) -> std::result::Result<Json, Refusal> {
    let response = client
        .post(url)
        .headers(headers.clone())
        .json(payload)
        .send()
        .map_err(|err| Refusal(format!("{}", outage(url, &err, remedy))))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|err| Refusal(format!("{url}: could not read response: {err}")))?;
    if !status.is_success() {
        let detail: String = body.chars().take(400).collect();
        let detail = if detail.is_empty() {
            status.to_string()
        } else {
            detail
        };
        return Err(Refusal(format!("{url} refused {status}: {detail}")));
    }
    serde_json::from_str(&body)
        .map_err(|err| Refusal(format!("{url}: invalid JSON response: {err}")))
}

fn get_json(client: &Client, url: &str, headers: &HeaderMap) -> std::result::Result<Json, Refusal> {
    let response = client
        .get(url)
        .headers(headers.clone())
        .send()
        .map_err(|err| Refusal(format!("{}", outage(url, &err, ""))))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|err| Refusal(format!("{url}: could not read response: {err}")))?;
    if !status.is_success() {
        return Err(Refusal(format!("{url} refused {status}")));
    }
    serde_json::from_str(&body)
        .map_err(|err| Refusal(format!("{url}: invalid JSON response: {err}")))
}

/// The identity an embroider host publishes at `GET /v1/models`.
#[derive(Debug, Clone, Default)]
pub struct Identity {
    pub raw: Json,
    pub engine: String,
    pub kind: String,
    pub id: String,
    pub dim: usize,
    pub max_length: usize,
    pub family: String,
    pub precision: String,
    pub space: Option<String>,
    pub text_partner: Option<String>,
    pub text_partner_space: Option<String>,
    pub revision: String,
    pub contract: String,
    pub device: String,
    pub engine_version: Option<String>,
}

fn text(row: &Json, key: &str) -> String {
    row.get(key)
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string()
}

fn number(row: &Json, key: &str) -> usize {
    row.get(key)
        .and_then(Json::as_i64)
        .map(|value| value.max(0) as usize)
        .unwrap_or_default()
}

impl Identity {
    fn from_row(row: &Json) -> Self {
        Self {
            engine: text(row, "engine"),
            kind: text(row, "kind"),
            id: text(row, "id"),
            dim: number(row, "dim"),
            max_length: number(row, "max_length"),
            family: text(row, "family"),
            precision: text(row, "precision"),
            space: row.get("space").and_then(Json::as_str).map(str::to_string),
            text_partner: row
                .get("text_partner")
                .and_then(Json::as_str)
                .map(str::to_string),
            text_partner_space: row
                .get("text_partner_space")
                .and_then(Json::as_str)
                .map(str::to_string),
            revision: text(row, "revision"),
            contract: text(row, "contract"),
            device: text(row, "device"),
            engine_version: row
                .get("engine_version")
                .and_then(Json::as_str)
                .map(str::to_string),
            raw: row.clone(),
        }
    }
}

fn probe_identity(
    client: &Client,
    endpoint: &str,
    headers: &HeaderMap,
) -> Result<Option<Identity>> {
    let url = format!("{}/models", normalize_endpoint(endpoint));
    match get_json(client, &url, headers) {
        Ok(body) => {
            let rows = body.get("data").and_then(Json::as_array);
            Ok(rows
                .into_iter()
                .flatten()
                .filter_map(Json::as_object)
                .find(|row| row.get("engine").and_then(Json::as_str) == Some("embroider"))
                .map(|row| Identity::from_row(&Json::Object(row.clone()))))
        }
        // Identity is optional for text/rerank: a server without the route
        // or with a non-JSON answer keeps the declared contract. A refused
        // identity probe (401/500) is a real configuration error, not an
        // invitation to guess.
        Err(Refusal(message))
            if message.contains(" refused 404")
                || message.contains(" refused 405")
                || message.contains(" refused 501")
                || message.contains("invalid JSON response") =>
        {
            Ok(None)
        }
        Err(Refusal(message)) if message.starts_with("EXTERNAL_UNAVAILABLE|") => {
            Err(OkfError::new(
                "EXTERNAL_UNAVAILABLE",
                message
                    .trim_start_matches("EXTERNAL_UNAVAILABLE|")
                    .to_string(),
            )
            .field("endpoint", Json::String(url))
            .remedy("start the host, or declare the lane's contract explicitly"))
        }
        Err(Refusal(message)) => Err(OkfError::new("CONFIG_INVALID", message)),
    }
}

/// Extract the `count` or `tokens` field from a `/tokenize` response.
fn token_count(body: &Json) -> Option<usize> {
    if let Some(count) = body.get("count").and_then(Json::as_u64) {
        return Some(count as usize);
    }
    body.get("tokens").and_then(Json::as_array).map(Vec::len)
}

fn http_client(timeout: Duration) -> Result<Client> {
    Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|err| OkfError::internal(err.to_string(), "reqwest::Client"))
}

/// A chars/token prior measured from a GGUF header's vocabulary.
///
/// Port of `okfgraph/components/gguf_meta.py`: no tokenizer crate enters the
/// graph; only the header's `tokenizer.ggml.tokens` array is read and matched
/// greedily against a fixed English probe.
fn chars_per_token_from_gguf(path: &str) -> std::result::Result<f64, String> {
    const MIN_VOCAB: u64 = 1000;
    const MAX_KV_PAIRS: u64 = 4096;
    const MAX_STRING: u64 = 1 << 22;
    const MAX_TOKEN_STR: usize = 48;
    const MAX_KV_SCAN: u64 = 4_000_000;
    const PROBE: &str = "Retrieval-augmented generation grounds an answer in the evidence a knowledge graph stores: every concept, title and chunk of text is mapped into one vector space, where cosine similarity finds the nearest neighbours. In 2024 open embedding models reached 768 or 1024 dimensions, with matryoshka truncation trading a little accuracy for much smaller vectors. The declared contract matters — prefixes such as search_query: and search_document: flavour each side of retrieval, mean pooling keeps long passages stable, and a careful batch budget keeps memory flat while encoding 4,096 rows. ";

    fn read_exact(
        file: &mut std::fs::File,
        len: usize,
        what: &str,
    ) -> std::result::Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)
            .map_err(|_| format!("truncated GGUF header ({what})"))?;
        Ok(bytes)
    }

    fn u32le(file: &mut std::fs::File) -> std::result::Result<u32, String> {
        let bytes = read_exact(file, 4, "u32")?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    }

    fn u64le(file: &mut std::fs::File) -> std::result::Result<u64, String> {
        let bytes = read_exact(file, 8, "u64")?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    fn string(file: &mut std::fs::File) -> std::result::Result<String, String> {
        let len = u64le(file)?;
        if len > MAX_STRING {
            return Err(format!("header string too long: {len}"));
        }
        let bytes = read_exact(file, len as usize, "string")?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn skip(file: &mut std::fs::File, kind: u32) -> std::result::Result<(), String> {
        let size = match kind {
            0 | 1 | 7 | 10 | 11 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            12 => 8,
            8 => {
                string(file)?;
                return Ok(());
            }
            9 => {
                let element = u32le(file)?;
                let count = u64le(file)?;
                if element == 9 {
                    return Err("nested arrays are not supported".into());
                }
                if element == 8 {
                    for _ in 0..count.min(MAX_KV_SCAN) {
                        string(file)?;
                    }
                    return Ok(());
                }
                let width = match element {
                    0 | 1 | 7 | 10 | 11 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    12 => 8,
                    other => return Err(format!("unknown GGUF value type {other}")),
                };
                file.seek(SeekFrom::Current((count * width) as i64))
                    .map_err(|err| err.to_string())?;
                return Ok(());
            }
            other => return Err(format!("unknown GGUF value type {other}")),
        };
        file.seek(SeekFrom::Current(size as i64))
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    let mut file = std::fs::File::open(path).map_err(|err| err.to_string())?;
    if read_exact(&mut file, 4, "magic")? != b"GGUF" {
        return Err("not a GGUF file (no GGUF magic)".into());
    }
    let version = u32le(&mut file)?;
    if version < 2 {
        return Err(format!("GGUF v{version} unsupported (need v2+)"));
    }
    let _tensors = u64le(&mut file)?;
    let kv_count = u64le(&mut file)?;
    for _ in 0..kv_count.min(MAX_KV_PAIRS) {
        let key = string(&mut file)?;
        let kind = u32le(&mut file)?;
        if key == "tokenizer.ggml.tokens" {
            if kind != 9 {
                return Err(format!(
                    "tokenizer.ggml.tokens is a type-{kind} array, not strings"
                ));
            }
            if u32le(&mut file)? != 8 {
                return Err("tokenizer.ggml.tokens is not a string array".into());
            }
            let count = u64le(&mut file)?;
            if !(MIN_VOCAB..=MAX_KV_SCAN).contains(&count) {
                return Err(format!("impossible vocab size: {count}"));
            }
            let mut tokens = Vec::with_capacity(count as usize);
            for _ in 0..count {
                tokens.push(string(&mut file)?);
            }
            let specials = tokens
                .iter()
                .filter(|token| token.starts_with('<') && token.ends_with('>'))
                .count();
            if specials * 3 >= tokens.len() {
                return Err(format!(
                    "vocab looks like special tokens only ({specials}/{})",
                    tokens.len()
                ));
            }
            let mut pieces = HashSet::new();
            for raw in tokens {
                if raw.is_empty() || raw.chars().count() > MAX_TOKEN_STR {
                    continue;
                }
                if raw.starts_with('<') && raw.ends_with('>') {
                    continue;
                }
                pieces.insert(raw.clone());
                if raw.contains('\u{2581}') {
                    pieces.insert(raw.replace('\u{2581}', " "));
                }
                if raw.contains('\u{120}') {
                    pieces.insert(raw.replace('\u{120}', " "));
                }
            }
            if pieces.is_empty() {
                return Err("no usable pieces in the vocabulary".into());
            }
            let max_len = pieces
                .iter()
                .map(|piece| piece.chars().count())
                .max()
                .unwrap_or(1);
            let probe: Vec<char> = PROBE.chars().collect();
            let mut index = 0;
            let (mut chars, mut count) = (0usize, 0usize);
            while index < probe.len() {
                let mut matched = None;
                for width in (1..=max_len.min(probe.len() - index)).rev() {
                    let candidate: String = probe[index..index + width].iter().collect();
                    if pieces.contains(&candidate) {
                        matched = Some(width);
                        break;
                    }
                }
                let width = matched.unwrap_or(1);
                chars += width;
                count += 1;
                index += width;
            }
            if count == 0 {
                return Err("the probe tokenised to nothing".into());
            }
            let prior = chars as f64 / count as f64;
            if !(1.0..=64.0).contains(&prior) {
                return Err(format!("implausible prior: {prior:.2} chars/token"));
            }
            return Ok(prior);
        }
        skip(&mut file, kind)?;
    }
    Err(format!("no tokenizer.ggml.tokens array in {path}'s header"))
}

// ── text embeddings ──────────────────────────────────────────────────────

fn nomic_prefixes() -> BTreeSet<String> {
    ["nomic-v1.5"].into_iter().map(str::to_string).collect()
}

fn parse_prefixes(settings: &Settings) -> Result<Map<String, Json>> {
    let raw = settings.embed_prefixes.trim();
    if raw.is_empty() || raw == "none" {
        return Ok(Map::new());
    }
    if nomic_prefixes().contains(raw) {
        return Ok([
            ("Query".to_string(), Json::String("search_query: ".into())),
            (
                "Document".to_string(),
                Json::String("search_document: ".into()),
            ),
        ]
        .into_iter()
        .collect());
    }
    if raw.starts_with('{') {
        return Err(OkfError::new(
            "CONFIG_INVALID",
            "embed_prefixes: inline JSON is not a declarations surface — use 'none', a built-in family name, or 'Task=prefix|Task=prefix' pairs",
        ));
    }
    let mut out = Map::new();
    for pair in raw.split('|') {
        let (key, value) = pair.split_once('=').ok_or_else(|| {
            OkfError::new(
                "CONFIG_INVALID",
                format!("embed_prefixes entry without '=': {pair}"),
            )
        })?;
        let key = key.trim();
        if !matches!(key, "Query" | "Document") {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!(
                    "embed_prefixes task keys must be in [\"Document\", \"Query\"], got '{key}'"
                ),
            ));
        }
        out.insert(key.to_string(), Json::String(value.to_string()));
    }
    Ok(out)
}

pub struct ExternalEmbedder {
    base: String,
    client: Client,
    headers: HeaderMap,
    model_id: String,
    family: String,
    dim: usize,
    native_dim: usize,
    ladder: Vec<usize>,
    max_tokens: usize,
    prefixes: Map<String, Json>,
    counter_declared: String,
    counter_gguf: String,
    chars_per_token: f64,
    identity: Option<Identity>,
    batch_size: usize,
    state: Mutex<ExternalEmbedState>,
}

#[derive(Default)]
struct ExternalEmbedState {
    counter: String,
    counter_ready: bool,
    cpt: f64,
    renorm_fallback: bool,
}

impl fmt::Debug for ExternalEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalEmbedder")
            .field("model_id", &self.model_id)
            .field("family", &self.family)
            .field("dim", &self.dim)
            .field("endpoint", &self.base)
            .finish()
    }
}

impl ExternalEmbedder {
    pub fn open(settings: &Settings) -> Result<Self> {
        let base = normalize_endpoint(&settings.endpoint_url);
        let client = http_client(DEFAULT_TIMEOUT)?;
        let headers = auth_headers(&settings.api_key_env, "embedding")?;
        let mut identity = None;
        if settings.embed_identity == "auto" {
            match probe_identity(&client, &base, &headers) {
                Ok(row) => identity = row,
                Err(err) if !settings.embed_family.is_empty() => {
                    tracing::warn!(
                        "[embedding] identity probe failed ({err}); using the declared '{}' contract",
                        settings.embed_family
                    );
                }
                Err(err) => {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!("[embedding] can't ask the endpoint who it is ({err}) and no embed_family is declared: start the host, or declare the family"),
                    ));
                }
            }
        }
        let (family, dim, native_dim, ladder, max_tokens, prefixes, counter, identity) =
            if let Some(row) = identity {
                if row.kind != "embed" {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "[embedding] {base} serves a '{}' model ({}), not text embeddings",
                            row.kind, row.id
                        ),
                    ));
                }
                if row.id != settings.model_id {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "[embedding] {base} serves {}, but embedding.model_id is {}: point endpoint_url at a host for {}, or set model_id to {}",
                            row.id, settings.model_id, row.id, row.id
                        ),
                    ));
                }
                if row.dim != settings.embedding_dim as usize {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "[embedding] {base} serves {}-wide vectors but the graph is {}-wide: open the host at width {}",
                            row.dim, settings.embedding_dim, settings.embedding_dim
                        ),
                    ));
                }
                if !settings.embed_family.is_empty() {
                    let host_family = if row.family.is_empty() {
                        row.id.rsplit('/').next().unwrap_or_default().to_string()
                    } else {
                        row.family.clone()
                    };
                    if settings.embed_family != host_family {
                        return Err(OkfError::new(
                            "CONFIG_INVALID",
                            format!(
                                "[embedding] embed_family='{}' but the embroider host at {base} serves family '{host_family}': drop embed_family or make it agree",
                                settings.embed_family
                            ),
                        ));
                    }
                }
                if !matches!(settings.embed_prefixes.trim(), "" | "none") {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "[embedding] embed_prefixes='{}', but the embroider host at {base} applies {}'s own prompts: drop embed_prefixes",
                            settings.embed_prefixes, settings.model_id
                        ),
                    ));
                }
                let family = if row.family.is_empty() {
                    settings.model_id.clone()
                } else {
                    row.family.clone()
                };
                (
                    family,
                    row.dim,
                    row.dim,
                    Vec::new(),
                    row.max_length.max(64),
                    Map::new(),
                    "server-tokenize".to_string(),
                    Some(row),
                )
            } else {
                let family = settings.embed_family.clone();
                if family.is_empty() {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        "[embedding] embed_family must be a short declared family name (no '/'), got '' (only an embroider host names its own family)",
                    ));
                }
                let prefixes = parse_prefixes(settings)?;
                if !matches!(
                    settings.embed_pooling.as_str(),
                    "" | "mean" | "last" | "cls"
                ) {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "[embedding] embed_pooling must be mean/last/cls, got '{}'",
                            settings.embed_pooling
                        ),
                    ));
                }
                let dim = settings.embedding_dim as usize;
                let (native_dim, ladder) = if family == "nomic-v1.5" {
                    (768, vec![64, 128, 256, 512, 768])
                } else {
                    (dim, Vec::new())
                };
                if dim != native_dim && !ladder.contains(&dim) {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "{family}: embedding_dim={dim} is neither its native dim {native_dim} nor on its declared Matryoshka ladder {ladder:?}"
                        ),
                    ));
                }
                (
                    family,
                    dim,
                    native_dim,
                    ladder,
                    settings.embed_max_tokens.max(64) as usize,
                    prefixes,
                    settings.embed_counter.clone(),
                    None,
                )
            };
        let state = ExternalEmbedState {
            counter: counter.clone(),
            ..Default::default()
        };
        Ok(Self {
            base,
            client,
            headers,
            model_id: settings.model_id.clone(),
            family,
            dim,
            native_dim,
            ladder,
            max_tokens,
            prefixes,
            counter_declared: counter,
            counter_gguf: settings.embed_counter_gguf.clone(),
            chars_per_token: settings.embed_chars_per_token,
            identity,
            batch_size: (settings.batch_size.max(1) as usize).min(256),
            state: Mutex::new(state),
        })
    }

    fn remedy(&self) -> String {
        format!(
            "start the embedding host at {}, or set embedding.backend='embroider' to run the crate lane in this process",
            self.base
        )
    }

    fn prefix(&self, text: &str, task: &str) -> String {
        let prefix = self.prefixes.get(task).and_then(Json::as_str).or_else(|| {
            if matches!(
                task,
                "Query" | "retrieval-query" | "searchquery" | "question"
            ) {
                self.prefixes.get("Query").and_then(Json::as_str)
            } else {
                None
            }
        });
        match prefix {
            Some(prefix) if !text.starts_with(prefix) => format!("{prefix}{text}"),
            _ => text.to_string(),
        }
    }

    fn post_embed(&self, texts: &[String], task: &str) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let want_mrl = !self.ladder.is_empty() && self.dim != self.native_dim;
        let mut out = Vec::with_capacity(texts.len());
        for part in texts.chunks(self.batch_size) {
            let input: Vec<String> = part.iter().map(|text| self.prefix(text, task)).collect();
            let mut payload = Map::new();
            payload.insert("model".into(), Json::String(self.model_id.clone()));
            payload.insert(
                "input".into(),
                Json::Array(input.iter().cloned().map(Json::String).collect()),
            );
            if self.identity.is_some() {
                payload.insert("task".into(), Json::String(task.to_string()));
            }
            let renorm_fallback = self
                .state
                .lock()
                .map_err(|_| OkfError::internal("external embed state poisoned", "mutex"))?
                .renorm_fallback;
            if want_mrl && !renorm_fallback {
                payload.insert("dimensions".into(), Json::from(self.dim as u64));
            }
            let url = format!("{}/embeddings", self.base);
            let body = post_json(
                &self.client,
                &url,
                &Json::Object(payload),
                &self.headers,
                &self.remedy(),
            )
            .map_err(|Refusal(message)| external_error(message))?;
            let rows = body
                .get("data")
                .and_then(Json::as_array)
                .ok_or_else(|| OkfError::new("INTERNAL", format!("{url}: no data in response")))?;
            if rows.len() != input.len() {
                return Err(OkfError::new(
                    "INTERNAL",
                    format!("expected {} embeddings, got {}", input.len(), rows.len()),
                ));
            }
            for row in rows {
                let vector = row
                    .get("embedding")
                    .and_then(Json::as_array)
                    .ok_or_else(|| {
                        OkfError::new("INTERNAL", format!("{url}: row without an embedding"))
                    })?;
                let values: std::result::Result<Vec<f32>, _> = vector
                    .iter()
                    .map(|value| {
                        value.as_f64().map(|number| number as f32).ok_or_else(|| {
                            OkfError::new("INTERNAL", "embedding value is not a number")
                        })
                    })
                    .collect();
                let values = values?;
                out.push(self.apply_mrl(values, want_mrl, &url)?);
            }
        }
        Ok(out)
    }

    fn apply_mrl(&self, vector: Vec<f32>, want_mrl: bool, url: &str) -> Result<Vec<f32>> {
        if vector.len() == self.dim {
            return Ok(vector);
        }
        if want_mrl
            && vector.len() == self.native_dim
            && (self.dim == self.native_dim || self.ladder.contains(&self.dim))
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| OkfError::internal("external embed state poisoned", "mutex"))?;
            if !state.renorm_fallback {
                state.renorm_fallback = true;
                tracing::warn!(
                    "[embedding] endpoint returned native dim {}; slicing to {} client-side (Matryoshka fallback)",
                    vector.len(),
                    self.dim
                );
            }
            let mut head: Vec<f32> = vector.into_iter().take(self.dim).collect();
            let norm: f32 = head.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in &mut head {
                    *value /= norm;
                }
            }
            return Ok(head);
        }
        Err(OkfError::new(
            "INTERNAL",
            format!(
                "embedding width {} != expected {} (native {}) at {url}",
                vector.len(),
                self.dim,
                self.native_dim
            ),
        ))
    }

    fn ensure_counter(&self) -> Result<()> {
        let ready = self
            .state
            .lock()
            .map_err(|_| OkfError::internal("external embed state poisoned", "mutex"))?
            .counter_ready;
        if ready {
            return Ok(());
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| OkfError::internal("external embed state poisoned", "mutex"))?;
        if state.counter_ready {
            return Ok(());
        }
        let declared = self.counter_declared.clone();
        let mut ok = true;
        match declared.as_str() {
            "server-tokenize" => {
                ok = self.probe_tokenize().is_ok();
            }
            "declared" => {
                state.cpt = self.chars_per_token;
            }
            "gguf-meta" => match chars_per_token_from_gguf(&self.counter_gguf) {
                Ok(prior) => {
                    state.cpt = prior;
                }
                Err(err) => {
                    tracing::warn!(
                        "[embedding] embed_counter_gguf={}: {err}; degrading to the heuristic",
                        self.counter_gguf
                    );
                    ok = false;
                }
            },
            "heuristic" => {}
            other => {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!("[embedding] embed_counter must be one of [server-tokenize, gguf-meta, declared, heuristic], got '{other}'"),
                ));
            }
        }
        if !ok {
            state.counter = "heuristic-fallback".into();
            tracing::warn!(
                "[embedding] embed_counter={declared} is unavailable; token counts degrade to the len/4 heuristic — budgets are approximate"
            );
        } else if declared == "heuristic" {
            tracing::warn!(
                "[embedding] external engine '{}' counts tokens heuristically (len/4); budgets are approximate",
                self.family
            );
        }
        state.counter_ready = true;
        Ok(())
    }

    fn probe_tokenize(&self) -> Result<usize> {
        self.server_count(EMBED_TOKENIZE_PROBE)
    }

    fn server_count(&self, text: &str) -> Result<usize> {
        let url = tokenize_url(&self.base);
        let payload = json!({"content": text});
        let body = post_json(&self.client, &url, &payload, &self.headers, &self.remedy())
            .map_err(|Refusal(message)| external_error(message))?;
        token_count(&body).ok_or_else(|| {
            OkfError::new("INTERNAL", format!("{url}: answer carried no token list"))
        })
    }

    fn count(&self, text: &str) -> Result<usize> {
        self.ensure_counter()?;
        let state = self
            .state
            .lock()
            .map_err(|_| OkfError::internal("external embed state poisoned", "mutex"))?;
        match state.counter.as_str() {
            "server-tokenize" => {
                drop(state);
                self.server_count(text)
            }
            "gguf-meta" | "declared" => {
                let cpt = state.cpt;
                Ok((text.chars().count() as f64 / cpt).ceil().max(1.0) as usize)
            }
            _ => Ok((text.chars().count() / 4).max(8)),
        }
    }

    pub fn counter(&self) -> Result<String> {
        self.ensure_counter()?;
        Ok(self
            .state
            .lock()
            .map_err(|_| OkfError::internal("external embed state poisoned", "mutex"))?
            .counter
            .clone())
    }

    pub fn honours_dimensions(&self) -> bool {
        self.state
            .lock()
            .map(|state| !state.renorm_fallback)
            .unwrap_or(true)
    }

    pub fn spec_report(&self) -> Result<Json> {
        let counter = self.counter()?;
        let mut report = json!({
            "backend": "external",
            "family": self.family,
            "model_id": self.model_id,
            "endpoint": self.base,
            "dim": self.dim,
            "native_dim": self.native_dim,
            "matryoshka": self.ladder,
            "max_tokens": self.max_tokens,
            "pooling": if self.identity.is_some() { "host" } else { "declared" },
            "prefixes": self.prefixes,
            "counter": self.counter_declared,
            "counter_gguf": self.counter_gguf,
            "counter_effective": counter,
            "identity": if self.identity.is_some() { "embroider" } else { "declared" },
        });
        if let Some(identity) = &self.identity {
            report["precision"] = Json::String(identity.precision.clone());
            report["space"] = identity.space.clone().map_or(Json::Null, Json::String);
            report["engine_version"] = identity
                .engine_version
                .clone()
                .map_or(Json::Null, Json::String);
            report["device"] = Json::String(identity.device.clone());
        }
        Ok(report)
    }

    pub fn encode_one(&self, text: &str, task: &str) -> Result<Vec<f32>> {
        self.post_embed(&[text.to_string()], task)?
            .into_iter()
            .next()
            .ok_or_else(|| OkfError::new("INTERNAL", "external embedder returned no vector"))
    }

    pub fn encode_many(&self, texts: &[String], task: &str) -> Result<Vec<Vec<f32>>> {
        self.post_embed(texts, task)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        self.count(text)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn precision(&self) -> &str {
        self.identity
            .as_ref()
            .map(|row| row.precision.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("external")
    }

    pub fn space(&self) -> Option<String> {
        self.identity.as_ref().and_then(|row| row.space.clone())
    }

    pub fn used_cuda(&self) -> bool {
        false
    }

    pub fn max_len(&self) -> usize {
        self.max_tokens
    }
}

// ── rerank ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RerankDialect {
    Standard,
    Tei,
}

pub struct ExternalReranker {
    base: String,
    standard_url: String,
    tei_url: String,
    count_url: String,
    client: Client,
    headers: HeaderMap,
    model_id: String,
    family: String,
    max_tokens: usize,
    count: String,
    identity: Option<Identity>,
    state: Mutex<RerankState>,
}

#[derive(Default)]
struct RerankState {
    dialect: Option<RerankDialect>,
    down: Option<String>,
    count: String,
    count_probe_done: bool,
}

impl fmt::Debug for ExternalReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalReranker")
            .field("model_id", &self.model_id)
            .field("endpoint", &self.standard_url)
            .field("family", &self.family)
            .finish()
    }
}

impl ExternalReranker {
    pub fn open(settings: &Settings) -> Result<Self> {
        let base = normalize_endpoint(&settings.rerank_endpoint_url)
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string();
        let client = http_client(DEFAULT_TIMEOUT)?;
        let headers = auth_headers(&settings.rerank_api_key_env, "rerank")?;
        let mut identity = None;
        if settings.rerank_identity == "auto" {
            match probe_identity(&client, &base, &headers) {
                Ok(row) => identity = row,
                Err(err) if !settings.rerank_family.is_empty() => {
                    tracing::warn!(
                        "[rerank] identity probe failed ({err}); using the declared '{}' contract",
                        settings.rerank_family
                    );
                }
                Err(err) => {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!("[rerank] can't ask the endpoint who it is ({err}) and no rerank_family is declared: start the host, or declare the family"),
                    ));
                }
            }
        }
        let (family, max_tokens, count, identity) = if let Some(row) = identity {
            if row.kind != "rerank" {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!(
                        "[rerank] {base} serves a '{}' model ({}), not a reranker",
                        row.kind, row.id
                    ),
                ));
            }
            if row.id != settings.rerank_model_id {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!(
                        "[rerank] {base} serves {}, but rerank_model_id is {}: point rerank_endpoint_url at a host for {}, or set rerank_model_id to {}",
                        row.id, settings.rerank_model_id, row.id, row.id
                    ),
                ));
            }
            let host_family = if row.family.is_empty() {
                row.id.rsplit('/').next().unwrap_or_default().to_string()
            } else {
                row.family.clone()
            };
            if !settings.rerank_family.is_empty() && settings.rerank_family != host_family {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!(
                        "[rerank] rerank_family='{}' but the embroider host at {base} serves family '{host_family}': drop rerank_family or make it agree",
                        settings.rerank_family
                    ),
                ));
            }
            (
                host_family,
                row.max_length.max(64),
                "server-tokenize".to_string(),
                Some(row),
            )
        } else {
            if settings.rerank_family.is_empty() {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    "[rerank] backend='external' needs rerank_family (the provenance label, e.g. 'bge-v2-m3')",
                ));
            }
            (
                settings.rerank_family.clone(),
                settings.rerank_max_tokens.max(64) as usize,
                settings.rerank_count.clone(),
                None,
            )
        };
        Ok(Self {
            standard_url: format!("{base}/v1/rerank"),
            tei_url: format!("{base}/rerank"),
            count_url: format!("{base}/tokenize"),
            base: base.to_string(),
            client,
            headers,
            model_id: settings.rerank_model_id.clone(),
            family,
            max_tokens,
            count,
            identity,
            state: Mutex::new(RerankState {
                count: String::new(),
                ..Default::default()
            }),
        })
    }

    fn remedy(&self) -> String {
        format!(
            "start the rerank host at {}, or set search.rerank_backend='embroider'",
            self.base
        )
    }

    fn trip(&self, reason: &str) -> OkfError {
        let mut state = self.state.lock().expect("rerank state mutex");
        if state.down.is_none() {
            state.down = Some(reason.chars().take(200).collect());
            tracing::warn!(
                "[rerank] external engine '{}' failed ({reason}); rerank disabled for this session — keyword + vector search still answers",
                self.family
            );
        }
        OkfError::new("INTERNAL", reason)
    }

    fn down_error(&self) -> OkfError {
        let reason = self
            .state
            .lock()
            .expect("rerank state mutex")
            .down
            .clone()
            .unwrap_or_default();
        OkfError::new(
            "INTERNAL",
            format!("rerank disabled for the session: {reason}"),
        )
    }

    fn parse_rows(rows: &[Json], n: usize, url: &str) -> std::result::Result<Vec<f64>, Refusal> {
        let mut scores = vec![None; n];
        for row in rows {
            let object = row
                .as_object()
                .ok_or_else(|| Refusal(format!("{url} row is not an object")))?;
            let index = object
                .get("index")
                .and_then(Json::as_u64)
                .ok_or_else(|| Refusal(format!("{url} bad rerank row")))?
                as usize;
            if index >= n || scores[index].is_some() {
                return Err(Refusal(format!(
                    "{url} rerank index coverage broken at {index}"
                )));
            }
            let score = object
                .get("relevance_score")
                .or_else(|| object.get("score"))
                .and_then(Json::as_f64)
                .ok_or_else(|| Refusal(format!("{url} row carries no score")))?;
            scores[index] = Some(score);
        }
        scores
            .into_iter()
            .map(|value| value.ok_or_else(|| Refusal(format!("{url} missing document score(s)"))))
            .collect()
    }

    fn post_standard(
        &self,
        query: &str,
        docs: &[String],
    ) -> std::result::Result<Vec<f64>, Refusal> {
        let payload = json!({
            "model": self.model_id,
            "query": query,
            "documents": docs,
            "top_n": docs.len(),
        });
        let body = post_json(
            &self.client,
            &self.standard_url,
            &payload,
            &self.headers,
            &self.remedy(),
        )?;
        let rows = body
            .get("results")
            .and_then(Json::as_array)
            .ok_or_else(|| {
                Refusal(format!(
                    "{}: unexpected rerank response shape",
                    self.standard_url
                ))
            })?;
        if rows.len() != docs.len() {
            return Err(Refusal(format!(
                "expected {} scored documents, got {}",
                docs.len(),
                rows.len()
            )));
        }
        Self::parse_rows(rows, docs.len(), &self.standard_url)
    }

    fn post_tei(&self, query: &str, docs: &[String]) -> std::result::Result<Vec<f64>, Refusal> {
        let payload = json!({
            "query": query,
            "texts": docs,
            "truncate": true,
        });
        let body = post_json(
            &self.client,
            &self.tei_url,
            &payload,
            &self.headers,
            &self.remedy(),
        )?;
        let rows = if let Some(rows) = body.as_array() {
            rows
        } else {
            body.get("results")
                .and_then(Json::as_array)
                .ok_or_else(|| {
                    Refusal(format!(
                        "{}: unexpected TEI rerank response shape",
                        self.tei_url
                    ))
                })?
        };
        if rows.len() != docs.len() {
            return Err(Refusal(format!(
                "expected {} scored documents, got {}",
                docs.len(),
                rows.len()
            )));
        }
        Self::parse_rows(rows, docs.len(), &self.tei_url)
    }

    fn post_rerank(&self, query: &str, docs: &[String]) -> Result<Vec<f64>> {
        let dialect = self.state.lock().expect("rerank state mutex").dialect;
        match dialect {
            Some(RerankDialect::Standard) => self
                .post_standard(query, docs)
                .map_err(|Refusal(message)| self.trip(&message)),
            Some(RerankDialect::Tei) => self
                .post_tei(query, docs)
                .map_err(|Refusal(message)| self.trip(&message)),
            None => match self.post_standard(query, docs) {
                Ok(scores) => {
                    self.state.lock().expect("rerank state mutex").dialect =
                        Some(RerankDialect::Standard);
                    Ok(scores)
                }
                Err(first) => match self.post_tei(query, docs) {
                    Ok(scores) => {
                        self.state.lock().expect("rerank state mutex").dialect =
                            Some(RerankDialect::Tei);
                        Ok(scores)
                    }
                    Err(second) => Err(self.trip(&format!(
                        "{} | TEI fallback at {} also failed: {}",
                        first.0, self.tei_url, second.0
                    ))),
                },
            },
        }
    }

    pub fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f64>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if self
            .state
            .lock()
            .expect("rerank state mutex")
            .down
            .is_some()
        {
            return Err(self.down_error());
        }
        // Group by query; this lane scores one query per request while the
        // caller may pass pairs from several queries.
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (index, doc) in docs.iter().enumerate() {
            let _ = doc;
            if let Some((_, indices)) = groups.last_mut() {
                // All scores in this call share the query argument.
                indices.push(index);
            } else {
                groups.push((query.to_string(), vec![index]));
            }
        }
        let mut out = vec![0.0; docs.len()];
        for (group_query, indices) in groups {
            let batch: Vec<String> = indices.iter().map(|index| docs[*index].clone()).collect();
            let scores = self.post_rerank(&group_query, &batch)?;
            for (index, score) in indices.into_iter().zip(scores) {
                out[index] = score;
            }
        }
        Ok(out)
    }

    fn server_count(&self, text: &str) -> std::result::Result<usize, Refusal> {
        let body = post_json(
            &self.client,
            &self.count_url,
            &json!({"content": text}),
            &self.headers,
            &self.remedy(),
        )?;
        token_count(&body).ok_or_else(|| {
            Refusal(format!(
                "{}: unexpected /tokenize response shape",
                self.count_url
            ))
        })
    }

    pub fn count_tokens(&self, text: &str, cap: usize) -> usize {
        let chars = text.chars().count();
        if self.count == "heuristic" {
            return (chars / 4).max(8);
        }
        let half = cap.saturating_sub(4) / 2;
        if chars <= half {
            return (chars / 4).max(8);
        }
        if chars > cap {
            return chars;
        }
        let mut state = self.state.lock().expect("rerank state mutex");
        if state.count == "heuristic-fallback" {
            return (chars / 4).max(8);
        }
        if !state.count_probe_done {
            if self.server_count(RERANK_TOKENIZE_PROBE).is_err() {
                state.count = "heuristic-fallback".into();
                tracing::warn!(
                    "[rerank] external engine '{}': token counting on {} is unavailable; token counts degrade to the len/4 heuristic",
                    self.family,
                    self.count_url
                );
                return (chars / 4).max(8);
            }
            state.count_probe_done = true;
        }
        drop(state);
        match self.server_count(text) {
            Ok(count) => count,
            Err(_) => {
                self.state.lock().expect("rerank state mutex").count = "heuristic-fallback".into();
                (chars / 4).max(8)
            }
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn max_length(&self) -> usize {
        self.max_tokens
    }

    pub fn precision(&self) -> Option<&str> {
        self.identity
            .as_ref()
            .map(|row| row.precision.as_str())
            .filter(|value| !value.is_empty())
    }

    pub fn spec_report(&self) -> Json {
        let state = self.state.lock().expect("rerank state mutex");
        json!({
            "backend": "external",
            "family": self.family,
            "model_id": self.model_id,
            "endpoint": self.standard_url,
            "max_tokens": self.max_tokens,
            "windowing": null,
            "count": if state.count.is_empty() { self.count.clone() } else { state.count.clone() },
            "count_declared": self.count,
            "dialect": state.dialect.map(|d| if d == RerankDialect::Standard { "standard" } else { "tei" }),
            "down": state.down.is_some(),
            "down_reason": state.down,
            "identity": if self.identity.is_some() { "embroider" } else { "declared" },
        })
    }
}

// ── vision ───────────────────────────────────────────────────────────────

pub struct ExternalVision {
    base: String,
    client: Client,
    headers: HeaderMap,
    model_id: String,
    dim: usize,
    precision: String,
    space: Option<String>,
    text_partner: String,
    text_space: Option<String>,
    used_cuda: bool,
    identity: Identity,
    broken: Mutex<Option<String>>,
}

impl fmt::Debug for ExternalVision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalVision")
            .field("model_id", &self.model_id)
            .field("endpoint", &self.base)
            .field("dim", &self.dim)
            .finish()
    }
}

impl ExternalVision {
    pub fn open(settings: &Settings) -> Result<Self> {
        let base = normalize_endpoint(&settings.image_endpoint_url);
        let client = http_client(DEFAULT_TIMEOUT)?;
        let headers = auth_headers(&settings.api_key_env, "embedding")?;
        let url = format!("{base}/models");
        let body = match get_json(&client, &url, &headers) {
            Ok(body) => body,
            Err(Refusal(message)) => {
                return Err(OkfError::new(
                    "EXTERNAL_UNAVAILABLE",
                    format!("{url} is unreachable: {message}"),
                )
                .field("endpoint", Json::String(url))
                .remedy("start the vision host, or set embedding.image_backend='embroider'"));
            }
        };
        let identity = body
            .get("data")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
            .filter_map(Json::as_object)
            .find(|row| row.get("engine").and_then(Json::as_str) == Some("embroider"))
            .map(|row| Identity::from_row(&Json::Object(row.clone())))
            .ok_or_else(|| {
                OkfError::new(
                    "CONFIG_INVALID",
                    format!("[embedding] {base} did not describe itself as an embroider host (GET /v1/models): an external image lane needs one, because image vectors are only meaningful against the text model the host names — set image_backend='embroider', or point image_endpoint_url at `python -m embroider.serve --kind vision`"),
                )
            })?;
        if identity.kind != "vision" {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!(
                    "[embedding] {base} serves a '{}' model ({}), not an image one: image_endpoint_url needs a vision host (embroider.serve --kind vision)",
                    identity.kind, identity.id
                ),
            ));
        }
        if identity.id.is_empty() {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!("[embedding] {base} named no model id; a vision host must describe itself (GET /v1/models)"),
            ));
        }
        if let Some(wanted) = &settings.image_model_id {
            if *wanted != identity.id {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!(
                        "[embedding] {base} serves {}, but embedding.image_model_id is {wanted}: point image_endpoint_url at a host for {wanted}, or drop image_model_id to accept the host's model",
                        identity.id
                    ),
                ));
            }
        }
        if identity.dim != settings.embedding_dim as usize {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!(
                    "[embedding] {base} serves {}-wide image vectors but the graph is {}-wide: the image route has no dimensions parameter, so open the host at the graph's width",
                    identity.dim, settings.embedding_dim
                ),
            ));
        }
        if settings.image_precision != "auto" && identity.precision != settings.image_precision {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!(
                    "[embedding] image_precision={} but the vision host at {base} landed {}: set image_precision='auto' or start the host on the matching device/precision",
                    settings.image_precision, identity.precision
                ),
            ));
        }
        let text_partner = identity.text_partner.clone().ok_or_else(|| {
            OkfError::new(
                "IMAGE_PIN_MISMATCH",
                format!(
                    "the vision host at {base} names no text partner for {} — image vectors would compare against nothing",
                    identity.id
                ),
            )
            .field("image_model", Json::String(identity.id.clone()))
            .remedy("use mode=text (captions), or host a partnered vision model")
        })?;
        Ok(Self {
            base,
            client,
            headers,
            model_id: identity.id.clone(),
            dim: identity.dim,
            precision: if identity.precision.is_empty() {
                "external".into()
            } else {
                identity.precision.clone()
            },
            space: identity.space.clone(),
            text_partner,
            text_space: identity.text_partner_space.clone(),
            used_cuda: identity.device == "cuda",
            identity,
            broken: Mutex::new(None),
        })
    }

    fn remedy(&self) -> String {
        format!(
            "start the vision host at {}, or set embedding.image_backend='embroider'",
            self.base
        )
    }

    pub fn encode_image(&self, data: &[u8]) -> Result<Vec<f32>> {
        self.encode_images(&[data])?
            .into_iter()
            .next()
            .ok_or_else(|| OkfError::new("INTERNAL", "external vision host returned no vector"))
    }

    pub fn encode_images(&self, items: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        if let Some(reason) = self.broken.lock().expect("vision state mutex").clone() {
            return Err(OkfError::new("EXTERNAL_UNAVAILABLE", reason));
        }
        let encoded: Vec<String> = items
            .iter()
            .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes))
            .collect();
        let payload = json!({"model": self.model_id, "image": encoded});
        let url = format!("{}/images/embeddings", self.base);
        let body = match post_json(&self.client, &url, &payload, &self.headers, &self.remedy()) {
            Ok(body) => body,
            Err(Refusal(message)) => {
                if message.contains("is unreachable") || message.contains("EXTERNAL_UNAVAILABLE") {
                    *self.broken.lock().expect("vision state mutex") = Some(message.clone());
                }
                return Err(external_error(message));
            }
        };
        let rows = body
            .get("data")
            .and_then(Json::as_array)
            .ok_or_else(|| OkfError::new("INTERNAL", format!("{url}: no data in response")))?;
        if rows.len() != items.len() {
            return Err(OkfError::new(
                "INTERNAL",
                format!(
                    "{url} answered {} embeddings for {} images",
                    rows.len(),
                    items.len()
                ),
            ));
        }
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let values: Vec<f32> =
                row.get("embedding")
                    .and_then(Json::as_array)
                    .ok_or_else(|| {
                        OkfError::new("INTERNAL", format!("{url}: row without an embedding"))
                    })?
                    .iter()
                    .map(|value| {
                        value.as_f64().map(|number| number as f32).ok_or_else(|| {
                            OkfError::new("INTERNAL", "embedding value is not a number")
                        })
                    })
                    .collect::<Result<_>>()?;
            if values.len() != self.dim {
                return Err(OkfError::new(
                    "INTERNAL",
                    format!(
                        "{url} returned a {}-wide vector, expected {}",
                        values.len(),
                        self.dim
                    ),
                ));
            }
            out.push(values);
        }
        Ok(out)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    pub fn precision(&self) -> &str {
        &self.precision
    }

    pub fn space(&self) -> Option<&str> {
        self.space.as_deref()
    }

    pub fn text_partner(&self) -> &str {
        &self.text_partner
    }

    pub fn text_space(&self) -> Option<&str> {
        self.text_space.as_deref()
    }

    pub fn spec_report(&self) -> Json {
        json!({
            "backend": "external",
            "model_id": self.model_id,
            "endpoint": self.base,
            "identity": "embroider",
            "kind": self.identity.kind,
            "precision": self.precision,
            "space": self.space,
            "text_space": self.text_space,
            "text_partner": self.text_partner,
            "dim": self.dim,
            "device": self.identity.device,
            "engine_version": self.identity.engine_version,
            "contract": self.identity.contract,
            "revision": self.identity.revision,
            "broken": self.broken.lock().expect("vision state mutex").is_some(),
        })
    }
}

impl Embedder for ExternalEmbedder {
    fn encode(&self, text: &str, task: Task) -> Result<Vec<f32>> {
        ExternalEmbedder::encode_one(self, text, task.as_str())
    }

    fn encode_many(&self, texts: &[String], task: Task) -> Result<Vec<Vec<f32>>> {
        ExternalEmbedder::encode_many(self, texts, task.as_str())
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        ExternalEmbedder::count_tokens(self, text)
    }

    fn dim(&self) -> usize {
        ExternalEmbedder::dim(self)
    }

    fn model_id(&self) -> &str {
        ExternalEmbedder::model_id(self)
    }

    fn used_cuda(&self) -> bool {
        ExternalEmbedder::used_cuda(self)
    }

    fn precision(&self) -> &str {
        ExternalEmbedder::precision(self)
    }

    fn max_len(&self) -> usize {
        ExternalEmbedder::max_len(self)
    }

    fn space(&self) -> Option<String> {
        ExternalEmbedder::space(self)
    }

    fn counter(&self) -> String {
        ExternalEmbedder::counter(self).unwrap_or_else(|_| self.counter_declared.clone())
    }

    fn honours_dimensions(&self) -> bool {
        ExternalEmbedder::honours_dimensions(self)
    }

    fn supports_dim(&self, dim: usize) -> bool {
        dim == self.native_dim || self.ladder.contains(&dim)
    }

    fn declared_ladder(&self) -> Vec<usize> {
        if self.identity.is_some() {
            Vec::new()
        } else if self.ladder.is_empty() {
            vec![self.native_dim]
        } else {
            self.ladder.clone()
        }
    }

    fn family(&self) -> String {
        self.family.clone()
    }

    fn spec_report(&self) -> Result<Json> {
        ExternalEmbedder::spec_report(self)
    }
}

impl Reranker for ExternalReranker {
    fn score_many(&self, pairs: &[(&str, &str)]) -> Result<Vec<f64>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        if self
            .state
            .lock()
            .expect("rerank state mutex")
            .down
            .is_some()
        {
            return Err(self.down_error());
        }
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (index, (query, _doc)) in pairs.iter().enumerate() {
            if let Some((known, indices)) = groups.iter_mut().find(|(known, _)| known == query) {
                let _ = known;
                indices.push(index);
            } else {
                groups.push(((*query).to_string(), vec![index]));
            }
        }
        let mut out = vec![0.0; pairs.len()];
        for (query, indices) in groups {
            let docs: Vec<String> = indices
                .iter()
                .map(|index| pairs[*index].1.to_string())
                .collect();
            let scores = self.post_rerank(&query, &docs)?;
            for (index, score) in indices.into_iter().zip(scores) {
                out[index] = score;
            }
        }
        Ok(out)
    }

    fn count_tokens(&self, text: &str, cap: usize) -> usize {
        ExternalReranker::count_tokens(self, text, cap)
    }

    fn model_id(&self) -> &str {
        ExternalReranker::model_id(self)
    }

    fn max_length(&self) -> usize {
        ExternalReranker::max_length(self)
    }

    fn precision(&self) -> Option<&str> {
        ExternalReranker::precision(self)
    }

    fn spec_report(&self) -> Json {
        ExternalReranker::spec_report(self)
    }
}

impl VisionEncoder for ExternalVision {
    fn encode_image(&self, data: &[u8]) -> Result<Vec<f32>> {
        ExternalVision::encode_image(self, data)
    }

    fn model_id(&self) -> &str {
        ExternalVision::model_id(self)
    }

    fn dim(&self) -> usize {
        ExternalVision::dim(self)
    }

    fn used_cuda(&self) -> bool {
        ExternalVision::used_cuda(self)
    }

    fn precision(&self) -> &str {
        ExternalVision::precision(self)
    }

    fn space(&self) -> Option<String> {
        ExternalVision::space(self).map(str::to_string)
    }

    fn text_partner(&self) -> Option<&str> {
        Some(ExternalVision::text_partner(self))
    }

    fn text_space(&self) -> Option<&str> {
        ExternalVision::text_space(self)
    }

    fn spec_report(&self) -> Json {
        ExternalVision::spec_report(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_counter_reads_a_vocabulary_header() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one KV pair
        let key = b"tokenizer.ggml.tokens";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array
        bytes.extend_from_slice(&8u32.to_le_bytes()); // string elements
        let tokens: Vec<String> = (0..1000).map(|index| format!("word{index} ")).collect();
        bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in &tokens {
            bytes.extend_from_slice(&(token.len() as u64).to_le_bytes());
            bytes.extend_from_slice(token.as_bytes());
        }
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();

        let prior = chars_per_token_from_gguf(path.to_str().unwrap()).unwrap();
        assert!((1.0..=64.0).contains(&prior), "{prior}");
    }

    #[test]
    fn endpoint_normalization_matches_oracle() {
        assert_eq!(normalize_endpoint("http://h:9"), "http://h:9/v1");
        assert_eq!(normalize_endpoint("http://h:9/v1"), "http://h:9/v1");
        assert_eq!(
            normalize_endpoint("http://h:9/v1/embeddings"),
            "http://h:9/v1"
        );
        assert_eq!(tokenize_url("http://h:9/v1"), "http://h:9/tokenize");
    }

    #[test]
    fn token_count_accepts_both_host_shapes() {
        assert_eq!(token_count(&json!({"count": 7})), Some(7));
        assert_eq!(token_count(&json!({"tokens": [1, 2, 3]})), Some(3));
        assert_eq!(token_count(&json!({})), None);
    }

    #[test]
    fn prefixes_are_declared_never_guessed() {
        let mut settings = Settings {
            embed_prefixes: "Query=search_query: |Document=search_document:".into(),
            ..Default::default()
        };
        let prefixes = parse_prefixes(&settings).unwrap();
        assert_eq!(prefixes["Query"], json!("search_query: "));
        settings.embed_prefixes = "{".into();
        assert!(parse_prefixes(&settings).is_err());
    }

    fn mock_embed_host() -> (String, std::sync::mpsc::Receiver<Json>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let mut stream = stream.unwrap();
                let mut buffer = [0u8; 8192];
                let len = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..len]).to_string();
                let marker = "\r\n\r\n";
                let body_start = request.find(marker).unwrap() + marker.len();
                let payload: Json = serde_json::from_str(&request[body_start..]).unwrap();
                let dim = payload
                    .get("dimensions")
                    .and_then(Json::as_u64)
                    .unwrap_or(768) as usize;
                sender.send(payload).unwrap();
                let response = json!({
                    "data": [{"embedding": vec![0.5; dim]}]
                });
                let body = response.to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
        });
        (url, receiver)
    }

    #[test]
    fn declared_embed_lane_sends_prefixes_and_dimensions() {
        let (url, receiver) = mock_embed_host();
        let settings = Settings {
            backend: "external".into(),
            endpoint_url: url,
            embed_identity: "declared".into(),
            embed_family: "nomic-v1.5".into(),
            embed_prefixes: "Query=search_query: |Document=search_document:".into(),
            embed_counter: "heuristic".into(),
            embedding_dim: 512,
            ..Default::default()
        };
        let engine = ExternalEmbedder::open(&settings).unwrap();
        let vector = engine.encode_one("fox", "Query").unwrap();
        assert_eq!(vector.len(), 512);
        let payload = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(payload["input"][0], json!("search_query: fox"));
        assert_eq!(payload["dimensions"], json!(512));
        assert!(payload.get("task").is_none());
        assert!(engine.honours_dimensions());
    }

    #[test]
    fn declared_embed_lane_maps_outages_to_external_unavailable() {
        let settings = Settings {
            backend: "external".into(),
            endpoint_url: "http://127.0.0.1:9".into(),
            embed_identity: "declared".into(),
            embed_family: "nomic-v1.5".into(),
            embed_counter: "heuristic".into(),
            embedding_dim: 512,
            ..Default::default()
        };
        let engine = ExternalEmbedder::open(&settings).unwrap();
        let err = engine.encode_one("fox", "Document").unwrap_err();
        assert_eq!(err.code, "EXTERNAL_UNAVAILABLE");
    }
}
