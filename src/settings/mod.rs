//! One settings table for OKFgraph, generated from the Python 0.15 oracle.
//!
//! Canonical names are the `OKFRouter` constructor kwargs. CLI flags, env vars,
//! TOML sections and validation all derive from the same table.

use crate::error::{OkfError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingKind {
    Str,
    Int,
    Float,
    Bool,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultValue {
    Str(&'static str),
    Int(i64),
    Float(f64),
    Bool(bool),
    List,
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SettingDef {
    pub name: &'static str,
    pub kind: SettingKind,
    pub default: DefaultValue,
    pub section: &'static str,
    pub help: &'static str,
    pub choices: &'static [&'static str],
    pub cli: bool,
    pub mcp: bool,
    pub flag: Option<&'static str>,
    pub dest: Option<&'static str>,
    pub invert: bool,
    pub required_mcp: bool,
}

impl SettingDef {
    pub fn env_name(&self) -> String {
        env_name(self.name)
    }

    pub fn flag_name(&self) -> String {
        self.flag
            .map(str::to_string)
            .unwrap_or_else(|| flag_name(self.name))
    }

    pub fn dest_name(&self) -> String {
        self.dest
            .map(str::to_string)
            .unwrap_or_else(|| self.name.to_string())
    }

    pub fn default_value(&self) -> Value {
        match self.default {
            DefaultValue::Str(s) => Value::String(s.to_string()),
            DefaultValue::Int(i) => Value::from(i),
            DefaultValue::Float(f) => serde_json::json!(f),
            DefaultValue::Bool(b) => Value::Bool(b),
            DefaultValue::List => Value::Array(Vec::new()),
            DefaultValue::Null => Value::Null,
        }
    }
}

include!("table.rs");

pub const DEVICES: &[&str] = &["auto", "cpu", "cuda"];
pub const PRECISIONS: &[&str] = &["auto", "fp32", "fp16", "int8"];
pub const DIM_LADDER: &[i64] = &[32, 64, 128, 256, 512, 768, 1024];
pub const GRAPH_HOME: &str = ".okfgraph";
pub const DEFAULT_DB_PATH: &str = ".okfgraph/okfgraph.db";
pub const LEGACY_DB_PATH: &str = "okfgraph.db";
pub const DEFAULT_MODEL_ID: &str = "google/embeddinggemma-2";
pub const DEFAULT_RERANK_MODEL_ID: &str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";

pub fn row(name: &str) -> Option<&'static SettingDef> {
    SETTINGS.iter().find(|row| row.name == name)
}

pub fn env_name(name: &str) -> String {
    format!("OKFGRAPH_{}", name.to_ascii_uppercase())
}

pub fn flag_name(name: &str) -> String {
    format!("--{}", name.replace('_', "-"))
}

/// Flat canonical settings. Field names are the documented Python kwargs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub db_path: String,
    pub bundle_root: Option<String>,
    pub wal_mode: bool,
    pub extension_dir: Option<String>,
    pub embedding_dim: i64,
    pub model_id: String,
    pub device: String,
    pub backend: String,
    pub endpoint_url: String,
    pub api_key_env: String,
    pub embed_identity: String,
    pub embed_family: String,
    pub embed_pooling: String,
    pub embed_prefixes: String,
    pub embed_max_tokens: i64,
    pub embed_counter: String,
    pub embed_counter_gguf: String,
    pub embed_chars_per_token: f64,
    pub precision: String,
    pub image_model_id: Option<String>,
    pub image_precision: String,
    pub image_backend: String,
    pub image_endpoint_url: String,
    pub cpu_arena: bool,
    pub cache_dir: Option<String>,
    pub max_length: Option<i64>,
    pub enable_chunking: bool,
    pub chunk_size: i64,
    pub chunk_overlap: i64,
    pub allow_remote_images: bool,
    pub allowed_image_domains: Vec<String>,
    pub mode: String,
    pub batch_size: i64,
    pub identity: String,
    pub pdf_device: String,
    pub pdf_precision: String,
    pub pdf_idle_seconds: i64,
    pub exclude_tags: Vec<String>,
    pub rerank_model_id: String,
    pub rerank_device: String,
    pub rerank_precision: String,
    pub rerank_top_n: i64,
    pub rerank_window: Option<i64>,
    pub rerank_stride: Option<i64>,
    pub rerank_backend: String,
    pub rerank_endpoint_url: String,
    pub rerank_api_key_env: String,
    pub rerank_identity: String,
    pub rerank_family: String,
    pub rerank_max_tokens: i64,
    pub rerank_count: String,
    pub efs: Option<i64>,
    pub roots: BTreeMap<String, String>,
}

impl Default for Settings {
    fn default() -> Self {
        let mut map = Map::new();
        for def in SETTINGS {
            map.insert(def.name.to_string(), def.default_value());
        }
        map.insert("roots".into(), Value::Object(Map::new()));
        Self::from_map(&map).expect("generated defaults are valid")
    }
}

impl Settings {
    /// Build from canonical JSON values; every omitted field uses its default.
    ///
    /// This is the single validation choke point for every public surface:
    /// unknown keys are refused, CLI-style strings/lists are normalized with
    /// the generated table, and `checks()` runs before a `Settings` value can
    /// reach a router.
    pub fn from_map(map: &Map<String, Value>) -> Result<Self> {
        let mut taken = Map::new();
        for (key, raw) in map {
            let key = if key == "embed_backend" {
                "backend"
            } else {
                key.as_str()
            };
            if key == "roots" {
                taken.insert("roots".into(), parse_roots(raw).map_err(config_error)?);
                continue;
            }
            if key == "model_path" || key == "tokenizer_path" {
                if !raw.is_null() {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!(
                            "{key} is not supported by the Rust core: explicit local ONNX files are unsupported; use a registry model id or an external backend"
                        ),
                    ));
                }
                continue;
            }
            let def = row(key).ok_or_else(|| {
                OkfError::new("CONFIG_INVALID", format!("unknown setting: {key}"))
            })?;
            if key == "bundle_root" {
                handle_bundle_root(raw).map_err(config_error)?;
            }
            let parsed = if raw.is_null() && matches!(def.default, DefaultValue::Null) {
                // Optional ints/strs legitimately use JSON null (`max_length`,
                // `efs`, rerank windows, ...). They bypass table parsing.
                Value::Null
            } else {
                parse_value(def, raw.clone(), &flag_name(def.name)).map_err(config_error)?
            };
            taken.insert(def.name.to_string(), parsed);
        }

        let mut full = Map::new();
        for def in SETTINGS {
            full.insert(
                def.name.to_string(),
                taken
                    .get(def.name)
                    .cloned()
                    .unwrap_or_else(|| def.default_value()),
            );
        }
        full.insert(
            "roots".to_string(),
            taken
                .get("roots")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
        );
        let settings: Self = serde_json::from_value(Value::Object(full))
            .map_err(|err| OkfError::new("CONFIG_INVALID", format!("invalid settings: {err}")))?;
        settings.validate_router()?;
        Ok(settings)
    }

    /// Constructor-level validation only. The Python `OKFRouter` validates
    /// the database path, embedding width/model, token ceiling, choices and
    /// explicit model files; it does not run the full `Settings.checks()`
    /// table (parity scenarios legitimately pass a 32-token `chunk_size`).
    /// CLI/env/TOML callers get the full checks through `Settings::load`.
    fn validate_router(&self) -> Result<()> {
        let mut errors = Vec::new();
        if self.db_path.trim().is_empty() {
            errors.push("database.db_path must be a non-empty string".into());
        }
        if !(32..=1024).contains(&self.embedding_dim) {
            errors.push("embedding.embedding_dim must be between 32 and 1024".into());
        }
        if !self.model_id.contains('/') {
            errors.push(format!(
                "embedding.model_id must be 'owner/name', got '{}'",
                self.model_id
            ));
        } else if ['\'', '"', '\\', ';']
            .iter()
            .any(|c| self.model_id.contains(*c))
        {
            errors.push(format!(
                "embedding.model_id must not contain quotes/semicolons, got '{}'",
                self.model_id
            ));
        }
        if let Some(max) = self.max_length {
            if !(1..=32768).contains(&max) {
                errors.push(format!(
                    "embedding.max_length must be within 1..=32768, got {max}"
                ));
            }
        }
        for def in SETTINGS {
            if def.choices.is_empty() {
                continue;
            }
            if let Some(value) = self
                .get(def.name)
                .and_then(|v| v.as_str().map(str::to_string))
            {
                if !def.choices.contains(&value.as_str()) {
                    errors.push(format!(
                        "{}.{} must be one of {:?}, got '{}'",
                        def.section, def.name, def.choices, value
                    ));
                }
            }
        }
        if !errors.is_empty() {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!("Configuration invalid:\n  - {}", errors.join("\n  - ")),
            ));
        }
        crate::roots::validate(self).map_err(config_error)
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("settings serialize")
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        self.to_value().get(name).cloned()
    }

    pub fn with(self, name: &str, value: Value) -> Result<Self> {
        let def = row(name)
            .ok_or_else(|| OkfError::new("CONFIG_INVALID", format!("unknown setting: {name}")))?;
        let parsed = parse_value(def, value, &flag_name(name))?;
        let mut map = Map::from_iter(
            self.to_value()
                .as_object()
                .expect("settings object")
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );
        map.insert(name.to_string(), parsed);
        Self::from_map(&map)
    }

    /// Merge TOML < env < CLI over defaults, with explicit `require` checks.
    ///
    /// Every configuration refusal is reported as `CONFIG_INVALID`, matching
    /// the Python CLI's `ValueError` -> `OKFError` translation.
    pub fn load(
        bundle_root: Option<&str>,
        cli: &Map<String, Value>,
        environ: &BTreeMap<String, String>,
        require: &[&str],
    ) -> Result<Self> {
        Self::load_inner(bundle_root, cli, environ, require).map_err(config_error)
    }

    fn load_inner(
        bundle_root: Option<&str>,
        cli: &Map<String, Value>,
        environ: &BTreeMap<String, String>,
        require: &[&str],
    ) -> Result<Self> {
        const LEGACY_ENV: &[(&str, &str)] = &[
            ("OKFGRAPH_DB", "OKFGRAPH_DB_PATH"),
            ("OKFGRAPH_DIM", "OKFGRAPH_EMBEDDING_DIM"),
            ("OKFGRAPH_MODEL", "OKFGRAPH_MODEL_ID"),
            ("OKFGRAPH_BUNDLE", "OKFGRAPH_BUNDLE_ROOT"),
            ("OKFGRAPH_IMAGE_MODEL", "OKFGRAPH_IMAGE_MODEL_ID"),
            (
                "OKFGRAPH_NO_CHUNKING",
                "OKFGRAPH_ENABLE_CHUNKING (inverted sense: 0 disables)",
            ),
            (
                "OKFGRAPH_OMNI_MODEL_ID",
                "OKFGRAPH_OMNI_MODEL_ID was removed (torch path, gone since 0.7.0) — unset it",
            ),
        ];
        for (old, new) in LEGACY_ENV {
            if environ.contains_key(*old) {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!("{old} was renamed: use {new}"),
                ));
            }
        }

        let mut taken: Map<String, Value> = Map::new();
        if let Some((table, path)) = read_toml(bundle_root)? {
            from_toml(&table, &path, &mut taken)?;
        }
        for def in SETTINGS {
            let key = def.env_name();
            if let Some(raw) = environ.get(&key).filter(|s| !s.is_empty()) {
                taken.insert(
                    def.name.to_string(),
                    parse_value(def, Value::String(raw.clone()), &key)?,
                );
            }
        }
        for (key, raw) in cli {
            if key == "roots" {
                taken.insert(key.clone(), raw.clone());
                continue;
            }
            let def = row(key).ok_or_else(|| {
                OkfError::new("CONFIG_INVALID", format!("unknown setting: {key}"))
            })?;
            if def.name == "bundle_root" {
                handle_bundle_root(raw)?;
            }
            taken.insert(key.clone(), parse_value(def, raw.clone(), &flag_name(key))?);
        }
        for name in require {
            if !taken.contains_key(*name) {
                let def = row(name).expect("required setting exists");
                let where_ = if def.section.is_empty() {
                    def.name.to_string()
                } else {
                    format!("[{}] {}", def.section, def.name)
                };
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!(
                        "{} must be set: {}, {} or okfgraph.toml {}",
                        name,
                        flag_name(name),
                        env_name(name),
                        where_
                    ),
                ));
            }
        }

        if !taken.contains_key("db_path") && legacy_home() {
            taken.insert("db_path".into(), Value::String(LEGACY_DB_PATH.into()));
        }
        let settings = Self::from_map(&taken)?;
        let (errors, _warnings) = settings.checks();
        if !errors.is_empty() {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!("Configuration invalid:\n  - {}", errors.join("\n  - ")),
            ));
        }
        Ok(settings)
    }

    /// Validate the merged settings. Returns `(errors, warnings)`.
    pub fn checks(&self) -> (Vec<String>, Vec<String>) {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if self.db_path.trim().is_empty() {
            errors.push("database.db_path must be a non-empty string".into());
        }
        if !(32..=1024).contains(&self.embedding_dim) {
            errors.push("embedding.embedding_dim must be between 32 and 1024".into());
        } else if !DIM_LADDER.contains(&self.embedding_dim) {
            warnings.push(format!(
                "embedding.embedding_dim={} is not a recommended Matryoshka dimension; consider 256 or 512",
                self.embedding_dim
            ));
        }
        if !self.model_id.contains('/') {
            errors.push(format!(
                "embedding.model_id must be 'owner/name', got '{}'",
                self.model_id
            ));
        } else if ['\'', '"', '\\', ';']
            .iter()
            .any(|c| self.model_id.contains(*c))
        {
            errors.push(format!(
                "embedding.model_id must not contain quotes/semicolons, got '{}'",
                self.model_id
            ));
        }
        for def in SETTINGS {
            if def.choices.is_empty() {
                continue;
            }
            if let Some(value) = self
                .get(def.name)
                .and_then(|v| v.as_str().map(str::to_string))
            {
                if !def.choices.contains(&value.as_str()) {
                    errors.push(format!(
                        "{}.{} must be one of {:?}, got '{}'",
                        def.section, def.name, def.choices, value
                    ));
                }
            }
        }
        if let Some(max) = self.max_length {
            if !(1..=32768).contains(&max) {
                errors.push(format!(
                    "embedding.max_length must be within 1..=32768, got {max}"
                ));
            }
        }
        if let Some(dir) = &self.cache_dir {
            if !Path::new(dir).is_absolute() {
                errors.push("embedding.cache_dir must be an absolute path".into());
            }
        }
        if self.pdf_idle_seconds < 0 {
            errors.push(format!(
                "import.pdf_idle_seconds must be >= 0, got {}",
                self.pdf_idle_seconds
            ));
        }
        if self.backend == "external" {
            if !is_http_url(&self.endpoint_url) {
                errors.push(
                    "embedding.endpoint_url must be an http(s) URL for backend='external'".into(),
                );
            }
            if self.embed_family.contains('/')
                || (self.embed_family.is_empty() && self.embed_identity == "declared")
            {
                errors.push(
                    "embedding.embed_family must be a short declared name for backend='external' with embed_identity='declared'".into(),
                );
            }
            if !(64..=1048576).contains(&self.embed_max_tokens) {
                errors.push("embedding.embed_max_tokens must be within 64..1048576".into());
            }
            if self.embed_counter == "gguf-meta"
                && !std::fs::metadata(&self.embed_counter_gguf)
                    .map(|m| m.is_file())
                    .unwrap_or(false)
            {
                errors.push(
                    "embedding.embed_counter_gguf must be the path to an existing GGUF file when embed_counter='gguf-meta'".into(),
                );
            }
            if self.embed_counter == "declared"
                && !(0.5..=64.0).contains(&self.embed_chars_per_token)
            {
                errors.push(
                    "embedding.embed_chars_per_token must be within 0.5..64 when embed_counter='declared'".into(),
                );
            }
        } else if !self.endpoint_url.is_empty()
            || !self.embed_family.is_empty()
            || self.embed_counter != "heuristic"
            || !self.embed_counter_gguf.is_empty()
            || self.embed_chars_per_token != 0.0
        {
            errors.push(
                "embedding.endpoint_url/embed_family/embed_counter* are external-backend settings — set embedding.backend='external' or drop them".into(),
            );
        }
        if self.image_backend == "external" {
            if !is_http_url(&self.image_endpoint_url) {
                errors.push(
                    "embedding.image_endpoint_url must be an http(s) URL for image_backend='external'".into(),
                );
            }
        } else if !self.image_endpoint_url.is_empty() {
            errors.push(
                "embedding.image_endpoint_url is an external-backend setting — set embedding.image_backend='external' or drop it".into(),
            );
        }
        if self.rerank_backend == "external" {
            if !is_http_url(&self.rerank_endpoint_url) {
                errors.push(
                    "search.rerank_endpoint_url must be an http(s) URL for rerank_backend='external'".into(),
                );
            }
            if self.rerank_family.contains('/')
                || (self.rerank_family.is_empty() && self.rerank_identity == "declared")
            {
                errors.push(
                    "search.rerank_family must be a short declared label for rerank_backend='external'".into(),
                );
            }
            if !(64..=1048576).contains(&self.rerank_max_tokens) {
                errors.push("search.rerank_max_tokens must be within 64..1048576".into());
            }
        } else if !self.rerank_endpoint_url.is_empty()
            || !self.rerank_family.is_empty()
            || self.rerank_count != "heuristic"
        {
            errors.push(
                "search.rerank_endpoint_url/rerank_family/rerank_count are external-lane settings — set search.rerank_backend='external' or drop them".into(),
            );
        }
        if !(1..=256).contains(&self.batch_size) {
            errors.push("import.batch_size must be between 1 and 256".into());
        }
        if !(64..=8192).contains(&self.chunk_size) {
            errors.push("import.chunk_size must be between 64 and 8192".into());
        }
        if self.chunk_overlap < 0 || self.chunk_overlap >= self.chunk_size {
            errors.push("import.chunk_overlap must be >= 0 and < chunk_size".into());
        }
        if self
            .allowed_image_domains
            .iter()
            .any(|d| d.trim().is_empty())
        {
            errors.push("allowed_image_domains contains empty entries".into());
        }
        if let Some(dir) = &self.extension_dir {
            if !Path::new(dir).is_dir() {
                errors.push(format!(
                    "database.extension_dir is not a directory: {dir} (create it with 'okf setup extensions')"
                ));
            }
        }
        if !self.rerank_model_id.contains('/') {
            errors.push(format!(
                "search.rerank_model_id must be 'owner/name', got '{}'",
                self.rerank_model_id
            ));
        }
        if !(1..=200).contains(&self.rerank_top_n) {
            errors.push(format!(
                "search.rerank_top_n must be within 1..200, got {}",
                self.rerank_top_n
            ));
        }
        if let Some(window) = self.rerank_window {
            if !(16..=8192).contains(&window) {
                errors.push(format!(
                    "search.rerank_window must be within 16..8192, got {window}"
                ));
            }
        }
        if let Some(stride) = self.rerank_stride {
            let ceiling = self.rerank_window.unwrap_or(8192);
            if stride < 1 || stride > ceiling {
                errors.push(format!(
                    "search.rerank_stride must be within 1..rerank_window, got {stride}"
                ));
            }
        }
        if let Some(efs) = self.efs {
            if !(16..=1000).contains(&efs) {
                errors.push(format!("search.efs must be within 16..1000, got {efs}"));
            }
        }
        (errors, warnings)
    }

    pub fn default_bundle_root(&self, fallback: &str) -> String {
        if let Some(root) = &self.bundle_root {
            return root.clone();
        }
        let db = Path::new(&self.db_path);
        if db
            .parent()
            .and_then(Path::file_name)
            .and_then(|s| s.to_str())
            != Some(GRAPH_HOME)
        {
            return fallback.to_string();
        }
        let bundle = db.parent().expect("parent").join("bundle");
        if !bundle.is_dir() {
            if db.exists() {
                return fallback.to_string();
            }
            let _ = std::fs::create_dir_all(&bundle);
        }
        bundle.to_string_lossy().into_owned()
    }

    /// Params passed by Python-surface shims into `Router::open`.
    pub fn router_params(&self) -> Value {
        let mut params = self
            .to_value()
            .as_object()
            .expect("settings object")
            .clone();
        let backend = params.remove("backend").unwrap_or(Value::Null);
        params.insert("embed_backend".into(), backend);
        if self.roots.is_empty() {
            params.insert("roots".into(), Value::Null);
        }
        Value::Object(params)
    }
}

fn is_http_url(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Loaders report every bad setting as `CONFIG_INVALID`, whatever the
/// underlying table parser called it.
fn config_error(err: OkfError) -> OkfError {
    if err.code == "CONFIG_INVALID" {
        err
    } else {
        OkfError::new("CONFIG_INVALID", err.message)
    }
}

/// `roots` arrives as a map from TOML/programmatic callers and as an array of
/// `ALIAS=PATH` strings from `--root`. Normalize both to the map the settings
/// struct expects; malformed entries get Python's `--root must be ALIAS=PATH`
/// wording.
fn parse_roots(raw: &Value) -> Result<Value> {
    match raw {
        Value::Null => Ok(Value::Object(Map::new())),
        Value::Object(map) => Ok(Value::Object(map.clone())),
        Value::Array(items) => {
            let mut roots = Map::new();
            for item in items {
                let Some(entry) = item.as_str() else {
                    return Err(OkfError::new(
                        "BAD_VALUE",
                        format!("--root must be ALIAS=PATH, got {item}"),
                    ));
                };
                let (alias, path) = entry.split_once('=').ok_or_else(|| {
                    OkfError::new(
                        "BAD_VALUE",
                        format!("--root must be ALIAS=PATH, got {entry:?}"),
                    )
                })?;
                if alias.is_empty() || path.is_empty() {
                    return Err(OkfError::new(
                        "BAD_VALUE",
                        format!("--root must be ALIAS=PATH, got {entry:?}"),
                    ));
                }
                // Python's dict assignment means a repeated alias keeps the
                // last path; keep that behavior for the CLI and TOML alike.
                roots.insert(alias.to_string(), Value::String(path.to_string()));
            }
            Ok(Value::Object(roots))
        }
        other => Err(OkfError::new(
            "BAD_VALUE",
            format!("--root must be ALIAS=PATH, got {other}"),
        )),
    }
}

fn parse_value(def: &SettingDef, raw: Value, origin: &str) -> Result<Value> {
    match def.kind {
        SettingKind::Bool => match &raw {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            _ => {
                let word = raw.as_str().unwrap_or_default().trim().to_ascii_lowercase();
                match word.as_str() {
                    "1" | "true" | "yes" | "on" => Ok(Value::Bool(true)),
                    "0" | "false" | "no" | "off" => Ok(Value::Bool(false)),
                    _ => Err(OkfError::new(
                        "BAD_VALUE",
                        format!("{origin}: expected one of 1, true, yes, on, 0, false, no, off (got {raw:?})"),
                    )),
                }
            }
        },
        SettingKind::Int => match &raw {
            Value::Number(n) if n.is_i64() => Ok(raw.clone()),
            Value::String(s) => s.trim().parse::<i64>().map(Value::from).map_err(|_| {
                OkfError::new(
                    "BAD_VALUE",
                    format!("{origin}: expected an integer (got {s:?})"),
                )
            }),
            _ => Err(OkfError::new(
                "BAD_VALUE",
                format!("{origin}: expected an integer (got {raw:?})"),
            )),
        },
        SettingKind::Float => match &raw {
            Value::Number(n) if n.is_f64() || n.is_i64() => {
                Ok(serde_json::json!(n.as_f64().unwrap()))
            }
            Value::String(s) => s
                .trim()
                .parse::<f64>()
                .map(|v| serde_json::json!(v))
                .map_err(|_| {
                    OkfError::new(
                        "BAD_VALUE",
                        format!("{origin}: expected a number (got {s:?})"),
                    )
                }),
            _ => Err(OkfError::new(
                "BAD_VALUE",
                format!("{origin}: expected a number (got {raw:?})"),
            )),
        },
        SettingKind::Csv => {
            let items: Vec<String> = match raw {
                Value::Array(items) => items
                    .iter()
                    .map(|value| match value {
                        Value::String(text) => text.trim().to_string(),
                        other => other.to_string().trim().to_string(),
                    })
                    .collect(),
                other => other
                    .as_str()
                    .unwrap_or("")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            };
            Ok(Value::Array(
                items
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .map(Value::String)
                    .collect(),
            ))
        }
        SettingKind::Str => {
            if raw.is_null() {
                Ok(Value::Null)
            } else {
                Ok(Value::String(match raw {
                    Value::String(s) => s,
                    other => other.to_string(),
                }))
            }
        }
    }
}

fn handle_bundle_root(value: &Value) -> Result<()> {
    let text = value.as_str().unwrap_or_default();
    if text.contains('=') && !Path::new(text).exists() {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!(
                "--bundle-root ALIAS=PATH was retired: the primary root is --bundle-root PATH, named roots move to --root ALIAS=PATH (got {text:?})"
            ),
        ));
    }
    Ok(())
}

fn legacy_home() -> bool {
    Path::new(LEGACY_DB_PATH).exists() && !Path::new(GRAPH_HOME).exists()
}

fn read_toml(bundle_root: Option<&str>) -> Result<Option<(toml::Value, PathBuf)>> {
    let mut candidates = vec![PathBuf::from("okfgraph.toml")];
    if let Some(root) = bundle_root {
        candidates.push(Path::new(root).join("okfgraph.toml"));
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    if !home.is_empty() {
        candidates.push(Path::new(&home).join(".config/okfgraph/config.toml"));
    }
    for path in candidates {
        if path.is_file() {
            let text = std::fs::read_to_string(&path)?;
            let table = toml::from_str(&text).map_err(|err| {
                OkfError::new(
                    "CONFIG_INVALID",
                    format!("{}: invalid TOML: {err}", path.display()),
                )
            })?;
            return Ok(Some((table, path)));
        }
    }
    Ok(None)
}

fn from_toml(table: &toml::Value, path: &Path, out: &mut Map<String, Value>) -> Result<()> {
    let legacy = [
        (
            Some("database"),
            "path",
            "[database] path -> [database] db_path",
        ),
        (
            Some("database"),
            "dim",
            "[database] dim -> [embedding] embedding_dim",
        ),
        (None, "bundle", "top-level 'bundle' -> 'bundle_root'"),
        (
            Some("import"),
            "no_chunking",
            "[import] no_chunking -> [import] enable_chunking",
        ),
        (
            Some("embedding"),
            "omni_model_id",
            "embedding.omni_model_id was removed (torch path, gone since 0.7.0) — delete the key",
        ),
    ];
    let root = table.as_table().expect("toml root");
    for (key, value) in root {
        if key == "roots" {
            let entries = value.as_array().ok_or_else(|| {
                OkfError::new(
                    "CONFIG_INVALID",
                    "[[roots]] must be a list of {alias, path} tables",
                )
            })?;
            let mut roots = Map::new();
            for entry in entries {
                let item = entry.as_table().ok_or_else(|| {
                    OkfError::new(
                        "CONFIG_INVALID",
                        "[[roots]] entries need string 'alias' and 'path'",
                    )
                })?;
                let alias = item.get("alias").and_then(|v| v.as_str()).ok_or_else(|| {
                    OkfError::new(
                        "CONFIG_INVALID",
                        "[[roots]] entries need string 'alias' and 'path'",
                    )
                })?;
                let raw_path = item.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
                    OkfError::new(
                        "CONFIG_INVALID",
                        "[[roots]] entries need string 'alias' and 'path'",
                    )
                })?;
                if roots.contains_key(alias) {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!("duplicate [[roots]] alias {alias:?}"),
                    ));
                }
                let resolved = Path::new(raw_path);
                let value = if resolved.is_absolute() {
                    raw_path.to_string()
                } else {
                    path.parent()
                        .unwrap_or(Path::new("."))
                        .join(resolved)
                        .to_string_lossy()
                        .into_owned()
                };
                roots.insert(alias.to_string(), Value::String(value));
            }
            out.insert("roots".into(), Value::Object(roots));
        } else if let Some(section) = value.as_table() {
            for (field, raw) in section {
                if let Some((_, _, message)) = legacy
                    .iter()
                    .find(|(s, f, _)| *s == Some(key.as_str()) && *f == field.as_str())
                {
                    return Err(OkfError::new(
                        "CONFIG_INVALID",
                        format!("okfgraph.toml: {message}"),
                    ));
                }
                let def = row(field).filter(|def| def.section == key);
                if let Some(def) = def {
                    out.insert(
                        def.name.into(),
                        parse_value(
                            def,
                            toml_to_json(raw),
                            &format!("okfgraph.toml: [{key}] {field}"),
                        )?,
                    );
                }
            }
        } else if let Some((_, _, message)) = legacy
            .iter()
            .find(|(s, f, _)| s.is_none() && *f == key.as_str())
        {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!("okfgraph.toml: {message}"),
            ));
        } else if let Some(def) = row(key).filter(|def| def.section.is_empty()) {
            out.insert(
                def.name.into(),
                parse_value(def, toml_to_json(value), &format!("okfgraph.toml: {key}"))?,
            );
        }
    }
    Ok(())
}

fn toml_to_json(value: &toml::Value) -> Value {
    match value {
        toml::Value::String(s) => Value::String(s.clone()),
        toml::Value::Integer(i) => Value::from(*i),
        toml::Value::Float(f) => serde_json::json!(f),
        toml::Value::Boolean(b) => Value::Bool(*b),
        toml::Value::Array(items) => Value::Array(items.iter().map(toml_to_json).collect()),
        toml::Value::Table(table) => Value::Object(
            table
                .iter()
                .map(|(k, v)| (k.clone(), toml_to_json(v)))
                .collect(),
        ),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_oracle_size() {
        assert_eq!(SETTINGS.len(), 52);
        assert!(row("db_path").unwrap().required_mcp);
        assert_eq!(row("enable_chunking").unwrap().flag_name(), "--no-chunking");
    }

    #[test]
    fn defaults_validate() {
        let settings = Settings::default();
        assert!(settings.checks().0.is_empty(), "{:?}", settings.checks().0);
        assert_eq!(settings.db_path, DEFAULT_DB_PATH);
    }

    #[test]
    fn env_overrides_defaults() {
        let env = BTreeMap::from([("OKFGRAPH_CHUNK_OVERLAP".to_string(), "0".to_string())]);
        let settings = Settings::load(None, &Map::new(), &env, &[]).expect("load");
        assert_eq!(settings.chunk_overlap, 0);
    }

    #[test]
    fn from_map_normalizes_cli_shapes() {
        let map = Map::from_iter([
            (
                "roots".to_string(),
                Value::Array(vec![
                    Value::String("other=parity/corpus/multi-root/other".into()),
                    Value::String("docs=docs".into()),
                ]),
            ),
            (
                "allowed_image_domains".to_string(),
                Value::String("a.example,b.example,, c.example ".into()),
            ),
        ]);
        let settings = Settings::from_map(&map).expect("normalized settings");
        assert_eq!(
            settings.roots.get("other").map(String::as_str),
            Some("parity/corpus/multi-root/other")
        );
        assert_eq!(
            settings.allowed_image_domains,
            vec![
                "a.example".to_string(),
                "b.example".into(),
                "c.example".into()
            ]
        );
    }

    #[test]
    fn from_map_rejects_bad_settings() {
        for (key, raw) in [
            ("bogus_setting", Value::from(1)),
            ("embedding_dim", Value::from(8)),
            ("device", Value::String("tpu".into())),
            ("model_path", Value::String("x.onnx".into())),
            ("tokenizer_path", Value::String("y.json".into())),
        ] {
            let map = Map::from_iter([(key.to_string(), raw)]);
            let err = Settings::from_map(&map).expect_err("must refuse");
            assert_eq!(err.code, "CONFIG_INVALID", "{key}: {}", err.message);
        }
        let roots = Map::from_iter([(
            "roots".to_string(),
            Value::Array(vec![Value::String("no-equals-here".into())]),
        )]);
        let err = Settings::from_map(&roots).expect_err("must refuse bad roots");
        assert_eq!(err.code, "CONFIG_INVALID");
        assert!(err.message.contains("ALIAS=PATH"), "{}", err.message);
    }

    #[test]
    fn router_settings_allow_small_chunk_size_but_loader_does_not() {
        let map = Map::from_iter([("chunk_size".to_string(), Value::from(32))]);
        assert_eq!(
            Settings::from_map(&map)
                .expect("router settings")
                .chunk_size,
            32
        );

        let env = BTreeMap::from([("OKFGRAPH_CHUNK_SIZE".to_string(), "32".into())]);
        let err = Settings::load(None, &Map::new(), &env, &[]).expect_err("loader validates");
        assert_eq!(err.code, "CONFIG_INVALID");
        assert!(err.message.contains("chunk_size"));
    }

    #[test]
    fn loader_merges_toml_env_and_cli() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            dir.path().join("okfgraph.toml"),
            "[database]\ndb_path = \"from-toml.db\"\n[embedding]\nembedding_dim = 256\n",
        )
        .expect("write toml");
        let cli = Map::from_iter([("db_path".to_string(), Value::String("from-cli.db".into()))]);
        let env = BTreeMap::from([("OKFGRAPH_CHUNK_OVERLAP".to_string(), "0".to_string())]);
        let settings =
            Settings::load(dir.path().to_str(), &cli, &env, &[]).expect("merged settings");
        assert_eq!(settings.db_path, "from-cli.db");
        assert_eq!(settings.embedding_dim, 256);
        assert_eq!(settings.chunk_overlap, 0);
    }

    #[test]
    fn loader_refuses_legacy_env_and_missing_required() {
        let env = BTreeMap::from([("OKFGRAPH_DB".to_string(), "old.db".into())]);
        let err = Settings::load(None, &Map::new(), &env, &[]).expect_err("legacy env");
        assert_eq!(err.code, "CONFIG_INVALID");
        assert!(err.message.contains("OKFGRAPH_DB_PATH"));

        let err = Settings::load(None, &Map::new(), &BTreeMap::new(), &["db_path"])
            .expect_err("required db_path");
        assert_eq!(err.code, "CONFIG_INVALID");
        assert!(err.message.contains("db_path must be set"));
    }

    #[test]
    fn default_bundle_root_uses_graph_home() {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = dir.path().join(".okfgraph/okfgraph.db");
        let settings = Settings::from_map(&Map::from_iter([(
            "db_path".to_string(),
            Value::String(db.to_string_lossy().into_owned()),
        )]))
        .expect("settings");
        let bundle = settings.default_bundle_root("fallback");
        let expected = dir.path().join(".okfgraph").join("bundle");
        assert_eq!(Path::new(&bundle), expected.as_path());
        assert!(Path::new(&bundle).is_dir());
    }
}
