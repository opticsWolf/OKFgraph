//! Ladybug wrapper, schema v10, migrations, indexes, and locking.
//!
//! M1 uses a fresh connection per call. That avoids self-referential borrows
//! (`Connection<'db>` borrows `Database`) and keeps the store `Send + Sync`.
//! A pooled/owned connection can be introduced later without changing the ops.

use crate::error::{OkfError, Result};
use crate::model::python_datetime_str;
use crate::settings::Settings;
use lbug::{Connection, Database, LogicalType, SystemConfig, Value};
use serde_json::{Map, Value as Json};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u16 = crate::SCHEMA_VERSION;

pub const NODE_TABLES: &[&str] = &[
    "Concept",
    "Directory",
    "ImageAsset",
    "Chunk",
    "BrokenLink",
    "DeletedConcept",
    "Meta",
    "MetaText",
    "FileHash",
    "DirHash",
    "DeletedPath",
    "SourceRoot",
    "ProducerCursor",
];

pub const REL_TABLES: &[&str] = &["CONTAINS", "LINKS_TO", "INCLUDES_ASSET", "PART_OF"];

/// A read/write handle to a Ladybug v10 graph.
#[derive(Debug, Clone)]
pub struct Store {
    path: PathBuf,
    settings: Settings,
    read_only: bool,
}

impl Store {
    pub fn open(settings: &Settings) -> Result<Self> {
        let path = PathBuf::from(&settings.db_path);
        if !path.exists() {
            return Err(OkfError::new(
                "FILE_NOT_FOUND",
                format!("database not found: {}", path.display()),
            )
            .field("path", Json::String(path.to_string_lossy().into_owned())));
        }
        let store = Self {
            path,
            settings: settings.clone(),
            read_only: true,
        };
        let mut settings = settings.clone();
        if let Some(dim) = store.meta_int("embedding_dim")? {
            settings.embedding_dim = dim;
        }
        let version = store.schema_version()?;
        if version > SCHEMA_VERSION {
            return Err(OkfError::new(
                "INTERNAL",
                format!("database schema v{version} is newer than this build (v{SCHEMA_VERSION})"),
            ));
        }
        Ok(Self { settings, ..store })
    }

    /// Create a v10 database at `path`. Content writes remain disabled in M1.
    pub fn init(path: impl AsRef<Path>, embedding_dim: i64) -> Result<Vec<String>> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let store = Self {
            path,
            settings: Settings::default(),
            read_only: false,
        };
        let mut warnings = Vec::new();
        store.create_schema(embedding_dim, &mut warnings)?;
        Ok(warnings)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    fn with_conn<T>(&self, f: impl for<'a> FnOnce(&Connection<'a>) -> Result<T>) -> Result<T> {
        let config = SystemConfig::default().read_only(self.read_only);
        let db = Database::new(&self.path, config).map_err(map_lbug)?;
        let conn = Connection::new(&db).map_err(map_lbug)?;
        f(&conn)
    }

    fn create_schema(&self, embedding_dim: i64, _warnings: &mut Vec<String>) -> Result<()> {
        let dim = embedding_dim.max(1) as usize;
        self.with_conn(|conn| {
            for ddl in schema_ddl(dim) {
                conn.query(&ddl).map_err(map_lbug)?;
            }
            conn.query(&format!(
                "MERGE (m:Meta {{key: 'schema_version'}}) SET m.value = {SCHEMA_VERSION}"
            ))
            .map_err(map_lbug)?;
            Ok(())
        })
    }

    pub fn schema_version(&self) -> Result<u16> {
        let rows = self.query(
            "MATCH (m:Meta {key: 'schema_version'}) RETURN m.value AS v",
            vec![],
        )?;
        let version = rows
            .first()
            .and_then(|row| row.get("v"))
            .and_then(Json::as_i64)
            .unwrap_or(0);
        Ok(version as u16)
    }

    /// Query rows as JSON objects with stable column names.
    pub fn query(&self, sql: &str, params: Vec<(&str, Value)>) -> Result<Vec<Map<String, Json>>> {
        self.with_conn(|conn| {
            let mut result = if params.is_empty() {
                conn.query(sql).map_err(map_lbug)?
            } else {
                let mut stmt = conn.prepare(sql).map_err(map_lbug)?;
                conn.execute(&mut stmt, params).map_err(map_lbug)?
            };
            let columns = result.get_column_names();
            let mut rows = Vec::new();
            for values in result.by_ref() {
                let mut row = Map::new();
                for (name, value) in columns.iter().zip(values) {
                    row.insert(name.clone(), value_to_json(value));
                }
                rows.push(row);
            }
            Ok(rows)
        })
    }

    fn scalar(&self, sql: &str, params: Vec<(&str, Value)>) -> Result<Option<Json>> {
        Ok(self
            .query(sql, params)?
            .into_iter()
            .next()
            .and_then(|row| row.into_values().next()))
    }

    /// Full normalized DB dump: all node and relationship tables, sorted.
    pub fn dump_all(&self) -> Result<Json> {
        let mut dump = Map::new();
        for name in NODE_TABLES.iter().chain(REL_TABLES.iter()) {
            let node_sql = format!("MATCH (n:`{name}`) RETURN n");
            let rel_sql = format!("MATCH ()-[r:`{name}`]->() RETURN r");
            let rows = self
                .query(&node_sql, vec![])
                .or_else(|_| self.query(&rel_sql, vec![]))?;
            let mut values: Vec<Json> = rows
                .into_iter()
                .map(|row| row.into_values().next().unwrap_or(Json::Null))
                .collect();
            values.sort_by_key(|value| serde_json::to_string(value).unwrap_or_default());
            dump.insert((*name).to_string(), Json::Array(values));
        }
        Ok(Json::Object(dump))
    }

    pub fn get_by_id(&self, concept_id: &str) -> Result<Option<Json>> {
        let rows = self.query(
            "MATCH (c:Concept {id: $id}) RETURN c.*",
            vec![("id", Value::String(concept_id.into()))],
        )?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let mut flat = Map::new();
        for (key, value) in row {
            let key = key.strip_prefix("c.").unwrap_or(&key).to_string();
            if key.starts_with("extra.") {
                continue;
            }
            flat.insert(key, value);
        }
        Ok(Some(public_concept(&flat)))
    }

    pub fn node_exists(&self, id: &str) -> Result<bool> {
        for label in ["Concept", "Directory"] {
            let sql = format!("MATCH (n:{label} {{id: $id}}) RETURN n.id AS id LIMIT 1");
            if !self
                .query(&sql, vec![("id", Value::String(id.into()))])?
                .is_empty()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn list_directory(&self, directory_id: &str) -> Result<Vec<Json>> {
        let (dir_sql, concept_sql, params): (String, String, Vec<(&str, Value)>) =
            if directory_id.is_empty() {
                (
                    "MATCH (d:Directory) WHERE NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(d) } \
                     RETURN d.id AS id, 'Directory' AS type, d.id AS title"
                        .into(),
                    "MATCH (c:Concept) WHERE NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(c) } \
                     RETURN c.id AS id, c.type AS type, c.title AS title"
                        .into(),
                    vec![],
                )
            } else {
                (
                    "MATCH (p:Directory {id: $id})-[:CONTAINS]->(d:Directory) \
                     RETURN d.id AS id, 'Directory' AS type, d.id AS title"
                        .into(),
                    "MATCH (p:Directory {id: $id})-[:CONTAINS]->(c:Concept) \
                     RETURN c.id AS id, c.type AS type, c.title AS title"
                        .into(),
                    vec![("id", Value::String(directory_id.into()))],
                )
            };
        let mut out: Vec<Json> = self
            .query(&dir_sql, params.clone())?
            .into_iter()
            .map(Json::Object)
            .collect();
        out.extend(
            self.query(&concept_sql, params)?
                .into_iter()
                .map(Json::Object),
        );
        Ok(out)
    }

    pub fn get_chunks(&self, concept_id: &str) -> Result<Vec<Json>> {
        let rows = self.query(
            "MATCH (c:Concept {id: $cid})-[:PART_OF]->(k:Chunk)              RETURN k.* ORDER BY k.chunk_index",
            vec![("cid", Value::String(concept_id.into()))],
        )?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let mut chunk = Map::new();
                for (key, value) in row {
                    let key = key.strip_prefix("k.").unwrap_or(&key).to_string();
                    if key != "embedding" {
                        chunk.insert(key, value);
                    }
                }
                Json::Object(chunk)
            })
            .collect())
    }

    pub fn traverse(
        &self,
        start_id: &str,
        relationship: &str,
        direction: &str,
        depth: i64,
        node_type: Option<&str>,
        link_kinds: Option<&[String]>,
    ) -> Result<Vec<Json>> {
        let depth = depth.clamp(1, 5);
        let target_label = match relationship {
            "PART_OF" => "Chunk",
            "INCLUDES_ASSET" => "ImageAsset",
            _ => "Concept",
        };
        let mut rel = format!("{relationship}*1..{depth}");
        let mut params: Vec<(&str, Value)> = vec![("start_id", Value::String(start_id.into()))];
        if let Some(kinds) = link_kinds {
            rel = format!("e:{rel} (r, n | WHERE r.kind IN $link_kinds)");
            params.push((
                "link_kinds",
                Value::List(
                    LogicalType::String,
                    kinds.iter().cloned().map(Value::String).collect(),
                ),
            ));
        }
        let pattern = match direction {
            "INCOMING" => format!("<-[{rel}]-(target:{target_label})"),
            "BOTH" => format!("-[{rel}]-(target:{target_label})"),
            _ => format!("-[{rel}]->(target:{target_label})"),
        };
        let where_clause = if node_type.is_some() {
            "WHERE target.type = $node_type"
        } else {
            ""
        };
        if let Some(node_type) = node_type {
            params.push(("node_type", Value::String(node_type.into())));
        }
        let cypher = format!(
            "MATCH (start {{id: $start_id}}){pattern} {where_clause} RETURN target.* LIMIT 100"
        );
        let rows = self.query(&cypher, params)?;
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for row in rows {
            let mut entry = Map::new();
            for (key, value) in row {
                let prop = key.split('.').next_back().unwrap_or(&key).to_string();
                entry.insert(prop, value);
            }
            if let Some(id) = entry.get("id").and_then(Json::as_str) {
                if !seen.insert(id.to_string()) {
                    continue;
                }
            }
            out.push(Json::Object(entry));
        }
        Ok(out)
    }

    pub fn list_images(&self, concept_id: &str) -> Result<Vec<Json>> {
        let rows = self.query(
            "MATCH (c:Concept {id: $cid})-[:INCLUDES_ASSET]->(i:ImageAsset) \
             RETURN i.id AS id, i.file_name AS file_name, i.mime_type AS mime_type, \
                    i.alt_text AS alt_text, i.embed_route AS embed_route",
            vec![("cid", Value::String(concept_id.into()))],
        )?;
        Ok(rows.into_iter().map(Json::Object).collect())
    }

    pub fn get_image(&self, asset_id: &str) -> Result<Option<Json>> {
        let rows = self.query(
            "MATCH (i:ImageAsset {id: $iid}) \
             RETURN i.id AS id, i.file_name AS file_name, i.mime_type AS mime_type, \
                    i.alt_text AS alt_text, i.embed_route AS embed_route, i.data AS data",
            vec![("iid", Value::String(asset_id.into()))],
        )?;
        let Some(mut row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let data = row.remove("data").unwrap_or(Json::Null);
        let encoded = match data {
            Json::String(s) => parse_python_bytes(&s)
                .map(|raw| base64::Engine::encode(&base64::engine::general_purpose::STANDARD, raw)),
            Json::Array(bytes) => {
                let raw: Vec<u8> = bytes
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u8))
                    .collect();
                Some(base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    raw,
                ))
            }
            _ => None,
        };
        row.insert("data".into(), encoded.map_or(Json::Null, Json::String));
        Ok(Some(Json::Object(row)))
    }

    pub fn list_deleted(&self) -> Result<Vec<Json>> {
        let rows = self.query(
            "MATCH (d:DeletedConcept) RETURN d.id AS id, d.original_id AS original_id, \
             d.title AS title, d.deleted_at AS deleted_at, d.type AS type, d.tags AS tags",
            vec![],
        )?;
        Ok(rows.into_iter().map(Json::Object).collect())
    }

    pub fn list_broken_links(&self) -> Result<Vec<Json>> {
        let rows = self.query(
            "MATCH (b:BrokenLink) RETURN b.source_id AS source, b.target_id AS target, \
             b.kind AS kind, b.reason AS reason",
            vec![],
        )?;
        Ok(rows.into_iter().map(Json::Object).collect())
    }

    pub fn counts(&self) -> Result<(i64, i64)> {
        let concepts = self
            .scalar("MATCH (c:Concept) RETURN count(c) AS n", vec![])?
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let chunks = self
            .scalar("MATCH (k:Chunk) RETURN count(k) AS n", vec![])?
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        Ok((concepts, chunks))
    }

    fn meta_text(&self, key: &str) -> Result<Option<String>> {
        let value = self.scalar(
            "MATCH (m:MetaText {key: $key}) RETURN m.value AS v",
            vec![("key", Value::String(key.into()))],
        )?;
        Ok(value.and_then(|v| v.as_str().map(str::to_string)))
    }

    fn meta_int(&self, key: &str) -> Result<Option<i64>> {
        let value = self.scalar(
            "MATCH (m:Meta {key: $key}) RETURN m.value AS v",
            vec![("key", Value::String(key.into()))],
        )?;
        Ok(value.and_then(|v| v.as_i64()))
    }

    pub fn pin_info(&self) -> Result<Json> {
        let model_id = self.meta_text("embedding_model")?;
        let space = self.meta_text("embedding_space")?;
        let precision = self
            .meta_int("embedding_precision")?
            .and_then(|n| match n {
                8 => Some("int8"),
                16 => Some("fp16"),
                32 => Some("fp32"),
                _ => None,
            })
            .map(str::to_string);
        let (concepts, chunks) = self.counts()?;
        let configured_precision = if self.settings.precision == "auto" {
            None
        } else {
            Some(self.settings.precision.clone())
        };
        let configured_image_model = self
            .settings
            .image_model_id
            .clone()
            .unwrap_or_else(|| format!("{}-vision", self.settings.model_id));
        let pinned_image_model = self.meta_text("embedding_image_model")?;
        let image_mismatch = if pinned_image_model.is_some()
            && pinned_image_model.as_deref() != Some(configured_image_model.as_str())
        {
            vec!["model_id"]
        } else {
            vec![]
        };
        let mismatch = if !image_mismatch.is_empty() {
            vec!["image_model_id"]
        } else if model_id.is_some() && model_id.as_deref() != Some(&self.settings.model_id) {
            vec!["model_id"]
        } else {
            vec![]
        };
        Ok(serde_json::json!({
            "pinned": {
                "model_id": model_id,
                "precision": precision,
                "embedding_dim": self.settings.embedding_dim,
                "space": space
            },
            "configured": {
                "model_id": self.settings.model_id,
                "expected_precision": configured_precision,
                "embedding_dim": self.settings.embedding_dim,
                "space": null,
                "backend": self.settings.backend,
                "embedding_family": self.settings.embed_family
            },
            "image": {
                "pinned_model": pinned_image_model,
                "pinned_precision": self.meta_int("embedding_image_precision")?,
                "pinned_space": self.meta_text("embedding_image_space")?,
                "pinned_text_space": self.meta_text("embedding_image_text_space")?,
                "backend": self.settings.image_backend,
                "configured_model": configured_image_model,
                "configured_precision": null,
                "configured_space": null,
                "configured_text_space": null
            },
            "image_mismatch": image_mismatch,
            "mismatch": mismatch,
            "empty": concepts == 0,
            "concepts": concepts,
            "chunks": chunks
        }))
    }

    pub fn lanes(&self) -> Result<Json> {
        Ok(serde_json::json!({
            "embed": {
                "backend": self.settings.backend,
                "model_id": self.settings.model_id,
                "endpoint": if self.settings.backend == "external" { Some(self.settings.endpoint_url.clone()) } else { None },
                "identity": if self.settings.backend == "external" { self.settings.embed_identity.clone() } else { "in-process".into() },
                "family": if self.settings.embed_family.is_empty() { None } else { Some(self.settings.embed_family.clone()) },
                "precision": if self.settings.precision == "auto" { None } else { Some(self.settings.precision.clone()) },
                "space": null,
                "pinned_space": null
            },
            "image": {
                "backend": self.settings.image_backend,
                "endpoint": null,
                "identity": "in-process",
                "model_id": if self.settings.image_model_id.is_some() { self.settings.image_model_id.clone() } else { Some(format!("{}-vision", self.settings.model_id)) },
                "pinned_space": null,
                "precision": null,
                "space": null,
                "text_partner": null
            },
            "rerank": {
                "backend": self.settings.rerank_backend,
                "model_id": self.settings.rerank_model_id,
                "endpoint": if self.settings.rerank_backend == "external" { Some(self.settings.rerank_endpoint_url.clone()) } else { None },
                "identity": if self.settings.rerank_backend == "external" { self.settings.rerank_identity.clone() } else { "in-process".into() },
                "family": if self.settings.rerank_family.is_empty() { None } else { Some(self.settings.rerank_family.clone()) },
                "precision": if self.settings.rerank_precision == "auto" { None } else { Some(self.settings.rerank_precision.clone()) }
            }
        }))
    }
}

fn public_concept(row: &Map<String, Json>) -> Json {
    let mut flat = row.clone();
    let extra = flat.remove("extra").unwrap_or(Json::Null);
    flat.remove("embedding");
    flat.remove("content_sig");
    for key in ["created_at", "updated_at", "effective_at"] {
        if flat.get(key).is_none() || flat.get(key) == Some(&Json::Null) {
            flat.remove(key);
        }
    }
    if let Some(obj) = extra.as_object() {
        for (key, value) in obj {
            let decoded = match value {
                Json::String(s) if s.starts_with('{') || s.starts_with('[') => {
                    serde_json::from_str(s).unwrap_or_else(|_| value.clone())
                }
                other => other.clone(),
            };
            flat.entry(key.clone()).or_insert(decoded);
        }
    }
    let tags = flat.get("tags").cloned().unwrap_or(Json::Array(Vec::new()));
    flat.insert(
        "tags".into(),
        Json::Array(
            crate::model::normalize_tags(&tags)
                .into_iter()
                .map(Json::String)
                .collect(),
        ),
    );
    Json::Object(flat)
}

fn python_bytes_repr(bytes: &[u8]) -> String {
    let mut out = String::from("b'");
    for byte in bytes {
        match *byte {
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(*byte as char),
            other => out.push_str(&format!("\\x{other:02x}")),
        }
    }
    out.push('\'');
    out
}

fn parse_python_bytes(text: &str) -> Option<Vec<u8>> {
    let inner = text.strip_prefix("b'")?.strip_suffix('\'')?;
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch as u8);
            continue;
        }
        match chars.next()? {
            '\\' => out.push(b'\\'),
            '\'' => out.push(b'\''),
            't' => out.push(b'\t'),
            'n' => out.push(b'\n'),
            'r' => out.push(b'\r'),
            'x' => {
                let hex: String = [chars.next()?, chars.next()?].into_iter().collect();
                out.push(u8::from_str_radix(&hex, 16).ok()?);
            }
            _ => return None,
        }
    }
    Some(out)
}

fn map_lbug(err: lbug::Error) -> OkfError {
    OkfError::internal(err.to_string(), "lbug::Error")
}

/// Convert an `lbug::Value` to JSON using Python-facing spellings.
pub fn value_to_json(value: Value) -> Json {
    match value {
        Value::Null(_) => Json::Null,
        Value::Bool(b) => Json::Bool(b),
        Value::Int8(i) => Json::from(i),
        Value::Int16(i) => Json::from(i),
        Value::Int32(i) => Json::from(i),
        Value::Int64(i) => Json::from(i),
        Value::UInt8(i) => Json::from(i),
        Value::UInt16(i) => Json::from(i),
        Value::UInt32(i) => Json::from(i),
        Value::UInt64(i) => Json::from(i),
        Value::Int128(i) => Json::from(i as i64),
        Value::Float(f) => serde_json::json!(f),
        Value::Double(f) => serde_json::json!(f),
        Value::Date(date) => Json::String(format!(
            "{:04}-{:02}-{:02}",
            date.year(),
            date.month() as u8,
            date.day()
        )),
        Value::Timestamp(dt)
        | Value::TimestampTz(dt)
        | Value::TimestampNs(dt)
        | Value::TimestampMs(dt)
        | Value::TimestampSec(dt) => {
            let utc = dt.to_offset(time::UtcOffset::UTC);
            let iso = format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                utc.year(),
                utc.month() as u8,
                utc.day(),
                utc.hour(),
                utc.minute(),
                utc.second()
            );
            Json::String(python_datetime_str(&iso))
        }
        Value::Interval(d) => Json::String(d.to_string()),
        Value::String(s) => Json::String(s),
        Value::Json(v) => v,
        Value::Blob(bytes) => Json::String(python_bytes_repr(&bytes)),
        Value::List(_, items) | Value::Array(_, items) => {
            Json::Array(items.into_iter().map(value_to_json).collect())
        }
        Value::Struct(fields) => Json::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k, value_to_json(v)))
                .collect(),
        ),
        Value::Node(node) => {
            let mut obj = Map::new();
            obj.insert(
                "_ID".into(),
                serde_json::json!({
                    "offset": node.get_node_id().offset,
                    "table": node.get_node_id().table_id
                }),
            );
            obj.insert("_LABEL".into(), Json::String(node.get_label_name().clone()));
            for (key, value) in node.get_properties() {
                obj.insert(key.clone(), value_to_json(value.clone()));
            }
            Json::Object(obj)
        }
        Value::Rel(rel) => {
            let mut obj = Map::new();
            obj.insert("_LABEL".into(), Json::String(rel.get_label_name().clone()));
            for (key, value) in rel.get_properties() {
                obj.insert(key.clone(), value_to_json(value.clone()));
            }
            Json::Object(obj)
        }
        Value::Map(_, entries) => Json::Object(
            entries
                .into_iter()
                .filter_map(|(k, v)| {
                    let key = value_to_json(k);
                    let name = key.as_str().unwrap_or_default().to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((name, value_to_json(v)))
                    }
                })
                .collect(),
        ),
        Value::Union { value, .. } => value_to_json(*value),
        Value::UUID(id) => Json::String(id.to_string()),
        Value::Decimal(d) => Json::String(d.to_string()),
        Value::InternalID(id) => serde_json::json!({
            "offset": id.offset,
            "table": id.table_id
        }),
        Value::RecursiveRel { nodes, rels } => serde_json::json!({
            "nodes": nodes.into_iter().map(|n| {
                let mut obj = Map::new();
                for (k, v) in n.get_properties() {
                    obj.insert(k.clone(), value_to_json(v.clone()));
                }
                Json::Object(obj)
            }).collect::<Vec<_>>(),
            "rels": rels.into_iter().map(|r| {
                let mut obj = Map::new();
                for (k, v) in r.get_properties() {
                    obj.insert(k.clone(), value_to_json(v.clone()));
                }
                Json::Object(obj)
            }).collect::<Vec<_>>(),
        }),
    }
}

fn schema_ddl(dim: usize) -> Vec<String> {
    vec![
        format!(
            "CREATE NODE TABLE IF NOT EXISTS Concept (
                id STRING PRIMARY KEY, type STRING, title STRING,
                description STRING, resource STRING, tags STRING[],
                timestamp TIMESTAMP, body STRING, embedding FLOAT[{dim}],
                extra MAP(STRING, STRING), created_at TIMESTAMP,
                updated_at TIMESTAMP, effective_at TIMESTAMP, content_sig STRING)"
        ),
        "CREATE NODE TABLE IF NOT EXISTS Directory (id STRING PRIMARY KEY)".into(),
        format!(
            "CREATE NODE TABLE IF NOT EXISTS ImageAsset (
                id STRING PRIMARY KEY, file_name STRING, mime_type STRING,
                alt_text STRING, caption STRING, embed_route STRING,
                content_hash STRING, data BLOB, embedding FLOAT[{dim}])"
        ),
        format!(
            "CREATE NODE TABLE IF NOT EXISTS Chunk (
                id STRING PRIMARY KEY, parent_doc_id STRING, chunk_index INT64,
                chunk_text STRING, block_type STRING, start_offset INT64,
                end_offset INT64, embedding FLOAT[{dim}])"
        ),
        "CREATE NODE TABLE IF NOT EXISTS BrokenLink (
            id STRING PRIMARY KEY, source_id STRING, target_id STRING,
            timestamp TIMESTAMP, kind STRING, reason STRING,
            candidates STRING, raw STRING)"
            .into(),
        "CREATE NODE TABLE IF NOT EXISTS DeletedConcept (
            id STRING PRIMARY KEY, original_id STRING, title STRING, body STRING,
            deleted_at STRING, type STRING, tags STRING, snapshot STRING)"
            .into(),
        "CREATE NODE TABLE IF NOT EXISTS Meta (key STRING PRIMARY KEY, value INT64)".into(),
        "CREATE NODE TABLE IF NOT EXISTS MetaText (key STRING PRIMARY KEY, value STRING)".into(),
        "CREATE NODE TABLE IF NOT EXISTS FileHash (
            path STRING PRIMARY KEY, hash STRING, concept_id STRING, dir STRING)"
            .into(),
        "CREATE NODE TABLE IF NOT EXISTS DirHash (
            path STRING PRIMARY KEY, hash STRING, files STRING)"
            .into(),
        "CREATE NODE TABLE IF NOT EXISTS DeletedPath (
            path STRING PRIMARY KEY, detected_at INT64, dir STRING)"
            .into(),
        "CREATE NODE TABLE IF NOT EXISTS SourceRoot (
            alias STRING PRIMARY KEY, path STRING, file_count INT64, last_seen INT64)"
            .into(),
        "CREATE NODE TABLE IF NOT EXISTS ProducerCursor (
            name STRING PRIMARY KEY, cursor STRING, updated_at INT64)"
            .into(),
        "CREATE REL TABLE IF NOT EXISTS CONTAINS (
            FROM Directory TO Directory, FROM Directory TO Concept)"
            .into(),
        "CREATE REL TABLE IF NOT EXISTS LINKS_TO (
            FROM Concept TO Concept, kind STRING DEFAULT 'mention')"
            .into(),
        "CREATE REL TABLE IF NOT EXISTS INCLUDES_ASSET (
            FROM Concept TO ImageAsset)"
            .into(),
        "CREATE REL TABLE IF NOT EXISTS PART_OF (FROM Concept TO Chunk)".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_table_names() {
        assert_eq!(NODE_TABLES.len(), 13);
        assert_eq!(REL_TABLES.len(), 4);
    }

    #[test]
    fn opens_oracle_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("parity/fixtures/oracle-v10.db");
        let settings = Settings::default()
            .with("db_path", Json::String(path.to_string_lossy().into_owned()))
            .expect("settings");
        let store = Store::open(&settings).expect("open");
        assert_eq!(store.schema_version().expect("version"), 10);
        assert!(store.get_by_id("alpha").expect("get").is_some());
    }
}
