//! Typed operation requests/results plus the `call(Op, Value)` escape hatch.
//!
//! M1 implements the read-only slice; write/ingest ops land in Stage B.

#[cfg(feature = "embed")]
use crate::embed::RerankerPool;
use crate::embed::{check_pins, EmbedderPool, Task};
use crate::error::{OkfError, Result};
use crate::markdown;
use crate::pycompat::from_json;
use crate::search::{self, DEDUPE_COSINE};
use crate::settings::Settings;
use crate::store::Store;
use serde_json::{Map, Value as Json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The canonical operation vocabulary (30 requests; `init` is local-only).
pub const OP_NAMES: &[&str] = &[
    "search",
    "read",
    "traverse",
    "embed",
    "ingest",
    "export_bundle",
    "export_concept",
    "import_bundle",
    "import_file",
    "import_files",
    "sync",
    "update",
    "supersede",
    "delete",
    "doctor",
    "diff",
    "reindex",
    "reembed",
    "pin_info",
    "lanes",
    "list_broken_links",
    "repair_links",
    "list_deleted",
    "recover_deleted",
    "purge_deleted",
    "detach",
    "list_images",
    "get_image",
    "get_by_id",
    "list_directory",
];

const READ_OPS: &[&str] = &[
    "search",
    "embed",
    "read",
    "traverse",
    "doctor",
    "pin_info",
    "lanes",
    "list_broken_links",
    "list_deleted",
    "list_images",
    "get_image",
    "get_by_id",
    "list_directory",
];

/// Stage B content-write ops implemented by the ingest pipeline.
pub const WRITE_OPS: &[&str] = &["import_file", "import_files"];

pub fn is_op(name: &str) -> bool {
    OP_NAMES.contains(&name)
}

#[derive(Debug, Clone)]
pub struct Router {
    store: Store,
    settings: Settings,
    embedder: EmbedderPool,
    #[cfg(feature = "embed")]
    reranker: RerankerPool,
}

impl Router {
    pub fn open(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open(settings)?,
            settings: settings.clone(),
            embedder: EmbedderPool::new(),
            #[cfg(feature = "embed")]
            reranker: RerankerPool::new(),
        })
    }

    /// Open an existing v10 database with write permission. Stage B callers
    /// must own the graph; never mix Python and Rust cores in one process.
    pub fn open_writable(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open_writable(settings)?,
            settings: settings.clone(),
            embedder: EmbedderPool::new(),
            #[cfg(feature = "embed")]
            reranker: RerankerPool::new(),
        })
    }

    /// Create a new v10 database.
    pub fn init(path: impl AsRef<Path>, embedding_dim: i64) -> Result<Vec<String>> {
        Store::init(path, embedding_dim)
    }

    fn import_files(&self, params: &Map<String, Json>) -> Result<Json> {
        let paths = match params.get("paths") {
            Some(Json::Array(items)) => items
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        if paths.is_empty() && params.contains_key("file_path") {
            return self.import_file(params);
        }
        crate::ingest::import_files(&self.store, &self.settings, &self.embedder, &paths)
            .map_err(|err| err.op("import_files"))
    }

    fn import_file(&self, params: &Map<String, Json>) -> Result<Json> {
        let path = str_param(params, "file_path")?.to_string();
        let report = crate::ingest::import_files(
            &self.store,
            &self.settings,
            &self.embedder,
            std::slice::from_ref(&path),
        )
        .map_err(|err| err.op("import_file"))?;
        let id = report
            .get("concept_ids")
            .and_then(Json::as_array)
            .and_then(|ids| ids.first())
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        Ok(serde_json::json!({
            "concept_id": id,
            "images": 0,
        }))
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn call(&self, op: &str, params: &Map<String, Json>) -> Result<Json> {
        if !is_op(op) {
            return Err(OkfError::new("BAD_VALUE", format!("unknown op: {op}"))
                .field("op", Json::String(op.into())));
        }
        if !READ_OPS.contains(&op) && !WRITE_OPS.contains(&op) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("op '{op}' is not implemented in Stage B"),
            )
            .field("op", Json::String(op.into())));
        }
        match op {
            "get_by_id" => self.get_by_id(str_param(params, "concept_id")?),
            "list_directory" => {
                let directory_id = opt_str(params, "directory_id")?.unwrap_or_default();
                self.list_directory(&directory_id)
            }
            "list_images" => self.list_images(str_param(params, "concept_id")?),
            "get_image" => self.get_image(str_param(params, "asset_id")?),
            "list_deleted" => self.list_deleted(),
            "list_broken_links" => self.list_broken_links(),
            "pin_info" => self.pin_info(),
            "lanes" => self.lanes(),
            "doctor" => self.doctor(),
            "search" => self.search(params),
            "embed" => self.embed(params),
            "read" => self.read(params),
            "traverse" => self.traverse(params),
            "import_files" => self.import_files(params),
            "import_file" => self.import_file(params),
            _ => unreachable!("read op without dispatch"),
        }
    }

    /// Python's `get_by_id` op returns `None` for a missing concept.
    pub fn get_by_id(&self, concept_id: &str) -> Result<Json> {
        require_nonempty("get_by_id", "concept_id", concept_id)?;
        Ok(self.store.get_by_id(concept_id)?.unwrap_or(Json::Null))
    }

    pub fn list_directory(&self, directory_id: &str) -> Result<Json> {
        Ok(Json::Array(self.store.list_directory(directory_id)?))
    }

    pub fn list_images(&self, concept_id: &str) -> Result<Json> {
        require_nonempty("list_images", "concept_id", concept_id)?;
        if self.store.get_by_id(concept_id)?.is_none() {
            return Err(unknown_concept("list_images", "concept_id", concept_id));
        }
        Ok(Json::Array(self.store.list_images(concept_id)?))
    }

    pub fn get_image(&self, asset_id: &str) -> Result<Json> {
        require_nonempty("get_image", "asset_id", asset_id)?;
        self.store.get_image(asset_id)?.ok_or_else(|| {
            OkfError::new(
                "UNKNOWN_ASSET",
                format!("image asset '{asset_id}' does not exist"),
            )
            .op("get_image")
            .field("asset_id", Json::String(asset_id.into()))
            .remedy("list_images(concept_id) names the asset ids")
        })
    }

    pub fn list_deleted(&self) -> Result<Json> {
        Ok(Json::Array(self.store.list_deleted()?))
    }

    pub fn list_broken_links(&self) -> Result<Json> {
        Ok(Json::Array(self.store.list_broken_links()?))
    }

    pub fn pin_info(&self) -> Result<Json> {
        let mut info = self.store.pin_info()?;
        let empty = info.get("empty").and_then(Json::as_bool).unwrap_or(true);
        if let Some(embedder) = self.embedder.loaded() {
            if let Some(configured) = info.get_mut("configured").and_then(Json::as_object_mut) {
                configured.insert(
                    "expected_precision".into(),
                    Json::String(embedder.precision().into()),
                );
                configured.insert(
                    "space".into(),
                    embedder.space().map(Json::String).unwrap_or(Json::Null),
                );
            }
        }
        let pinned = info.get("pinned").cloned().unwrap_or(Json::Null);
        let configured = info.get("configured").cloned().unwrap_or(Json::Null);
        let mut mismatch = Vec::new();
        if !empty {
            if let (Some(a), Some(b)) = (
                pinned.get("model_id").and_then(Json::as_str),
                configured.get("model_id").and_then(Json::as_str),
            ) {
                if a != b {
                    mismatch.push("model_id".to_string());
                }
            }
            if let (Some(a), Some(b)) = (
                pinned.get("precision").and_then(Json::as_str),
                configured.get("expected_precision").and_then(Json::as_str),
            ) {
                if a != b {
                    mismatch.push("precision".to_string());
                }
            }
            if let (Some(a), Some(b)) = (
                pinned.get("space").and_then(Json::as_str),
                configured.get("space").and_then(Json::as_str),
            ) {
                if a != b {
                    mismatch.push("space".to_string());
                }
            }
        }
        if let Some(Json::Array(image)) = info.get("image_mismatch") {
            for entry in image {
                if let Some(name) = entry.as_str() {
                    mismatch.push(format!("image_{name}"));
                }
            }
        }
        if let Some(obj) = info.as_object_mut() {
            obj.insert(
                "mismatch".into(),
                Json::Array(mismatch.into_iter().map(Json::String).collect()),
            );
        }
        Ok(info)
    }

    pub fn lanes(&self) -> Result<Json> {
        let mut lanes = self.store.lanes()?;
        if let Some(embedder) = self.embedder.loaded() {
            let pins = self.store.pin_info()?;
            let pinned_space = pins
                .get("pinned")
                .and_then(|p| p.get("space"))
                .cloned()
                .unwrap_or(Json::Null);
            if let Some(Json::Object(embed)) = lanes.get_mut("embed") {
                embed.insert("model_id".into(), Json::String(embedder.model_id().into()));
                embed.insert(
                    "precision".into(),
                    Json::String(embedder.precision().into()),
                );
                embed.insert(
                    "space".into(),
                    embedder.space().map_or(Json::Null, Json::String),
                );
                embed.insert("pinned_space".into(), pinned_space);
            }
        }
        #[cfg(feature = "embed")]
        if let Some(precision) = self.reranker.precision() {
            if let Some(Json::Object(rerank)) = lanes.get_mut("rerank") {
                rerank.insert("precision".into(), Json::String(precision.into()));
            }
        }
        Ok(lanes)
    }

    pub fn doctor(&self) -> Result<Json> {
        let (concepts, chunks) = self.store.counts()?;
        let pins = self.store.pin_info()?;
        let mut findings = Vec::new();
        if concepts == 0 {
            findings.push("graph has no concepts");
        }
        Ok(serde_json::json!({
            "ok": findings.is_empty(),
            "schema_version": self.store.schema_version()?,
            "concepts": concepts,
            "chunks": chunks,
            "pins": pins,
            "findings": findings
        }))
    }

    pub fn read(&self, params: &Map<String, Json>) -> Result<Json> {
        let concept_id = str_param(params, "concept_id")?;
        let include = opt_str(params, "include")?.unwrap_or_else(|| "body".into());
        const INCLUDES: &[&str] = &["body", "chunks", "document", "context"];
        if !INCLUDES.contains(&include.as_str()) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!(
                    "include must be one of 'body', 'chunks', 'document', 'context', got '{include}'"
                ),
            )
            .op("read")
            .field("include", Json::String(include)));
        }
        let max_tokens = opt_i64(params, "max_tokens")?;
        if let Some(budget) = max_tokens {
            if budget < 1 {
                return Err(bad_value2(
                    "read",
                    format!("max_tokens must be >= 1, got {budget}"),
                    vec![("max_tokens", Json::from(budget))],
                ));
            }
        }
        let concept = self.store.get_by_id(concept_id)?;
        if concept.is_none() {
            return Err(unknown_concept("read", "concept", concept_id));
        }
        if let Some(budget) = max_tokens {
            return self.read_with_budget(concept_id, &include, budget, &concept.expect("checked"));
        }
        match include.as_str() {
            "chunks" => Ok(Json::Array(self.store.get_chunks(concept_id)?)),
            "document" => {
                let chunks = self.store.get_chunks(concept_id)?;
                if chunks.is_empty() {
                    return Err(OkfError::new(
                        "UNKNOWN_CONCEPT",
                        format!("concept '{concept_id}' has no chunks to rebuild"),
                    )
                    .op("read")
                    .field("concept_id", Json::String(concept_id.into()))
                    .remedy("use include='body' for unchunked concepts"));
                }
                let typed: Vec<markdown::Chunk> =
                    serde_json::from_value(Json::Array(chunks.clone())).unwrap_or_default();
                let markdown = markdown::reconstruct_document(&typed).unwrap_or_default();
                Ok(serde_json::json!({"concept_id": concept_id, "markdown": markdown}))
            }
            "context" => {
                let incoming = self
                    .store
                    .traverse(concept_id, "LINKS_TO", "INCOMING", 1, None, None)?;
                let outgoing = self
                    .store
                    .traverse(concept_id, "LINKS_TO", "OUTGOING", 1, None, None)?;
                let (ancestry, siblings) = self.store.context(concept_id, 10)?;
                Ok(serde_json::json!({
                    "incoming_links": truncate(incoming, 10),
                    "outgoing_links": truncate(outgoing, 10),
                    "ancestry": ancestry,
                    "siblings": siblings
                }))
            }
            _ => Ok(concept.expect("checked")),
        }
    }

    /// Python `SearchEngine.read_with_budget`: ordered sections, exact
    /// tokenizer counts, proportional character truncation on the boundary.
    fn read_with_budget(
        &self,
        concept_id: &str,
        include: &str,
        max_tokens: i64,
        concept: &Json,
    ) -> Result<Json> {
        let mut sections: Vec<Json> = Vec::new();
        if include == "chunks" {
            for chunk in self.store.get_chunks(concept_id)? {
                let index = chunk.get("chunk_index").and_then(Json::as_i64).unwrap_or(0);
                let block = chunk.get("block_type").and_then(Json::as_str).unwrap_or("");
                sections.push(section(
                    chunk.get("id").and_then(Json::as_str).unwrap_or(""),
                    &format!("#{index} [{block}]"),
                    "chunk",
                    chunk.get("chunk_text").and_then(Json::as_str).unwrap_or(""),
                ));
            }
        } else if include == "document" {
            let chunks = self.store.get_chunks(concept_id)?;
            let typed: Vec<markdown::Chunk> =
                serde_json::from_value(Json::Array(chunks)).unwrap_or_default();
            let text = markdown::reconstruct_document(&typed).unwrap_or_default();
            sections.push(section(
                concept_id,
                concept.get("title").and_then(Json::as_str).unwrap_or(""),
                "document",
                &text,
            ));
        } else {
            if include == "context" {
                let mut index_rows = self.store.query(
                    "MATCH (c:Concept) WHERE c.id ENDS WITH 'index' \
                     RETURN c.id AS id, c.title AS title, c.body AS body",
                    vec![],
                )?;
                index_rows.sort_by_key(|row| {
                    let id = row.get("id").and_then(Json::as_str).unwrap_or("");
                    (if id == "index" { 0 } else { 1 }, id.to_string())
                });
                if let Some(index) = index_rows.first() {
                    let id = index.get("id").and_then(Json::as_str).unwrap_or("");
                    if id != concept_id {
                        sections.push(section_with_title(
                            id,
                            index.get("title").cloned().unwrap_or(Json::Null),
                            "index",
                            index.get("body").and_then(Json::as_str).unwrap_or(""),
                        ));
                    }
                }
            }
            sections.push(section_with_title(
                concept_id,
                concept.get("title").cloned().unwrap_or(Json::Null),
                "body",
                concept.get("body").and_then(Json::as_str).unwrap_or(""),
            ));
            if include == "context" {
                let (nodes, edges) = self.store.link_graph(None)?;
                let ranked = search::ppr(
                    &nodes,
                    &edges,
                    &[(concept_id.to_string(), 1.0)],
                    0.85,
                    1e-12,
                    200,
                    None,
                );
                let ids: Vec<String> = ranked
                    .into_iter()
                    .map(|(id, _)| id)
                    .filter(|id| id != concept_id)
                    .collect();
                if !ids.is_empty() {
                    let values = lbug::Value::List(
                        lbug::LogicalType::String,
                        ids.iter().cloned().map(lbug::Value::String).collect(),
                    );
                    let rows = self.store.query(
                        "MATCH (c:Concept) WHERE c.id IN $ids \
                         RETURN c.id AS id, c.title AS title, \
                                c.description AS description, c.body AS body",
                        vec![("ids", values)],
                    )?;
                    let meta: BTreeMap<String, Json> = rows
                        .into_iter()
                        .filter_map(|row| {
                            let id = row.get("id")?.as_str()?.to_string();
                            Some((id, Json::Object(row)))
                        })
                        .collect();
                    for id in ids {
                        let Some(row) = meta.get(&id) else { continue };
                        let title = row.get("title").and_then(Json::as_str).unwrap_or(&id);
                        let text = format!(
                            "{}\n{}\n{}",
                            title,
                            row.get("description").and_then(Json::as_str).unwrap_or(""),
                            row.get("body").and_then(Json::as_str).unwrap_or(""),
                        );
                        sections.push(section_with_title(
                            &id,
                            row.get("title").cloned().unwrap_or(Json::Null),
                            "neighbour",
                            &text,
                        ));
                    }
                }
            }
        }

        let tokenizer = self.embedder.get(&self.settings).ok();
        let count = |text: &str| -> usize {
            let fallback = || (text.chars().count() / 4).max(1);
            tokenizer
                .as_ref()
                .and_then(|embedder| embedder.count_tokens(text).ok())
                .unwrap_or_else(fallback)
        };
        let mut kept = Vec::new();
        let mut used = 0i64;
        let mut truncated = false;
        for mut sec in sections {
            let text = sec
                .get("text")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string();
            let cost = count(&text) as i64;
            if used + cost <= max_tokens {
                used += cost;
                kept.push(sec);
                continue;
            }
            let budget = (max_tokens - used).max(0) as usize;
            sec.as_object_mut().expect("section object").insert(
                "text".to_string(),
                Json::String(truncate_to_tokens(&text, budget, &count)),
            );
            kept.push(sec);
            used = max_tokens;
            truncated = true;
            break;
        }
        Ok(serde_json::json!({
            "concept_id": concept_id,
            "include": include,
            "budget": max_tokens,
            "used": used,
            "truncated": truncated,
            "sections": kept,
        }))
    }

    /// Pydantic `str(ConceptModel)` used by the oracle harness via
    /// `json.dumps(..., default=str)`. Field order follows the Python model
    /// declaration, then graph times, then `extra` keys.
    fn concept_model_str(&self, id: &str) -> Result<String> {
        let rows = self.store.query(
            "MATCH (c:Concept {id: $id}) RETURN c.*",
            vec![("id", lbug::Value::String(id.to_string()))],
        )?;
        let Some(row) = rows.first() else {
            return Ok(String::new());
        };
        const BASE: &[(&str, Json)] = &[
            ("id", Json::String(String::new())),
            ("type", Json::String(String::new())),
            ("title", Json::Null),
            ("description", Json::Null),
            ("resource", Json::Null),
            ("tags", Json::Array(Vec::new())),
            ("timestamp", Json::Null),
            ("body", Json::String(String::new())),
            ("embedding", Json::Null),
        ];
        let mut fields: Vec<(String, Json)> = Vec::new();
        for (name, default) in BASE {
            fields.push((
                (*name).to_string(),
                row.get(&format!("c.{name}"))
                    .cloned()
                    .unwrap_or_else(|| default.clone()),
            ));
        }
        for name in ["created_at", "updated_at", "effective_at"] {
            if let Some(value) = row.get(&format!("c.{name}")) {
                fields.push((name.to_string(), value.clone()));
            }
        }
        if let Some(Json::Object(extra)) = row.get("c.extra") {
            for (key, value) in extra {
                let decoded = match value {
                    Json::String(text) if text.starts_with('{') || text.starts_with('[') => {
                        serde_json::from_str(text).unwrap_or_else(|_| value.clone())
                    }
                    other => other.clone(),
                };
                fields.push((key.clone(), decoded));
            }
        }
        Ok(fields
            .iter()
            .map(|(name, value)| {
                let datetime = ["timestamp", "created_at", "updated_at", "effective_at"]
                    .contains(&name.as_str());
                format!("{name}={}", model_value_repr(value, datetime))
            })
            .collect::<Vec<_>>()
            .join(" "))
    }

    pub fn traverse(&self, params: &Map<String, Json>) -> Result<Json> {
        let start_id = opt_str(params, "start_id")?.unwrap_or_default();
        let relationship = opt_str(params, "relationship")?.unwrap_or_else(|| "CONTAINS".into());
        let direction = opt_str(params, "direction")?.unwrap_or_else(|| "OUTGOING".into());
        let depth = opt_i64(params, "depth")?.unwrap_or(1);
        let node_type = opt_str(params, "node_type")?;
        let target =
            opt_str(params, "target_id")?.or_else(|| opt_str(params, "target").unwrap_or(None));
        let max_path_length = opt_i64(params, "max_path_length")?.unwrap_or(6);
        let link_kinds = match params.get("link_kinds") {
            Some(Json::Array(items)) => Some(
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };

        const RELS: &[&str] = &["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"];
        const DIRS: &[&str] = &["OUTGOING", "INCOMING", "BOTH"];
        if !RELS.contains(&relationship.as_str()) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("relationship must be one of {RELS:?}, got '{relationship}'"),
            )
            .op("traverse"));
        }
        if !DIRS.contains(&direction.as_str()) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("direction must be one of {DIRS:?}, got '{direction}'"),
            )
            .op("traverse"));
        }
        if depth < 1 {
            return Err(
                OkfError::new("BAD_VALUE", format!("depth must be >= 1, got {depth}"))
                    .op("traverse"),
            );
        }
        if link_kinds.is_some() && relationship != "LINKS_TO" {
            return Err(
                OkfError::new("BAD_VALUE", "link_kinds needs relationship='LINKS_TO'")
                    .op("traverse"),
            );
        }

        if start_id.is_empty() {
            return self.list_directory("");
        }
        if let Some(target) = target {
            for (field, id) in [("start_id", start_id.as_str()), ("target", target.as_str())] {
                if !self.store.node_exists(id)? {
                    return Err(unknown_concept("traverse", field, id));
                }
            }
            return self.find_path(&start_id, &target, max_path_length);
        }
        if !self.store.node_exists(&start_id)? {
            return Err(unknown_concept("traverse", "start_id", &start_id));
        }
        Ok(Json::Array(self.store.traverse(
            &start_id,
            &relationship,
            &direction,
            depth,
            node_type.as_deref(),
            link_kinds.as_deref(),
        )?))
    }

    fn find_path(&self, start: &str, target: &str, max_len: i64) -> Result<Json> {
        let max_len = max_len.clamp(1, 6) as usize;
        let mut queue: Vec<(String, Vec<Json>)> = vec![(start.to_string(), Vec::new())];
        let mut visited = std::collections::BTreeSet::new();
        visited.insert(start.to_string());
        while !queue.is_empty() {
            let (current, path) = queue.remove(0);
            if current.as_str() == target {
                return Ok(Json::Array(path));
            }
            if path.len() >= max_len {
                continue;
            }
            for neighbor in self
                .store
                .traverse(&current, "LINKS_TO", "BOTH", 1, None, None)?
                .into_iter()
                .chain(
                    self.store
                        .traverse(&current, "CONTAINS", "BOTH", 1, None, None)?,
                )
            {
                let Some(id) = neighbor
                    .get("id")
                    .and_then(Json::as_str)
                    .map(str::to_string)
                else {
                    continue;
                };
                if visited.insert(id.clone()) {
                    let mut next = path.clone();
                    next.push(neighbor);
                    queue.push((id, next));
                }
            }
        }
        Ok(Json::Array(Vec::new()))
    }

    /// Embed texts with the configured native model. No database writes.
    pub fn embed(&self, params: &Map<String, Json>) -> Result<Json> {
        let texts = match params.get("texts") {
            Some(Json::Array(items)) if !items.is_empty() => items
                .iter()
                .map(|v| {
                    v.as_str().map(str::to_string).ok_or_else(|| {
                        OkfError::new("BAD_VALUE", "texts must be a list of strings").op("embed")
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => {
                return Err(
                    OkfError::new("BAD_VALUE", "texts must be a non-empty list of strings")
                        .op("embed"),
                )
            }
        };
        let task = match opt_str(params, "task")?.as_deref() {
            None | Some("Query") => Task::Query,
            Some("Document") => Task::Document,
            Some(other) => {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    format!("task must be 'Query' or 'Document', got '{other}'"),
                )
                .op("embed"))
            }
        };
        let embedder = self.embedder.get(&self.settings)?;
        check_pins(&self.settings, &self.store.pin_info()?)?;
        let vectors = embedder.encode_many(&texts, task)?;
        let mut vectors: Vec<Vec<f32>> = vectors;
        let mut renormalised = false;
        if let Some(dim) = opt_i64(params, "dim")? {
            const LADDER: &[i64] = &[32, 64, 128, 256, 512, 768, 1024];
            if !LADDER.contains(&dim) || dim as usize > embedder.dim() {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    format!(
                        "dim must be a Matryoshka level <= {}, got {dim}",
                        embedder.dim()
                    ),
                )
                .op("embed"));
            }
            let dim = dim as usize;
            renormalised = dim < embedder.dim();
            for vector in &mut vectors {
                vector.truncate(dim);
                let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for value in vector.iter_mut() {
                        *value /= norm;
                    }
                }
            }
        }
        let dim = vectors.first().map_or(embedder.dim(), Vec::len);
        Ok(serde_json::json!({
            "vectors": vectors,
            "dim": dim,
            "model_id": embedder.model_id(),
            "task": task.as_str(),
            "renormalised": renormalised,
            "honours_dimensions": true,
            "counter": "crate",
        }))
    }

    /// Hybrid concept/chunk search. The real query vector comes from the GPU
    /// embedder unless a scenario supplies `query_vector` explicitly.
    pub fn search(&self, raw: &Map<String, Json>) -> Result<Json> {
        let mut params = SearchParams::parse(raw)?;
        if params.hub_rerank || params.expand {
            // chunks:hub and chunks:expand do not honour filters, including
            // the configured default exclude_tags list.
            params.exclude_tags.clear();
        } else if !params.exclude_tags_set {
            params.exclude_tags = self.settings.exclude_tags.clone();
        }
        if let Some(id) = &params.near_id {
            if self.store.get_by_id(id)?.is_none() {
                return Err(unknown_concept("search", "near_id", id));
            }
        }
        let query_vector = self.query_vector(&params.query, params.query_vector.clone())?;
        match params.target.as_str() {
            "concepts" => self.search_concepts(&params, &query_vector),
            "chunks" => {
                let rows = if params.hub_rerank {
                    self.search_chunks_hub(&params, &query_vector, params.limit)?
                } else if params.expand {
                    self.search_with_context(&params, &query_vector, params.limit.min(20))?
                } else {
                    self.search_chunks(&params, &query_vector, params.limit)?
                };
                Ok(Json::Array(rows))
            }
            "images" => self.search_images(&params, &query_vector),
            _ => unreachable!("validated target"),
        }
    }

    /// Text-query search over the image-vector index. The oracle encodes the
    /// query with the paired text model; stored rows may be caption or vision
    /// route vectors, both in the same space.
    fn search_images(&self, params: &SearchParams, query_vector: &[f32]) -> Result<Json> {
        let hits = self.store.vector_hits_raw(
            "ImageAsset",
            "image_omni_idx",
            query_vector,
            params.limit as i64,
            None,
        )?;
        let asset_ids: Vec<String> = hits
            .iter()
            .filter_map(|(node, _)| node.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();
        let mut owners: BTreeMap<String, String> = BTreeMap::new();
        if !asset_ids.is_empty() {
            let values = lbug::Value::List(
                lbug::LogicalType::String,
                asset_ids.iter().cloned().map(lbug::Value::String).collect(),
            );
            for row in self.store.query(
                "MATCH (c:Concept)-[:INCLUDES_ASSET]->(i:ImageAsset) \
                 WHERE i.id IN $aids RETURN i.id AS aid, c.id AS cid",
                vec![("aids", values)],
            )? {
                let (Some(aid), Some(cid)) = (
                    row.get("aid").and_then(Json::as_str),
                    row.get("cid").and_then(Json::as_str),
                ) else {
                    continue;
                };
                let current = owners.get(aid).map(String::as_str).unwrap_or("");
                if current.is_empty() || cid < current {
                    owners.insert(aid.to_string(), cid.to_string());
                }
            }
        }
        let mut rows = Vec::new();
        for (node, distance) in hits {
            let asset_id = node.get("id").and_then(Json::as_str).unwrap_or("");
            let owner = owners
                .get(asset_id)
                .map(|id| Json::String(id.clone()))
                .unwrap_or(Json::Null);
            rows.push(serde_json::json!({
                "id": owner,
                "concept_id": owners.get(asset_id).map(|id| Json::String(id.clone())).unwrap_or(Json::Null),
                "asset_id": asset_id,
                "file_name": node.get("file_name").cloned().unwrap_or(Json::Null),
                "alt_text": node.get("alt_text").cloned().unwrap_or(Json::Null),
                "embed_route": node.get("embed_route").cloned().unwrap_or(Json::Null),
                "distance": json_number(distance),
                "relevance_score": json_number(1.0 - distance),
            }));
        }
        Ok(Json::Array(rows))
    }

    fn query_vector(&self, query: &str, explicit: Option<Vec<f32>>) -> Result<Vec<f32>> {
        if let Some(vector) = explicit {
            return Ok(vector);
        }
        #[cfg(feature = "embed")]
        {
            check_pins(&self.settings, &self.store.pin_info()?)?;
            let embedder = self.embedder.get(&self.settings)?;
            embedder.encode(query, Task::Query)
        }
        #[cfg(not(feature = "embed"))]
        {
            let _ = query;
            Err(OkfError::new(
                "NO_ORT_RUNTIME",
                "this build has no embedder; pass query_vector explicitly",
            )
            .op("search"))
        }
    }

    fn search_concepts(&self, params: &SearchParams, query_vector: &[f32]) -> Result<Json> {
        if params.rank == "ppr" {
            let (nodes, edges) = self.store.link_graph(params.link_kinds.as_deref())?;
            let all = self.store.concepts_by_ids(&nodes)?;
            let ranked = search::ppr(
                &nodes,
                &edges,
                &search::seeds(&all, &params.query, 20),
                0.85,
                1e-12,
                200,
                None,
            );
            let ids: Vec<String> = ranked.iter().map(|(id, _)| id.clone()).collect();
            let scored: Vec<(String, f64)> = ranked;
            let mut rows = self.concept_rows(&ids, &scored, params)?;
            rows.truncate(params.limit);
            return Ok(Json::Array(rows));
        }
        let want = params.limit + params.offset;
        let filtered = params.filtered();
        let within = match &params.near_id {
            Some(id) => Some(self.store.near_ids(id, params.hops)?),
            None => None,
        };
        let (combined, full) = if let Some(within) = within {
            (
                self.rrf_within(
                    "Concept",
                    "concept_fts",
                    query_vector,
                    &params.query,
                    &within,
                )?,
                false,
            )
        } else {
            let k = search::fetch_k(want.max(search::PAGE_WINDOW), filtered);
            let vec = self.store.vector_hits(
                "Concept",
                "concept_embedding",
                query_vector,
                k,
                params.efs,
            )?;
            let fts = self
                .store
                .fts_hits("Concept", "concept_fts", &params.query, k)?;
            let combined = search::rrf_fuse(&vec, &fts);
            let full = vec.len() as i64 >= k || fts.len() as i64 >= k;
            (combined, full)
        };
        let mut rows = self.concept_rows(
            &combined
                .iter()
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>(),
            &combined,
            params,
        )?;
        if filtered {
            if let Some(deeper) = search::deeper_k(
                search::fetch_k(want.max(search::PAGE_WINDOW), filtered),
                rows.len(),
                want.max(search::PAGE_WINDOW),
                full,
            ) {
                let vec = self.store.vector_hits(
                    "Concept",
                    "concept_embedding",
                    query_vector,
                    deeper,
                    params.efs,
                )?;
                let fts = self
                    .store
                    .fts_hits("Concept", "concept_fts", &params.query, deeper)?;
                let combined = search::rrf_fuse(&vec, &fts);
                rows = self.concept_rows(
                    &combined
                        .iter()
                        .map(|(id, _)| id.clone())
                        .collect::<Vec<_>>(),
                    &combined,
                    params,
                )?;
            }
        }

        if params.rank == "hub" && !rows.is_empty() {
            let (_, edges) = self.store.link_graph(params.link_kinds.as_deref())?;
            let ids: Vec<String> = rows
                .iter()
                .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
                .collect();
            let hubs = search::hub_scores(&ids, &edges);
            let max_hub = hubs.values().copied().max().unwrap_or(0);
            for row in &mut rows {
                let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
                    continue;
                };
                let hub = hubs.get(&id).copied().unwrap_or(0);
                let norm = if max_hub > 0 {
                    hub as f64 / max_hub as f64
                } else {
                    0.0
                };
                let base = row
                    .get("relevance_score")
                    .and_then(Json::as_f64)
                    .unwrap_or(0.0);
                let score = (1.0 - params.hub_weight) * base + params.hub_weight * norm;
                let obj = row.as_object_mut().expect("row object");
                obj.insert("hub_score".into(), Json::from(norm));
                obj.insert("relevance_score".into(), json_number(score));
            }
            rows.sort_by(|a, b| {
                let sa = a
                    .get("relevance_score")
                    .and_then(Json::as_f64)
                    .unwrap_or(0.0);
                let sb = b
                    .get("relevance_score")
                    .and_then(Json::as_f64)
                    .unwrap_or(0.0);
                sb.total_cmp(&sa).then_with(|| {
                    a.get("id")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .cmp(b.get("id").and_then(Json::as_str).unwrap_or(""))
                })
            });
        }
        if params.recency.is_some() {
            self.recency_blend(&mut rows, params)?;
        }
        if params.rank == "cross" {
            self.cross_rerank(&mut rows, &params.query)?;
        }
        let start = params.offset.min(rows.len());
        let end = (start + params.limit).min(rows.len());
        let mut page = rows[start..end].to_vec();
        if params.include_chunks {
            for row in &mut page {
                let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
                    continue;
                };
                let mut child = params.clone();
                child.parent_id = Some(id.clone());
                child.limit = 3;
                child.offset = 0;
                let chunks = self.search_chunks(&child, query_vector, 3)?;
                row.as_object_mut()
                    .expect("row object")
                    .insert("matched_chunks".into(), Json::Array(chunks));
            }
        }
        Ok(Json::Array(page))
    }

    fn concept_rows(
        &self,
        ids: &[String],
        scores: &[(String, f64)],
        params: &SearchParams,
    ) -> Result<Vec<Json>> {
        let rows = self.store.concepts_by_ids(ids)?;
        let score_by_id: BTreeMap<&str, f64> = scores
            .iter()
            .map(|(id, score)| (id.as_str(), *score))
            .collect();
        let descendants = match &params.parent_id {
            Some(id) => Some(
                self.store
                    .descendant_concepts(id, 3)?
                    .into_iter()
                    .collect::<BTreeSet<_>>(),
            ),
            None => None,
        };
        let within = match &params.near_id {
            Some(id) => Some(
                self.store
                    .near_ids(id, params.hops)?
                    .into_iter()
                    .collect::<BTreeSet<_>>(),
            ),
            None => None,
        };
        let mut by_id: BTreeMap<String, Json> = BTreeMap::new();
        for row in rows {
            let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
                continue;
            };
            if id.ends_with("index") || id.ends_with("log") {
                continue;
            }
            if params.concept_type.is_some()
                && row.get("type").and_then(Json::as_str) != params.concept_type.as_deref()
            {
                continue;
            }
            let tags = json_strings(row.get("tags"));
            if !params.tags.is_empty() && !params.tags.iter().any(|tag| tags.contains(tag)) {
                continue;
            }
            if !params.exclude_tags.is_empty()
                && params.exclude_tags.iter().any(|tag| tags.contains(tag))
            {
                continue;
            }
            if let Some(set) = &descendants {
                if !set.contains(&id) {
                    continue;
                }
            }
            if let Some(set) = &within {
                if !set.contains(&id) {
                    continue;
                }
            }
            let effective = row.get("effective_at").and_then(Json::as_str).unwrap_or("");
            if let Some(since) = &params.since {
                if effective < since.as_str() {
                    continue;
                }
            }
            if let Some(until) = &params.until {
                if effective >= until.as_str() {
                    continue;
                }
            }
            if !params.where_map.is_empty() {
                let extra = row.get("extra").and_then(Json::as_object);
                let mut matches = true;
                for (key, expected) in &params.where_map {
                    let actual = extra
                        .and_then(|map| map.get(key))
                        .map(|value| match value {
                            Json::String(s) => s.clone(),
                            other => from_json(other).to_python_json(false),
                        })
                        .unwrap_or_default();
                    if actual != *expected {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    continue;
                }
            }
            let Some(score) = score_by_id.get(id.as_str()) else {
                continue;
            };
            let description = row.get("description").and_then(Json::as_str).unwrap_or("");
            let description = if description.chars().count() > 200 {
                format!("{}...", description.chars().take(200).collect::<String>())
            } else {
                description.to_string()
            };
            by_id.insert(
                id.clone(),
                serde_json::json!({
                    "id": id,
                    "title": row.get("title").cloned().unwrap_or(Json::Null),
                    "type": row.get("type").cloned().unwrap_or(Json::Null),
                    "description": description,
                    "tags": tags,
                    "relevance_score": json_number(*score),
                }),
            );
        }
        // RRF order from `ids`, not the DB's natural order.
        Ok(ids.iter().filter_map(|id| by_id.get(id).cloned()).collect())
    }

    fn search_chunks(
        &self,
        params: &SearchParams,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<Json>> {
        let offset = params.offset;
        let want = [
            limit + offset,
            search::PAGE_WINDOW,
            if params.rerank {
                self.settings.rerank_top_n.max(0) as usize
            } else {
                0
            },
            if params.dedupe {
                2 * (limit + offset)
            } else {
                0
            },
        ]
        .into_iter()
        .max()
        .unwrap_or(search::PAGE_WINDOW);
        let within = params
            .near_id
            .as_ref()
            .map(|id| self.store.near_ids(id, params.hops))
            .transpose()?;
        let filtered = params.filtered();
        let mut rows = if let Some(within) = within.as_deref() {
            let combined =
                self.rrf_within("Chunk", "chunk_fts", query_vector, &params.query, within)?;
            self.chunk_rows(&combined, params, want)?
        } else {
            let k = search::fetch_k(want, filtered);
            let vec =
                self.store
                    .vector_hits("Chunk", "chunk_embedding", query_vector, k, params.efs)?;
            let fts = self
                .store
                .fts_hits("Chunk", "chunk_fts", &params.query, k)?;
            let combined = search::rrf_fuse(&vec, &fts);
            let mut rows = self.chunk_rows(&combined, params, want)?;
            if filtered {
                if let Some(deeper) = search::deeper_k(
                    k,
                    rows.len(),
                    want,
                    vec.len() as i64 >= k || fts.len() as i64 >= k,
                ) {
                    let vec = self.store.vector_hits(
                        "Chunk",
                        "chunk_embedding",
                        query_vector,
                        deeper,
                        params.efs,
                    )?;
                    let fts = self
                        .store
                        .fts_hits("Chunk", "chunk_fts", &params.query, deeper)?;
                    rows = self.chunk_rows(&search::rrf_fuse(&vec, &fts), params, want)?;
                }
            }
            rows
        };
        if params.recency.is_some() {
            self.recency_blend(&mut rows, params)?;
        }
        if params.rerank {
            self.cross_rerank(&mut rows, &params.query)?;
        }
        if params.dedupe {
            self.collapse_duplicates(&mut rows)?;
        }
        Ok(rows.into_iter().skip(offset).take(limit).collect())
    }

    fn search_chunks_hub(
        &self,
        params: &SearchParams,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<Json>> {
        let pool = [
            limit * 2,
            self.settings.rerank_top_n.max(0) as usize,
            if params.dedupe { limit * 4 } else { 0 },
        ]
        .into_iter()
        .max()
        .unwrap_or(limit * 2);
        let mut child = params.clone();
        child.offset = 0;
        child.rerank = false;
        child.dedupe = false;
        child.hub_rerank = false;
        let mut rows = self.search_chunks(&child, query_vector, pool)?;
        let parents: Vec<String> = rows
            .iter()
            .filter_map(|row| {
                row.get("parent_doc_id")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let (_, edges) = self.store.link_graph(None)?;
        let hubs = search::hub_scores(&parents, &edges);
        let max_hub = hubs.values().copied().max().unwrap_or(0).max(1) as f64;
        for row in &mut rows {
            let parent_id = row
                .get("parent_doc_id")
                .and_then(Json::as_str)
                .unwrap_or("");
            let hub = hubs.get(parent_id).copied().unwrap_or(0) as f64 / max_hub;
            let base = row.get("rrf_score").and_then(Json::as_f64).unwrap_or(0.0);
            let score = (1.0 - params.hub_weight) * base + params.hub_weight * hub;
            let obj = row.as_object_mut().expect("row object");
            obj.insert("hub_score".into(), json_number(hub));
            obj.insert("final_score".into(), json_number(score));
        }
        rows.sort_by(|a, b| {
            let sa = a.get("final_score").and_then(Json::as_f64).unwrap_or(0.0);
            let sb = b.get("final_score").and_then(Json::as_f64).unwrap_or(0.0);
            sb.total_cmp(&sa)
        });
        if params.rerank {
            self.cross_rerank(&mut rows, &params.query)?;
        }
        if params.dedupe {
            self.collapse_duplicates(&mut rows)?;
        }
        rows.truncate(limit);
        Ok(rows)
    }

    /// Python `search_with_context`: plain chunk search plus graph context
    /// around each chunk's owning concept.
    fn search_with_context(
        &self,
        params: &SearchParams,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<Json>> {
        let chunks = self.search_chunks(params, query_vector, limit * 2)?;
        let mut enriched = Vec::new();
        for chunk in chunks.into_iter().take(limit) {
            let parent_id = chunk
                .get("parent_doc_id")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string();
            let incoming = self.store.traverse(
                &parent_id,
                "LINKS_TO",
                "INCOMING",
                params.context_hops,
                None,
                None,
            )?;
            let outgoing = self.store.traverse(
                &parent_id,
                "LINKS_TO",
                "OUTGOING",
                params.context_hops,
                None,
                None,
            )?;
            let (ancestry, siblings) = self.store.context(&parent_id, 10)?;
            let document = Json::String(self.concept_model_str(&parent_id)?);
            enriched.push(serde_json::json!({
                "id": parent_id,
                "concept_id": parent_id,
                "chunk": chunk,
                "document": document,
                "incoming_links": truncate(incoming, 5),
                "outgoing_links": truncate(outgoing, 5),
                "ancestry": ancestry,
                "siblings": truncate(siblings, 5),
            }));
        }
        Ok(enriched)
    }

    fn chunk_rows(
        &self,
        combined: &[(String, f64)],
        params: &SearchParams,
        limit: usize,
    ) -> Result<Vec<Json>> {
        let ids: Vec<String> = combined.iter().map(|(id, _)| id.clone()).collect();
        let rows = self.store.chunks_by_ids(&ids)?;
        let by_id: BTreeMap<String, Json> = rows
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                Some((id, row))
            })
            .collect();
        let parents: Vec<String> = by_id
            .values()
            .filter_map(|row| {
                row.get("parent_doc_id")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let parent_by_id: BTreeMap<String, Json> = self
            .store
            .concepts_by_ids(&parents)?
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                Some((id, row))
            })
            .collect();
        let within = params
            .near_id
            .as_ref()
            .map(|id| {
                self.store
                    .near_ids(id, params.hops)
                    .map(|ids| ids.into_iter().collect::<BTreeSet<_>>())
            })
            .transpose()?;
        let mut out = Vec::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (chunk_id, score) in combined {
            let Some(row) = by_id.get(chunk_id) else {
                continue;
            };
            let parent_id = row
                .get("parent_doc_id")
                .and_then(Json::as_str)
                .unwrap_or("");
            let parent = parent_by_id.get(parent_id);
            if let Some(wanted) = &params.parent_id {
                if parent_id != wanted {
                    continue;
                }
            }
            if let Some(concept_type) = &params.concept_type {
                if parent.and_then(|p| p.get("type").and_then(Json::as_str))
                    != Some(concept_type.as_str())
                {
                    continue;
                }
            }
            let parent_tags = parent
                .map(|p| json_strings(p.get("tags")))
                .unwrap_or_default();
            if !params.tags.is_empty() && !params.tags.iter().any(|tag| parent_tags.contains(tag)) {
                continue;
            }
            if !params.exclude_tags.is_empty()
                && params
                    .exclude_tags
                    .iter()
                    .any(|tag| parent_tags.contains(tag))
            {
                continue;
            }
            if let Some(set) = &within {
                if !set.contains(parent_id) {
                    continue;
                }
            }
            if let Some(parent) = parent {
                let effective = parent
                    .get("effective_at")
                    .and_then(Json::as_str)
                    .unwrap_or("");
                if params
                    .since
                    .as_ref()
                    .is_some_and(|since| effective < since.as_str())
                    || params
                        .until
                        .as_ref()
                        .is_some_and(|until| effective >= until.as_str())
                {
                    continue;
                }
                if !params.where_map.is_empty() {
                    let extra = parent.get("extra").and_then(Json::as_object);
                    if !params.where_map.iter().all(|(key, expected)| {
                        extra
                            .and_then(|map| map.get(key))
                            .map(|value| match value {
                                Json::String(s) => s.clone(),
                                other => from_json(other).to_python_json(false),
                            })
                            .unwrap_or_default()
                            == *expected
                    }) {
                        continue;
                    }
                }
            } else if !params.where_map.is_empty()
                || params.since.is_some()
                || params.until.is_some()
            {
                continue;
            }
            let count = counts.entry(parent_id.to_string()).or_default();
            *count += 1;
            if *count > params.max_chunks_per_doc {
                continue;
            }
            let text = row.get("chunk_text").and_then(Json::as_str).unwrap_or("");
            let text = if text.chars().count() > 500 {
                format!("{}...", text.chars().take(500).collect::<String>())
            } else {
                text.to_string()
            };
            out.push(serde_json::json!({
                "id": parent_id,
                "concept_id": parent_id,
                "chunk_id": chunk_id,
                "chunk_text": text,
                "block_type": row.get("block_type").cloned().unwrap_or(Json::Null),
                "chunk_index": row.get("chunk_index").cloned().unwrap_or(Json::Null),
                "parent_doc_id": parent_id,
                "rrf_score": json_number(*score),
                "parent_title": parent.and_then(|p| p.get("title")).cloned().unwrap_or(Json::Null),
                "parent_type": parent.and_then(|p| p.get("type")).cloned().unwrap_or(Json::Null),
                "parent_tags": Json::Array(parent_tags.into_iter().map(Json::String).collect()),
            }));
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    fn collapse_duplicates(&self, rows: &mut Vec<Json>) -> Result<()> {
        let ids: Vec<String> = rows
            .iter()
            .filter_map(|row| {
                row.get("chunk_id")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            })
            .collect();
        if ids.len() < 2 {
            return Ok(());
        }
        let sql =
            "MATCH (n:`Chunk`) WHERE n.id IN $ids RETURN n.id AS id, n.embedding AS embedding";
        let embeddings: BTreeMap<String, Vec<f32>> = self
            .store
            .query(
                sql,
                vec![(
                    "ids",
                    lbug::Value::List(
                        lbug::LogicalType::String,
                        ids.iter().cloned().map(lbug::Value::String).collect(),
                    ),
                )],
            )?
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                Some((id, json_floats(row.get("embedding"))))
            })
            .collect();
        let mut kept: Vec<(String, Vec<f32>)> = Vec::new();
        let mut out = Vec::with_capacity(rows.len());
        for row in rows.drain(..) {
            let Some(chunk_id) = row
                .get("chunk_id")
                .and_then(Json::as_str)
                .map(str::to_string)
            else {
                out.push(row);
                continue;
            };
            let embedding = embeddings.get(&chunk_id).cloned().unwrap_or_default();
            let twin = if embedding.is_empty() {
                None
            } else {
                kept.iter()
                    .find(|(_, other)| search::cosine(&embedding, other) >= DEDUPE_COSINE)
                    .map(|(id, _)| id.clone())
            };
            if let Some(twin) = twin {
                for kept_row in &mut out {
                    if kept_row.get("chunk_id").and_then(Json::as_str) == Some(twin.as_str()) {
                        let entry = kept_row
                            .as_object_mut()
                            .expect("row object")
                            .entry("collapsed".to_string())
                            .or_insert_with(|| Json::Array(Vec::new()));
                        if let Some(items) = entry.as_array_mut() {
                            items.push(Json::String(chunk_id));
                        }
                        break;
                    }
                }
            } else {
                kept.push((chunk_id, embedding));
                out.push(row);
            }
        }
        *rows = out;
        Ok(())
    }

    fn rrf_within(
        &self,
        table: &str,
        fts_index: &str,
        query_vector: &[f32],
        query: &str,
        within: &[String],
    ) -> Result<Vec<(String, f64)>> {
        // Score the small neighbourhood exactly instead of post-filtering the
        // global index window.
        let ids_param = (
            "ids",
            lbug::Value::List(
                lbug::LogicalType::String,
                within.iter().cloned().map(lbug::Value::String).collect(),
            ),
        );
        let sql = if table == "Chunk" {
            "MATCH (n:`Chunk`) WHERE n.parent_doc_id IN $ids AND n.embedding IS NOT NULL \
             RETURN n.id AS id, n.embedding AS embedding"
        } else {
            "MATCH (n:`Concept`) WHERE n.id IN $ids AND n.embedding IS NOT NULL \
             RETURN n.id AS id, n.embedding AS embedding"
        };
        let mut vec_scores = Vec::new();
        for row in self.store.query(sql, vec![ids_param.clone()])? {
            let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
                continue;
            };
            let stored = json_floats(row.get("embedding"));
            let score = search::cosine(query_vector, &stored);
            vec_scores.push((id, score));
        }
        let mut members: BTreeSet<String> = vec_scores.iter().map(|(id, _)| id.clone()).collect();
        if table == "Chunk" {
            let sql = "MATCH (n:`Chunk`) WHERE n.parent_doc_id IN $ids RETURN n.id AS id";
            for row in self.store.query(sql, vec![ids_param])? {
                if let Some(id) = row.get("id").and_then(Json::as_str) {
                    members.insert(id.to_string());
                }
            }
        } else {
            members.extend(within.iter().cloned());
        }
        let fts = self
            .store
            .fts_hits(table, fts_index, query, search::FETCH_CAP)?;
        let fts: Vec<(String, f64)> = fts
            .into_iter()
            .filter(|(id, _)| members.contains(id))
            .collect();
        Ok(search::rrf_fuse(&vec_scores, &fts))
    }

    #[cfg(feature = "embed")]
    fn cross_rerank(&self, rows: &mut [Json], query: &str) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let top_n = self.settings.rerank_top_n.max(1) as usize;
        let head_len = rows.len().min(top_n);
        let (head, _) = rows.split_at_mut(head_len);
        let chunk_ids: Vec<String> = head
            .iter()
            .filter_map(|row| {
                row.get("chunk_id")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            })
            .collect();
        let concept_ids: Vec<String> = head
            .iter()
            .filter(|row| row.get("chunk_id").is_none())
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();
        let chunks = self.store.chunks_by_ids(&chunk_ids)?;
        let chunk_text: BTreeMap<String, String> = chunks
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                let text = row.get("chunk_text")?.as_str().unwrap_or("").to_string();
                Some((id, text))
            })
            .collect();
        let concepts = self.store.concepts_by_ids(&concept_ids)?;
        let concept_text: BTreeMap<String, String> = concepts
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                let text = row
                    .get("body")
                    .and_then(Json::as_str)
                    .filter(|s| !s.is_empty())
                    .or_else(|| row.get("description").and_then(Json::as_str))
                    .unwrap_or("")
                    .to_string();
                Some((id, text))
            })
            .collect();
        let mut docs = Vec::with_capacity(head.len());
        for row in &*head {
            let (title, text) = if let Some(chunk_id) = row.get("chunk_id").and_then(Json::as_str) {
                (
                    row.get("parent_title").and_then(Json::as_str),
                    chunk_text
                        .get(chunk_id)
                        .map(String::as_str)
                        .unwrap_or(row.get("chunk_text").and_then(Json::as_str).unwrap_or("")),
                )
            } else {
                let id = row.get("id").and_then(Json::as_str).unwrap_or("");
                (
                    row.get("title").and_then(Json::as_str),
                    concept_text
                        .get(id)
                        .map(String::as_str)
                        .unwrap_or(row.get("description").and_then(Json::as_str).unwrap_or("")),
                )
            };
            docs.push(if let Some(title) = title {
                format!("{title}: {text}")
            } else {
                text.to_string()
            });
        }
        let scores = self.reranker.score(&self.settings, query, &docs)?;
        for (row, score) in head.iter_mut().zip(scores) {
            row.as_object_mut()
                .expect("row object")
                .insert("cross_score".into(), json_number(score));
        }
        head.sort_by(|a, b| {
            let sa = a.get("cross_score").and_then(Json::as_f64).unwrap_or(0.0);
            let sb = b.get("cross_score").and_then(Json::as_f64).unwrap_or(0.0);
            sb.total_cmp(&sa)
                .then_with(|| {
                    a.get("id")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .cmp(b.get("id").and_then(Json::as_str).unwrap_or(""))
                })
                .then_with(|| {
                    a.get("chunk_id")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .cmp(b.get("chunk_id").and_then(Json::as_str).unwrap_or(""))
                })
        });
        Ok(())
    }

    #[cfg(not(feature = "embed"))]
    fn cross_rerank(&self, _rows: &mut [Json], _query: &str) -> Result<()> {
        Err(
            OkfError::new("BAD_VALUE", "rerank requires a build with --features embed")
                .op("search"),
        )
    }

    fn recency_blend(&self, rows: &mut [Json], params: &SearchParams) -> Result<()> {
        let Some(half_life) = params.recency else {
            return Ok(());
        };
        if rows.is_empty() {
            return Ok(());
        }
        let ids: Vec<String> = rows
            .iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let effective = self.store.concepts_by_ids(&ids)?;
        let by_id: BTreeMap<String, String> = effective
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                let value = row.get("effective_at")?.as_str()?.to_string();
                Some((id, value))
            })
            .collect();
        let now = crate::model::utc_now_string();
        let top = rows
            .iter()
            .filter_map(|row| row.get("relevance_score").and_then(Json::as_f64))
            .fold(0.0_f64, f64::max);
        for row in rows.iter_mut() {
            let id = row.get("id").and_then(Json::as_str).unwrap_or("");
            let decay =
                crate::model::recency_decay(by_id.get(id).map(String::as_str), &now, half_life);
            let base = row
                .get("relevance_score")
                .and_then(Json::as_f64)
                .unwrap_or(0.0);
            let rel = if top > 0.0 { base / top } else { 0.0 };
            let score = (1.0 - params.recency_weight) * rel + params.recency_weight * decay;
            let obj = row.as_object_mut().expect("row object");
            obj.insert("recency_score".into(), json_number(decay));
            obj.insert("relevance_score".into(), json_number(score));
        }
        rows.sort_by(|a, b| {
            let sa = a
                .get("relevance_score")
                .and_then(Json::as_f64)
                .unwrap_or(0.0);
            let sb = b
                .get("relevance_score")
                .and_then(Json::as_f64)
                .unwrap_or(0.0);
            sb.total_cmp(&sa).then_with(|| {
                a.get("id")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .cmp(b.get("id").and_then(Json::as_str).unwrap_or(""))
                    .then_with(|| {
                        a.get("chunk_id")
                            .and_then(Json::as_str)
                            .unwrap_or("")
                            .cmp(b.get("chunk_id").and_then(Json::as_str).unwrap_or(""))
                    })
            })
        });
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct SearchParams {
    query: String,
    target: String,
    limit: usize,
    concept_type: Option<String>,
    tags: Vec<String>,
    parent_id: Option<String>,
    include_chunks: bool,
    max_chunks_per_doc: usize,
    #[allow(dead_code)]
    expand: bool,
    #[allow(dead_code)]
    context_hops: i64,
    hub_rerank: bool,
    hub_weight: f64,
    rank: String,
    where_map: BTreeMap<String, String>,
    since: Option<String>,
    until: Option<String>,
    exclude_tags: Vec<String>,
    exclude_tags_set: bool,
    link_kinds: Option<Vec<String>>,
    near_id: Option<String>,
    hops: i64,
    offset: usize,
    recency: Option<f64>,
    recency_weight: f64,
    efs: Option<i64>,
    dedupe: bool,
    rerank: bool,
    query_vector: Option<Vec<f32>>,
}

impl SearchParams {
    fn parse(params: &Map<String, Json>) -> Result<Self> {
        let query = str_param(params, "query")?.to_string();
        let target = opt_str(params, "target")?.unwrap_or_else(|| "concepts".into());
        if !["concepts", "chunks", "images"].contains(&target.as_str()) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("target must be one of 'concepts', 'chunks', 'images', got '{target}'"),
            )
            .op("search")
            .field("target", Json::String(target)));
        }
        let rank = opt_str(params, "rank")?.unwrap_or_else(|| "none".into());
        if !["none", "hub", "ppr", "cross"].contains(&rank.as_str()) {
            return Err(bad_value(
                format!("rank must be one of 'none', 'hub', 'ppr', 'cross', got '{rank}'"),
                vec![("rank", Json::String(rank))],
            ));
        }
        let limit = required_int(params, "limit", 10, |v| v < 1, "limit must be >= 1, got")?;
        let offset = required_int(params, "offset", 0, |v| v < 0, "offset must be >= 0, got")?;
        let hops = required_int(
            params,
            "hops",
            2,
            |v| !(1..=3).contains(&v),
            "hops must be 1..3, got",
        )?;
        let near_id = opt_str(params, "near_id")?;
        if hops != 2 && near_id.is_none() {
            return Err(bad_value(
                "hops needs near_id",
                vec![("hops", Json::from(hops))],
            ));
        }
        let recency_raw = params.get("recency").cloned().unwrap_or(Json::Null);
        let recency = match &recency_raw {
            Json::Null => None,
            Json::Bool(_) | Json::String(_) | Json::Array(_) | Json::Object(_) => {
                return Err(bad_value(
                    format!(
                        "recency must be a half-life in days > 0, got {}",
                        py_repr(&recency_raw)
                    ),
                    vec![("recency", recency_raw.clone())],
                ));
            }
            value => {
                let number = value.as_f64().unwrap_or(0.0);
                if number <= 0.0 {
                    return Err(bad_value(
                        format!(
                            "recency must be a half-life in days > 0, got {}",
                            py_repr(value)
                        ),
                        vec![("recency", value.clone())],
                    ));
                }
                Some(number)
            }
        };
        let recency_weight_raw = params
            .get("recency_weight")
            .cloned()
            .unwrap_or_else(|| Json::from(0.3));
        let recency_weight = recency_weight_raw.as_f64().unwrap_or(0.3);
        if !(0.0..=1.0).contains(&recency_weight) {
            return Err(bad_value(
                format!(
                    "recency_weight must be 0..1, got {}",
                    py_repr(&recency_weight_raw)
                ),
                vec![("recency_weight", recency_weight_raw.clone())],
            ));
        }
        if recency_weight != 0.3 && recency.is_none() {
            return Err(bad_value(
                "recency_weight needs recency",
                vec![("recency_weight", recency_weight_raw.clone())],
            ));
        }
        let efs_raw = params.get("efs").cloned().unwrap_or(Json::Null);
        let efs = match &efs_raw {
            Json::Null => None,
            value => match value.as_i64() {
                Some(number) if (16..=1000).contains(&number) => Some(number),
                _ => {
                    return Err(bad_value(
                        format!("efs must be an integer 16..1000, got {}", py_repr(value)),
                        vec![("efs", value.clone())],
                    ));
                }
            },
        };
        if efs.is_some() && near_id.is_some() {
            return Err(bad_value(
                "efs has no effect with near_id: the neighbourhood is scored exactly, without the vector index",
                vec![
                    ("efs", efs_raw.clone()),
                    ("near_id", Json::String(near_id.clone().unwrap_or_default())),
                ],
            ));
        }
        let dedupe = required_bool(params, "dedupe")?;
        let rerank = required_bool(params, "rerank")?;
        if rerank && target == "concepts" {
            return Err(bad_value(
                "rerank is for target='chunks'; concepts use rank='cross'",
                vec![
                    ("rerank", Json::Bool(true)),
                    ("target", Json::String(target)),
                ],
            ));
        }
        if (rerank || rank == "cross") && recency.is_some() {
            return Err(bad_value(
                "recency does not combine with the cross-encoder rerank (rerank / rank='cross'); use one or the other",
                vec![
                    ("recency", recency_raw.clone()),
                    ("rerank", Json::Bool(rerank)),
                    ("rank", Json::String(rank)),
                ],
            ));
        }
        if let Some(value) = params.get("where") {
            if !matches!(value, Json::Null | Json::Object(_)) {
                return Err(bad_value(
                    "where must be a mapping of extra key -> value",
                    vec![("where", value.clone())],
                ));
            }
        }
        let hub_rerank = required_bool(params, "hub_rerank")?;
        let expand = required_bool(params, "expand")?;
        if target == "chunks" && rank != "none" {
            return Err(bad_value(
                "rank is concepts-only; target='chunks' uses hub_rerank/expand/rerank",
                vec![
                    ("rank", Json::String(rank)),
                    ("target", Json::String(target)),
                ],
            ));
        }
        let path = if target == "chunks" {
            if hub_rerank {
                "chunks:hub".to_string()
            } else if expand {
                "chunks:expand".to_string()
            } else {
                "chunks:plain".to_string()
            }
        } else if target == "images" {
            "images".to_string()
        } else {
            format!("concepts:{rank}")
        };
        let mut ignored = Vec::new();
        for name in PASSED_ORDER {
            if !passed(params, name) {
                continue;
            }
            let honoured = match path.as_str() {
                "chunks:hub" => HUB_PARAMS.contains(name),
                "chunks:expand" => EXPAND_PARAMS.contains(name),
                "chunks:plain" => {
                    FILTER_PARAMS.contains(name)
                        || PAGED_PARAMS.contains(name)
                        || ["max_chunks_per_doc", "rerank", "efs", "dedupe"].contains(name)
                }
                "images" => false,
                "concepts:ppr" => {
                    FILTER_PARAMS.contains(name)
                        || PAGED_PARAMS.contains(name)
                        || ["rank", "link_kinds"].contains(name)
                }
                "concepts:hub" => {
                    FILTER_PARAMS.contains(name)
                        || PAGED_PARAMS.contains(name)
                        || ["include_chunks", "rank", "hub_weight", "link_kinds", "efs"]
                            .contains(name)
                }
                "concepts:cross" => {
                    FILTER_PARAMS.contains(name)
                        || PAGED_PARAMS.contains(name)
                        || ["include_chunks", "rank", "efs"].contains(name)
                }
                _ => {
                    FILTER_PARAMS.contains(name)
                        || PAGED_PARAMS.contains(name)
                        || ["include_chunks", "rank", "efs"].contains(name)
                }
            };
            if !honoured {
                ignored.push((*name).to_string());
            }
        }
        if !ignored.is_empty() {
            let suffix = path.split_once(':').map_or("default", |(_, rest)| rest);
            let message = format!(
                "target='{target}' ({suffix} path) ignores: {}",
                ignored.join(", ")
            );
            let mut err = bad_value(
                message,
                vec![
                    ("target", Json::String(target)),
                    ("path", Json::String(path.clone())),
                    (
                        "ignored",
                        Json::Array(ignored.into_iter().map(Json::String).collect()),
                    ),
                ],
            );
            err.remedy = Some("drop the ignored params or use a path that honours them".into());
            return Err(err);
        }
        let hub_weight = opt_f64(params, "hub_weight")?.unwrap_or(0.3);
        let mut where_map = BTreeMap::new();
        if let Some(Json::Object(obj)) = params.get("where") {
            for (key, value) in obj {
                let encoded = match value {
                    Json::String(s) => s.clone(),
                    other => from_json(other).to_python_json(false),
                };
                where_map.insert(key.clone(), encoded);
            }
        }
        let query_vector = params
            .get("query_vector")
            .and_then(Json::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_f64().map(|n| n as f32))
                    .collect::<Vec<_>>()
            });
        Ok(Self {
            query,
            target,
            limit: limit as usize,
            concept_type: opt_str(params, "concept_type")?,
            tags: opt_strings(params, "tags")?,
            parent_id: opt_str(params, "parent_id")?,
            include_chunks: required_bool(params, "include_chunks")?,
            max_chunks_per_doc: opt_i64(params, "max_chunks_per_doc")?.unwrap_or(3).max(1) as usize,
            expand,
            context_hops: opt_i64(params, "context_hops")?.unwrap_or(1),
            hub_rerank,
            hub_weight,
            rank,
            where_map,
            since: opt_str(params, "since")?,
            until: opt_str(params, "until")?,
            exclude_tags: opt_strings(params, "exclude_tags")?,
            exclude_tags_set: params.contains_key("exclude_tags"),
            link_kinds: opt_strings(params, "link_kinds")?.into_version(),
            near_id,
            hops,
            offset: offset as usize,
            recency,
            recency_weight,
            efs,
            dedupe,
            rerank,
            query_vector,
        })
    }

    fn filtered(&self) -> bool {
        self.concept_type.is_some()
            || !self.tags.is_empty()
            || self.parent_id.is_some()
            || !self.where_map.is_empty()
            || self.since.is_some()
            || self.until.is_some()
            || !self.exclude_tags.is_empty()
            || self.near_id.is_some()
    }
}

trait IntoOptionalVec {
    fn into_version(self) -> Option<Vec<String>>;
}

impl IntoOptionalVec for Vec<String> {
    fn into_version(self) -> Option<Vec<String>> {
        if self.is_empty() {
            None
        } else {
            Some(self)
        }
    }
}

fn json_number(value: f64) -> Json {
    serde_json::Number::from_f64(value)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

fn json_strings(value: Option<&Json>) -> Vec<String> {
    match value {
        Some(Json::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Json::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn json_floats(value: Option<&Json>) -> Vec<f32> {
    match value {
        Some(Json::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_f64().map(|n| n as f32))
            .collect(),
        _ => Vec::new(),
    }
}

fn truncate(mut items: Vec<Json>, n: usize) -> Vec<Json> {
    items.truncate(n);
    items
}

const PASSED_ORDER: &[&str] = &[
    "concept_type",
    "tags",
    "parent_id",
    "include_chunks",
    "max_chunks_per_doc",
    "expand",
    "context_hops",
    "hub_rerank",
    "hub_weight",
    "rank",
    "where",
    "since",
    "until",
    "exclude_tags",
    "link_kinds",
    "near_id",
    "hops",
    "offset",
    "recency",
    "recency_weight",
    "rerank",
    "efs",
    "dedupe",
];
const FILTER_PARAMS: &[&str] = &[
    "concept_type",
    "tags",
    "parent_id",
    "where",
    "since",
    "until",
    "exclude_tags",
    "near_id",
];
const PAGED_PARAMS: &[&str] = &["hops", "offset", "recency", "recency_weight"];
const HUB_PARAMS: &[&str] = &[
    "hub_rerank",
    "hub_weight",
    "expand",
    "context_hops",
    "rerank",
    "efs",
    "dedupe",
];
const EXPAND_PARAMS: &[&str] = &["expand", "context_hops"];

fn bad_value(message: impl Into<String>, fields: Vec<(&str, Json)>) -> OkfError {
    bad_value2("search", message, fields)
}

fn bad_value2(op: &str, message: impl Into<String>, fields: Vec<(&str, Json)>) -> OkfError {
    let mut err = OkfError::new("BAD_VALUE", message).op(op);
    for (key, value) in fields {
        err = err.field(key, value);
    }
    err
}

fn section(id: &str, title: &str, kind: &str, text: &str) -> Json {
    section_with_title(id, Json::String(title.to_string()), kind, text)
}

fn section_with_title(id: &str, title: Json, kind: &str, text: &str) -> Json {
    serde_json::json!({
        "id": id,
        "title": title,
        "kind": kind,
        "text": text,
    })
}

fn model_value_repr(value: &Json, datetime: bool) -> String {
    if datetime {
        if let Some(text) = value.as_str() {
            if let Some(repr) = py_datetime_repr(text) {
                return repr;
            }
        }
    }
    match value {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::String(text) => py_string_repr(text),
        Json::Number(number) if number.is_f64() => {
            crate::pycompat::python_float_repr(number.as_f64().unwrap_or(0.0))
        }
        Json::Number(number) => number.to_string(),
        Json::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| model_value_repr(item, false))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Json::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(key, item)| {
                    format!("{}: {}", py_string_repr(key), model_value_repr(item, false))
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn py_string_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_datetime_repr(text: &str) -> Option<String> {
    if text.len() < 19 {
        return None;
    }
    let number = |start: usize, end: usize| text.get(start..end)?.parse::<u32>().ok();
    let year = number(0, 4)?;
    let month = number(5, 7)?;
    let day = number(8, 10)?;
    let hour = number(11, 13)?;
    let minute = number(14, 16)?;
    let second = number(17, 19)?;
    let mut args = vec![year, month, day, hour, minute];
    let rest = &text[19..];
    let micros = if let Some(fraction) = rest.strip_prefix('.') {
        let digits: String = fraction
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .take(6)
            .collect();
        if digits.is_empty() {
            0
        } else {
            format!("{digits:0<6}").parse::<u32>().ok()?
        }
    } else {
        0
    };
    if second != 0 || micros != 0 {
        args.push(second);
    }
    if micros != 0 {
        args.push(micros);
    }
    Some(format!(
        "datetime.datetime({})",
        args.into_iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn truncate_to_tokens(text: &str, budget: usize, count: &impl Fn(&str) -> usize) -> String {
    if budget == 0 {
        return String::new();
    }
    let total = count(text);
    if total <= budget {
        return text.to_string();
    }
    let chars = text.chars().count();
    let take = ((chars as f64 * budget as f64 / total as f64) as usize).max(1);
    let head: String = text.chars().take(take).collect();
    format!("{}\n[…truncated to budget]", head.trim_end())
}

fn py_repr(value: &Json) -> String {
    match value {
        Json::String(s) => format!("'{s}'"),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Null => "None".into(),
        Json::Number(number) => number.to_string(),
        other => other.to_string(),
    }
}

fn required_int(
    params: &Map<String, Json>,
    name: &str,
    default: i64,
    invalid: impl Fn(i64) -> bool,
    message: &str,
) -> Result<i64> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(default),
        Some(value) => match value.as_i64() {
            Some(number) if !invalid(number) => Ok(number),
            _ => Err(bad_value(
                format!("{message} {}", py_repr(value)),
                vec![(name, value.clone())],
            )),
        },
    }
}

fn required_bool(params: &Map<String, Json>, name: &str) -> Result<bool> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(false),
        Some(Json::Bool(value)) => Ok(*value),
        Some(value) => Err(bad_value(
            format!("{name} must be true or false, got {}", py_repr(value)),
            vec![(name, value.clone())],
        )),
    }
}

fn passed(params: &Map<String, Json>, name: &str) -> bool {
    let default = match name {
        "include_chunks" | "expand" | "hub_rerank" | "rerank" | "dedupe" => Json::Bool(false),
        "max_chunks_per_doc" => Json::from(3),
        "context_hops" => Json::from(1),
        "hops" => Json::from(2),
        "offset" => Json::from(0),
        "hub_weight" | "recency_weight" => Json::from(0.3),
        "rank" => Json::String("none".into()),
        _ => Json::Null,
    };
    params.get(name).is_some_and(|value| *value != default)
}

fn unknown_concept(op: &str, field: &str, id: &str) -> OkfError {
    OkfError::new("UNKNOWN_CONCEPT", format!("{field} '{id}' does not exist"))
        .op(op)
        .field(field, Json::String(id.into()))
        .remedy("search first to find IDs")
}

fn require_nonempty(op: &str, field: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(
            OkfError::new("MISSING_PARAM", format!("{field} is required"))
                .op(op)
                .field(field, Json::String(String::new())),
        );
    }
    Ok(())
}

fn str_param<'a>(params: &'a Map<String, Json>, name: &str) -> Result<&'a str> {
    params.get(name).and_then(Json::as_str).ok_or_else(|| {
        OkfError::new("MISSING_PARAM", format!("{name} is required")).field(name, Json::Null)
    })
}

fn opt_str(params: &Map<String, Json>, name: &str) -> Result<Option<String>> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(OkfError::new(
            "BAD_VALUE",
            format!("{name} must be a string, got {other}"),
        )),
    }
}

fn opt_f64(params: &Map<String, Json>, name: &str) -> Result<Option<f64>> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(None),
        Some(value) => value.as_f64().map(Some).ok_or_else(|| {
            OkfError::new("BAD_VALUE", format!("{name} must be a number, got {value}"))
        }),
    }
}

fn opt_strings(params: &Map<String, Json>, name: &str) -> Result<Vec<String>> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    OkfError::new("BAD_VALUE", format!("{name} must be a list of strings"))
                })
            })
            .collect(),
        Some(Json::String(s)) => Ok(vec![s.clone()]),
        Some(other) => Err(OkfError::new(
            "BAD_VALUE",
            format!("{name} must be a list of strings, got {other}"),
        )),
    }
}

fn opt_i64(params: &Map<String, Json>, name: &str) -> Result<Option<i64>> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(None),
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            OkfError::new(
                "BAD_VALUE",
                format!("{name} must be an integer, got {value}"),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_vocabulary() {
        assert_eq!(OP_NAMES.len(), 30);
        assert!(is_op("get_by_id"));
        assert!(!is_op("init"));
    }

    #[test]
    fn opens_fixture_and_reads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("parity/fixtures/oracle-v10.db");
        let settings = Settings::default()
            .with("db_path", Json::String(path.to_string_lossy().into_owned()))
            .expect("settings");
        let router = Router::open(&settings).expect("router");
        let id = Json::String("alpha".into());
        let concept = router
            .call("get_by_id", &Map::from_iter([("concept_id".into(), id)]))
            .expect("get_by_id");
        assert_eq!(concept["id"], "alpha");
        assert_eq!(concept["type"], "note");
    }
}
