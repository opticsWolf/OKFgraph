//! Typed operation requests/results plus the `call(Op, Value)` escape hatch.
//!
//! M1 implements the read-only slice; write/ingest ops land in Stage B.

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

pub fn is_op(name: &str) -> bool {
    OP_NAMES.contains(&name)
}

#[derive(Debug, Clone)]
pub struct Router {
    store: Store,
    settings: Settings,
    embedder: EmbedderPool,
}

impl Router {
    pub fn open(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open(settings)?,
            settings: settings.clone(),
            embedder: EmbedderPool::new(),
        })
    }

    /// Create a new v10 database. This is the only schema write Stage A allows.
    pub fn init(path: impl AsRef<Path>, embedding_dim: i64) -> Result<Vec<String>> {
        Store::init(path, embedding_dim)
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
        if !READ_OPS.contains(&op) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("op '{op}' is not implemented in Stage A"),
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
        self.store.pin_info()
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
        let concept = self.store.get_by_id(concept_id)?;
        if concept.is_none() {
            return Err(unknown_concept("read", "concept_id", concept_id));
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
    pub fn search(&self, params: &Map<String, Json>) -> Result<Json> {
        let params = SearchParams::parse(params)?;
        let query_vector = self.query_vector(&params.query, params.query_vector.clone())?;
        match params.target.as_str() {
            "concepts" => self.search_concepts(&params, &query_vector),
            "chunks" => Ok(Json::Array(self.search_chunks(
                &params,
                &query_vector,
                params.limit,
            )?)),
            "images" => Err(OkfError::new(
                "BAD_VALUE",
                "image search needs the vision lane; Stage C exposes it",
            )
            .op("search")),
            _ => unreachable!("validated target"),
        }
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
        if params.rank == "cross" {
            return Err(OkfError::new(
                "BAD_VALUE",
                "rank='cross' needs the rerank lane; Stage C exposes it",
            )
            .op("search"));
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
        let k = search::fetch_k(limit.max(search::PAGE_WINDOW), params.filtered());
        let vec =
            self.store
                .vector_hits("Chunk", "chunk_embedding", query_vector, k, params.efs)?;
        let fts = self
            .store
            .fts_hits("Chunk", "chunk_fts", &params.query, k)?;
        let combined = search::rrf_fuse(&vec, &fts);
        let rows = self.store.chunks_by_ids(
            &combined
                .iter()
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>(),
        )?;
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
        let parent_rows = self.store.concepts_by_ids(&parents)?;
        let parent_by_id: BTreeMap<String, Json> = parent_rows
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                Some((id, row))
            })
            .collect();
        let score_by_id: BTreeMap<&str, f64> = combined
            .iter()
            .map(|(id, score)| (id.as_str(), *score))
            .collect();
        let mut out = Vec::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut kept: Vec<(Vec<f32>, Vec<String>)> = Vec::new();
        for row in rows {
            let Some(chunk_id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
                continue;
            };
            let parent_id = row
                .get("parent_doc_id")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(wanted) = &params.parent_id {
                if parent_id != *wanted {
                    continue;
                }
            }
            let parent = parent_by_id.get(&parent_id);
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
            let count = counts.entry(parent_id.clone()).or_default();
            *count += 1;
            if *count > params.max_chunks_per_doc {
                continue;
            }
            let Some(score) = score_by_id.get(chunk_id.as_str()) else {
                continue;
            };
            let embedding = json_floats(row.get("embedding"));
            if params.dedupe {
                let mut duplicate = false;
                for (kept_vec, _) in &kept {
                    if search::cosine(&embedding, kept_vec) >= DEDUPE_COSINE {
                        duplicate = true;
                        break;
                    }
                }
                if duplicate {
                    continue;
                }
                kept.push((embedding, Vec::new()));
            }
            let text = row.get("chunk_text").and_then(Json::as_str).unwrap_or("");
            let text = if text.chars().count() > 500 {
                format!("{}...", text.chars().take(500).collect::<String>())
            } else {
                text.to_string()
            };
            let mut result = serde_json::json!({
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
            });
            if params.hub_rerank {
                let (_, edges) = self.store.link_graph(params.link_kinds.as_deref())?;
                let hubs = search::hub_scores(std::slice::from_ref(&parent_id), &edges);
                let hub = hubs.get(&parent_id).copied().unwrap_or(0) as f64;
                let score = (1.0 - params.hub_weight) * score + params.hub_weight * hub;
                result
                    .as_object_mut()
                    .expect("row object")
                    .insert("rrf_score".into(), json_number(score));
                result
                    .as_object_mut()
                    .expect("row object")
                    .insert("hub_score".into(), json_number(hub));
            }
            out.push(result);
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
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
        let sql = format!(
            "MATCH (n:`{table}`) WHERE n.id IN $ids RETURN n.id AS id, n.embedding AS embedding"
        );
        let rows = self.store.query(
            &sql,
            vec![(
                "ids",
                lbug::Value::List(
                    lbug::LogicalType::String,
                    within.iter().cloned().map(lbug::Value::String).collect(),
                ),
            )],
        )?;
        let mut vec_scores = Vec::new();
        for row in rows {
            let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
                continue;
            };
            let stored = json_floats(row.get("embedding"));
            let score = search::cosine(query_vector, &stored);
            vec_scores.push((id, score));
        }
        let fts = self
            .store
            .fts_hits(table, fts_index, query, search::FETCH_CAP)?;
        let fts: Vec<(String, f64)> = fts
            .into_iter()
            .filter(|(id, _)| within.contains(id))
            .collect();
        Ok(search::rrf_fuse(&vec_scores, &fts))
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
    hub_rerank: bool,
    hub_weight: f64,
    rank: String,
    where_map: BTreeMap<String, String>,
    since: Option<String>,
    until: Option<String>,
    exclude_tags: Vec<String>,
    link_kinds: Option<Vec<String>>,
    near_id: Option<String>,
    hops: i64,
    offset: usize,
    recency: Option<f64>,
    recency_weight: f64,
    efs: Option<i64>,
    dedupe: bool,
    query_vector: Option<Vec<f32>>,
}

impl SearchParams {
    fn parse(params: &Map<String, Json>) -> Result<Self> {
        let query = str_param(params, "query")?.to_string();
        let target = opt_str(params, "target")?.unwrap_or_else(|| "concepts".into());
        if !["concepts", "chunks", "images"].contains(&target.as_str()) {
            return Err(
                OkfError::new("BAD_VALUE", format!("unknown search target: {target}")).op("search"),
            );
        }
        let limit = opt_i64(params, "limit")?.unwrap_or(10);
        if limit < 1 {
            return Err(OkfError::new("BAD_VALUE", "limit must be >= 1").op("search"));
        }
        let rank = opt_str(params, "rank")?.unwrap_or_else(|| "none".into());
        if !["none", "hub", "ppr", "cross"].contains(&rank.as_str()) {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("rank must be 'none', 'hub', 'ppr' or 'cross', got '{rank}'"),
            )
            .op("search"));
        }
        let hops = opt_i64(params, "hops")?.unwrap_or(2);
        if !(1..=3).contains(&hops) {
            return Err(OkfError::new("BAD_VALUE", "hops must be 1..3").op("search"));
        }
        let offset = opt_i64(params, "offset")?.unwrap_or(0);
        if offset < 0 {
            return Err(OkfError::new("BAD_VALUE", "offset must be >= 0").op("search"));
        }
        let recency = opt_f64(params, "recency")?;
        if recency.is_some_and(|value| value <= 0.0) {
            return Err(OkfError::new("BAD_VALUE", "recency must be > 0").op("search"));
        }
        let recency_weight = opt_f64(params, "recency_weight")?.unwrap_or(0.3);
        if !(0.0..=1.0).contains(&recency_weight) {
            return Err(OkfError::new("BAD_VALUE", "recency_weight must be 0..1").op("search"));
        }
        let hub_weight = opt_f64(params, "hub_weight")?.unwrap_or(0.3);
        let efs = opt_i64(params, "efs")?;
        if let Some(value) = efs {
            if !(16..=1000).contains(&value) {
                return Err(OkfError::new("BAD_VALUE", "efs must be 16..1000").op("search"));
            }
        }
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
            include_chunks: opt_bool(params, "include_chunks")?.unwrap_or(false),
            max_chunks_per_doc: opt_i64(params, "max_chunks_per_doc")?.unwrap_or(3).max(1) as usize,
            hub_rerank: opt_bool(params, "hub_rerank")?.unwrap_or(false),
            hub_weight,
            rank,
            where_map,
            since: opt_str(params, "since")?,
            until: opt_str(params, "until")?,
            exclude_tags: opt_strings(params, "exclude_tags")?,
            link_kinds: opt_strings(params, "link_kinds")?.into_version(),
            near_id: opt_str(params, "near_id")?,
            hops,
            offset: offset as usize,
            recency,
            recency_weight,
            efs,
            dedupe: opt_bool(params, "dedupe")?.unwrap_or(false),
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

fn opt_bool(params: &Map<String, Json>, name: &str) -> Result<Option<bool>> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(other) => Err(OkfError::new(
            "BAD_VALUE",
            format!("{name} must be a boolean, got {other}"),
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
