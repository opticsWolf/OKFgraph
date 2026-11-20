//! Typed operation requests/results plus the `call(Op, Value)` escape hatch.
//!
//! M1 implements the read-only slice; write/ingest ops land in Stage B.

#[cfg(feature = "embed")]
use crate::embed::RerankerPool;
use crate::embed::{check_engine_pins, EmbedderPool, Pools, Task};
use crate::error::{OkfError, Result};
use crate::markdown;
use crate::pycompat::from_json;
use crate::search::{self, DEDUPE_COSINE};
use crate::settings::Settings;
use crate::store::Store;
use serde_json::{Map, Value as Json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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
    "diff",
    "lint",
];

/// Stage B content-write ops implemented by the ingest pipeline.
pub const WRITE_OPS: &[&str] = &[
    "import_bundle",
    "import_file",
    "import_files",
    "repair_links",
    "export_bundle",
    "export_concept",
    "delete",
    "recover_deleted",
    "purge_deleted",
    "reindex",
    "reembed",
    "update",
    "supersede",
    "detach",
    "produce",
    "sync",
    "ingest",
];

pub fn is_op(name: &str) -> bool {
    // `lint` and `produce` are CLI/parity verbs implemented by this crate but
    // not part of the public 0.15 `OPS` vocabulary (which has 30 entries).
    OP_NAMES.contains(&name) || READ_OPS.contains(&name) || WRITE_OPS.contains(&name)
}

#[derive(Debug, Clone)]
pub struct Router {
    store: Store,
    settings: Settings,
    embedder: EmbedderPool,
    pools: Pools,
    #[cfg(feature = "embed")]
    reranker: RerankerPool,
    write_lock: std::sync::Arc<std::sync::Mutex<()>>,
}

impl Router {
    /// Open a v10 database for the normal public surface, including writes.
    /// Stage B callers must own the graph; never mix Python and Rust cores in
    /// one process.
    pub fn open(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open_writable(settings)?,
            settings: settings.clone(),
            embedder: EmbedderPool::new(),
            pools: Pools::new(),
            #[cfg(feature = "embed")]
            reranker: RerankerPool::new(),
            write_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
        })
    }

    /// Explicit read-only handle for tools that must never mutate the graph.
    pub fn open_read_only(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open(settings)?,
            settings: settings.clone(),
            embedder: EmbedderPool::new(),
            pools: Pools::new(),
            #[cfg(feature = "embed")]
            reranker: RerankerPool::new(),
            write_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
        })
    }

    /// Open an existing v10 database with write permission. Stage B callers
    /// must own the graph; never mix Python and Rust cores in one process.
    pub fn open_writable(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open_writable(settings)?,
            settings: settings.clone(),
            embedder: EmbedderPool::new(),
            pools: Pools::new(),
            #[cfg(feature = "embed")]
            reranker: RerankerPool::new(),
            write_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
        })
    }

    /// Create a new v10 database.
    pub fn init(path: impl AsRef<Path>, embedding_dim: i64) -> Result<Vec<String>> {
        Store::init(path, embedding_dim)
    }

    fn import_settings(&self, params: &Map<String, Json>) -> Result<Settings> {
        let mut settings = self.settings.clone();
        if let Some(mode) = opt_str(params, "mode")? {
            if !matches!(mode.as_str(), "text" | "optional" | "omni") {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    format!("mode must be text, optional, or omni, got '{mode}'"),
                ));
            }
            settings.mode = mode;
        }
        Ok(settings)
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
        let settings = self.import_settings(params)?;
        crate::ingest::import_files(&self.store, &settings, &self.embedder, &self.pools, &paths)
            .map_err(|err| err.op("import_files"))
    }

    fn delete(&self, params: &Map<String, Json>) -> Result<Json> {
        let concept_id = str_param(params, "concept_id")?.to_string();
        let remove_file = opt_bool(params, "remove_file")?.unwrap_or(false);
        if self.store.get_by_id(&concept_id)?.is_none() {
            return Err(unknown_concept("delete", "concept_id", &concept_id));
        }

        let keys = self.store.file_hash_keys_for_concept(&concept_id)?;
        let alive: Vec<PathBuf> = keys
            .iter()
            .flat_map(|key| self.store.resolve_file_paths(key))
            .filter(|path| path.is_file())
            .collect();
        if !alive.is_empty() && !remove_file {
            let sources: Vec<Json> = alive
                .iter()
                .map(|path| Json::String(path.to_string_lossy().into_owned()))
                .collect();
            return Err(OkfError::new(
                "BAD_VALUE",
                format!(
                    "'{concept_id}' is backed by existing source file(s): {}",
                    alive
                        .iter()
                        .map(|path| path.to_string_lossy())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
            .op("delete")
            .field("concept_id", Json::String(concept_id.clone()))
            .field("sources", Json::Array(sources))
            .remedy(
                "delete or move the file(s) from the bundle, then run \
                 'okf import --all --prune-missing' to drop the concept",
            ));
        }

        let trashed = if alive.is_empty() {
            Vec::new()
        } else {
            self.trash_files(&keys)?
        };
        if !crate::purge::soft_delete(&self.store, &concept_id)? {
            return Err(OkfError::new(
                "NOT_RECOVERABLE",
                format!("concept '{concept_id}' vanished mid-operation"),
            )
            .op("delete"));
        }
        crate::purge::remember_trashed(&self.store, &concept_id, &trashed)?;
        let until = crate::model::local_now_datetime().as_naive().assume_utc()
            + time::Duration::seconds(86_400);
        Ok(serde_json::json!({
            "concept_id": concept_id,
            "deleted": true,
            "recoverable_until": crate::model::format_py_datetime(until),
            "trashed_files": trashed
                .iter()
                .map(|(trash, _)| Json::String(trash.clone()))
                .collect::<Vec<_>>(),
        }))
    }

    /// Move backing files to `<root>/.okf-trash/<stamp>/...` and retire their
    /// delta ledger entries. Returns `(trash, source)` pairs.
    fn trash_files(&self, keys: &[String]) -> Result<Vec<(String, String)>> {
        let Some(root) = self.settings.bundle_root.as_ref() else {
            return Ok(Vec::new());
        };
        let root = crate::store::absolute_path(Path::new(root));
        let now = crate::model::local_now_datetime().as_naive().assume_utc();
        let stamp = format!(
            "{:04}{:02}{:02}T{:02}{:02}{:02}",
            now.year(),
            now.month() as u8,
            now.day(),
            now.hour(),
            now.minute(),
            now.second()
        );
        let mut moved: Vec<(String, String)> = Vec::new();
        let mut seen = BTreeSet::new();
        for key in keys {
            for source in self.store.resolve_file_paths(key) {
                if !source.is_file() || !seen.insert(source.clone()) {
                    continue;
                }
                let Ok(relative) = source.strip_prefix(&root) else {
                    continue;
                };
                let trash = root.join(".okf-trash").join(&stamp).join(relative);
                if let Some(parent) = trash.parent() {
                    std::fs::create_dir_all(parent).map_err(|err| {
                        OkfError::internal(
                            format!("cannot create {}: {err}", parent.display()),
                            "std::io::Error",
                        )
                    })?;
                }
                std::fs::rename(&source, &trash).map_err(|err| {
                    OkfError::internal(
                        format!(
                            "cannot move {} to {}: {err}",
                            source.display(),
                            trash.display()
                        ),
                        "std::io::Error",
                    )
                })?;
                self.store.forget_file_hash(key)?;
                moved.push((
                    trash.to_string_lossy().into_owned(),
                    source.to_string_lossy().into_owned(),
                ));
            }
        }
        Ok(moved)
    }

    fn recover_deleted(&self, params: &Map<String, Json>) -> Result<Json> {
        let concept_id = str_param(params, "concept_id")?.to_string();
        let trashed = crate::purge::read_trashed(&self.store, &concept_id)?;
        if !crate::purge::recover(&self.store, &concept_id)? {
            return Err(OkfError::new(
                "NOT_RECOVERABLE",
                format!("concept '{concept_id}' not found or past recovery window"),
            )
            .op("recover_deleted")
            .field("concept_id", Json::String(concept_id)));
        }

        let mut restored = Vec::new();
        for (trash, source) in trashed {
            let trash_path = PathBuf::from(&trash);
            let source_path = PathBuf::from(&source);
            if !trash_path.exists() || source_path.exists() {
                continue;
            }
            if let Some(parent) = source_path.parent() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    OkfError::internal(
                        format!("cannot create {}: {err}", parent.display()),
                        "std::io::Error",
                    )
                })?;
            }
            std::fs::rename(&trash_path, &source_path).map_err(|err| {
                OkfError::internal(
                    format!(
                        "cannot restore {} to {}: {err}",
                        trash_path.display(),
                        source_path.display()
                    ),
                    "std::io::Error",
                )
            })?;
            restored.push(source);
        }

        if restored.is_empty() {
            crate::ingest::extract_links_for_concept(&self.store, &concept_id)?;
            crate::ingest::repair_links_public(&self.store)?;
        } else {
            for path in &restored {
                let mut import_params = Map::new();
                import_params.insert("file_path".into(), Json::String(path.clone()));
                self.import_file(&import_params)?;
            }
        }
        Ok(serde_json::json!({
            "concept_id": concept_id,
            "recovered": true,
            "restored_files": restored,
        }))
    }

    fn purge_deleted(&self, params: &Map<String, Json>) -> Result<Json> {
        let older_than = match params.get("older_than") {
            Some(Json::Number(number)) => number.as_i64(),
            Some(Json::Null) | None => None,
            Some(_) => {
                return Err(
                    OkfError::new("BAD_VALUE", "older_than must be an integer").op("purge_deleted")
                )
            }
        };
        Ok(serde_json::json!({
            "purged": crate::purge::purge_deleted(&self.store, older_than)?,
        }))
    }

    fn diff(&self, params: &Map<String, Json>) -> Result<Json> {
        let old = opt_str(params, "old")?;
        let new = opt_str(params, "new")?;
        let report =
            crate::diagnostics::diff(&self.store, &self.settings, old.as_deref(), new.as_deref())?;
        if report.get("identical").and_then(Json::as_bool) == Some(false) {
            return Err(OkfError::outcome(
                "DIFF_DIFFERENT",
                "structures differ (report in data)",
                report,
            )
            .op("diff"));
        }
        Ok(report)
    }

    fn lint(&self, params: &Map<String, Json>) -> Result<Json> {
        let dir = opt_str(params, "bundle_dir")?.unwrap_or_else(|| ".".into());
        let report = crate::diagnostics::lint(&dir, &self.settings)?;
        let count = report
            .get("errors")
            .and_then(Json::as_array)
            .map(|items| items.len())
            .unwrap_or(0);
        if count > 0 {
            return Err(
                OkfError::outcome("LINT_ERRORS", format!("{count} lint error(s)"), report)
                    .op("lint")
                    .field("errors", Json::from(count)),
            );
        }
        Ok(report)
    }

    fn reembed(&self, params: &Map<String, Json>) -> Result<Json> {
        let dry_run = opt_bool(params, "dry_run")?.unwrap_or(false);
        let engine = self.embedder.get(&self.settings)?;
        let dim = engine.dim();
        if dim as i64 != self.settings.embedding_dim {
            return Err(OkfError::new(
                "DIM_MISMATCH",
                format!(
                    "engine dim {dim} != graph column dim {}; 0.12.0 reembed keeps the column width — open a fresh graph for a different dim",
                    self.settings.embedding_dim
                ),
            )
            .op("reembed")
            .field("engine_dim", Json::from(dim))
            .field("graph_dim", Json::from(self.settings.embedding_dim))
            .remedy(
                "re-import into a new database with the new dim, or reopen with the pinned model",
            ));
        }
        if self.store.count_image_assets()? > 0 {
            return Err(OkfError::new(
                "IMAGE_PIN_MISMATCH",
                "graph carries ImageAsset rows; a text-model reembed would leave the omni space under the old weights",
            )
            .op("reembed")
            .remedy("re-import into a fresh database instead"));
        }
        let concepts = self.store.reembed_texts("concept")?;
        let chunks = self.store.reembed_texts("chunk")?;
        let model_id = engine.model_id().to_string();
        let precision = engine.precision().to_string();
        let mut report = serde_json::json!({
            "dry_run": dry_run,
            "model_id": model_id,
            "model_name": Json::Null,
            "dimension": dim,
            "concepts": concepts.len(),
            "chunks": chunks.len(),
        });
        if dry_run {
            return Ok(report);
        }
        let started = std::time::Instant::now();
        self.store.drop_text_vector_indexes()?;
        let batch_size = (settings_batch_size(&self.settings) as usize).max(1);
        let mut rewritten = Map::new();
        for (table, rows) in [
            ("concept", concepts.as_slice()),
            ("chunk", chunks.as_slice()),
        ] {
            let mut done = 0;
            for part in rows.chunks(batch_size) {
                let texts: Vec<String> = part.iter().map(|(_, text)| text.clone()).collect();
                let vectors = engine.encode_many(&texts, Task::Document)?;
                for ((id, _), vector) in part.iter().zip(vectors) {
                    self.store.set_reembedded_vector(table, id, &vector)?;
                }
                done += part.len();
            }
            let label = if table == "concept" {
                "concepts"
            } else {
                "chunks"
            };
            rewritten.insert(label.to_string(), Json::from(done));
        }
        self.store.create_search_indexes()?;
        self.store.set_meta_text("embedding_model", &model_id)?;
        let code = match precision.as_str() {
            "fp16" => 16,
            "int8" => 8,
            "fp32" => 32,
            _ => 0,
        };
        if code > 0 {
            self.store.set_meta_int("embedding_precision", code)?;
        }
        if let Some(space) = engine.space() {
            self.store.set_meta_text("embedding_space", &space)?;
        }
        let write_epoch = self.store.meta_int("write_epoch")?.unwrap_or(0);
        self.store.set_meta_int("indexed_epoch", write_epoch)?;
        if let Some(object) = report.as_object_mut() {
            object.insert("reembedded".into(), Json::Object(rewritten));
            object.insert(
                "seconds".into(),
                Json::from((started.elapsed().as_secs_f64() * 100.0).round() / 100.0),
            );
            object.insert("precision".into(), Json::String(precision.clone()));
            let mut pins = self.store.pin_info()?;
            if let Some(configured) = pins.get_mut("configured").and_then(Json::as_object_mut) {
                configured.insert("expected_precision".into(), Json::String(precision));
                configured.insert(
                    "space".into(),
                    engine.space().map_or(Json::Null, Json::String),
                );
            }
            object.insert("pins_after".into(), pins);
        }
        Ok(report)
    }

    fn write_and_import(
        &self,
        op: &str,
        concept_id: &str,
        edit: impl FnOnce(&mut Map<String, Json>) -> Result<Vec<String>>,
    ) -> Result<Json> {
        let concept = self
            .store
            .get_by_id(concept_id)?
            .ok_or_else(|| unknown_write_concept(op, concept_id))?;
        let source = self.store.source_file_for_concept(concept_id)?;
        let (mut meta, body) = if let Some(path) = &source {
            let text = std::fs::read_to_string(path).map_err(|err| {
                OkfError::internal(
                    format!("cannot read {}: {err}", path.display()),
                    "std::io::Error",
                )
            })?;
            let parsed = markdown::parse_frontmatter(&text);
            (
                parsed.metadata.as_object().cloned().unwrap_or_default(),
                parsed.content,
            )
        } else {
            concept_frontmatter(&concept)
        };
        let changed = edit(&mut meta)?;
        if changed.is_empty() {
            return Ok(serde_json::json!({
                "concept_id": concept_id,
                "path": source.map(|path| path.to_string_lossy().into_owned()),
                "changed": Json::Array(Vec::new()),
            }));
        }
        let path = if let Some(path) = source {
            path
        } else if let Some(root) = self.settings.bundle_root.as_ref() {
            let mut relative = concept_id.to_string();
            if let Some(rest) = relative.strip_prefix('@') {
                relative = rest
                    .split_once('/')
                    .map(|(_, rest)| rest)
                    .unwrap_or(rest)
                    .to_string();
            }
            if self.settings.identity == "frontmatter" {
                meta.insert("id".into(), Json::String(relative.clone()));
            }
            if !crate::util::safe_bundle_id(&relative) {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    format!("{op}: concept id {concept_id:?} is not safe to materialize"),
                )
                .op(op)
                .field("concept_id", Json::String(concept_id.into())));
            }
            Path::new(root).join(format!("{relative}.md"))
        } else {
            return Err(OkfError::new(
                "MISSING_PARAM",
                "materialize needs a bundle root: this router is file-free (bundle_root=None)",
            )
            .op(op)
            .field("materialize", Json::Bool(true))
            .remedy("open the router with bundle_root, or omit materialize"));
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                OkfError::internal(
                    format!("cannot create {}: {err}", parent.display()),
                    "std::io::Error",
                )
            })?;
        }
        let yaml = crate::export::dump_yaml_sorted(&Json::Object(meta.clone()), 0);
        let text = format!("---\n{yaml}---\n\n{body}\n");
        crate::export::fs_write(&path, &text)?;
        let mut import_params = Map::new();
        import_params.insert(
            "file_path".into(),
            Json::String(path.to_string_lossy().into_owned()),
        );
        let imported = self.import_file(&import_params)?;
        let imported_id = imported
            .get("concept_id")
            .and_then(Json::as_str)
            .unwrap_or_default();
        if imported_id != concept_id {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!(
                    "{op}: the file {} now imports as '{imported_id}', not '{concept_id}' (its id moved?)",
                    path.display()
                ),
            )
            .op(op)
            .field("concept_id", Json::String(concept_id.into()))
            .field("imported", Json::String(imported_id.into())));
        }
        Ok(serde_json::json!({
            "concept_id": imported_id,
            "path": path.to_string_lossy().into_owned(),
            "changed": changed,
        }))
    }

    fn update(&self, params: &Map<String, Json>) -> Result<Json> {
        let concept_id = str_param(params, "concept_id")?.to_string();
        let result = self.write_and_import("update", &concept_id, |meta| {
            if params.contains_key("tags")
                && (params.contains_key("add_tags") || params.contains_key("remove_tags"))
            {
                return Err(bad_value2(
                    "update",
                    "pass tags or add_tags/remove_tags, not both",
                    vec![("concept_id", Json::String(concept_id.clone()))],
                ));
            }
            let mut changed = Vec::new();
            for key in ["title", "description", "type"] {
                if let Some(value) = params.get(key) {
                    if !value.is_null() && meta.get(key) != Some(value) {
                        meta.insert(key.to_string(), value.clone());
                        changed.push(key.to_string());
                    }
                }
            }
            let current = crate::model::normalize_tags(meta.get("tags").unwrap_or(&Json::Null));
            let mut new_tags = if let Some(tags) = params.get("tags") {
                crate::model::normalize_tags(tags)
            } else {
                current.clone()
            };
            for tag in params
                .get("add_tags")
                .map(crate::model::normalize_tags)
                .unwrap_or_default()
            {
                if !new_tags.contains(&tag) {
                    new_tags.push(tag);
                }
            }
            let drop: BTreeSet<String> = params
                .get("remove_tags")
                .map(|value| crate::model::normalize_tags(value).into_iter().collect())
                .unwrap_or_default();
            new_tags.retain(|tag| !drop.contains(tag));
            if new_tags != current {
                meta.insert(
                    "tags".into(),
                    Json::Array(new_tags.into_iter().map(Json::String).collect()),
                );
                changed.push("tags".into());
            }
            if let Some(extra) = params.get("extra").and_then(Json::as_object) {
                let mut reserved: Vec<String> = extra
                    .keys()
                    .filter(|key| WRITE_RESERVED.contains(&key.as_str()))
                    .cloned()
                    .collect();
                if !reserved.is_empty() {
                    reserved.sort();
                    return Err(bad_value2(
                        "update",
                        format!("extra may not set {}", reserved.join(", ")),
                        vec![(
                            "reserved",
                            Json::Array(reserved.into_iter().map(Json::String).collect()),
                        )],
                    )
                    .remedy("use title/description/type/tags/relations"));
                }
                for (key, value) in extra {
                    if value.is_null() {
                        if meta.remove(key).is_some() {
                            changed.push(key.clone());
                        }
                    } else if meta.get(key) != Some(value) {
                        meta.insert(key.clone(), value.clone());
                        changed.push(key.clone());
                    }
                }
            }
            if let Some(relations) = params.get("relations") {
                let problems = relation_problems(relations);
                if !problems.is_empty() {
                    return Err(bad_value2(
                        "update",
                        problems.join("; "),
                        vec![(
                            "relations",
                            Json::Array(problems.into_iter().map(Json::String).collect()),
                        )],
                    ));
                }
                if meta.get("relations") != Some(relations) {
                    if relations.as_object().is_some_and(|map| !map.is_empty()) {
                        meta.insert("relations".into(), relations.clone());
                    } else {
                        meta.remove("relations");
                    }
                    changed.push("relations".into());
                }
            }
            Ok(changed)
        })?;
        Ok(result)
    }

    fn supersede(&self, params: &Map<String, Json>) -> Result<Json> {
        let old_id = str_param(params, "old_id")?.to_string();
        let new_id = str_param(params, "new_id")?.to_string();
        if old_id == new_id {
            return Err(bad_value2(
                "supersede",
                "a concept cannot supersede itself",
                vec![("old", Json::String(old_id))],
            ));
        }
        for id in [&old_id, &new_id] {
            if self.store.get_by_id(id)?.is_none() {
                return Err(unknown_write_concept("supersede", id));
            }
        }
        let old = old_id.clone();
        let new_result = self.write_and_import("supersede", &new_id, |meta| {
            let mut relations = meta
                .get("relations")
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default();
            let mut current = match relations.get("supersedes") {
                Some(Json::String(text)) => vec![text.clone()],
                Some(Json::Array(items)) => items
                    .iter()
                    .filter_map(Json::as_str)
                    .map(str::to_string)
                    .collect(),
                _ => Vec::new(),
            };
            if current.contains(&old) {
                return Ok(Vec::new());
            }
            current.push(old.clone());
            relations.insert(
                "supersedes".into(),
                Json::Array(current.into_iter().map(Json::String).collect()),
            );
            meta.insert("relations".into(), Json::Object(relations));
            Ok(vec!["relations".into()])
        })?;
        let new = new_id.clone();
        let old_result = self.write_and_import("supersede", &old_id, |meta| {
            let mut changed = Vec::new();
            let mut tags = crate::model::normalize_tags(meta.get("tags").unwrap_or(&Json::Null));
            if !tags.iter().any(|tag| tag == "superseded") {
                tags.push("superseded".into());
                meta.insert(
                    "tags".into(),
                    Json::Array(tags.into_iter().map(Json::String).collect()),
                );
                changed.push("tags".into());
            }
            if meta.get("superseded_by") != Some(&Json::String(new.clone())) {
                meta.insert("superseded_by".into(), Json::String(new.clone()));
                changed.push("superseded_by".into());
            }
            Ok(changed)
        })?;
        Ok(serde_json::json!({
            "old": old_id,
            "new": new_id,
            "old_path": old_result.get("path").cloned().unwrap_or(Json::Null),
            "new_path": new_result.get("path").cloned().unwrap_or(Json::Null),
        }))
    }

    fn produce(&self, params: &Map<String, Json>) -> Result<Json> {
        let source_type = str_param(params, "source_type")?.to_string();
        let source = opt_str(params, "source_path")?;
        let output_dir = opt_str(params, "output_dir")?.unwrap_or_else(|| ".".into());
        let prefix = opt_str(params, "prefix")?;
        let overwrite = params
            .get("overwrite")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let bundle = crate::producers::run(
            &source_type,
            source.as_deref(),
            Path::new(&output_dir),
            prefix.as_deref(),
            overwrite,
            None,
            "produce",
        )?;
        let root = bundle.root.join(&bundle.prefix);
        let lint = crate::diagnostics::lint(
            &crate::producers::native(Path::new(&output_dir)),
            &self.settings,
        )?;
        Ok(serde_json::json!({
            "producer": bundle.producer,
            "files": bundle.files,
            "root": crate::producers::native(&root),
            "prefix": bundle.prefix,
            "lint": lint,
        }))
    }

    fn sync(&self, params: &Map<String, Json>) -> Result<Json> {
        let producer = str_param(params, "producer")?.to_string();
        if !crate::producers::PRODUCERS.contains(&producer.as_str()) {
            return Err(
                OkfError::new("BAD_VALUE", format!("unknown producer '{producer}'"))
                    .op("sync")
                    .field("producer", Json::String(producer.clone()))
                    .field(
                        "available",
                        Json::Array(
                            crate::producers::PRODUCERS
                                .iter()
                                .map(|name| Json::String((*name).into()))
                                .collect(),
                        ),
                    ),
            );
        }
        let source = opt_str(params, "source")?;
        let output_dir = opt_str(params, "output_dir")?
            .or_else(|| self.settings.bundle_root.clone())
            .ok_or_else(|| {
                OkfError::new(
                    "MISSING_PARAM",
                    "sync needs output_dir (or a router bundle_root)",
                )
                .op("sync")
                .field("output_dir", Json::Null)
            })?;
        let prefix = opt_str(params, "prefix")?;
        let key = format!(
            "{}{}",
            producer,
            prefix
                .as_ref()
                .map_or(String::new(), |value| format!(":{value}"))
        );
        let cursor = self
            .store
            .query(
                "MATCH (p:ProducerCursor {name: $name}) RETURN p.cursor AS c",
                vec![("name", lbug::Value::String(key.clone()))],
            )?
            .first()
            .and_then(|row| row.get("c"))
            .and_then(Json::as_str)
            .map(str::to_string);
        let bundle = crate::producers::run(
            &producer,
            source.as_deref(),
            Path::new(&output_dir),
            prefix.as_deref(),
            true,
            cursor.as_deref(),
            "sync",
        )?;
        let written: Vec<String> = bundle
            .written
            .iter()
            .filter(|path| {
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                matches!(
                    path.extension().and_then(|value| value.to_str()),
                    Some("md" | "markdown" | "txt")
                ) && name != "index.md"
                    && name != "log.md"
            })
            .map(|path| crate::producers::native(path))
            .collect();
        let concept_ids = if written.is_empty() {
            Vec::new()
        } else {
            // FileHash keys are relative to the tree that was scanned. `sync`
            // may write into an output_dir that differs from bundle_root.
            let mut sync_settings = self.settings.clone();
            sync_settings.bundle_root = Some(output_dir.clone());
            let report = crate::ingest::import_files(
                &self.store,
                &sync_settings,
                &self.embedder,
                &self.pools,
                &written,
            )
            .map_err(|err| err.op("sync"))?;
            report
                .get("concept_ids")
                .and_then(Json::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Json::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };
        {
            let new_cursor = &bundle.cursor;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|err| OkfError::internal(err.to_string(), "system time"))?
                .as_secs() as i64;
            self.store.write(
                "MERGE (p:ProducerCursor {name: $name}) SET p.cursor = $cursor, p.updated_at = $updated",
                vec![
                    ("name", lbug::Value::String(key)),
                    ("cursor", lbug::Value::String(new_cursor.clone())),
                    ("updated", lbug::Value::Int64(now)),
                ],
            )?;
        }
        Ok(serde_json::json!({
            "producer": producer,
            "cursor": bundle.cursor,
            "written": written,
            "concept_ids": concept_ids,
        }))
    }

    fn reindex(&self, params: &Map<String, Json>) -> Result<Json> {
        let if_dirty = opt_bool(params, "if_dirty")?.unwrap_or(false);
        let write_epoch = self.store.meta_int("write_epoch")?.unwrap_or(0);
        let indexed_epoch = self.store.meta_int("indexed_epoch")?.unwrap_or(0);
        if if_dirty && write_epoch <= indexed_epoch {
            return Ok(serde_json::json!({ "rebuilt": false }));
        }
        if !if_dirty {
            self.store.drop_search_indexes()?;
        }
        self.store.create_search_indexes()?;
        self.store.set_meta_int("indexed_epoch", write_epoch)?;
        Ok(serde_json::json!({ "rebuilt": true }))
    }

    fn export_bundle(&self, params: &Map<String, Json>) -> Result<Json> {
        let output_dir = str_param(params, "output_dir")?.to_string();
        let directory_id = opt_str(params, "directory_id")?;
        let concept_type = opt_str(params, "concept_type")?;
        let tags = match params.get("tags") {
            Some(Json::Array(items)) => Some(
                items
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect::<Vec<_>>(),
            ),
            Some(Json::Null) | None => None,
            Some(_) => {
                return Err(OkfError::new("BAD_VALUE", "tags must be a list of strings")
                    .op("export_bundle"))
            }
        };
        let flavor = opt_str(params, "flavor")?.unwrap_or_else(|| "okf".into());
        crate::export::export_bundle(
            &self.store,
            &output_dir,
            directory_id.as_deref(),
            concept_type.as_deref(),
            tags.as_deref(),
            &flavor,
        )
        .map_err(|err| err.op("export_bundle"))
    }

    fn export_concept(&self, params: &Map<String, Json>) -> Result<Json> {
        let concept_id = str_param(params, "concept_id")?.to_string();
        let output_dir = str_param(params, "output_dir")?.to_string();
        let flavor = opt_str(params, "flavor")?.unwrap_or_else(|| "okf".into());
        crate::export::export_concept(&self.store, &concept_id, &output_dir, &flavor)
            .map_err(|err| err.op("export_concept"))
    }

    fn repair_links(&self) -> Result<Json> {
        crate::ingest::repair_links_public(&self.store)
            .map(|n| serde_json::json!({ "repaired": n }))
    }

    fn import_bundle(&self, params: &Map<String, Json>) -> Result<Json> {
        let bundle_path = match params.get("bundle_path") {
            Some(Json::String(value)) => Some(value.as_str()),
            Some(Json::Null) | None => None,
            Some(_) => {
                return Err(OkfError::new("BAD_VALUE", "bundle_path must be a string")
                    .op("import_bundle")
                    .field("bundle_path", Json::String("expected a string".into())))
            }
        };
        let settings = self.import_settings(params)?;
        let options = crate::ingest::ImportOptions {
            prune_missing: params
                .get("prune_missing")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            force: params.get("force").and_then(Json::as_bool).unwrap_or(false),
        };
        crate::ingest::import_bundle(
            &self.store,
            &settings,
            &self.embedder,
            &self.pools,
            bundle_path,
            options,
        )
        .map_err(|err| err.op("import_bundle"))
    }

    fn import_file(&self, params: &Map<String, Json>) -> Result<Json> {
        let path = str_param(params, "file_path")?.to_string();
        if !Path::new(&path).is_file() {
            let shown = crate::util::python_path_str(&path);
            return Err(
                OkfError::new("FILE_NOT_FOUND", format!("file not found: {shown}"))
                    .op("import_file")
                    .field("file_path", Json::String(shown)),
            );
        }
        let settings = self.import_settings(params)?;
        let report = crate::ingest::import_files(
            &self.store,
            &settings,
            &self.embedder,
            &self.pools,
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
        let images = report
            .get("images")
            .and_then(Json::as_object)
            .and_then(|images| images.get(&id))
            .and_then(Json::as_i64)
            .unwrap_or(0);
        Ok(serde_json::json!({
            "concept_id": id,
            "images": images,
        }))
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Inter-process lock wait. The public contract is Python's 300 s; the
    /// environment override exists for lock-interoperability/regression tests.
    /// Invalid values fall back to the contract rather than disabling locking.
    fn write_lock_timeout() -> std::time::Duration {
        let secs = std::env::var("OKFGRAPH_WRITE_LOCK_TIMEOUT")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(300.0);
        std::time::Duration::from_secs_f64(secs)
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    fn lock_path(&self) -> PathBuf {
        std::path::absolute(format!("{}.lock", self.settings.db_path))
            .unwrap_or_else(|_| PathBuf::from(format!("{}.lock", self.settings.db_path)))
    }

    /// Run a mutating operation under both lock layers. Python holds the same
    /// `<db>.lock` with `fasteners`; the timeout and file name are contractual.
    fn with_write_lock<T>(&self, op: &str, run: impl FnOnce() -> Result<T>) -> Result<T> {
        let _write_guard = self
            .write_lock
            .lock()
            .map_err(|_| OkfError::internal("write lock poisoned", "mutex"))?;
        let lock_path = self.lock_path();
        let lock_text = lock_path.to_string_lossy().into_owned();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| {
                OkfError::internal(
                    format!("cannot open write lock {lock_text}: {err}"),
                    "std::io::Error",
                )
            })?;
        let mut lock = fd_lock::RwLock::new(file);
        let deadline = std::time::Instant::now() + Self::write_lock_timeout();
        let _guard = loop {
            match lock.try_write() {
                Ok(guard) => break guard,
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return Err(OkfError::new(
                            "WRITE_LOCK_TIMEOUT",
                            "write lock acquisition timed out (another process is writing)",
                        )
                        .op(op)
                        .remedy("close concurrent writers or raise the lock timeout"));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(err) => {
                    return Err(OkfError::internal(
                        format!("cannot acquire write lock {lock_text}: {err}"),
                        "std::io::Error",
                    ))
                }
            }
        };
        run()
    }

    /// Unlink an idle `<db>.lock` when no other process holds it, matching
    /// Python's `OKFRouter._remove_idle_lock_file`. Best effort: a failed
    /// probe or unlink leaves the sidecar in place (the `doctor` stale-lock
    /// probe will then report it as informational).
    pub fn cleanup_write_lock_file(&self) {
        let path = self.lock_path();
        let Ok(file) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        else {
            return;
        };
        let mut lock = fd_lock::RwLock::new(file);
        let Ok(guard) = lock.try_write() else {
            return;
        };
        drop(guard);
        let _ = std::fs::remove_file(path);
    }

    pub fn call(&self, op: &str, params: &Map<String, Json>) -> Result<Json> {
        if WRITE_OPS.contains(&op) {
            return self.with_write_lock(op, || self.call_inner(op, params));
        }
        self.call_inner(op, params)
    }

    fn call_inner(&self, op: &str, params: &Map<String, Json>) -> Result<Json> {
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
            "repair_links" => self.repair_links(),
            "export_bundle" => self.export_bundle(params),
            "export_concept" => self.export_concept(params),
            "delete" => self.delete(params),
            "recover_deleted" => self.recover_deleted(params),
            "purge_deleted" => self.purge_deleted(params),
            "reindex" => self.reindex(params),
            "reembed" => self.reembed(params),
            "update" => self.update(params),
            "supersede" => self.supersede(params),
            "detach" => crate::detach::detach(&self.store, &self.settings, params),
            "diff" => self.diff(params),
            "lint" => self.lint(params),
            "produce" => self.produce(params),
            "sync" => self.sync(params),
            "ingest" => crate::converters::ingest(
                &self.store,
                &self.settings,
                &self.embedder,
                &self.pools,
                params,
            ),
            "pin_info" => self.pin_info(),
            "lanes" => self.lanes(),
            "doctor" => self.doctor(params),
            "search" => self.search(params),
            "embed" => self.embed(params),
            "read" => self.read(params),
            "traverse" => self.traverse(params),
            "import_files" => self.import_files(params),
            "import_file" => self.import_file(params),
            "import_bundle" => self.import_bundle(params),
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
        #[cfg(feature = "vision")]
        if let Some(vision) = self.pools.vision.loaded() {
            if let Some(Json::Object(image)) = lanes.get_mut("image") {
                image.insert("model_id".into(), Json::String(vision.model_id().into()));
                image.insert("precision".into(), Json::String(vision.precision().into()));
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

    pub fn doctor(&self, params: &Map<String, Json>) -> Result<Json> {
        use crate::model::{utc_now_datetime, PyDateTime};

        let stale_days = opt_i64(params, "stale_days")?.unwrap_or(365);
        let strict = opt_bool(params, "strict")?.unwrap_or(false);
        let fix = opt_bool(params, "fix")?.unwrap_or(false);
        let fixed = if fix {
            let result = self.with_write_lock("doctor", || self.doctor_fix())?;
            // Python's doctor fix does not take the router write lock; to
            // preserve its informational stale-lock output while still being
            // safe, release the sidecar immediately after the repair.
            self.cleanup_write_lock_file();
            Some(result)
        } else {
            None
        };

        let concepts = self.store.query(
            "MATCH (c:Concept) RETURN c.id AS id, c.title AS title, \
             c.description AS description, c.timestamp AS timestamp",
            vec![],
        )?;
        let live_ids: BTreeSet<String> = concepts
            .iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();
        let tombstoned: BTreeSet<String> = self
            .store
            .query(
                "MATCH (d:DeletedConcept) RETURN d.original_id AS id",
                vec![],
            )
            .unwrap_or_default()
            .into_iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();

        let mut findings: Vec<Map<String, Json>> = Vec::new();
        let mut add = |path: &str, severity: &str, rule: &str, message: String| {
            let mut row = Map::new();
            row.insert("path".into(), Json::String(path.into()));
            row.insert("severity".into(), Json::String(severity.into()));
            row.insert("rule".into(), Json::String(rule.into()));
            row.insert("message".into(), Json::String(message));
            findings.push(row);
        };

        // Search-index inventory and missing-index findings. The query uses a
        // connection with vector/FTS loaded, like the Python schema manager.
        let mut index_info: Vec<Json> = Vec::new();
        match self.store.index_inventory() {
            Ok(rows) => {
                let dim = self
                    .store
                    .schema_embedding_dim()?
                    .unwrap_or(self.store.settings().embedding_dim);
                let mut present: BTreeMap<(String, String), Map<String, Json>> = BTreeMap::new();
                for row in rows {
                    let table = row
                        .get("table_name")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let name = row
                        .get("index_name")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string();
                    if name != "_PK" && !table.is_empty() && !name.is_empty() {
                        present.insert((table, name), row);
                    }
                }
                for ((table, name), row) in &present {
                    let kind = row.get("index_type").and_then(Json::as_str).unwrap_or("?");
                    let properties = row
                        .get("property_names")
                        .and_then(Json::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Json::as_str)
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    let mut detail = properties;
                    if kind == "HNSW" {
                        let definition = row
                            .get("index_definition")
                            .and_then(Json::as_str)
                            .unwrap_or_default();
                        detail.push_str(&format!("; {}, dim {}", hnsw_metric(definition), dim));
                    }
                    index_info.push(serde_json::json!({
                        "rule": "search_index",
                        "message": format!("index {table}.{name}: {kind} ({detail})"),
                    }));
                }
                for (table, name) in INDEX_SPECS {
                    if !present.contains_key(&(table.to_string(), name.to_string())) {
                        add(
                            "<database>",
                            "warning",
                            "missing_index",
                            format!(
                                "search index {table}.{name} is missing — \
                                search on that table falls back to one source; \
                                run 'okf reindex'"
                            ),
                        );
                    }
                }
            }
            Err(err) => {
                let source = self
                    .settings
                    .extension_dir
                    .as_deref()
                    .map(|dir| format!("extension_dir {dir}"))
                    .unwrap_or_else(|| "the Ladybug extension repository".into());
                let detail = format!("vector: {}", err.message);
                add(
                    "<database>",
                    "error",
                    "search_unavailable",
                    format!(
                        "vector/fts extensions not loaded from {source} — hybrid \
                        search is unavailable, only rank='ppr' works ({detail}). \
                        Run 'okf setup extensions' and set [database] extension_dir."
                    ),
                );
            }
        }

        for link in self.store.list_broken_links()? {
            let source = link.get("source").and_then(Json::as_str).unwrap_or("");
            let target = link.get("target").and_then(Json::as_str).unwrap_or("");
            let kind = link
                .get("kind")
                .and_then(Json::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or("mention");
            let reason = link
                .get("reason")
                .and_then(Json::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or("missing");
            let via = if kind == "mention" {
                String::new()
            } else {
                format!(" ({kind})")
            };
            if reason == "ambiguous" {
                let mut candidates: Vec<String> = link
                    .get("candidates")
                    .and_then(Json::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Json::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                candidates.sort();
                let matched = if candidates.is_empty() {
                    "several concepts".to_string()
                } else {
                    candidates.join(", ")
                };
                add(
                    source,
                    "warning",
                    "ambiguous_link",
                    format!(
                        "link '{target}'{via} is ambiguous: matches {matched} — use an id or a path link"
                    ),
                );
            } else {
                add(
                    source,
                    "error",
                    "broken_link",
                    format!("links to missing concept '{target}'{via}"),
                );
            }
        }

        for row in self.store.query(
            "MATCH (f:FileHash) RETURN f.path AS path, f.concept_id AS cid",
            vec![],
        )? {
            let path = row.get("path").and_then(Json::as_str).unwrap_or("");
            let cid = row.get("cid").and_then(Json::as_str);
            let known = cid.is_some_and(|id| live_ids.contains(id) || tombstoned.contains(id));
            if !known {
                let label = cid.unwrap_or("None");
                add(
                    path,
                    "error",
                    "orphan_hash",
                    format!(
                        "hash tracked for missing concept '{label}' (crashed import?) — cleared by --fix"
                    ),
                );
            }
        }

        let mut titles: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for row in &concepts {
            let title = row.get("title").and_then(Json::as_str).unwrap_or("");
            let id = row.get("id").and_then(Json::as_str).unwrap_or("");
            if !title.is_empty() {
                titles
                    .entry(title.to_string())
                    .or_default()
                    .push(id.to_string());
            }
        }
        for (title, ids) in &titles {
            if ids.len() > 1 {
                let mut sorted = ids.clone();
                sorted.sort();
                for cid in &sorted {
                    let others: Vec<&str> = sorted
                        .iter()
                        .filter(|other| *other != cid)
                        .map(String::as_str)
                        .collect();
                    add(
                        cid,
                        "warn",
                        "duplicate_title",
                        format!(
                            "title '{title}' also used by {} ([[{title}]] links resolve to neither)",
                            others.join(", ")
                        ),
                    );
                }
            }
        }

        let mut degrees: BTreeMap<String, (i64, i64)> = BTreeMap::new();
        for row in self.store.query(
            "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) \
             RETURN a.id AS src, b.id AS dst",
            vec![],
        )? {
            let src = row.get("src").and_then(Json::as_str).unwrap_or("");
            let dst = row.get("dst").and_then(Json::as_str).unwrap_or("");
            degrees.entry(src.to_string()).or_default().0 += 1;
            degrees.entry(dst.to_string()).or_default().1 += 1;
        }

        let now = utc_now_datetime();
        for row in &concepts {
            let cid = row.get("id").and_then(Json::as_str).unwrap_or("");
            if cid.ends_with("index") || cid.ends_with("log") {
                continue;
            }
            let (out_d, in_d) = degrees.get(cid).copied().unwrap_or((0, 0));
            if out_d == 0 && in_d == 0 {
                add(
                    cid,
                    "warn",
                    "orphan",
                    "no incoming or outgoing links".into(),
                );
            }
            if row
                .get("description")
                .and_then(Json::as_str)
                .unwrap_or("")
                .trim()
                .is_empty()
            {
                add(
                    cid,
                    "warn",
                    "missing_description",
                    "empty description".into(),
                );
            }
            if let Some(timestamp) = row.get("timestamp").and_then(Json::as_str) {
                if let Ok(parsed) = PyDateTime::parse(timestamp) {
                    let age_days = (now.as_naive() - parsed.as_naive()).as_seconds_f64() / 86_400.0;
                    if age_days > stale_days as f64 {
                        add(
                            cid,
                            "warn",
                            "stale",
                            format!("timestamp {timestamp} is {} days old", age_days as i64),
                        );
                    }
                }
            }
        }

        findings.sort_by(|left, right| {
            let key = |row: &Map<String, Json>| {
                (
                    row.get("rule")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                    row.get("path")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                )
            };
            key(left).cmp(&key(right))
        });

        let mut summary: BTreeMap<String, i64> = BTreeMap::new();
        for finding in &findings {
            let rule = finding
                .get("rule")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string();
            *summary.entry(rule).or_default() += 1;
        }
        let deductions: &[(&str, i64, i64)] = &[
            ("search_unavailable", 20, 20),
            ("broken_link", 5, 40),
            ("ambiguous_link", 3, 15),
            ("orphan_hash", 5, 30),
            ("orphan", 2, 20),
            ("duplicate_title", 3, 15),
            ("stale", 1, 10),
            ("missing_description", 1, 15),
        ];
        let mut score = 100i64;
        for (rule, count) in &summary {
            if let Some((points, cap)) = deductions
                .iter()
                .find(|(name, _, _)| name == rule)
                .map(|(_, points, cap)| (*points, *cap))
            {
                score -= (points * count).min(cap);
            }
        }
        score = score.max(0);

        let mut info = Vec::new();
        let mut incoming: Vec<(String, i64)> = degrees
            .iter()
            .filter(|(_, (_, ind))| *ind > 0)
            .map(|(id, (_, ind))| (id.clone(), *ind))
            .collect();
        incoming.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
        for (id, count) in incoming.into_iter().take(3) {
            info.push(serde_json::json!({
                "rule": "hub_concentration",
                "message": format!("'{id}' has {count} incoming link(s)")
            }));
        }
        info.extend(index_info);

        // Pre-0.7 image rows still on the removed `omni` route.
        let stale_omni = self
            .store
            .query(
                "MATCH (i:ImageAsset) WHERE i.embed_route = 'omni' \
                 RETURN i.id AS iid",
                vec![],
            )?
            .len();
        if stale_omni > 0 {
            info.push(serde_json::json!({
                "rule": "stale_omni_images",
                "message": format!(
                    "{stale_omni} image asset(s) use the removed omni route — \
                    re-import to re-embed by caption"
                ),
            }));
        }

        // Image route counts plus the pinned image model/precision.
        let routes = self.store.query(
            "MATCH (i:ImageAsset) RETURN i.embed_route AS route, count(*) AS n",
            vec![],
        )?;
        if !routes.is_empty() {
            let mut counts: Vec<(String, i64)> = routes
                .into_iter()
                .map(|row| {
                    let route = row
                        .get("route")
                        .map(|value| match value {
                            Json::String(text) => text.clone(),
                            Json::Null => "None".into(),
                            other => other.to_string(),
                        })
                        .unwrap_or_else(|| "None".into());
                    let n = row.get("n").and_then(Json::as_i64).unwrap_or(0);
                    (route, n)
                })
                .collect();
            counts.sort_by(|left, right| left.0.cmp(&right.0));
            let summary = counts
                .into_iter()
                .map(|(route, n)| format!("{route}={n}"))
                .collect::<Vec<_>>()
                .join(", ");
            let pins = self.store.pin_info()?;
            let image = pins.get("image");
            let model = image
                .and_then(|obj| obj.get("pinned_model"))
                .and_then(Json::as_str)
                .unwrap_or("(unpinned)");
            let precision = image
                .and_then(|obj| obj.get("pinned_precision"))
                .and_then(Json::as_str)
                .unwrap_or("(unpinned)");
            info.push(serde_json::json!({
                "rule": "image_routes",
                "message": format!(
                    "image assets: {summary}; image model={model} precision={precision}"
                ),
            }));
        }

        // A leftover `<db>.lock` file is only informational: probe it without
        // blocking, exactly like the Python `fasteners` check.
        let lock_path = self.lock_path();
        if lock_path.is_file() {
            if let Ok(file) = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&lock_path)
            {
                let mut probe = fd_lock::RwLock::new(file);
                let name = lock_path
                    .file_name()
                    .map(|value| value.to_string_lossy().into_owned())
                    .unwrap_or_default();
                match probe.try_write() {
                    Ok(guard) => {
                        drop(guard);
                        info.push(serde_json::json!({
                            "rule": "stale_lock",
                            "message": format!(
                                "lock file '{name}' exists but no live holder \
                                (leftover from a crash/kill?) — harmless, but safe to \
                                remove after checking no okf/mcp process is running"
                            ),
                        }))
                    }
                    Err(_) => info.push(serde_json::json!({
                        "rule": "db_lock_held",
                        "message": format!(
                            "lock file '{name}' is held by a live process \
                            (another okf/mcp writer?) — run one writer per database"
                        ),
                    })),
                };
            }
        }

        // The Python admin layer appends one line per configured lane after
        // the schema/env probes. Report the same lane identity without opening
        // any engine.
        info.extend(doctor_lane_info(&self.lanes()?));

        let detached = if self.store.meta_int("detached")?.unwrap_or(0) == 1 {
            let roots = self
                .store
                .query(
                    "MATCH (s:SourceRoot) RETURN s.alias AS alias, s.path AS path, \
                     s.file_count AS file_count, s.last_seen AS last_seen",
                    vec![],
                )
                .unwrap_or_default();
            Some(serde_json::json!({
                "detached_at": self.store.meta_int("detached_at")?.unwrap_or(0),
                "roots": roots,
            }))
        } else {
            None
        };

        let file_paths: Vec<String> = self.store.file_hashes()?.into_keys().collect();
        let mut targets: Vec<(String, String)> = Vec::new();
        if let Some(root) = self.settings.bundle_root.as_ref() {
            targets.push((String::new(), root.clone()));
        }
        targets.extend(
            self.settings
                .roots
                .iter()
                .map(|(alias, path)| (alias.clone(), path.clone())),
        );
        let roots: Vec<Json> = targets
            .into_iter()
            .map(|(alias, path)| {
                let prefix = if alias.is_empty() {
                    String::new()
                } else {
                    format!("@{alias}/")
                };
                let tracked_files = if alias.is_empty() {
                    file_paths.iter().filter(|p| !p.starts_with('@')).count()
                } else {
                    file_paths.iter().filter(|p| p.starts_with(&prefix)).count()
                };
                let concepts = if alias.is_empty() {
                    live_ids.iter().filter(|id| !id.starts_with('@')).count()
                } else {
                    live_ids.iter().filter(|id| id.starts_with(&prefix)).count()
                };
                serde_json::json!({
                    "alias": if alias.is_empty() { "<primary>" } else { &alias },
                    "path": path,
                    "present": Path::new(&path).is_dir(),
                    "tracked_files": tracked_files,
                    "concepts": concepts,
                })
            })
            .collect();

        let report = serde_json::json!({
            "concepts": concepts.len(),
            "detached": detached,
            "findings": findings,
            "info": info,
            "roots": roots,
            "score": score,
            "summary": summary,
        });
        let result = serde_json::json!({"report": report, "fixed": fixed});
        if strict && !findings.is_empty() {
            return Err(OkfError::outcome(
                "DOCTOR_FINDINGS",
                format!("doctor found {} finding(s)", findings.len()),
                result,
            )
            .op("doctor")
            .field("findings", Json::from(findings.len() as i64))
            .field("score", Json::from(score)));
        }
        Ok(result)
    }

    /// Safe doctor repairs: link repair plus orphan delta rows. Timestamp
    /// normalization and reviewed-concept skipping are the remaining
    /// compatibility refinements tracked by BUG-045.
    fn doctor_fix(&self) -> Result<Json> {
        let mut skipped: BTreeSet<String> = BTreeSet::new();
        for row in self.store.query(
            "MATCH (c:Concept) RETURN c.id AS id, c.extra AS extra",
            vec![],
        )? {
            let reviewed = row
                .get("extra")
                .and_then(Json::as_object)
                .and_then(|extra| extra.get("reviewed"))
                .map(|value| match value {
                    Json::String(text) => text.trim().to_ascii_lowercase(),
                    Json::Bool(true) => "true".into(),
                    Json::Number(number) => number.to_string(),
                    _ => String::new(),
                })
                .is_some_and(|value| matches!(value.as_str(), "true" | "1" | "yes"));
            if reviewed {
                if let Some(id) = row.get("id").and_then(Json::as_str) {
                    skipped.insert(id.to_string());
                }
            }
        }
        let repaired = crate::ingest::repair_links_public_skip(&self.store, &skipped)?;
        let live: BTreeSet<String> = self
            .store
            .query("MATCH (c:Concept) RETURN c.id AS id", vec![])?
            .into_iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();
        let tombstoned: BTreeSet<String> = self
            .store
            .query(
                "MATCH (d:DeletedConcept) RETURN d.original_id AS id",
                vec![],
            )
            .unwrap_or_default()
            .into_iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();
        let mut cleared_files = 0i64;
        let mut cleared_dirs = 0i64;
        let mut dirs: BTreeSet<String> = BTreeSet::new();
        for row in self.store.query(
            "MATCH (f:FileHash) RETURN f.path AS path, f.concept_id AS cid, f.dir AS dir",
            vec![],
        )? {
            let path = row.get("path").and_then(Json::as_str).unwrap_or("");
            let cid = row.get("cid").and_then(Json::as_str);
            let orphan = cid.is_none_or(|id| !live.contains(id) && !tombstoned.contains(id));
            if !orphan {
                continue;
            }
            let dir = row
                .get("dir")
                .and_then(Json::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| {
                    Path::new(path)
                        .parent()
                        .map(|parent| parent.to_string_lossy().into_owned())
                        .filter(|parent| !parent.is_empty())
                        .unwrap_or_else(|| ".".to_string())
                });
            self.store.forget_file_hash(path)?;
            cleared_files += 1;
            dirs.insert(dir);
        }
        // Python counts the attempted parent deletions even when a legacy
        // graph never wrote the DirHash row (the fixture's orphan is such a
        // case); the DELETE itself is idempotent.
        for dir in dirs {
            self.store.delete_dir_hash(&dir)?;
            cleared_dirs += 1;
        }
        Ok(serde_json::json!({
            "repaired_links": repaired,
            "normalized_timestamps": 0,
            "skipped_reviewed": skipped,
            "cleared_orphan_hashes": cleared_files,
            "cleared_dir_hashes": cleared_dirs,
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
        let concept = self
            .store
            .get_by_id(concept_id)?
            .ok_or_else(|| unknown_concept("read", "concept", concept_id))?;
        if let Some(budget) = max_tokens {
            return self.read_with_budget(concept_id, &include, budget, &concept);
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
            _ => Ok(concept),
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
                    .op("traverse")
                    .field("depth", Json::from(depth)),
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
                        .op("embed")
                        .field("n", Json::from(0)),
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
        check_engine_pins(&self.settings, &self.store.pin_info()?, embedder.as_ref())?;
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
            let embedder = self.embedder.get(&self.settings)?;
            check_engine_pins(&self.settings, &self.store.pin_info()?, embedder.as_ref())?;
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

fn unknown_write_concept(op: &str, id: &str) -> OkfError {
    OkfError::new("UNKNOWN_CONCEPT", format!("concept '{id}' does not exist"))
        .op(op)
        .field("concept_id", Json::String(id.into()))
        .remedy("search first to find IDs")
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

const WRITE_RESERVED: &[&str] = &[
    "id",
    "type",
    "title",
    "description",
    "tags",
    "timestamp",
    "body",
    "embedding",
    "uid",
    "source_path",
    "relations",
    "resource",
];

fn concept_frontmatter(concept: &Json) -> (Map<String, Json>, String) {
    let mut meta = Map::new();
    if let Some(object) = concept.as_object() {
        for (key, value) in object {
            if key == "body"
                || key == "id"
                || key == "source_path"
                || crate::model::GRAPH_TIMES.contains(&key.as_str())
            {
                continue;
            }
            if key == "uid" {
                meta.insert("id".into(), value.clone());
            } else {
                meta.insert(key.clone(), value.clone());
            }
        }
    }
    let body = concept
        .get("body")
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string();
    (meta, body)
}

fn relation_problems(value: &Json) -> Vec<String> {
    let Some(object) = value.as_object() else {
        return vec![format!(
            "relations must be a mapping of kind -> targets, got {}",
            if value.is_null() {
                "NoneType"
            } else {
                "object"
            }
        )];
    };
    let mut problems = Vec::new();
    for (kind, targets) in object {
        let bytes = kind.as_bytes();
        let valid = !kind.is_empty()
            && kind.len() <= 64
            && bytes[0].is_ascii_lowercase()
            && bytes[1..]
                .iter()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"-_".contains(b));
        if kind == "mention" || !valid {
            problems.push(format!(
                "relation kind {} is not allowed",
                py_repr(&Json::String(kind.clone()))
            ));
            continue;
        }
        let items: Vec<&Json> = match targets {
            Json::Array(values) => values.iter().collect(),
            other => vec![other],
        };
        for target in items {
            if target.is_null() {
                continue;
            }
            let Some(text) = target.as_str() else {
                problems.push(format!(
                    "relation {}: target {} is not a string",
                    kind,
                    py_repr(target)
                ));
                continue;
            };
            if text.trim().is_empty() {
                problems.push(format!(
                    "relation {}: target {} is not a string",
                    kind,
                    py_repr(target)
                ));
            }
        }
    }
    problems
}

fn settings_batch_size(settings: &Settings) -> i64 {
    settings.batch_size.max(1)
}

fn opt_bool(params: &Map<String, Json>, name: &str) -> Result<Option<bool>> {
    match params.get(name) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Bool(value)) => Ok(Some(*value)),
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

const INDEX_SPECS: &[(&str, &str)] = &[
    ("Concept", "concept_embedding"),
    ("Concept", "concept_fts"),
    ("ImageAsset", "image_omni_idx"),
    ("Chunk", "chunk_embedding"),
    ("Chunk", "chunk_fts"),
];

fn hnsw_metric(definition: &str) -> &str {
    let marker = "metric := '";
    definition
        .find(marker)
        .and_then(|start| {
            let rest = &definition[start + marker.len()..];
            rest.find('\'').map(|end| &rest[..end])
        })
        .filter(|value| !value.is_empty())
        .unwrap_or("?")
}

fn json_text(value: Option<&Json>) -> Option<String> {
    match value {
        Some(Json::String(text)) if !text.is_empty() => Some(text.clone()),
        Some(Json::Bool(true)) => Some("true".into()),
        Some(Json::Number(number)) => Some(number.to_string()),
        _ => None,
    }
}

fn doctor_lane_info(lanes: &Json) -> Vec<Json> {
    let mut out = Vec::new();
    for name in ["embed", "rerank", "image"] {
        let Some(lane) = lanes.get(name) else {
            continue;
        };
        let endpoint = json_text(lane.get("endpoint"));
        let source = endpoint.clone().unwrap_or_else(|| "in-process".into());
        let who = json_text(lane.get("model_id")).unwrap_or_else(|| {
            if endpoint.is_some() {
                "the host's model".into()
            } else {
                "unopened".into()
            }
        });
        let identity = json_text(lane.get("identity")).unwrap_or_default();
        let mut message = format!("{who} via {source} (identity={identity}");
        for field in ["family", "precision", "space"] {
            if let Some(value) = json_text(lane.get(field)) {
                message.push_str(&format!(", {field}={value}"));
            }
        }
        message.push(')');
        out.push(serde_json::json!({
            "rule": format!("{name}_lane"),
            "message": message,
        }));
    }
    out
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
    fn hnsw_metric_reads_index_definition() {
        assert_eq!(
            hnsw_metric("CALL CREATE_VECTOR_INDEX(... metric := 'cosine', efc := 200)"),
            "cosine"
        );
        assert_eq!(hnsw_metric(""), "?");
    }

    #[test]
    fn doctor_lane_info_matches_admin_shape() {
        let lanes = serde_json::json!({
            "embed": {"model_id": "text", "identity": "in-process"},
            "rerank": {"endpoint": "http://host", "identity": "declared"},
            "image": {"model_id": "vision", "identity": "in-process"},
        });
        let info = doctor_lane_info(&lanes);
        assert_eq!(info.len(), 3);
        assert_eq!(
            info[0]["message"],
            "text via in-process (identity=in-process)"
        );
        assert_eq!(
            info[1]["message"],
            "the host's model via http://host (identity=declared)"
        );
        assert_eq!(info[2]["rule"], "image_lane");
    }

    #[test]
    fn opens_fixture_and_reads() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("parity/fixtures/oracle-v10.db");
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("oracle.db");
        std::fs::copy(&source, &path).expect("copy fixture");
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
