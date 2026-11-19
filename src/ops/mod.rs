//! Typed operation requests/results plus the `call(Op, Value)` escape hatch.
//!
//! M1 implements the read-only slice; write/ingest ops land in Stage B.

use crate::error::{OkfError, Result};
use crate::settings::Settings;
use crate::store::Store;
use serde_json::{Map, Value as Json};
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
}

impl Router {
    pub fn open(settings: &Settings) -> Result<Self> {
        Ok(Self {
            store: Store::open(settings)?,
            settings: settings.clone(),
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
        self.store.lanes()
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
                let markdown = chunks
                    .iter()
                    .filter_map(|c| c.get("chunk_text").and_then(Json::as_str))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                Ok(serde_json::json!({"concept_id": concept_id, "markdown": markdown}))
            }
            "context" => {
                let incoming = self
                    .store
                    .traverse(concept_id, "LINKS_TO", "INCOMING", 1, None, None)?;
                let outgoing = self
                    .store
                    .traverse(concept_id, "LINKS_TO", "OUTGOING", 1, None, None)?;
                let ancestry = self.store.traverse(
                    concept_id,
                    "CONTAINS",
                    "INCOMING",
                    5,
                    Some("Directory"),
                    None,
                )?;
                let siblings = self.store.traverse(
                    concept_id,
                    "CONTAINS",
                    "INCOMING",
                    1,
                    Some("Directory"),
                    None,
                )?;
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
