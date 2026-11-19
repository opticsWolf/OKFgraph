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
use std::collections::{BTreeMap, BTreeSet};
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

/// Concept ids and directed LINKS_TO edges for ranking algorithms.
pub type LinkGraph = (Vec<String>, Vec<(String, String)>);

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

    /// Open the same database with write permission. Stage B imports use
    /// this only in a process that owns the graph; never mix runtimes.
    pub fn open_writable(settings: &Settings) -> Result<Self> {
        let mut store = Self::open(settings)?;
        store.read_only = false;
        Ok(store)
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

    /// Writable connection with the index extensions loaded. Ladybug
    /// maintains indexes in place but refuses unindexed writes when the
    /// extension was not loaded in the writing process.
    fn with_write_conn<T>(
        &self,
        f: impl for<'a> FnOnce(&Connection<'a>) -> Result<T>,
    ) -> Result<T> {
        let config = SystemConfig::default().read_only(false);
        let db = Database::new(&self.path, config).map_err(map_lbug)?;
        let conn = Connection::new(&db).map_err(map_lbug)?;
        for extension in ["vector", "fts"] {
            conn.query(&format!("LOAD {extension}")).map_err(map_lbug)?;
        }
        f(&conn)
    }

    pub fn write(&self, sql: &str, params: Vec<(&str, Value)>) -> Result<()> {
        self.with_write_conn(|conn| {
            if params.is_empty() {
                conn.query(sql).map_err(map_lbug)?;
            } else {
                let mut stmt = conn.prepare(sql).map_err(map_lbug)?;
                conn.execute(&mut stmt, params).map_err(map_lbug)?;
            }
            Ok(())
        })
    }

    pub fn bump_write_epoch(&self) -> Result<()> {
        self.write(
            "MERGE (m:Meta {key: 'write_epoch'}) \
             ON CREATE SET m.value = 1 ON MATCH SET m.value = m.value + 1",
            vec![],
        )
    }

    pub fn set_meta_int(&self, key: &str, value: i64) -> Result<()> {
        self.write(
            "MERGE (m:Meta {key: $k}) SET m.value = $v",
            vec![("k", Value::String(key.into())), ("v", Value::Int64(value))],
        )
    }

    pub fn set_meta_text(&self, key: &str, value: &str) -> Result<()> {
        self.write(
            "MERGE (m:MetaText {key: $k}) SET m.value = $v",
            vec![
                ("k", Value::String(key.into())),
                ("v", Value::String(value.into())),
            ],
        )
    }

    pub fn drop_search_indexes(&self) -> Result<()> {
        self.with_write_conn(|conn| {
            for sql in [
                "CALL DROP_VECTOR_INDEX('Concept', 'concept_embedding')",
                "CALL DROP_FTS_INDEX('Concept', 'concept_fts')",
                "CALL DROP_VECTOR_INDEX('ImageAsset', 'image_omni_idx')",
                "CALL DROP_VECTOR_INDEX('Chunk', 'chunk_embedding')",
                "CALL DROP_FTS_INDEX('Chunk', 'chunk_fts')",
            ] {
                // A missing index is benign: drop is the recovery path.
                let _ = conn.query(sql);
            }
            Ok(())
        })
    }

    pub fn create_search_indexes(&self) -> Result<()> {
        self.with_write_conn(|conn| {
            for sql in [
                "CALL CREATE_VECTOR_INDEX('Concept', 'concept_embedding', 'embedding', mu := 30, ml := 60, metric := 'cosine', efc := 200)",
                "CALL CREATE_FTS_INDEX('Concept', 'concept_fts', ['title', 'description', 'body'])",
                "CALL CREATE_VECTOR_INDEX('ImageAsset', 'image_omni_idx', 'embedding', mu := 30, ml := 60, metric := 'cosine', efc := 200)",
                "CALL CREATE_VECTOR_INDEX('Chunk', 'chunk_embedding', 'embedding', mu := 30, ml := 60, metric := 'cosine', efc := 200)",
                "CALL CREATE_FTS_INDEX('Chunk', 'chunk_fts', ['chunk_text'])",
            ] {
                // Already-exists is benign: indexes are maintained in place.
                let _ = conn.query(sql);
            }
            Ok(())
        })
    }

    /// Connections used by search must load the index extensions first.
    fn with_search_conn<T>(
        &self,
        f: impl for<'a> FnOnce(&Connection<'a>) -> Result<T>,
    ) -> Result<T> {
        let config = SystemConfig::default().read_only(true);
        let db = Database::new(&self.path, config).map_err(map_search_lbug)?;
        let conn = Connection::new(&db).map_err(map_search_lbug)?;
        for extension in ["vector", "fts"] {
            conn.query(&format!("LOAD {extension}"))
                .map_err(map_search_lbug)?;
        }
        f(&conn)
    }

    /// Vector index hits as `(id, 1 - distance)`, one row per node.
    pub fn vector_hits(
        &self,
        table: &str,
        index: &str,
        vector: &[f32],
        k: i64,
        efs: Option<i64>,
    ) -> Result<Vec<(String, f64)>> {
        Ok(self
            .vector_hits_raw(table, index, vector, k, efs)?
            .into_iter()
            .map(|(node, distance)| {
                (
                    node.get("id")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                    1.0 - distance,
                )
            })
            .collect())
    }

    /// Index hits with the full node and raw cosine distance, used by image
    /// search which exposes both ``distance`` and ``relevance_score``.
    pub fn vector_hits_raw(
        &self,
        table: &str,
        index: &str,
        vector: &[f32],
        k: i64,
        efs: Option<i64>,
    ) -> Result<Vec<(Json, f64)>> {
        let values = Value::List(
            LogicalType::Float,
            vector.iter().copied().map(Value::Float).collect(),
        );
        let sql = if efs.is_some() {
            format!(
                "CALL QUERY_VECTOR_INDEX('{table}', '{index}', $vec, $k, efs := $efs) \
                 RETURN node, distance"
            )
        } else {
            format!(
                "CALL QUERY_VECTOR_INDEX('{table}', '{index}', $vec, $k) \
                 RETURN node, distance"
            )
        };
        self.with_search_conn(|conn| {
            let mut stmt = conn.prepare(&sql).map_err(map_search_lbug)?;
            let mut params: Vec<(&str, Value)> = vec![("vec", values), ("k", Value::Int64(k))];
            if let Some(efs) = efs {
                params.push(("efs", Value::Int64(efs)));
            }
            let result = conn.execute(&mut stmt, params).map_err(map_search_lbug)?;
            let columns = result.get_column_names();
            let mut hits = Vec::new();
            for row in result {
                let mut node_json = None;
                let mut distance = 0.0;
                for (name, value) in columns.iter().zip(row) {
                    match name.as_str() {
                        "node" => node_json = Some(value_to_json(value)),
                        "distance" => distance = json_f64(&value_to_json(value)).unwrap_or(0.0),
                        _ => {}
                    }
                }
                if let Some(node) = node_json {
                    hits.push((node, distance));
                }
            }
            Ok(hits)
        })
    }

    /// FTS index hits as `(id, score)`, bounded with `top := k`.
    pub fn fts_hits(
        &self,
        table: &str,
        index: &str,
        query: &str,
        k: i64,
    ) -> Result<Vec<(String, f64)>> {
        let sql = format!(
            "CALL QUERY_FTS_INDEX('{table}', '{index}', $query, top := $k) \
             RETURN node, score"
        );
        self.with_search_conn(|conn| {
            let mut stmt = conn.prepare(&sql).map_err(map_search_lbug)?;
            let result = conn
                .execute(
                    &mut stmt,
                    vec![
                        ("query", Value::String(query.into())),
                        ("k", Value::Int64(k)),
                    ],
                )
                .map_err(map_search_lbug)?;
            let columns = result.get_column_names();
            let mut hits = Vec::new();
            for row in result {
                let mut node_json = None;
                let mut score = 0.0;
                for (name, value) in columns.iter().zip(row) {
                    match name.as_str() {
                        "node" => node_json = Some(value_to_json(value)),
                        "score" => score = json_f64(&value_to_json(value)).unwrap_or(0.0),
                        _ => {}
                    }
                }
                if let Some(id) = node_json
                    .as_ref()
                    .and_then(|n| n.get("id"))
                    .and_then(Json::as_str)
                {
                    hits.push((id.to_string(), score));
                }
            }
            Ok(hits)
        })
    }

    /// Metadata for ranked concept ids, preserving no inherent order.
    pub fn concepts_by_ids(&self, ids: &[String]) -> Result<Vec<Json>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.query(
            "MATCH (c:Concept) \
             WHERE c.id IN $ids \
             RETURN c.id AS id, c.title AS title, c.type AS type, \
                    c.description AS description, c.tags AS tags, c.extra AS extra, \
                    c.effective_at AS effective_at, c.body AS body",
            vec![(
                "ids",
                Value::List(
                    LogicalType::String,
                    ids.iter().cloned().map(Value::String).collect(),
                ),
            )],
        )?;
        Ok(rows.into_iter().map(Json::Object).collect())
    }

    /// Chunk rows for ranked chunk ids.
    pub fn chunks_by_ids(&self, ids: &[String]) -> Result<Vec<Json>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.query(
            "MATCH (k:Chunk) \
             WHERE k.id IN $ids \
             RETURN k.id AS id, k.parent_doc_id AS parent_doc_id, k.chunk_index AS chunk_index, \
                    k.chunk_text AS chunk_text, k.block_type AS block_type, \
                    k.start_offset AS start_offset, k.end_offset AS end_offset, \
                    k.embedding AS embedding",
            vec![(
                "ids",
                Value::List(
                    LogicalType::String,
                    ids.iter().cloned().map(Value::String).collect(),
                ),
            )],
        )?;
        Ok(rows.into_iter().map(Json::Object).collect())
    }

    /// Concept ids below `parent_id` up to `depth` CONTAINS hops.
    pub fn descendant_concepts(&self, parent_id: &str, depth: i64) -> Result<Vec<String>> {
        let depth = depth.clamp(1, 3);
        let sql = format!(
            "MATCH (:Directory {{id: $id}})-[:CONTAINS*1..{depth}]->(c:Concept) \
             RETURN DISTINCT c.id AS id"
        );
        Ok(self
            .query(&sql, vec![("id", Value::String(parent_id.into()))])?
            .into_iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect())
    }

    /// `near_id` plus every concept within `hops` LINKS_TO edges.
    pub fn near_ids(&self, near_id: &str, hops: i64) -> Result<Vec<String>> {
        let hops = hops.clamp(1, 3);
        let sql = format!(
            "MATCH (s:Concept {{id: $id}})-[:LINKS_TO*1..{hops}]-(c:Concept) \
             RETURN DISTINCT c.id AS id"
        );
        let mut ids: BTreeSet<String> = BTreeSet::new();
        ids.insert(near_id.to_string());
        for row in self.query(&sql, vec![("id", Value::String(near_id.into()))])? {
            if let Some(id) = row.get("id").and_then(Json::as_str) {
                ids.insert(id.to_string());
            }
        }
        Ok(ids.into_iter().collect())
    }

    /// Every concept id plus incoming link counts for hub/PPR ranking.
    pub fn link_graph(&self, link_kinds: Option<&[String]>) -> Result<LinkGraph> {
        let nodes = self
            .query("MATCH (c:Concept) RETURN c.id AS id", vec![])?
            .into_iter()
            .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
            .collect();
        let (sql, params) = if let Some(kinds) = link_kinds {
            (
                "MATCH (a:Concept)-[r:LINKS_TO]->(b:Concept) \
                 WHERE r.kind IN $kinds RETURN a.id AS src, b.id AS dst"
                    .to_string(),
                vec![(
                    "kinds",
                    Value::List(
                        LogicalType::String,
                        kinds.iter().cloned().map(Value::String).collect(),
                    ),
                )],
            )
        } else {
            (
                "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) \
                 RETURN a.id AS src, b.id AS dst"
                    .to_string(),
                vec![],
            )
        };
        let edges = self
            .query(&sql, params)?
            .into_iter()
            .filter_map(|row| {
                let src = row.get("src")?.as_str()?;
                let dst = row.get("dst")?.as_str()?;
                Some((src.to_string(), dst.to_string()))
            })
            .collect();
        Ok((nodes, edges))
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

    pub fn concept_raw(&self, concept_id: &str) -> Result<Option<Map<String, Json>>> {
        let rows = self.query(
            "MATCH (c:Concept {id: $id}) RETURN c.*",
            vec![("id", Value::String(concept_id.into()))],
        )?;
        Ok(rows.into_iter().next().map(|row| {
            row.into_iter()
                .map(|(key, value)| (key.strip_prefix("c.").unwrap_or(&key).to_string(), value))
                .collect()
        }))
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
                chunk.entry("rrf_score").or_insert(Json::Null);
                chunk.entry("hub_score").or_insert(Json::Null);
                chunk.entry("final_score").or_insert(Json::Null);
                chunk.entry("parent_title").or_insert(Json::Null);
                chunk.entry("parent_type").or_insert(Json::Null);
                chunk
                    .entry("parent_tags")
                    .or_insert_with(|| Json::Array(Vec::new()));
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

    /// Ancestry and sibling summary used by `read(include='context')`.
    pub fn context(&self, concept_id: &str, cap: i64) -> Result<(Vec<Json>, Vec<Json>)> {
        let ancestry = self.query(
            "MATCH (d:Directory)-[:CONTAINS*1..5]->(c:Concept {id: $cid}) \
             RETURN d.id AS id LIMIT 1",
            vec![("cid", Value::String(concept_id.into()))],
        )?;
        let parents = self.query(
            "MATCH (d:Directory)-[:CONTAINS]->(c:Concept {id: $cid}) \
             RETURN d.id AS parent_id",
            vec![("cid", Value::String(concept_id.into()))],
        )?;
        let (sql, params) = if let Some(parent) = parents
            .first()
            .and_then(|row| row.get("parent_id"))
            .and_then(Json::as_str)
        {
            (
                "MATCH (d:Directory {id: $pid})-[:CONTAINS]->(s:Concept) \
                 WHERE s.id <> $cid RETURN s.id AS id, s.title AS title, s.type AS type \
                 LIMIT $cap"
                    .to_string(),
                vec![
                    ("pid", Value::String(parent.to_string())),
                    ("cid", Value::String(concept_id.into())),
                    ("cap", Value::Int64(cap)),
                ],
            )
        } else {
            (
                "MATCH (c:Concept) \
                 WHERE c.id <> $cid \
                   AND NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(c) } \
                 RETURN c.id AS id, c.title AS title, c.type AS type LIMIT $cap"
                    .to_string(),
                vec![
                    ("cid", Value::String(concept_id.into())),
                    ("cap", Value::Int64(cap)),
                ],
            )
        };
        let siblings = self
            .query(&sql, params)?
            .into_iter()
            .map(Json::Object)
            .collect();
        Ok((ancestry.into_iter().map(Json::Object).collect(), siblings))
    }

    pub fn image_asset_fingerprint(&self, asset_id: &str) -> Result<Option<(String, bool)>> {
        let rows = self.query(
            "MATCH (i:ImageAsset {id: $iid}) \
             RETURN i.content_hash AS hash, i.data AS data",
            vec![("iid", Value::String(asset_id.into()))],
        )?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let hash = row
            .get("hash")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        let has_data = match row.get("data") {
            Some(Json::Null) | None => false,
            Some(Json::String(text)) => text != "b''",
            Some(Json::Array(items)) => !items.is_empty(),
            Some(_) => true,
        };
        Ok(Some((hash, has_data)))
    }

    pub fn image_asset_fingerprints_for_concept(
        &self,
        concept_id: &str,
    ) -> Result<Vec<(String, String)>> {
        let rows = self.query(
            "MATCH (c:Concept {id: $cid})-[:INCLUDES_ASSET]->(i:ImageAsset) \
             RETURN i.id AS id, i.content_hash AS hash",
            vec![("cid", Value::String(concept_id.into()))],
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                Some((
                    row.get("id")?.as_str()?.to_string(),
                    row.get("hash")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ))
            })
            .collect())
    }

    pub fn delete_image_asset(&self, concept_id: &str, asset_id: &str) -> Result<()> {
        self.write(
            "MATCH (c:Concept {id: $cid})-[r:INCLUDES_ASSET]->(i:ImageAsset {id: $iid}) \
             DELETE r \
             WITH i OPTIONAL MATCH (i)<-[:INCLUDES_ASSET]-(:Concept) \
             WITH i, count(*) AS n WHERE n = 0 DELETE i",
            vec![
                ("cid", Value::String(concept_id.into())),
                ("iid", Value::String(asset_id.into())),
            ],
        )
    }

    pub fn link_image_asset(&self, concept_id: &str, asset_id: &str) -> Result<()> {
        self.write(
            "MATCH (c:Concept {id: $cid}), (i:ImageAsset {id: $iid}) \
             MERGE (c)-[:INCLUDES_ASSET]->(i)",
            vec![
                ("cid", Value::String(concept_id.into())),
                ("iid", Value::String(asset_id.into())),
            ],
        )
    }

    pub fn count_image_assets(&self) -> Result<i64> {
        Ok(self
            .scalar("MATCH (i:ImageAsset) RETURN count(i) AS n", vec![])?
            .and_then(|value| value.as_i64())
            .unwrap_or(0))
    }

    pub fn reembed_texts(&self, table: &str) -> Result<Vec<(String, String)>> {
        let sql = if table == "concept" {
            "MATCH (c:Concept) RETURN c.id AS id, c.title AS t,              c.description AS d, c.body AS b"
        } else {
            "MATCH (k:Chunk) RETURN k.id AS id, k.chunk_text AS t"
        };
        let rows = self.query(sql, vec![])?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                let text = if table == "concept" {
                    [
                        row.get("t").and_then(Json::as_str).unwrap_or(""),
                        row.get("d").and_then(Json::as_str).unwrap_or(""),
                        row.get("b").and_then(Json::as_str).unwrap_or(""),
                    ]
                    .join(" ")
                } else {
                    row.get("t")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string()
                };
                (!text.trim().is_empty()).then_some((id, text))
            })
            .collect())
    }

    pub fn set_reembedded_vector(&self, table: &str, id: &str, vector: &[f32]) -> Result<()> {
        let value = Value::List(
            LogicalType::Float,
            vector.iter().copied().map(Value::Float).collect(),
        );
        let sql = if table == "concept" {
            "MATCH (c:Concept {id: $id}) SET c.embedding = $v"
        } else {
            "MATCH (k:Chunk {id: $id}) SET k.embedding = $v"
        };
        self.write(sql, vec![("id", Value::String(id.into())), ("v", value)])
    }

    pub fn drop_text_vector_indexes(&self) -> Result<()> {
        self.with_write_conn(|conn| {
            for sql in [
                "CALL DROP_VECTOR_INDEX('Concept', 'concept_embedding')",
                "CALL DROP_VECTOR_INDEX('Chunk', 'chunk_embedding')",
            ] {
                let _ = conn.query(sql);
            }
            Ok(())
        })
    }

    pub fn diagnostic_concepts(&self) -> Result<Vec<Map<String, Json>>> {
        self.query(
            "MATCH (c:Concept) RETURN c.id AS id, c.title AS title, \
             c.type AS type, c.body AS body",
            vec![],
        )
    }

    pub fn diagnostic_edges(&self) -> Result<Vec<(String, String)>> {
        let rows = self.query(
            "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) \
             RETURN a.id AS src, b.id AS dst",
            vec![],
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                Some((
                    row.get("src")?.as_str()?.to_string(),
                    row.get("dst")?.as_str()?.to_string(),
                ))
            })
            .collect())
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
            "MATCH (d:DeletedConcept) \
             RETURN d.original_id AS id, d.title AS title, d.deleted_at AS deleted_at, \
                    d.type AS type ORDER BY d.deleted_at DESC",
            vec![],
        )?;
        let now = crate::model::utc_now_datetime();
        Ok(rows
            .into_iter()
            .map(|row| {
                let deleted_at = row
                    .get("deleted_at")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_string();
                let parsed = crate::model::PyDateTime::parse(&deleted_at);
                let age = parsed
                    .map(|value| (now.as_naive() - value.as_naive()).as_seconds_f64())
                    .unwrap_or(0.0);
                serde_json::json!({
                    "concept_id": row.get("id").cloned().unwrap_or(Json::Null),
                    "title": row.get("title").cloned().unwrap_or(Json::Null),
                    "type": row.get("type").cloned().unwrap_or(Json::Null),
                    "deleted_at": deleted_at,
                    "age_seconds": age,
                    "recoverable": age <= 24.0 * 60.0 * 60.0,
                })
            })
            .collect())
    }

    pub fn list_broken_links(&self) -> Result<Vec<Json>> {
        let rows = self.query(
            "MATCH (b:BrokenLink) RETURN b.source_id AS source, b.target_id AS target, \
             b.timestamp AS timestamp, b.kind AS kind, b.reason AS reason, \
             b.candidates AS candidates, b.id AS id, b.raw AS raw \
             ORDER BY b.timestamp, b.id",
            vec![],
        )?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let candidates = row
                    .get("candidates")
                    .and_then(Json::as_str)
                    .and_then(|text| serde_json::from_str::<Vec<Json>>(text).ok())
                    .unwrap_or_default();
                let mut out = Map::new();
                for (key, value) in row {
                    if key != "candidates" {
                        out.insert(key, value);
                    }
                }
                out.insert("candidates".into(), Json::Array(candidates));
                out.insert(
                    "kind".into(),
                    Json::String(
                        out.get("kind")
                            .and_then(Json::as_str)
                            .filter(|value| !value.is_empty())
                            .unwrap_or("mention")
                            .to_string(),
                    ),
                );
                out.insert(
                    "reason".into(),
                    Json::String(
                        out.get("reason")
                            .and_then(Json::as_str)
                            .filter(|value| !value.is_empty())
                            .unwrap_or("missing")
                            .to_string(),
                    ),
                );
                Json::Object(out)
            })
            .collect())
    }

    pub fn clear_broken_from(&self, source: &str) -> Result<()> {
        self.write(
            "MATCH (b:BrokenLink {source_id: $s}) DELETE b",
            vec![("s", Value::String(source.into()))],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_broken_link(
        &self,
        source: &str,
        recorded: &str,
        kind: &str,
        reason: &str,
        candidates_json: &str,
        raw: &str,
        timestamp: Value,
    ) -> Result<()> {
        let mut id = format!("{source}\u{2192}{recorded}");
        if kind != "mention" {
            id.push_str(&format!("#{kind}"));
        }
        self.write(
            "MERGE (b:BrokenLink {id: $id}) SET b.source_id = $s, b.target_id = $t, \
             b.timestamp = $ts, b.kind = $k, b.reason = $r, b.candidates = $c, b.raw = $raw",
            vec![
                ("id", Value::String(id)),
                ("s", Value::String(source.into())),
                ("t", Value::String(recorded.into())),
                ("ts", timestamp),
                ("k", Value::String(kind.into())),
                ("r", Value::String(reason.into())),
                (
                    "c",
                    if candidates_json.is_empty() {
                        Value::Null(LogicalType::String)
                    } else {
                        Value::String(candidates_json.into())
                    },
                ),
                ("raw", Value::String(raw.into())),
            ],
        )
    }

    pub fn update_broken_reason(
        &self,
        id: &str,
        reason: &str,
        candidates_json: &str,
    ) -> Result<()> {
        self.write(
            "MATCH (b:BrokenLink {id: $id}) SET b.reason = $r, b.candidates = $c",
            vec![
                ("id", Value::String(id.into())),
                ("r", Value::String(reason.into())),
                (
                    "c",
                    if candidates_json.is_empty() {
                        Value::Null(LogicalType::String)
                    } else {
                        Value::String(candidates_json.into())
                    },
                ),
            ],
        )
    }

    pub fn delete_broken_link(&self, id: &str) -> Result<()> {
        self.write(
            "MATCH (b:BrokenLink {id: $id}) DELETE b",
            vec![("id", Value::String(id.into()))],
        )
    }

    pub fn link_concepts(&self, source: &str, target: &str, kind: &str) -> Result<()> {
        self.write(
            "MATCH (a:Concept {id: $s}), (b:Concept {id: $t}) \
             MERGE (a)-[:LINKS_TO {kind: $k}]->(b)",
            vec![
                ("s", Value::String(source.into())),
                ("t", Value::String(target.into())),
                ("k", Value::String(kind.into())),
            ],
        )
    }

    pub fn incoming_links(&self, ids: &[&str]) -> Result<Vec<(String, String, String)>> {
        let values = ids.iter().map(|id| Value::String((*id).into())).collect();
        let rows = self.query(
            "MATCH (s:Concept)-[r:LINKS_TO]->(c:Concept) \
             WHERE c.id IN $ids AND NOT s.id IN $ids \
             RETURN s.id AS source, c.id AS target, r.kind AS kind",
            vec![("ids", Value::List(LogicalType::String, values))],
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                Some((
                    row.get("source")?.as_str()?.to_string(),
                    row.get("target")?.as_str()?.to_string(),
                    row.get("kind")
                        .and_then(Json::as_str)
                        .filter(|value| !value.is_empty())
                        .unwrap_or("mention")
                        .to_string(),
                ))
            })
            .collect())
    }

    pub fn export_concepts(&self) -> Result<Vec<Json>> {
        let rows = self.query("MATCH (c:Concept) RETURN c.*", vec![])?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let flat: Map<String, Json> = row
                    .into_iter()
                    .map(|(key, value)| (key.strip_prefix("c.").unwrap_or(&key).to_string(), value))
                    .collect();
                public_concept(&flat)
            })
            .collect())
    }

    pub fn link_titles(&self, concept_id: &str, outgoing: bool) -> Result<Vec<(String, String)>> {
        let sql = if outgoing {
            "MATCH (s:Concept {id: $id})-[:LINKS_TO]->(t:Concept) \
             RETURN t.id AS id, t.title AS title ORDER BY t.title"
        } else {
            "MATCH (s:Concept)-[:LINKS_TO]->(t:Concept {id: $id}) \
             RETURN s.id AS id, s.title AS title ORDER BY s.title"
        };
        Ok(self
            .query(sql, vec![("id", Value::String(concept_id.into()))])?
            .into_iter()
            .filter_map(|row| {
                Some((
                    row.get("id")?.as_str()?.to_string(),
                    row.get("title")
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ))
            })
            .collect())
    }

    pub fn link_index(&self) -> Result<Vec<Json>> {
        Ok(self
            .query(
                "MATCH (c:Concept) RETURN c.id AS id, c.title AS title, c.extra AS extra",
                vec![],
            )?
            .into_iter()
            .map(Json::Object)
            .collect())
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

    /// Persisted directory hashes for delta detection.
    pub fn dir_hashes(&self) -> Result<BTreeMap<String, (String, String)>> {
        let rows = self.query(
            "MATCH (d:DirHash) RETURN d.path AS path, d.hash AS hash, d.files AS files",
            vec![],
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let path = row.get("path")?.as_str()?.to_string();
                let hash = row
                    .get("hash")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_string();
                let files = row
                    .get("files")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_string();
                Some((path, (hash, files)))
            })
            .collect())
    }

    pub fn set_dir_hash(&self, path: &str, hash: &str, files_json: &str) -> Result<()> {
        self.write(
            "MERGE (d:DirHash {path: $p}) SET d.hash = $h, d.files = $f",
            vec![
                ("p", Value::String(path.into())),
                ("h", Value::String(hash.into())),
                ("f", Value::String(files_json.into())),
            ],
        )
    }

    /// Persisted file hashes for delta detection: path -> (hash, concept id).
    pub fn file_hashes(&self) -> Result<BTreeMap<String, (String, String)>> {
        let rows = self.query(
            "MATCH (f:FileHash) RETURN f.path AS path, f.hash AS hash, f.concept_id AS cid",
            vec![],
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let path = row.get("path")?.as_str()?.to_string();
                let hash = row
                    .get("hash")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_string();
                let cid = row
                    .get("cid")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_string();
                Some((path, (hash, cid)))
            })
            .collect())
    }

    pub fn set_file_hash(&self, path: &str, hash: &str, concept_id: &str, dir: &str) -> Result<()> {
        self.write(
            "MERGE (f:FileHash {path: $p}) SET f.hash = $h, f.concept_id = $c, f.dir = $d",
            vec![
                ("p", Value::String(path.into())),
                ("h", Value::String(hash.into())),
                ("c", Value::String(concept_id.into())),
                ("d", Value::String(dir.into())),
            ],
        )
    }

    pub fn dir_hash_files(&self, path: &str) -> Result<Option<String>> {
        Ok(self
            .scalar(
                "MATCH (d:DirHash {path: $p}) RETURN d.files AS f",
                vec![("p", Value::String(path.into()))],
            )?
            .and_then(|value| value.as_str().map(str::to_string)))
    }

    pub fn set_dir_hash_files(&self, path: &str, files_json: &str) -> Result<()> {
        self.write(
            "MATCH (d:DirHash {path: $p}) SET d.files = $f",
            vec![
                ("p", Value::String(path.into())),
                ("f", Value::String(files_json.into())),
            ],
        )
    }

    pub fn delete_dir_hash(&self, path: &str) -> Result<()> {
        self.write(
            "MATCH (d:DirHash {path: $p}) DELETE d",
            vec![("p", Value::String(path.into()))],
        )
    }

    pub fn delete_deleted_path(&self, path: &str) -> Result<()> {
        self.write(
            "MATCH (d:DeletedPath {path: $p}) DELETE d",
            vec![("p", Value::String(path.into()))],
        )
    }

    pub fn live_dir_keys(&self) -> Result<BTreeSet<String>> {
        let mut out = BTreeSet::new();
        for sql in [
            "MATCH (f:FileHash) RETURN f.dir AS dir",
            "MATCH (d:DeletedPath) RETURN d.dir AS dir",
        ] {
            for row in self.query(sql, vec![])? {
                if let Some(dir) = row.get("dir").and_then(Json::as_str) {
                    out.insert(dir.to_string());
                }
            }
        }
        Ok(out)
    }

    pub fn forget_file_hash(&self, path: &str) -> Result<()> {
        self.write(
            "MATCH (f:FileHash {path: $p}) DELETE f",
            vec![("p", Value::String(path.into()))],
        )
    }

    pub fn record_deleted_path(&self, path: &str, detected_at: i64) -> Result<()> {
        let dir = Path::new(path)
            .parent()
            .map(|value| value.to_string_lossy().into_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| ".".into());
        self.write(
            "MERGE (d:DeletedPath {path: $p}) SET d.detected_at = $t, d.dir = $d",
            vec![
                ("p", Value::String(path.into())),
                ("t", Value::Int64(detected_at)),
                ("d", Value::String(dir)),
            ],
        )
    }

    pub fn meta_text(&self, key: &str) -> Result<Option<String>> {
        let value = self.scalar(
            "MATCH (m:MetaText {key: $key}) RETURN m.value AS v",
            vec![("key", Value::String(key.into()))],
        )?;
        Ok(value.and_then(|v| v.as_str().map(str::to_string)))
    }

    pub fn meta_int(&self, key: &str) -> Result<Option<i64>> {
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
            .unwrap_or_else(|| crate::embed::default_image_model_id(&self.settings.model_id));
        let pinned_image_model = self.meta_text("embedding_image_model")?;
        let pinned_image_precision = match self.meta_int("embedding_image_precision")? {
            Some(8) => Some("int8".to_string()),
            Some(16) => Some("fp16".to_string()),
            Some(32) => Some("fp32".to_string()),
            Some(other) => Some(other.to_string()),
            None => None,
        };
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
                "pinned_precision": pinned_image_precision,
                "pinned_space": self.meta_text("embedding_image_space")?,
                "pinned_text_space": self.meta_text("embedding_image_text_space")?,
                "backend": self.settings.image_backend,
                "configured_model": configured_image_model,
                "configured_precision": if self.settings.image_precision == "auto" {
                    None
                } else {
                    Some(self.settings.image_precision.clone())
                },
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

fn map_search_lbug(err: lbug::Error) -> OkfError {
    OkfError::new(
        "SEARCH_UNAVAILABLE",
        format!("vector/fts index query failed: {err}"),
    )
    .remedy("reopen the router so the vector/fts extensions load")
}

fn json_f64(value: &Json) -> Option<f64> {
    match value {
        Json::Number(n) => n.as_f64(),
        _ => None,
    }
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
            let mut iso = format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                utc.year(),
                utc.month() as u8,
                utc.day(),
                utc.hour(),
                utc.minute(),
                utc.second()
            );
            let micros = utc.nanosecond() / 1000;
            if micros != 0 {
                iso.push_str(&format!(".{micros:06}"));
            }
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
