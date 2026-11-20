//! Soft-delete tombstones and recovery for the Stage B write path.

use crate::error::{OkfError, Result};
use crate::model::{local_now_datetime, normalize_tags, PyDateTime};
use crate::pycompat::PyValue;
use crate::store::Store;
use lbug::{LogicalType, Value};
use serde_json::{Map, Value as Json};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SOFT_DELETE_WINDOW: f64 = 24.0 * 60.0 * 60.0;

pub fn soft_delete(store: &Store, concept_id: &str) -> Result<bool> {
    let Some(raw) = store.concept_raw(concept_id)? else {
        return Ok(false);
    };
    let now = local_now_datetime();
    let deleted_at = now.isoformat();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!("deleted_{concept_id}_{stamp}");
    let title = raw.get("title").and_then(Json::as_str).unwrap_or_default();
    let concept_type = raw.get("type").and_then(Json::as_str).unwrap_or_default();
    let body = raw.get("body").and_then(Json::as_str).unwrap_or_default();
    let snapshot = snapshot_json(&raw);
    let tags = PyValue::List(
        normalize_tags(raw.get("tags").unwrap_or(&Json::Null))
            .into_iter()
            .map(PyValue::Str)
            .collect(),
    )
    .to_python_json(false);
    store.write(
        "CREATE (d:DeletedConcept {id: $id, original_id: $oid, title: $title, \
         body: $body, deleted_at: $deleted_at, type: $type, tags: $tags, snapshot: $snap})",
        vec![
            ("id", Value::String(id)),
            ("oid", Value::String(concept_id.into())),
            ("title", maybe_string(Some(title))),
            ("body", Value::String(body.into())),
            ("deleted_at", Value::String(deleted_at)),
            ("type", Value::String(concept_type.into())),
            ("tags", Value::String(tags)),
            ("snap", Value::String(snapshot)),
        ],
    )?;
    purge_concept(store, concept_id)?;
    Ok(true)
}

/// Trashed files recorded on the latest tombstone: `(trash, source)`.
pub fn read_trashed(store: &Store, concept_id: &str) -> Result<Vec<(String, String)>> {
    let rows = store.query(
        "MATCH (d:DeletedConcept {original_id: $id})          RETURN d.snapshot AS snapshot ORDER BY d.deleted_at DESC",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    let Some(snapshot) = rows
        .first()
        .and_then(|row| row.get("snapshot"))
        .and_then(Json::as_str)
    else {
        return Ok(Vec::new());
    };
    let Ok(snapshot) = serde_json::from_str::<Map<String, Json>>(snapshot) else {
        return Ok(Vec::new());
    };
    Ok(snapshot
        .get("trashed_files")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let trash = entry.get("trash").and_then(Json::as_str)?;
            let source = entry.get("source").and_then(Json::as_str)?;
            Some((trash.to_string(), source.to_string()))
        })
        .collect())
}

/// Store the trash list on the latest tombstone so recovery can restore it.
pub fn remember_trashed(
    store: &Store,
    concept_id: &str,
    trashed: &[(String, String)],
) -> Result<()> {
    if trashed.is_empty() {
        return Ok(());
    }
    let rows = store.query(
        "MATCH (d:DeletedConcept {original_id: $id})          RETURN d.id AS id, d.snapshot AS snapshot ORDER BY d.deleted_at DESC",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    let Some(row) = rows.first() else {
        return Ok(());
    };
    let Some(tombstone_id) = row.get("id").and_then(Json::as_str) else {
        return Ok(());
    };
    let snapshot = row.get("snapshot").and_then(Json::as_str).unwrap_or("{}");
    let mut snapshot: Map<String, Json> = serde_json::from_str(snapshot).unwrap_or_default();
    snapshot.insert(
        "trashed_files".into(),
        Json::Array(
            trashed
                .iter()
                .map(|(trash, source)| {
                    Json::Object(Map::from_iter([
                        ("trash".into(), Json::String(trash.clone())),
                        ("source".into(), Json::String(source.clone())),
                    ]))
                })
                .collect(),
        ),
    );
    let serialized = serde_json::to_string(&snapshot)
        .map_err(|err| OkfError::internal(err.to_string(), "serde_json::Error"))?;
    store.write(
        "MATCH (d:DeletedConcept {id: $id}) SET d.snapshot = $snapshot",
        vec![
            ("id", Value::String(tombstone_id.into())),
            ("snapshot", Value::String(serialized)),
        ],
    )
}

pub fn recover(store: &Store, concept_id: &str) -> Result<bool> {
    let rows = store.query(
        "MATCH (d:DeletedConcept {original_id: $id}) \
         RETURN d.snapshot AS snapshot ORDER BY d.deleted_at DESC",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    let Some(snapshot) = rows
        .first()
        .and_then(|row| row.get("snapshot"))
        .and_then(Json::as_str)
    else {
        return Ok(false);
    };
    let snapshot: Map<String, Json> = serde_json::from_str(snapshot).map_err(|err| {
        OkfError::internal(
            format!("invalid recovery snapshot: {err}"),
            "serde_json::Error",
        )
        .remedy("purge the corrupt deleted snapshot and re-import the file")
    })?;
    let id = snapshot
        .get("id")
        .and_then(Json::as_str)
        .unwrap_or(concept_id)
        .to_string();
    let mut params: Vec<(&str, Value)> = vec![
        ("id", Value::String(id)),
        (
            "type",
            Value::String(
                snapshot
                    .get("type")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .into(),
            ),
        ),
        (
            "title",
            maybe_string(snapshot.get("title").and_then(Json::as_str)),
        ),
        (
            "description",
            maybe_string(snapshot.get("description").and_then(Json::as_str)),
        ),
        (
            "resource",
            maybe_string(snapshot.get("resource").and_then(Json::as_str)),
        ),
        (
            "tags",
            Value::List(
                LogicalType::String,
                normalize_tags(snapshot.get("tags").unwrap_or(&Json::Null))
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ),
        ),
        ("timestamp", json_timestamp(snapshot.get("timestamp"))),
        (
            "body",
            Value::String(
                snapshot
                    .get("body")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .into(),
            ),
        ),
        (
            "embedding",
            Value::List(
                LogicalType::Float,
                snapshot
                    .get("embedding")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|value| value.as_f64())
                    .map(|value| Value::Float(value as f32))
                    .collect(),
            ),
        ),
    ];
    let mut sql = String::from(
        "CREATE (:Concept {id: $id, type: $type, title: $title, \
         description: $description, resource: $resource, tags: $tags, \
         timestamp: $timestamp, body: $body, embedding: $embedding",
    );
    let mut fields = Vec::new();
    for key in ["created_at", "updated_at", "effective_at", "content_sig"] {
        let value = if key == "content_sig" {
            json_to_lbug(snapshot.get(key).unwrap_or(&Json::Null))
        } else {
            json_timestamp(snapshot.get(key))
        };
        params.push((box_key(key), value));
        fields.push(format!("{key}: ${key}"));
    }
    if !fields.is_empty() {
        sql.push_str(", ");
        sql.push_str(&fields.join(", "));
    }
    if let Some(extra) = snapshot.get("extra").and_then(Json::as_object) {
        if !extra.is_empty() {
            let (keys, values): (Vec<_>, Vec<_>) = extra
                .iter()
                .map(|(key, value)| {
                    (
                        Value::String(key.clone()),
                        Value::String(value.as_str().unwrap_or(&value.to_string()).into()),
                    )
                })
                .unzip();
            params.push(("extra_keys", Value::List(LogicalType::String, keys)));
            params.push(("extra_values", Value::List(LogicalType::String, values)));
            sql.push_str(", extra: MAP($extra_keys, $extra_values)");
        }
    }
    sql.push_str("})");
    store.write(&sql, params)?;
    store.write(
        "MATCH (d:DeletedConcept {original_id: $id}) DELETE d",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    Ok(true)
}

pub fn purge_deleted(store: &Store, older_than: Option<i64>) -> Result<usize> {
    let window = older_than.unwrap_or(SOFT_DELETE_WINDOW as i64) as f64;
    let now = local_now_datetime();
    let rows = store.query(
        "MATCH (d:DeletedConcept) RETURN d.original_id AS id, d.deleted_at AS deleted_at",
        vec![],
    )?;
    let mut count = 0;
    for row in rows {
        let deleted_at = row
            .get("deleted_at")
            .and_then(Json::as_str)
            .unwrap_or_default();
        let age = PyDateTime::parse(deleted_at)
            .map(|value| (now.as_naive() - value.as_naive()).as_seconds_f64())
            .unwrap_or(0.0);
        if age <= window {
            continue;
        }
        let id = row.get("id").and_then(Json::as_str).unwrap_or_default();
        let _ = purge_concept(store, id);
        store.write(
            "MATCH (d:DeletedConcept {original_id: $id}) DELETE d",
            vec![("id", Value::String(id.into()))],
        )?;
        count += 1;
    }
    Ok(count)
}

pub fn purge_concept(store: &Store, concept_id: &str) -> Result<()> {
    let incoming = store.incoming_links(&[concept_id])?;
    let assets = store.query(
        "MATCH (c:Concept {id: $id})-[:INCLUDES_ASSET]->(i:ImageAsset) RETURN i.id AS id",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    let paths = store.query(
        "MATCH (f:FileHash {concept_id: $id}) RETURN f.path AS path, f.dir AS dir",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    store.write(
        "MATCH (ch:Chunk {parent_doc_id: $id}) DETACH DELETE ch",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    store.write(
        "MATCH (c:Concept {id: $id}) DETACH DELETE c",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    for row in assets {
        let asset_id = row.get("id").and_then(Json::as_str).unwrap_or_default();
        let refs = store.query(
            "MATCH (i:ImageAsset {id: $id}) OPTIONAL MATCH (i)<-[:INCLUDES_ASSET]-(c:Concept) \
             RETURN count(c) AS n",
            vec![("id", Value::String(asset_id.into()))],
        )?;
        let refs = refs
            .first()
            .and_then(|row| row.get("n"))
            .and_then(Json::as_i64)
            .unwrap_or(0);
        if refs == 0 {
            store.write(
                "MATCH (i:ImageAsset {id: $id}) DELETE i",
                vec![("id", Value::String(asset_id.into()))],
            )?;
        }
    }
    store.write(
        "MATCH (b:BrokenLink {source_id: $id}) DELETE b",
        vec![("id", Value::String(concept_id.into()))],
    )?;
    let now = local_now_datetime();
    for (source, _, kind) in incoming {
        let mut link_id = format!("{source}\u{2192}{concept_id}");
        if kind != "mention" {
            link_id.push_str(&format!("#{kind}"));
        }
        store.write(
            "MERGE (b:BrokenLink {id: $id}) SET b.source_id = $s, b.target_id = $t, \
             b.timestamp = $ts, b.kind = $k, b.reason = 'missing', b.candidates = NULL",
            vec![
                ("id", Value::String(link_id)),
                ("s", Value::String(source)),
                ("t", Value::String(concept_id.into())),
                ("ts", Value::Timestamp(now.as_naive().assume_utc())),
                ("k", Value::String(kind)),
            ],
        )?;
    }
    let mut parents = Vec::new();
    for row in &paths {
        let Some(path) = row.get("path").and_then(Json::as_str) else {
            continue;
        };
        let parent = row
            .get("dir")
            .and_then(Json::as_str)
            .unwrap_or_else(|| {
                std::path::Path::new(path)
                    .parent()
                    .and_then(|value| value.to_str())
                    .unwrap_or(".")
            })
            .to_string();
        store.delete_deleted_path(path)?;
        if let Some(files) = store.dir_hash_files(&parent)? {
            let mut names: Vec<String> = serde_json::from_str(&files).unwrap_or_default();
            let name = std::path::Path::new(path)
                .file_name()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_default();
            names.retain(|value| value != &name);
            store.set_dir_hash_files(
                &parent,
                &PyValue::List(names.into_iter().map(PyValue::Str).collect()).to_python_json(false),
            )?;
        }
        store.forget_file_hash(path)?;
        parents.push(parent);
    }
    if !parents.is_empty() {
        let live = store.live_dir_keys()?;
        for parent in parents {
            if !live.contains(&parent) {
                store.delete_dir_hash(&parent)?;
            }
        }
    }
    Ok(())
}

fn snapshot_json(raw: &Map<String, Json>) -> String {
    let mut fields = Vec::new();
    let mut push = |key: &str, value: PyValue| {
        fields.push((key.to_string(), value));
    };
    push(
        "id",
        PyValue::from_golden(raw.get("id").unwrap_or(&Json::Null)),
    );
    push(
        "type",
        PyValue::from_golden(raw.get("type").unwrap_or(&Json::Null)),
    );
    push(
        "title",
        PyValue::from_golden(raw.get("title").unwrap_or(&Json::Null)),
    );
    push(
        "description",
        PyValue::from_golden(raw.get("description").unwrap_or(&Json::Null)),
    );
    push(
        "resource",
        PyValue::from_golden(raw.get("resource").unwrap_or(&Json::Null)),
    );
    push(
        "tags",
        PyValue::List(
            normalize_tags(raw.get("tags").unwrap_or(&Json::Null))
                .into_iter()
                .map(PyValue::Str)
                .collect(),
        ),
    );
    push(
        "timestamp",
        PyValue::from_golden(raw.get("timestamp").unwrap_or(&Json::Null)),
    );
    push(
        "body",
        PyValue::from_golden(raw.get("body").unwrap_or(&Json::Null)),
    );
    push(
        "embedding",
        PyValue::from_golden(raw.get("embedding").unwrap_or(&Json::Null)),
    );
    push(
        "extra",
        PyValue::from_golden(raw.get("extra").unwrap_or(&Json::Null)),
    );
    for key in ["created_at", "updated_at", "effective_at", "content_sig"] {
        push(
            key,
            PyValue::from_golden(raw.get(key).unwrap_or(&Json::Null)),
        );
    }
    PyValue::Dict(fields).to_python_json(false)
}

fn box_key(key: &str) -> &'static str {
    match key {
        "created_at" => "created_at",
        "updated_at" => "updated_at",
        "effective_at" => "effective_at",
        _ => "content_sig",
    }
}

fn maybe_string(value: Option<&str>) -> Value {
    value
        .map(|text| Value::String(text.to_string()))
        .unwrap_or_else(|| Value::Null(LogicalType::String))
}

fn json_timestamp(value: Option<&Json>) -> Value {
    match value.and_then(Json::as_str).map(PyDateTime::parse) {
        Some(Ok(dt)) => Value::Timestamp(dt.as_naive().assume_utc()),
        _ => Value::Null(LogicalType::Timestamp),
    }
}

fn json_to_lbug(value: &Json) -> Value {
    match value {
        Json::Null => Value::Null(LogicalType::String),
        Json::Bool(flag) => Value::Bool(*flag),
        Json::Number(number) if number.is_i64() => Value::Int64(number.as_i64().unwrap_or(0)),
        Json::Number(number) => Value::Float(number.as_f64().unwrap_or(0.0) as f32),
        Json::String(text) => Value::String(text.clone()),
        Json::Array(items) => Value::List(
            LogicalType::String,
            items
                .iter()
                .map(|item| Value::String(item.as_str().unwrap_or(&item.to_string()).into()))
                .collect(),
        ),
        Json::Object(_) => Value::String(value.to_string()),
    }
}
