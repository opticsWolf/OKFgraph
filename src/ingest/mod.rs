//! Walk/parse/import pipeline. Stage B implements markdown import parity for
//! the differential corpus: frontmatter identity, synthesized titles/types,
//! chunk rows, file hashes, directory hierarchy, and wikilink resolution.

use crate::embed::{check_pins, Embedder, EmbedderPool, Task};
use crate::error::{OkfError, Result};
use crate::markdown::{parse_frontmatter, split_into_chunks, Chunk as MdChunk};
use crate::model::{extra_value, normalize_tags, utc_now_datetime, PyDateTime};
use crate::pycompat::{content_sig, python_datetime_isoformat, PyValue};
use crate::settings::Settings;
use crate::store::Store;
use lbug::{LogicalType, Value};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const STRUCTURAL_BLOCKS: &[&str] = &[
    "Heading",
    "CodeBlock",
    "List",
    "Blockquote",
    "Table",
    "Diagram",
];

#[derive(Debug)]
struct Parsed {
    id: String,
    title: String,
    concept_type: String,
    description: Option<String>,
    resource: Option<String>,
    tags: Vec<String>,
    timestamp: Option<PyDateTime>,
    authored: BTreeMap<String, PyValue>,
    body: String,
    extra: BTreeMap<String, String>,
    sig: String,
    path: PathBuf,
}

pub fn import_files(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    paths: &[String],
) -> Result<Json> {
    let mut missing = Vec::new();
    for path in paths {
        if !Path::new(path).is_file() {
            missing.push(path.clone());
        }
    }
    if !missing.is_empty() {
        return Err(OkfError::new(
            "FILE_NOT_FOUND",
            format!("file(s) not found: {}", missing.join(", ")),
        )
        .op("import_files")
        .field(
            "files",
            Json::Array(missing.into_iter().map(Json::String).collect()),
        ));
    }

    let engine = embedder.get(settings)?;
    pin_graph(store, settings, engine.as_ref())?;
    let parsed: Vec<Parsed> = paths
        .iter()
        .map(|path| parse_file(Path::new(path), settings))
        .collect::<Result<Vec<_>>>()?;
    let mut ids = Vec::new();
    let mut images = Map::new();
    for item in &parsed {
        import_one(store, settings, engine.as_ref(), item)?;
        ids.push(Json::String(item.id.clone()));
        images.insert(item.id.clone(), Json::from(0));
    }
    create_links(store, &parsed)?;
    create_directories(store, &parsed)?;
    for name in [
        "schema_version",
        "embedding_precision",
        "write_epoch",
        "indexed_epoch",
    ] {
        // No-op read keeps the pin contract visible to callers/debuggers.
        let _ = store.meta_int(name)?;
    }
    let epoch = store.meta_int("write_epoch")?.unwrap_or(0);
    store.create_search_indexes()?;
    store.set_meta_int("indexed_epoch", epoch)?;
    Ok(serde_json::json!({
        "concept_ids": ids,
        "images": Json::Object(images),
    }))
}

fn pin_graph(store: &Store, settings: &Settings, engine: &dyn Embedder) -> Result<()> {
    if store.meta_int("schema_version")?.is_none() {
        return Err(OkfError::new(
            "STATE_INVALID",
            "database has no v10 schema; call okf init first",
        ));
    }
    let pins = store_pin_json(store)?;
    check_pins(settings, &pins)?;
    let code = match engine.precision() {
        "fp16" => 16,
        "fp32" => 32,
        "int8" => 8,
        other => {
            return Err(OkfError::new(
                "PRECISION_PIN_MISMATCH",
                format!("cannot pin unknown precision {other:?}"),
            ))
        }
    };
    if store.meta_int("embedding_precision")?.is_none() {
        store.set_meta_int("embedding_precision", code)?;
    }
    if store.meta_text("embedding_model")?.is_none() {
        store.set_meta_text("embedding_model", engine.model_id())?;
    }
    if store.meta_text("embedding_space")?.is_none() {
        if let Some(space) = engine.space() {
            store.set_meta_text("embedding_space", &space)?;
        }
    }
    Ok(())
}

fn store_pin_json(store: &Store) -> Result<Json> {
    let precision = match store.meta_int("embedding_precision")? {
        Some(16) => Json::String("fp16".into()),
        Some(32) => Json::String("fp32".into()),
        Some(8) => Json::String("int8".into()),
        _ => Json::Null,
    };
    Ok(serde_json::json!({
        "pinned": {
            "model_id": store.meta_text("embedding_model")?,
            "precision": precision,
        },
        "empty": store.counts()?.0 == 0,
    }))
}

fn relative_id(path: &Path, settings: &Settings) -> String {
    let root = settings.bundle_root.as_ref().map(PathBuf::from);
    if let Some(root) = root {
        if let Ok(rel) = path.strip_prefix(&root) {
            return rel.with_extension("").to_string_lossy().replace('\\', "/");
        }
    }
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn parse_file(path: &Path, settings: &Settings) -> Result<Parsed> {
    let text = fs::read_to_string(path)?;
    let front = parse_frontmatter(&text);
    let mut meta = front.metadata.as_object().cloned().unwrap_or_default();
    let id = relative_id(path, settings);
    if id.is_empty() {
        return Err(OkfError::new("BAD_VALUE", "empty concept id").op("import_files"));
    }
    let title = string_field(&meta, "title")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            path.file_stem()
                .map(|stem| {
                    stem.to_string_lossy()
                        .replace(['_', '-'], " ")
                        .trim()
                        .to_string()
                })
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| id.clone())
        });
    let concept_type = string_field(&meta, "type")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "note".into());
    let description = string_field(&meta, "description");
    let resource = string_field(&meta, "resource");
    let tags = meta.get("tags").map(normalize_tags).unwrap_or_default();
    let timestamp = meta.get("timestamp").and_then(parse_tagged_datetime);

    let mut authored = BTreeMap::new();
    for (key, value) in &meta {
        if matches!(key.as_str(), "body" | "embedding") {
            continue;
        }
        authored.insert(key.clone(), PyValue::from_golden(value));
    }
    let mut sig_data = authored.clone();
    let fm_id = meta.remove("id");
    if let Some(value) = &fm_id {
        let written = value.as_str().unwrap_or_default();
        if !written.is_empty() && written != id {
            sig_data.remove("id");
            sig_data.insert("uid".into(), PyValue::from_golden(value));
        }
    }
    sig_data.insert("id".into(), PyValue::Str(id.clone()));
    sig_data.insert("type".into(), PyValue::Str(concept_type.clone()));
    sig_data.insert("title".into(), PyValue::Str(title.clone()));
    sig_data.insert(
        "description".into(),
        description
            .as_ref()
            .map(|value| PyValue::Str(value.clone()))
            .unwrap_or(PyValue::Null),
    );
    sig_data.insert(
        "resource".into(),
        resource
            .as_ref()
            .map(|value| PyValue::Str(value.clone()))
            .unwrap_or(PyValue::Null),
    );
    sig_data.insert(
        "tags".into(),
        PyValue::List(tags.iter().cloned().map(PyValue::Str).collect()),
    );
    sig_data.insert(
        "timestamp".into(),
        timestamp
            .map(|value| PyValue::DateTime(value.isoformat()))
            .unwrap_or(PyValue::Null),
    );
    let sig = content_sig(
        &PyValue::Dict(sig_data.into_iter().collect()),
        &front.content,
    );

    let mut extra = BTreeMap::new();
    if let Some(value) = fm_id {
        let written = value.as_str().unwrap_or_default();
        if written != id && !written.is_empty() {
            extra.insert("uid".into(), value_to_extra(&value));
        }
    }
    for (key, value) in &meta {
        if [
            "type",
            "title",
            "description",
            "resource",
            "tags",
            "timestamp",
        ]
        .contains(&key.as_str())
        {
            continue;
        }
        extra.insert(key.clone(), value_to_extra(value));
    }

    Ok(Parsed {
        id,
        title,
        concept_type,
        description,
        resource,
        tags,
        timestamp,
        authored,
        body: front.content,
        extra,
        sig,
        path: path.to_path_buf(),
    })
}

fn string_field(meta: &Map<String, Json>, key: &str) -> Option<String> {
    meta.get(key).and_then(Json::as_str).map(str::to_string)
}

fn parse_tagged_datetime(value: &Json) -> Option<PyDateTime> {
    let obj = value.as_object()?;
    let kind = obj.get("__type__")?.as_str()?;
    let text = obj.get("value")?.as_str()?;
    match kind {
        "datetime" => PyDateTime::parse(text).ok(),
        "date" => PyDateTime::parse(&format!("{text}T00:00:00")).ok(),
        _ => None,
    }
}

fn value_to_extra(value: &Json) -> String {
    let tagged = PyValue::from_golden(value);
    match &tagged {
        PyValue::Str(text) => text.clone(),
        PyValue::Date(text) | PyValue::DateTime(text) => {
            python_datetime_isoformat(text.as_str()).replace('T', " ")
        }
        other => extra_value(other),
    }
}

fn timestamp_value(value: &PyDateTime) -> Value {
    Value::Timestamp(value.as_naive().assume_utc())
}

fn effective_at(item: &Parsed, updated: &PyDateTime) -> PyDateTime {
    for key in ["timestamp", "updated", "date", "created"] {
        if let Some(value) = item.authored.get(key) {
            if let Some(parsed) = parse_tagged_datetime(&tag_to_json(value)) {
                return parsed;
            }
        }
    }
    *updated
}

fn tag_to_json(value: &PyValue) -> Json {
    match value {
        PyValue::Date(text) => serde_json::json!({"__type__": "date", "value": text}),
        PyValue::DateTime(text) => serde_json::json!({"__type__": "datetime", "value": text}),
        PyValue::Str(text) => Json::String(text.clone()),
        PyValue::Int(n) => Json::from(*n),
        PyValue::UInt(n) => Json::from(*n),
        PyValue::Float(n) => serde_json::json!(*n),
        PyValue::Bool(b) => Json::Bool(*b),
        PyValue::Null => Json::Null,
        PyValue::List(items) => Json::Array(items.iter().map(tag_to_json).collect()),
        PyValue::Dict(items) => Json::Object(
            items
                .iter()
                .map(|(key, value)| (key.clone(), tag_to_json(value)))
                .collect(),
        ),
        PyValue::Other(text) => Json::String(text.clone()),
    }
}

fn import_one(
    store: &Store,
    settings: &Settings,
    engine: &dyn Embedder,
    item: &Parsed,
) -> Result<()> {
    store.write(
        "MATCH (ch:Chunk {parent_doc_id: $id}) DETACH DELETE ch",
        vec![("id", Value::String(item.id.clone()))],
    )?;
    store.write(
        "MATCH (c:Concept {id: $id}) DETACH DELETE c",
        vec![("id", Value::String(item.id.clone()))],
    )?;

    let chunks = if settings.enable_chunking {
        let counter = |text: &str| {
            engine
                .count_tokens(text)
                .unwrap_or_else(|_| text.split_whitespace().count())
        };
        split_into_chunks(
            &item.body,
            &item.id,
            settings.chunk_size.max(1) as usize,
            counter,
        )
    } else {
        Vec::new()
    };
    if !chunks.is_empty() {
        let payloads = overlap_payloads(&chunks, settings.chunk_overlap.max(0) as usize);
        let vectors = engine.encode_many(&payloads, Task::Document)?;
        for (chunk, vector) in chunks.iter().zip(vectors) {
            write_chunk(store, chunk, &vector)?;
        }
        store.bump_write_epoch()?;
    }

    let search_text = format!(
        "{} {} {}",
        item.title,
        item.description.clone().unwrap_or_default(),
        item.body
    );
    let embedding = engine.encode(&search_text, Task::Document)?;
    write_concept(store, item, &embedding)?;
    store.bump_write_epoch()?;

    for chunk in &chunks {
        store.write(
            "MATCH (d:Concept {id: $doc}), (ch:Chunk {id: $cid}) MERGE (d)-[:PART_OF]->(ch)",
            vec![
                ("doc", Value::String(item.id.clone())),
                (
                    "cid",
                    Value::String(format!("{}#chunk:{}", item.id, chunk.chunk_index)),
                ),
            ],
        )?;
    }
    record_file_hash(store, settings, item)?;
    Ok(())
}

fn write_chunk(store: &Store, chunk: &MdChunk, embedding: &[f32]) -> Result<()> {
    let vector = Value::List(
        LogicalType::Float,
        embedding.iter().copied().map(Value::Float).collect(),
    );
    store.write(
        "CREATE (:Chunk {id: $id, parent_doc_id: $doc, chunk_index: $idx, \
         chunk_text: $text, block_type: $block, start_offset: $start_offset, \
         end_offset: $end_offset, embedding: $emb})",
        vec![
            (
                "id",
                Value::String(format!(
                    "{}#chunk:{}",
                    chunk.parent_doc_id, chunk.chunk_index
                )),
            ),
            ("doc", Value::String(chunk.parent_doc_id.clone())),
            ("idx", Value::Int64(chunk.chunk_index)),
            ("text", Value::String(chunk.chunk_text.clone())),
            ("block", Value::String(chunk.block_type.clone())),
            ("start_offset", Value::Int64(chunk.start_offset)),
            ("end_offset", Value::Int64(chunk.end_offset)),
            ("emb", vector),
        ],
    )
}

fn write_concept(store: &Store, item: &Parsed, embedding: &[f32]) -> Result<()> {
    let now = utc_now_datetime();
    let effective = effective_at(item, &now);
    let mut params: Vec<(&str, Value)> = vec![
        ("id", Value::String(item.id.clone())),
        ("type", Value::String(item.concept_type.clone())),
        ("title", maybe_string(Some(&item.title))),
        ("description", maybe_string(item.description.as_deref())),
        ("resource", maybe_string(item.resource.as_deref())),
        (
            "tags",
            Value::List(
                LogicalType::String,
                item.tags.iter().cloned().map(Value::String).collect(),
            ),
        ),
        (
            "timestamp",
            item.timestamp
                .map(|value| timestamp_value(&value))
                .unwrap_or_else(|| Value::Null(LogicalType::Timestamp)),
        ),
        ("body", Value::String(item.body.clone())),
        (
            "embedding",
            Value::List(
                LogicalType::Float,
                embedding.iter().copied().map(Value::Float).collect(),
            ),
        ),
        ("created_at", timestamp_value(&now)),
        ("updated_at", timestamp_value(&now)),
        ("effective_at", timestamp_value(&effective)),
        ("content_sig", Value::String(item.sig.clone())),
    ];
    let mut sql = String::from(
        "CREATE (:Concept {id: $id, type: $type, title: $title, \
         description: $description, resource: $resource, tags: $tags, \
         timestamp: $timestamp, body: $body, embedding: $embedding, \
         created_at: $created_at, updated_at: $updated_at, \
         effective_at: $effective_at, content_sig: $content_sig",
    );
    if !item.extra.is_empty() {
        let (keys, values): (Vec<_>, Vec<_>) = item
            .extra
            .iter()
            .map(|(key, value)| (Value::String(key.clone()), Value::String(value.clone())))
            .unzip();
        params.push(("extra_keys", Value::List(LogicalType::String, keys)));
        params.push(("extra_values", Value::List(LogicalType::String, values)));
        sql.push_str(", extra: MAP($extra_keys, $extra_values)");
    }
    sql.push_str("})");
    store.write(&sql, params)
}

fn maybe_string(value: Option<&str>) -> Value {
    value
        .map(|text| Value::String(text.to_string()))
        .unwrap_or_else(|| Value::Null(LogicalType::String))
}

fn record_file_hash(store: &Store, settings: &Settings, item: &Parsed) -> Result<()> {
    let digest = Sha256::digest(fs::read(&item.path)?);
    let hash: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    let rel = settings
        .bundle_root
        .as_ref()
        .and_then(|root| item.path.strip_prefix(root).ok())
        .unwrap_or(&item.path)
        .to_string_lossy()
        .into_owned();
    let dir = Path::new(&rel)
        .parent()
        .map(|value| value.to_string_lossy().into_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| ".".into());
    store.write(
        "MERGE (f:FileHash {path: $path}) \
         SET f.hash = $hash, f.concept_id = $cid, f.dir = $dir",
        vec![
            ("path", Value::String(rel)),
            ("hash", Value::String(hash)),
            ("cid", Value::String(item.id.clone())),
            ("dir", Value::String(dir)),
        ],
    )
}

fn create_links(store: &Store, parsed: &[Parsed]) -> Result<()> {
    let mut by_name: BTreeMap<String, String> = BTreeMap::new();
    for item in parsed {
        by_name.insert(item.id.clone(), item.id.clone());
        by_name.insert(item.title.clone(), item.id.clone());
        if let Some(stem) = item.id.rsplit('/').next() {
            by_name.insert(stem.to_string(), item.id.clone());
        }
    }
    for item in parsed {
        for target in wikilinks(&item.body) {
            let Some(target_id) = by_name.get(&target) else {
                continue;
            };
            if target_id == &item.id {
                continue;
            }
            store.write(
                "MATCH (a:Concept {id: $src}), (b:Concept {id: $dst}) \
                 MERGE (a)-[:LINKS_TO {kind: 'mention'}]->(b)",
                vec![
                    ("src", Value::String(item.id.clone())),
                    ("dst", Value::String(target_id.clone())),
                ],
            )?;
        }
    }
    Ok(())
}

fn wikilinks(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("[[") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("]]") else { break };
        let raw = &rest[..end];
        let name = raw
            .split(['|', '#'])
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !name.is_empty() {
            out.push(name);
        }
        rest = &rest[end + 2..];
    }
    out
}

fn create_directories(store: &Store, parsed: &[Parsed]) -> Result<()> {
    for item in parsed {
        let parts: Vec<&str> = item.id.split('/').collect();
        for index in 1..parts.len() {
            let parent = parts[..index].join("/");
            let child = parts[..=index].join("/");
            store.write(
                "MERGE (:Directory {id: $id})",
                vec![("id", Value::String(parent.clone()))],
            )?;
            if index + 1 == parts.len() {
                store.write(
                    "MATCH (d:Directory {id: $parent}), (c:Concept {id: $child}) \
                     MERGE (d)-[:CONTAINS]->(c)",
                    vec![
                        ("parent", Value::String(parent)),
                        ("child", Value::String(child)),
                    ],
                )?;
            } else {
                store.write(
                    "MATCH (d:Directory {id: $parent}), (s:Directory {id: $child}) \
                     MERGE (d)-[:CONTAINS]->(s)",
                    vec![
                        ("parent", Value::String(parent)),
                        ("child", Value::String(child)),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

fn overlap_payloads(chunks: &[MdChunk], overlap: usize) -> Vec<String> {
    let mut payloads = Vec::new();
    let mut prev_tail: Vec<String> = Vec::new();
    for chunk in chunks {
        let base = chunk
            .block_type
            .strip_suffix('+')
            .unwrap_or(&chunk.block_type);
        if STRUCTURAL_BLOCKS.contains(&base) {
            prev_tail.clear();
        }
        let words: Vec<&str> = chunk.chunk_text.split_whitespace().collect();
        let text = if prev_tail.is_empty() {
            chunk.chunk_text.clone()
        } else {
            let keep = words.len().max(1);
            let start = prev_tail.len().saturating_sub(keep);
            let tail = prev_tail[start..].join("  ");
            format!("{tail}\n\n{}", chunk.chunk_text)
        };
        payloads.push(text);
        if overlap > 0 && !STRUCTURAL_BLOCKS.contains(&base) {
            let start = words.len().saturating_sub(overlap);
            prev_tail = words[start..]
                .iter()
                .map(|word| (*word).to_string())
                .collect();
        } else {
            prev_tail.clear();
        }
    }
    payloads
}
