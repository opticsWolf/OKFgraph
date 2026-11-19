//! Walk/parse/import pipeline. Stage B implements markdown import parity for
//! the differential corpus: frontmatter identity, synthesized titles/types,
//! chunk rows, file hashes, directory hierarchy, and wikilink resolution.

use crate::embed::{check_pins, Embedder, EmbedderPool, Pools, Task};
use crate::error::{OkfError, Result};
use crate::markdown::{parse_frontmatter, split_into_chunks, Chunk as MdChunk};
use crate::model::{extra_value, normalize_tags, utc_now_datetime, PyDateTime};
use crate::pycompat::{content_sig, python_datetime_isoformat, PyValue};
use crate::settings::Settings;
use crate::store::Store;
use lbug::{LogicalType, Value};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const STRUCTURAL_BLOCKS: &[&str] = &[
    "Heading",
    "CodeBlock",
    "List",
    "Blockquote",
    "Table",
    "Diagram",
];

#[derive(Debug, Clone)]
pub(crate) struct Parsed {
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) concept_type: String,
    pub(crate) description: Option<String>,
    pub(crate) resource: Option<String>,
    pub(crate) tags: Vec<String>,
    pub(crate) timestamp: Option<PyDateTime>,
    pub(crate) authored: BTreeMap<String, PyValue>,
    pub(crate) body: String,
    pub(crate) extra: BTreeMap<String, String>,
    pub(crate) sig: String,
    pub(crate) path: PathBuf,
}

pub fn import_files(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
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
        .map(|path| parse_file(Path::new(path), settings, None))
        .collect::<Result<Vec<_>>>()?;
    commit_parsed(
        store,
        settings,
        engine.as_ref(),
        pools,
        &parsed,
        None,
        false,
    )?;
    import_report(store, &parsed, Json::Null)
}

/// Import a whole bundle with Python's directory-delta detection. Unchanged
/// trees are skipped; changed directories are compared per file, so a single
/// edit does not re-embed its untouched siblings.
pub fn import_bundle(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    bundle_path: Option<&str>,
) -> Result<Json> {
    let configured = settings.bundle_root.as_deref().map(PathBuf::from);
    let root = match bundle_path {
        Some(value) => PathBuf::from(value),
        None => configured.clone().ok_or_else(|| {
            OkfError::new(
                "MISSING_PARAM",
                "import_bundle needs a bundle path or configured bundle_root",
            )
            .op("import_bundle")
        })?,
    };
    if !root.is_dir() {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("bundle path is not a directory: {}", root.display()),
        )
        .op("import_bundle")
        .field(
            "bundle_path",
            Json::String(root.to_string_lossy().into_owned()),
        ));
    }
    let root = canonicalish(&root);
    let engine = embedder.get(settings)?;
    pin_graph(store, settings, engine.as_ref())?;

    let files = walk_source_files(&root)?;
    let stored_dirs = store.dir_hashes()?;
    let stored_files = store.file_hashes()?;
    let mut current_dirs: BTreeMap<String, DirState> = BTreeMap::new();
    for file in &files {
        let rel = native_rel(file, &root);
        let dir = parent_key(&rel);
        let entry = current_dirs.entry(dir).or_default();
        let name = file
            .file_name()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default();
        entry.files.push(name.clone());
        entry.hashes.push((name, file_digest(file)?));
    }

    let mut changed = Vec::new();
    let mut changed_dir_names = BTreeSet::new();
    let mut actually_changed = 0usize;
    for (dir, state) in &mut current_dirs {
        state.files.sort();
        state.hashes.sort_by(|a, b| a.0.cmp(&b.0));
        let hash = directory_hash(&state.hashes);
        state.write_hash = Some(hash.clone());
        let stored = stored_dirs.get(dir).map(|value| value.0.as_str());
        if stored != Some(hash.as_str()) {
            changed_dir_names.insert(dir.clone());
            for file in &files {
                if parent_key(&native_rel(file, &root)) != *dir {
                    continue;
                }
                let rel = native_rel(file, &root);
                let digest = file_digest(file)?;
                let unchanged = stored_files
                    .get(&rel)
                    .map(|value| value.0 == digest && store.node_exists(&value.1).unwrap_or(false))
                    .unwrap_or(false);
                if !unchanged {
                    changed.push(file.clone());
                    actually_changed += 1;
                }
            }
        }
    }
    let mut deleted = Vec::new();
    for dir in stored_dirs.keys() {
        if !current_dirs.contains_key(dir) {
            if let Some((_, files_json)) = stored_dirs.get(dir) {
                if let Ok(names) = serde_json::from_str::<Vec<String>>(files_json) {
                    for name in names {
                        deleted.push(join_native(dir, &name));
                    }
                }
            }
        }
    }
    for dir in &changed_dir_names {
        let Some(state) = current_dirs.get(dir) else {
            continue;
        };
        let Some((_, files_json)) = stored_dirs.get(dir) else {
            continue;
        };
        let Ok(stored_names) = serde_json::from_str::<Vec<String>>(files_json) else {
            continue;
        };
        // Python caches a recursive listing here; match it so nested files
        // are not tombstoned just because their parent directory is direct.
        let current: BTreeSet<String> = walk_source_files(&root.join(dir))?
            .iter()
            .map(|file| native_rel(file, &root.join(dir)))
            .collect();
        for name in stored_names {
            if !current.contains(&name) {
                deleted.push(join_native(dir, &name));
            }
        }
        let _ = state;
    }
    deleted.sort();
    deleted.dedup();

    let stats = serde_json::json!({
        "changed_dirs": changed_dir_names.len(),
        "queued": changed.len(),
        "actually_changed": actually_changed,
    });

    let mut parsed: Vec<Parsed> = changed
        .iter()
        .map(|path| parse_file(path, settings, Some(&root)))
        .collect::<Result<Vec<_>>>()?;
    let (mut duplicates, moved) =
        resolve_identity_conflicts(store, &mut parsed, &stored_files, &root)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    for path in &deleted {
        let removed = moved.iter().any(|value| value.replace('/', "\\") == *path);
        if !removed {
            store.record_deleted_path(path, now)?;
        }
    }

    if !parsed.is_empty() {
        commit_parsed(
            store,
            settings,
            engine.as_ref(),
            pools,
            &parsed,
            Some(&root),
            true,
        )?;
        let mut file_hashes: BTreeMap<String, (String, String)> = BTreeMap::new();
        for item in &parsed {
            let rel = native_rel(&item.path, &root);
            file_hashes.insert(rel.clone(), (file_digest(&item.path)?, item.id.clone()));
        }
        for (path, (hash, cid)) in file_hashes {
            store.set_file_hash(&path, &hash, &cid, &parent_key(&path))?;
        }
    } else {
        finish_indexes(store)?;
    }
    for (dir, state) in &current_dirs {
        let json = PyValue::List(state.files.iter().cloned().map(PyValue::Str).collect())
            .to_python_json(false);
        store.set_dir_hash(dir, state.write_hash.as_deref().unwrap_or_default(), &json)?;
    }
    duplicates.sort();
    let mut detail = serde_json::json!({
        "duplicates": duplicates,
        "moved": moved,
        "delta": stats,
    });
    if let Some(object) = detail.as_object_mut() {
        object.insert(
            "concept_ids".into(),
            Json::Array(
                parsed
                    .iter()
                    .map(|item| Json::String(item.id.clone()))
                    .collect(),
            ),
        );
        let mut images = Map::new();
        for item in &parsed {
            images.insert(
                item.id.clone(),
                Json::from(store.list_images(&item.id)?.len() as i64),
            );
        }
        object.insert("images".into(), Json::Object(images));
    }
    Ok(detail)
}

#[derive(Default)]
struct DirState {
    files: Vec<String>,
    hashes: Vec<(String, String)>,
    write_hash: Option<String>,
}

fn finish_indexes(store: &Store) -> Result<()> {
    let epoch = store.meta_int("write_epoch")?.unwrap_or(0);
    store.create_search_indexes()?;
    store.set_meta_int("indexed_epoch", epoch)
}

fn commit_parsed(
    store: &Store,
    settings: &Settings,
    engine: &dyn Embedder,
    pools: &Pools,
    parsed: &[Parsed],
    root: Option<&Path>,
    bulk: bool,
) -> Result<()> {
    let ids: Vec<&str> = parsed.iter().map(|item| item.id.as_str()).collect();
    let incoming = if ids.is_empty() {
        Vec::new()
    } else {
        store.incoming_links(&ids)?
    };
    for item in parsed {
        import_one(store, settings, engine, item, root, bulk)?;
    }
    if bulk && !parsed.is_empty() {
        // Python's batch upsert bumps once for the whole concept transaction.
        store.bump_write_epoch()?;
    }
    create_links(store, parsed)?;
    for (source, target, kind) in incoming {
        store.link_concepts(&source, &target, &kind)?;
    }
    create_directories(store, parsed)?;
    repair_links(store)?;
    for item in parsed {
        let base_dir = item.path.parent();
        crate::image_ingest::ingest_concept_images(
            store, settings, engine, pools, &item.id, &item.body, base_dir, root,
        )?;
    }
    finish_indexes(store)
}

fn import_report(store: &Store, parsed: &[Parsed], detail: Json) -> Result<Json> {
    let mut images = BTreeMap::new();
    for item in parsed {
        images.insert(
            item.id.clone(),
            Json::from(store.list_images(&item.id)?.len() as i64),
        );
    }
    let mut result = serde_json::json!({
        "concept_ids": parsed
            .iter()
            .map(|item| Json::String(item.id.clone()))
            .collect::<Vec<_>>(),
        "images": images,
    });
    if !detail.is_null() {
        if let (Some(target), Some(extra)) = (result.as_object_mut(), detail.as_object()) {
            for (key, value) in extra {
                target.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(result)
}

fn resolve_identity_conflicts(
    store: &Store,
    parsed: &mut Vec<Parsed>,
    stored_files: &BTreeMap<String, (String, String)>,
    root: &Path,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut seen = BTreeMap::new();
    let mut duplicates = Vec::new();
    let mut moved = Vec::new();
    let mut keep = Vec::new();
    for item in std::mem::take(parsed) {
        if seen.contains_key(&item.id) {
            duplicates.push(native_rel(&item.path, root).replace('\\', "/"));
            continue;
        }
        seen.insert(item.id.clone(), item.path.clone());
        let rel = native_rel(&item.path, root);
        let old = stored_files
            .iter()
            .filter(|(path, (_, cid))| cid == &item.id && path.as_str() != rel)
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        for path in old {
            if !root.join(&path).exists() {
                moved.push(path.replace('\\', "/"));
                store.forget_file_hash(&path)?;
            } else {
                duplicates.push(native_rel(&item.path, root).replace('\\', "/"));
            }
        }
        keep.push(item);
    }
    *parsed = keep;
    Ok((duplicates, moved))
}

fn canonicalish(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub(crate) fn walk_source_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk_dir(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_dir(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let skip = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| {
                    name.starts_with('.')
                        || matches!(
                            name,
                            "target" | "node_modules" | "__pycache__" | "venv" | "dist" | "build"
                        )
                })
                .unwrap_or(false);
            if !skip {
                walk_dir(&path, root, out)?;
            }
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if matches!(lower.as_str(), "index.md" | "log.md") {
            continue;
        }
        let source = matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("md" | "markdown" | "txt")
        );
        if source {
            let _ = root;
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) fn native_rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn parent_key(rel: &str) -> String {
    Path::new(rel)
        .parent()
        .map(|value| value.to_string_lossy().into_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| ".".into())
}

fn join_native(dir: &str, name: &str) -> String {
    if dir == "." {
        name.to_string()
    } else {
        Path::new(dir).join(name).to_string_lossy().into_owned()
    }
}

fn file_digest(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn directory_hash(hashes: &[(String, String)]) -> String {
    let combined = hashes
        .iter()
        .map(|(name, hash)| format!("{name}:{hash}"))
        .collect::<Vec<_>>()
        .join("|");
    Sha256::digest(combined.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

fn relative_id(path: &Path, root: Option<&Path>) -> String {
    if let Some(root) = root {
        if let Ok(rel) = path.strip_prefix(root) {
            return rel.with_extension("").to_string_lossy().replace('\\', "/");
        }
    }
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub(crate) fn parse_file(path: &Path, settings: &Settings, root: Option<&Path>) -> Result<Parsed> {
    let text = fs::read_to_string(path)?;
    let front = parse_frontmatter(&text);
    let mut meta = front.metadata.as_object().cloned().unwrap_or_default();
    let effective_root = root
        .map(Path::to_path_buf)
        .or_else(|| settings.bundle_root.as_ref().map(PathBuf::from));
    let id = relative_id(path, effective_root.as_deref());
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
    root: Option<&Path>,
    bulk: bool,
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
    if !bulk {
        store.bump_write_epoch()?;
    }

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
    record_file_hash(store, settings, root, item)?;
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

fn record_file_hash(
    store: &Store,
    settings: &Settings,
    root: Option<&Path>,
    item: &Parsed,
) -> Result<()> {
    let effective_root = root
        .map(Path::to_path_buf)
        .or_else(|| settings.bundle_root.as_ref().map(PathBuf::from));
    let Some(effective_root) = effective_root else {
        // File-free imports mint bare IDs and do not touch the shared delta
        // ledger (Python `_root is None`).
        return Ok(());
    };
    let rel = item
        .path
        .strip_prefix(&effective_root)
        .unwrap_or(&item.path)
        .to_string_lossy()
        .into_owned();
    let hash = file_digest(&item.path)?;
    store.set_file_hash(&rel, &hash, &item.id, &parent_key(&rel))
}

fn create_links(store: &Store, parsed: &[Parsed]) -> Result<()> {
    let index = load_link_index(store)?;
    for item in parsed {
        store.clear_broken_from(&item.id)?;
    }
    let mut links = Vec::new();
    for item in parsed {
        links.extend(resolve_concept_links(item, &index));
    }
    let now = utc_now_datetime();
    for link in links {
        record_link(store, link, &now)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct LinkRecord {
    pub(crate) source: String,
    pub(crate) target: Option<String>,
    pub(crate) recorded: String,
    pub(crate) kind: String,
    pub(crate) reason: Option<String>,
    pub(crate) candidates: Vec<String>,
    pub(crate) raw: String,
}

#[derive(Default)]
pub(crate) struct LinkIndex {
    ids: BTreeSet<String>,
    uid: BTreeMap<String, String>,
    alias: BTreeMap<String, String>,
    title: BTreeMap<String, String>,
    stem: BTreeMap<String, String>,
    candidates: BTreeMap<String, Vec<String>>,
}

impl LinkIndex {
    fn add(kind: &str, key: &str, id: &str, index: &mut LinkIndex) {
        let key = key.trim().to_ascii_lowercase();
        if key.is_empty() {
            return;
        }
        let map = match kind {
            "uid" => &mut index.uid,
            "alias" => &mut index.alias,
            "title" => &mut index.title,
            _ => &mut index.stem,
        };
        if let Some(existing) = map.get(&key) {
            if existing != id {
                let entry = index.candidates.entry(format!("{kind}:{key}")).or_default();
                if !entry.contains(&existing.to_string()) {
                    entry.push(existing.to_string());
                }
                if !entry.contains(&id.to_string()) {
                    entry.push(id.to_string());
                }
                map.remove(&key);
            }
        } else if !index
            .candidates
            .get(&format!("{kind}:{key}"))
            .map(|ids| ids.contains(&id.to_string()))
            .unwrap_or(false)
        {
            map.insert(key, id.to_string());
        }
    }

    pub(crate) fn from_parsed(items: &[Parsed]) -> Self {
        let mut index = Self::default();
        for item in items {
            index.ids.insert(item.id.clone());
            if let Some(uid) = item.extra.get("uid") {
                Self::add("uid", uid, &item.id, &mut index);
            }
            for key in ["aliases", "alias"] {
                let Some(value) = item.extra.get(key) else {
                    continue;
                };
                let values = if value.trim_start().starts_with('[') {
                    serde_json::from_str::<Vec<String>>(value)
                        .unwrap_or_else(|_| vec![value.clone()])
                } else {
                    vec![value.clone()]
                };
                for alias in values {
                    Self::add("alias", &alias, &item.id, &mut index);
                }
            }
            Self::add("title", &item.title, &item.id, &mut index);
            if let Some(stem) = item.id.rsplit('/').next() {
                Self::add("stem", stem, &item.id, &mut index);
            }
        }
        index
    }

    pub(crate) fn resolve_wiki(
        &self,
        reference: &str,
    ) -> (Option<String>, Option<String>, Vec<String>) {
        let ref_text = reference.split('#').next().unwrap_or("").trim();
        if ref_text.is_empty() {
            return (None, Some("missing".into()), Vec::new());
        }
        if self.ids.contains(ref_text) {
            return (Some(ref_text.to_string()), None, Vec::new());
        }
        let without_md = if ref_text.to_ascii_lowercase().ends_with(".md") {
            &ref_text[..ref_text.len() - 3]
        } else {
            ref_text
        };
        if self.ids.contains(without_md) {
            return (Some(without_md.to_string()), None, Vec::new());
        }
        let key = without_md.to_ascii_lowercase();
        for (kind, map) in [
            ("uid", &self.uid),
            ("alias", &self.alias),
            ("title", &self.title),
            ("stem", &self.stem),
        ] {
            if let Some(id) = map.get(&key) {
                return (Some(id.clone()), None, Vec::new());
            }
            if let Some(ids) = self.candidates.get(&format!("{kind}:{key}")) {
                return (None, Some("ambiguous".into()), ids.clone());
            }
        }
        (None, Some("missing".into()), Vec::new())
    }
}

fn load_link_index(store: &Store) -> Result<LinkIndex> {
    let mut index = LinkIndex::default();
    for row in store.link_index()? {
        let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
            continue;
        };
        index.ids.insert(id.clone());
        let extra = row.get("extra").cloned().unwrap_or(Json::Null);
        let get = |key: &str| -> Option<String> {
            match &extra {
                Json::Object(map) => map.get(key).and_then(Json::as_str).map(str::to_string),
                Json::String(text) => serde_json::from_str::<Map<String, Json>>(text)
                    .ok()
                    .and_then(|map| map.get(key).and_then(Json::as_str).map(str::to_string)),
                _ => None,
            }
        };
        if let Some(uid) = get("uid") {
            LinkIndex::add("uid", &uid, &id, &mut index);
        }
        for key in ["aliases", "alias"] {
            let Some(value) = get(key) else { continue };
            let values = if value.trim_start().starts_with('[') {
                serde_json::from_str::<Vec<String>>(&value).unwrap_or_else(|_| vec![value.clone()])
            } else {
                vec![value.clone()]
            };
            for alias in values {
                LinkIndex::add("alias", &alias, &id, &mut index);
            }
        }
        if let Some(title) = row.get("title").and_then(Json::as_str) {
            LinkIndex::add("title", title, &id, &mut index);
        }
        if let Some(stem) = id.rsplit('/').next() {
            LinkIndex::add("stem", stem, &id, &mut index);
        }
    }
    Ok(index)
}

pub(crate) fn resolve_concept_links(item: &Parsed, index: &LinkIndex) -> Vec<LinkRecord> {
    let mut out = Vec::new();
    for raw in markdown_links(&item.body) {
        if is_external(&raw) {
            continue;
        }
        let (target, recorded) = link_target(&raw, &item.id, index, None);
        out.push(LinkRecord {
            source: item.id.clone(),
            target,
            recorded,
            kind: "mention".into(),
            reason: None,
            candidates: Vec::new(),
            raw,
        });
    }
    for raw in wikilinks(&item.body) {
        if raw.is_empty() || is_external(&raw) {
            continue;
        }
        let (target, reason, candidates) = index.resolve_wiki(&raw);
        out.push(LinkRecord {
            source: item.id.clone(),
            target,
            recorded: raw.clone(),
            kind: "mention".into(),
            reason,
            candidates,
            raw,
        });
    }
    if let Some(relations) = item.extra.get("relations") {
        if let Ok(value) = serde_json::from_str::<Json>(relations) {
            if let Some(map) = value.as_object() {
                for (kind, targets) in map {
                    if kind == "mention"
                        || !kind
                            .chars()
                            .next()
                            .map(|c| c.is_ascii_lowercase())
                            .unwrap_or(false)
                        || !kind.chars().all(|c| {
                            c.is_ascii_lowercase() || c.is_ascii_digit() || "_-".contains(c)
                        })
                    {
                        continue;
                    }
                    let values: Vec<&Json> = match targets {
                        Json::Array(items) => items.iter().collect(),
                        Json::Null => Vec::new(),
                        other => vec![other],
                    };
                    for target in values {
                        let Some(raw) = target.as_str().map(str::trim).filter(|s| !s.is_empty())
                        else {
                            continue;
                        };
                        let (target_id, reason, candidates, recorded) =
                            resolve_relation_target(raw, &item.id, index, None);
                        out.push(LinkRecord {
                            source: item.id.clone(),
                            target: target_id,
                            recorded,
                            kind: kind.clone(),
                            reason,
                            candidates,
                            raw: raw.to_string(),
                        });
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn resolve_relation_target(
    raw: &str,
    source: &str,
    index: &LinkIndex,
    path_ids: Option<&BTreeMap<String, String>>,
) -> (Option<String>, Option<String>, Vec<String>, String) {
    let mut reference = raw.trim();
    if reference.starts_with("[[") && reference.ends_with("]]") {
        reference = reference[2..reference.len() - 2]
            .split('|')
            .next()
            .unwrap_or("")
            .trim();
    }
    if index.ids.contains(reference) {
        return (
            Some(reference.to_string()),
            None,
            Vec::new(),
            reference.to_string(),
        );
    }
    if is_source_path(reference) {
        let (target, recorded) = link_target(reference, source, index, path_ids);
        let reason = if target.is_some() {
            None
        } else {
            Some("missing".into())
        };
        return (target, reason, Vec::new(), recorded);
    }
    let (target, reason, candidates) = index.resolve_wiki(reference);
    (
        target.clone(),
        reason,
        candidates,
        target.unwrap_or_else(|| reference.to_string()),
    )
}

fn record_link(store: &Store, link: LinkRecord, now: &crate::model::PyDateTime) -> Result<()> {
    if let Some(target) = link.target {
        store.link_concepts(&link.source, &target, &link.kind)?;
        return Ok(());
    }
    let cands = if link.candidates.is_empty() {
        String::new()
    } else {
        PyValue::List(link.candidates.iter().cloned().map(PyValue::Str).collect())
            .to_python_json(false)
    };
    let timestamp = timestamp_value(now);
    store.set_broken_link(
        &link.source,
        &link.recorded,
        &link.kind,
        link.reason.as_deref().unwrap_or("missing"),
        &cands,
        &link.raw,
        timestamp,
    )
}

pub fn extract_links_for_concept(store: &Store, concept_id: &str) -> Result<()> {
    let Some(concept) = store.get_by_id(concept_id)? else {
        return Ok(());
    };
    let mut extra = BTreeMap::new();
    if let Some(map) = concept.as_object() {
        for (key, value) in map {
            if [
                "id",
                "type",
                "title",
                "description",
                "resource",
                "tags",
                "timestamp",
                "body",
                "created_at",
                "updated_at",
                "effective_at",
                "content_sig",
            ]
            .contains(&key.as_str())
            {
                continue;
            }
            extra.insert(key.clone(), extra_value(&PyValue::from_golden(value)));
        }
    }
    let item = Parsed {
        id: concept_id.to_string(),
        title: concept
            .get("title")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string(),
        concept_type: concept
            .get("type")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string(),
        description: None,
        resource: None,
        tags: Vec::new(),
        timestamp: None,
        authored: BTreeMap::new(),
        body: concept
            .get("body")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string(),
        extra,
        sig: String::new(),
        path: PathBuf::new(),
    };
    create_links(store, std::slice::from_ref(&item))?;
    repair_links(store)
}

pub fn repair_links_public(store: &Store) -> Result<usize> {
    let before = store.list_broken_links()?.len();
    repair_links(store)?;
    let after = store.list_broken_links()?.len();
    Ok(before.saturating_sub(after))
}

fn repair_links(store: &Store) -> Result<()> {
    let index = load_link_index(store)?;
    for broken in store.list_broken_links()? {
        let source = broken
            .get("source")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        if source.is_empty() || !index.ids.contains(&source) {
            continue;
        }
        let target_text = broken
            .get("target")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        let raw = broken
            .get("raw")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        let kind = broken
            .get("kind")
            .and_then(Json::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("mention")
            .to_string();
        let mut target = if index.ids.contains(&target_text) {
            Some(target_text.clone())
        } else {
            None
        };
        if target.is_none() && is_source_path(&raw) {
            target = link_target(&raw, &source, &index, None).0;
        }
        let (wiki_target, reason, candidates) = index.resolve_wiki(&target_text);
        if target.is_none() {
            target = wiki_target;
        }
        let id = broken
            .get("id")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        if let Some(target) = target {
            store.link_concepts(&source, &target, &kind)?;
            store.delete_broken_link(&id)?;
            continue;
        }
        let old_reason = broken
            .get("reason")
            .and_then(Json::as_str)
            .unwrap_or("missing");
        let old_candidates = broken
            .get("candidates")
            .and_then(Json::as_array)
            .cloned()
            .unwrap_or_default();
        let reason = reason.unwrap_or_else(|| "missing".into());
        let cands = PyValue::List(candidates.iter().cloned().map(PyValue::Str).collect())
            .to_python_json(false);
        if reason != old_reason || old_candidates.len() != candidates.len() {
            store.update_broken_reason(&id, &reason, &cands)?;
        }
    }
    Ok(())
}

pub(crate) fn markdown_links(body: &str) -> Vec<String> {
    let stripped = strip_code(body);
    let mut out = Vec::new();
    let mut rest = stripped.as_str();
    while let Some(start) = rest.find("](") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find(')') else { break };
        let inner = rest[..end].trim();
        let raw = inner
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(['<', '>']);
        if is_source_path(raw) {
            out.push(raw.to_string());
        }
        rest = &rest[end + 1..];
    }
    out
}

fn strip_code(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut in_fence = false;
    for line in body.lines() {
        let fence = line.trim_start().starts_with("```") || line.trim_start().starts_with("~~~");
        if fence {
            in_fence = !in_fence;
            out.push_str(&" ".repeat(line.len()));
        } else if in_fence {
            out.push_str(&" ".repeat(line.len()));
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn is_source_path(raw: &str) -> bool {
    let path = raw
        .split(['#', '?'])
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    path.ends_with(".md") || path.ends_with(".markdown") || path.ends_with(".txt")
}

pub(crate) fn is_external(raw: &str) -> bool {
    let head = raw.split('#').next().unwrap_or("").trim();
    let Some(colon) = head.find(':') else {
        return false;
    };
    colon > 0
        && head[..colon]
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic())
            .unwrap_or(false)
}

pub(crate) fn link_target(
    raw: &str,
    source: &str,
    index: &LinkIndex,
    path_ids: Option<&BTreeMap<String, String>>,
) -> (Option<String>, String) {
    let path_part = raw.split(['#', '?']).next().unwrap_or("").trim();
    let joined = if let Some(rest) = path_part.strip_prefix('/') {
        rest.to_string()
    } else {
        let parent = source.rsplit_once('/').map(|value| value.0).unwrap_or("");
        if parent.is_empty() {
            path_part.to_string()
        } else {
            format!("{parent}/{path_part}")
        }
    };
    let normalized = joined.replace('\\', "/");
    let mut parts: Vec<&str> = Vec::new();
    for part in normalized.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    let mut candidate = parts.join("/");
    for ext in [".md", ".markdown", ".txt"] {
        if candidate.to_ascii_lowercase().ends_with(ext) {
            candidate.truncate(candidate.len() - ext.len());
            break;
        }
    }
    if candidate == ".." || candidate.starts_with("../") {
        return (None, candidate);
    }
    let legacy = path_part
        .trim_start_matches(['.', '/'])
        .replace('\\', "/")
        .replace(".md", "")
        .trim_start_matches('/')
        .to_string();
    for reading in [candidate.as_str(), legacy.as_str()] {
        if reading.is_empty() {
            continue;
        }
        if index.ids.contains(reading) {
            return (Some(reading.to_string()), reading.to_string());
        }
        if let Some(map) = path_ids {
            if let Some(id) = map.get(reading) {
                return (Some(id.clone()), id.clone());
            }
        }
    }
    (None, candidate)
}

pub(crate) fn wikilinks(body: &str) -> Vec<String> {
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
