//! Whole-graph detach: end the mirror relationship and keep the DB as the
//! artifact. The report and provenance rows mirror the Python 0.15 oracle.

use crate::error::{OkfError, Result};
use crate::ingest::{native_rel, parse_file, walk_source_files};
use crate::model::normalize_tags;
use crate::settings::Settings;
use crate::store::Store;
use lbug::Value;
use serde_json::{Map, Value as Json};
use std::fs;
use std::path::{Path, PathBuf};

const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "__pycache__",
    "venv",
    "dist",
    "build",
];
const SOURCE_EXTS: &[&str] = &["md", "markdown", "txt"];
const RESERVED: &[&str] = &["index.md", "log.md"];

fn native_display(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn skipped(path: &Path, root: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    relative.components().any(|component| {
        let name = component.as_os_str().to_string_lossy();
        name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref())
    })
}

fn is_db_file(path: &Path, settings: &Settings) -> bool {
    let db = PathBuf::from(&settings.db_path);
    let Ok(path) = fs::canonicalize(path) else {
        return false;
    };
    let Ok(db) = fs::canonicalize(&db) else {
        return false;
    };
    if path == db {
        return true;
    }
    let Some(parent) = db.parent() else {
        return false;
    };
    if path.parent() != Some(parent) {
        return false;
    }
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    let base = db
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    name.starts_with(&format!("{base}.")) || name.starts_with(&format!("{base}-"))
}

fn all_files(root: &Path, settings: &Settings) -> Vec<PathBuf> {
    fn walk(dir: &Path, root: &Path, settings: &Settings, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                if !skipped(&path, root) {
                    walk(&path, root, settings, out);
                }
                continue;
            }
            if skipped(&path, root) || is_db_file(&path, settings) {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if RESERVED.contains(&name.as_str()) {
                continue;
            }
            if !SOURCE_EXTS.contains(
                &path
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default(),
            ) {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, settings, &mut out);
    out.sort();
    out
}

fn verify(
    root: &Path,
    report: &mut Map<String, Json>,
    store: &Store,
    settings: &Settings,
) -> Result<()> {
    if !root.is_dir() {
        return Err(OkfError::new(
            "FILE_NOT_FOUND",
            format!(
                "bundle '{}' not found - nothing to verify against.",
                root.display()
            ),
        )
        .field("bundle", Json::String(native_display(root)))
        .remedy("pass the source tree, or --no-verify to detach an already-removed tree"));
    }
    let files = walk_source_files(root)?;
    report.insert("file_count".into(), Json::from(files.len()));
    for path in &files {
        let rel = native_rel(path, root);
        let parsed = match parse_file(path, settings, Some(root)) {
            Ok(parsed) => parsed,
            Err(err) => {
                push_object(
                    report,
                    "untracked",
                    serde_json::json!({"path": rel, "reason": format!("unparseable: {err}")}),
                );
                continue;
            }
        };
        let Some(stored) = store.concept_raw(&parsed.id)? else {
            push_object(
                report,
                "untracked",
                serde_json::json!({"path": rel, "concept_id": parsed.id}),
            );
            continue;
        };
        let mut fields = Vec::new();
        for (key, expected) in [
            ("type", Json::String(parsed.concept_type.clone())),
            ("title", Json::String(parsed.title.clone())),
            (
                "description",
                parsed.description.clone().map_or(Json::Null, Json::String),
            ),
            (
                "resource",
                parsed.resource.clone().map_or(Json::Null, Json::String),
            ),
        ] {
            let actual = stored.get(key).cloned().unwrap_or(Json::Null);
            let actual = if actual.is_null() {
                Json::String(String::new())
            } else {
                actual
            };
            let expected = if expected.is_null() {
                Json::String(String::new())
            } else {
                expected
            };
            if actual != expected {
                fields.push(Json::String(key.to_string()));
            }
        }
        let actual_tags = normalize_tags(stored.get("tags").unwrap_or(&Json::Null));
        if actual_tags != parsed.tags {
            fields.push(Json::String("tags".to_string()));
        }
        let actual_body = stored
            .get("body")
            .and_then(Json::as_str)
            .unwrap_or_default();
        if actual_body != parsed.body {
            fields.push(Json::String("body".to_string()));
        }
        if !fields.is_empty() {
            push_object(
                report,
                "mismatched",
                serde_json::json!({"path": rel, "concept_id": parsed.id, "fields": fields}),
            );
        }
    }
    for path in all_files(root, settings) {
        let rel = native_rel(&path, root);
        push_array(report, "non_source_files", Json::String(rel));
    }
    let mut sourceless = 0;
    let rows = store.query("MATCH (f:FileHash) RETURN f.path AS p", vec![])?;
    for row in rows {
        if let Some(path) = row.get("p").and_then(Json::as_str) {
            let native = path.replace('\\', "/");
            let relative = native
                .strip_prefix('@')
                .and_then(|rest| rest.split_once('/'))
                .map(|(_, rest)| rest)
                .unwrap_or(&native);
            if !root.join(relative).exists() {
                sourceless += 1;
            }
        }
    }
    report.insert("already_sourceless".into(), Json::from(sourceless));
    Ok(())
}

fn push_array(report: &mut Map<String, Json>, key: &str, value: Json) {
    if let Some(Json::Array(items)) = report.get_mut(key) {
        items.push(value);
    }
}

fn push_object(report: &mut Map<String, Json>, key: &str, value: Json) {
    push_array(report, key, value);
}

fn summary(report: &Map<String, Json>) -> String {
    let count = |key: &str| match report.get(key).and_then(Json::as_array) {
        Some(items) => items.len(),
        None => 0,
    };
    format!(
        "{} mismatched, {} untracked, {} source-only",
        count("mismatched"),
        count("untracked"),
        count("non_source_files")
    )
}

pub fn detach(store: &Store, settings: &Settings, params: &Map<String, Json>) -> Result<Json> {
    if let Some(_since) = store.meta_int("detached")? {
        let since_at = store.meta_int("detached_at")?;
        let suffix = match since_at {
            Some(value) if value != 0 => format!(" (since epoch {value})"),
            _ => String::new(),
        };
        return Err(
            OkfError::new("DETACHED", format!("graph is already detached{suffix}."))
                .op("detach")
                .field("since", since_at.map_or(Json::Null, Json::from)),
        );
    }
    let verify_enabled = params.get("verify").and_then(Json::as_bool).unwrap_or(true);
    let force = params.get("force").and_then(Json::as_bool).unwrap_or(false);
    let root = params
        .get("bundle_path")
        .and_then(Json::as_str)
        .map(PathBuf::from)
        .or_else(|| settings.bundle_root.as_ref().map(PathBuf::from))
        .ok_or_else(|| {
            OkfError::new("MISSING_PARAM", "detach needs a bundle root")
                .op("detach")
                .field("bundle", Json::Null)
        })?;
    let mut report = serde_json::json!({
        "bundle": native_display(&root),
        "verified": false,
        "mismatched": Json::Array(Vec::new()),
        "untracked": Json::Array(Vec::new()),
        "non_source_files": Json::Array(Vec::new()),
        "already_sourceless": 0,
        "file_count": 0,
    });
    let map = report.as_object_mut().expect("object");
    let baseline = store
        .query("MATCH (f:FileHash) RETURN count(f) AS n", vec![])?
        .first()
        .and_then(|row| row.get("n"))
        .and_then(Json::as_i64)
        .unwrap_or(0);
    if verify_enabled {
        verify(&root, map, store, settings)?;
        let summary = summary(map);
        let mismatched = map
            .get("mismatched")
            .and_then(Json::as_array)
            .is_some_and(|items| !items.is_empty());
        let untracked = map
            .get("untracked")
            .and_then(Json::as_array)
            .is_some_and(|items| !items.is_empty());
        let source_only = map
            .get("non_source_files")
            .and_then(Json::as_array)
            .is_some_and(|items| !items.is_empty());
        if mismatched && !force {
            let sample = map
                .get("mismatched")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
                .take(5)
                .map(|item| {
                    let path = item.get("path").and_then(Json::as_str).unwrap_or("");
                    let fields: Vec<String> = item
                        .get("fields")
                        .and_then(Json::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(Json::as_str)
                        .map(str::to_string)
                        .collect();
                    format!("{path} [{}]", fields.join(", "))
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(OkfError::new(
                "DETACH_REFUSED",
                format!(
                    "detach refused: {summary}. Re-import first so the graph matches the files, or pass --force to declare the database the artifact anyway. e.g. {sample}"
                ),
            )
            .op("detach"));
        }
        if (untracked || source_only) && !force {
            return Err(OkfError::new(
                "DETACH_REFUSED",
                format!(
                    "detach refused: {summary} - these files have no counterpart in the graph and would not survive source deletion (WILL-NOT-SURVIVE). Pass --force to acknowledge."
                ),
            )
            .op("detach"));
        }
        map.insert("verified".into(), Json::Bool(true));
    } else {
        map.insert("file_count".into(), Json::from(baseline));
    }
    let file_count = map
        .get("file_count")
        .and_then(Json::as_i64)
        .unwrap_or(baseline);
    for sql in [
        "MATCH (f:FileHash) DELETE f",
        "MATCH (d:DirHash) DELETE d",
        "MATCH (d:DeletedPath) DELETE d",
    ] {
        store.write(sql, vec![])?;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| OkfError::internal(err.to_string(), "system time"))?
        .as_secs() as i64;
    let resolved = fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
    store.write(
        "MERGE (s:SourceRoot {alias: $alias}) \
         SET s.path = $path, s.file_count = $count, s.last_seen = $seen",
        vec![
            ("alias", Value::String(String::new())),
            ("path", Value::String(native_display(&resolved))),
            ("count", Value::Int64(file_count)),
            ("seen", Value::Int64(now)),
        ],
    )?;
    store.write("MERGE (m:Meta {key: 'detached'}) SET m.value = 1", vec![])?;
    store.write(
        "MERGE (m:Meta {key: 'detached_at'}) SET m.value = $now",
        vec![("now", Value::Int64(now))],
    )?;
    map.insert(
        "provenance".into(),
        serde_json::json!({
            "alias": "",
            "path": native_display(&resolved),
            "file_count": file_count,
            "last_seen": now,
            "detached_at": now,
        }),
    );
    Ok(report)
}
