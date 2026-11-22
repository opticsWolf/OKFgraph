//! Whole-graph detach: end the mirror relationship and keep the DB as the
//! artifact. The report and provenance rows mirror the Python 0.15 oracle.

use crate::error::{OkfError, Result};
use crate::ingest::{native_rel, parse_file_alias, walk_source_files};
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
            if !crate::util::is_concept_file(&path) {
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
    explicit_root: bool,
) -> Result<()> {
    let roots = crate::roots::Roots::from_settings(settings);
    let mut checked = 0;
    let mut total_files = 0;
    if explicit_root {
        // An explicit `bundle_path` pins one tree, like Python's detach
        // call; configured named roots are not verified in that mode.
        if root.is_dir() {
            checked += 1;
            total_files += verify_one(root, "", report, store, settings)?;
        }
    } else {
        for (alias, tree) in roots.all_roots() {
            if !tree.is_dir() {
                // An absent/unmounted named root is a legal liveness state;
                // the primary root reports FILE_NOT_FOUND below when none
                // of the configured trees can be checked.
                continue;
            }
            checked += 1;
            total_files += verify_one(tree, alias, report, store, settings)?;
        }
    }
    if checked == 0 {
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
    report.insert("file_count".into(), Json::from(total_files));
    let mut sourceless = 0;
    let rows = store.query("MATCH (f:FileHash) RETURN f.path AS p", vec![])?;
    for row in rows {
        if let Some(path) = row.get("p").and_then(Json::as_str) {
            let candidates = if explicit_root {
                // An explicit tree owns every ledger key for this check;
                // Python strips the `@alias/` namespace before joining.
                let (_, native) = crate::roots::split_namespaced(path);
                vec![root.join(native)]
            } else {
                roots.resolve_key(path)
            };
            if !candidates.into_iter().any(|candidate| candidate.exists()) {
                sourceless += 1;
            }
        }
    }
    report.insert("already_sourceless".into(), Json::from(sourceless));
    Ok(())
}

fn verify_one(
    root: &Path,
    alias: &str,
    report: &mut Map<String, Json>,
    store: &Store,
    settings: &Settings,
) -> Result<usize> {
    let files = walk_source_files(root)?;
    let file_count = files.len();
    for path in &files {
        let rel = native_rel(path, root);
        let parsed = match parse_file_alias(path, settings, Some(root), alias) {
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
    Ok(file_count)
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

/// Mirror-write guard. Returns true when a forced write should clear the
/// detached state after it succeeds (Python's `_require_attached`).
pub fn require_attached(
    store: &Store,
    settings: &Settings,
    force: bool,
    root: Option<&Path>,
) -> Result<bool> {
    if store.meta_int("detached")?.unwrap_or(0) == 0 {
        return Ok(false);
    }
    if !force {
        return Err(OkfError::new(
            "DETACHED",
            "graph is detached from its bundle (see `okf detach`): mirror writes are refused. Re-attach with --force (the bundle root must match the recorded source tree).",
        )
        .remedy("re-attach with --force, or pass --force to import"));
    }
    let Some(root) = root else {
        // Rootless addressed writes never carry a mirror relationship.
        return Ok(false);
    };
    let want = crate::roots::canonicalish(root);
    let rows = store.query("MATCH (s:SourceRoot) RETURN s.path AS path", vec![])?;
    let known: Vec<PathBuf> = rows
        .into_iter()
        .filter_map(|row| {
            row.get("path")
                .and_then(Json::as_str)
                .map(|path| crate::roots::canonicalish(Path::new(path)))
        })
        .collect();
    if !known.iter().any(|path| path == &want) {
        let shown = known
            .first()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unknown>".into());
        return Err(OkfError::new(
            "DETACHED",
            format!(
                "graph was detached from a different source tree ({shown}); refusing --force import from {}",
                want.display()
            ),
        )
        .field("detached_from", Json::String(shown))
        .field("want", Json::String(want.to_string_lossy().into_owned()))
        .remedy("create a new database for a different tree"));
    }
    // A multi-root graph may only reattach when every configured tree is
    // present in the recorded provenance.
    if !settings.roots.is_empty() {
        let roots = crate::roots::Roots::from_settings(settings);
        if roots.all_roots().iter().any(|(_, path)| {
            let resolved = crate::roots::canonicalish(path);
            !known.iter().any(|candidate| candidate == &resolved)
        }) {
            return Err(OkfError::new(
                "DETACHED",
                "graph was detached from a different set of source roots; refusing --force import",
            )
            .remedy("create a new database for a different tree"));
        }
    }
    Ok(true)
}

/// Clear detached state and provenance after a successful forced reattach.
pub fn clear_detached(store: &Store) -> Result<()> {
    for sql in [
        "MATCH (m:Meta {key: 'detached'}) DELETE m",
        "MATCH (m:Meta {key: 'detached_at'}) DELETE m",
        "MATCH (s:SourceRoot) DELETE s",
    ] {
        store.write(sql, vec![])?;
    }
    Ok(())
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
    let explicit_root = params.get("bundle_path").and_then(Json::as_str).is_some();
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
        verify(&root, map, store, settings, explicit_root)?;
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
    let roots = crate::roots::Roots::from_settings(settings);
    store.write("MATCH (s:SourceRoot) DELETE s", vec![])?;
    for (alias, tree) in roots.all_roots() {
        let count = if tree.is_dir() {
            walk_source_files(tree)?.len() as i64
        } else {
            0
        };
        let resolved = crate::roots::canonicalish(tree);
        store.write(
            "MERGE (s:SourceRoot {alias: $alias}) \
             SET s.path = $path, s.file_count = $count, s.last_seen = $seen",
            vec![
                ("alias", Value::String(alias.to_string())),
                ("path", Value::String(native_display(&resolved))),
                ("count", Value::Int64(count)),
                ("seen", Value::Int64(now)),
            ],
        )?;
    }
    let resolved = crate::roots::canonicalish(&root);
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
