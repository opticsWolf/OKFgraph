//! Structural diff and pre-import lint. Both are filesystem/DB comparisons;
//! neither writes to the graph or loads a model.

use crate::error::{OkfError, Result};
use crate::ingest::{
    is_external, link_target, markdown_links, native_rel, parse_file, resolve_relation_target,
    walk_source_files, wikilinks, LinkIndex,
};
use crate::markdown::parse_frontmatter;
use crate::settings::Settings;
use crate::store::Store;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct Meta {
    title: Option<String>,
    kind: Option<String>,
    hash: String,
}

#[derive(Default)]
struct State {
    concepts: BTreeMap<String, Meta>,
    edges: BTreeSet<(String, String)>,
    broken: BTreeSet<(String, String)>,
}

struct ParsedDir {
    items: Vec<crate::ingest::Parsed>,
    metas: Vec<Map<String, Json>>,
    index: LinkIndex,
    path_ids: BTreeMap<String, String>,
}

fn content_hash(body: &str) -> String {
    let normalized = body.replace("\r\n", "\n");
    format!("{:x}", Sha256::digest(normalized.trim().as_bytes()))
}

fn string_or_null(value: Option<&String>) -> Json {
    value
        .map(|text| Json::String(text.clone()))
        .unwrap_or(Json::Null)
}

fn extra_string(extra: &BTreeMap<String, String>, key: &str) -> Option<String> {
    extra.get(key).cloned().filter(|value| !value.is_empty())
}

fn path_ids(items: &[crate::ingest::Parsed], root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for item in items {
        let rel = native_rel(&item.path, root).replace('\\', "/");
        let stem = rel.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(&rel);
        if stem != item.id {
            out.insert(stem.to_string(), item.id.clone());
        }
    }
    out
}

fn parse_dir(dir: &Path, settings: &Settings) -> Result<ParsedDir> {
    let mut items = Vec::new();
    let mut metas = Vec::new();
    for path in walk_source_files(dir)? {
        let mut item = parse_file(&path, settings, Some(dir))?;
        if settings.identity == "frontmatter" {
            if let Some(uid) = extra_string(&item.extra, "uid") {
                item.id = uid;
            }
        }
        let text = fs::read_to_string(&path)?;
        metas.push(
            parse_frontmatter(&text)
                .metadata
                .as_object()
                .cloned()
                .unwrap_or_default(),
        );
        items.push(item);
    }
    let index = LinkIndex::from_parsed(&items);
    let ids = path_ids(&items, dir);
    Ok(ParsedDir {
        items,
        metas,
        index,
        path_ids: ids,
    })
}

fn state_from_dir(dir: &Path, settings: &Settings) -> Result<State> {
    let parsed = parse_dir(dir, settings)?;
    let mut state = State::default();
    for item in &parsed.items {
        state.concepts.insert(
            item.id.clone(),
            Meta {
                title: Some(item.title.clone()),
                kind: Some(item.concept_type.clone()),
                hash: content_hash(&item.body),
            },
        );
    }
    for (item, meta) in parsed.items.iter().zip(parsed.metas.iter()) {
        let source = item.id.as_str();
        for raw in markdown_links(&item.body) {
            if is_external(&raw) {
                continue;
            }
            let (target, recorded) =
                link_target(&raw, source, &parsed.index, Some(&parsed.path_ids));
            push_link(&mut state, source, target, &recorded);
        }
        for raw in wikilinks(&item.body) {
            if raw.is_empty() || is_external(&raw) {
                continue;
            }
            let (target, _, _) = parsed.index.resolve_wiki(&raw);
            let recorded = target.clone().unwrap_or(raw);
            push_link(&mut state, source, target, &recorded);
        }
        if let Some(value) = meta.get("relations") {
            let (pairs, _) = relation_pairs(value);
            for (_, raw) in pairs {
                let (target, _, _, recorded) =
                    resolve_relation_target(&raw, source, &parsed.index, Some(&parsed.path_ids));
                push_link(&mut state, source, target, &recorded);
            }
        }
    }
    Ok(state)
}

fn push_link(state: &mut State, source: &str, target: Option<String>, recorded: &str) {
    match target {
        Some(target) => {
            state.edges.insert((source.to_string(), target));
        }
        None => {
            state
                .broken
                .insert((source.to_string(), recorded.to_string()));
        }
    }
}

fn state_from_db(store: &Store) -> Result<State> {
    let mut state = State::default();
    for row in store.diagnostic_concepts()? {
        let Some(id) = row.get("id").and_then(Json::as_str).map(str::to_string) else {
            continue;
        };
        state.concepts.insert(
            id,
            Meta {
                title: row.get("title").and_then(Json::as_str).map(str::to_string),
                kind: row.get("type").and_then(Json::as_str).map(str::to_string),
                hash: content_hash(row.get("body").and_then(Json::as_str).unwrap_or("")),
            },
        );
    }
    for (source, target) in store.diagnostic_edges()? {
        state.edges.insert((source, target));
    }
    for row in store.list_broken_links()? {
        let source = row.get("source").and_then(Json::as_str).unwrap_or_default();
        let target = row.get("target").and_then(Json::as_str).unwrap_or_default();
        state
            .broken
            .insert((source.to_string(), target.to_string()));
    }
    Ok(state)
}

fn pairs_json(items: &BTreeSet<(String, String)>) -> Json {
    Json::Array(
        items
            .iter()
            .map(|(left, right)| {
                Json::Array(vec![
                    Json::String(left.clone()),
                    Json::String(right.clone()),
                ])
            })
            .collect(),
    )
}

fn compare(old: &State, new: &State) -> Json {
    let old_ids: BTreeSet<_> = old.concepts.keys().cloned().collect();
    let new_ids: BTreeSet<_> = new.concepts.keys().cloned().collect();
    let added: Vec<_> = new_ids.difference(&old_ids).cloned().collect();
    let removed: Vec<_> = old_ids.difference(&new_ids).cloned().collect();
    let mut changed = Vec::new();
    let mut retitled = Vec::new();
    let mut retyped = Vec::new();
    for id in old_ids.intersection(&new_ids) {
        let before = &old.concepts[id];
        let after = &new.concepts[id];
        if before.hash != after.hash {
            changed.push(id.clone());
        }
        if before.title != after.title {
            retitled.push(serde_json::json!({
                "id": id,
                "old": string_or_null(before.title.as_ref()),
                "new": string_or_null(after.title.as_ref()),
            }));
        }
        if before.kind != after.kind {
            retyped.push(serde_json::json!({
                "id": id,
                "old": string_or_null(before.kind.as_ref()),
                "new": string_or_null(after.kind.as_ref()),
            }));
        }
    }
    let edges_added: BTreeSet<_> = new.edges.difference(&old.edges).cloned().collect();
    let edges_removed: BTreeSet<_> = old.edges.difference(&new.edges).cloned().collect();
    let broken_new: BTreeSet<_> = new.broken.difference(&old.broken).cloned().collect();
    let broken_fixed: BTreeSet<_> = old.broken.difference(&new.broken).cloned().collect();
    let identical = added.is_empty()
        && removed.is_empty()
        && changed.is_empty()
        && retitled.is_empty()
        && retyped.is_empty()
        && edges_added.is_empty()
        && edges_removed.is_empty()
        && broken_new.is_empty()
        && broken_fixed.is_empty();
    serde_json::json!({
        "added": added,
        "removed": removed,
        "changed": changed,
        "retitled": retitled,
        "retyped": retyped,
        "edges_added": pairs_json(&edges_added),
        "edges_removed": pairs_json(&edges_removed),
        "broken_new": pairs_json(&broken_new),
        "broken_fixed": pairs_json(&broken_fixed),
        "identical": identical,
    })
}

fn native_display(path: &Path) -> String {
    path.components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("\\")
}

fn require_dir(path: &str, op: &str, name: &str) -> Result<PathBuf> {
    let dir = PathBuf::from(path);
    if !dir.is_dir() {
        return Err(OkfError::new(
            "FILE_NOT_FOUND",
            format!("not a bundle directory: {}", dir.display()),
        )
        .op(op)
        .field(name, Json::String(native_display(&dir))));
    }
    Ok(dir)
}

pub fn diff_dirs(old: &str, new: &str, settings: &Settings) -> Result<Json> {
    let old = require_dir(old, "diff", "old")?;
    let new = require_dir(new, "diff", "new")?;
    Ok(compare(
        &state_from_dir(&old, settings)?,
        &state_from_dir(&new, settings)?,
    ))
}

pub fn diff(
    store: &Store,
    settings: &Settings,
    old: Option<&str>,
    new: Option<&str>,
) -> Result<Json> {
    if let (Some(old), Some(new)) = (old, new) {
        return diff_dirs(old, new, settings);
    }
    let side = old
        .or(new)
        .map(str::to_string)
        .or_else(|| settings.bundle_root.clone().filter(|root| !root.is_empty()))
        .ok_or_else(|| {
            OkfError::new(
            "BAD_VALUE",
            "diff needs a bundle root: this router was opened without bundle_root (file-free mode)",
        )
        .op("diff")
        })?;
    let side = require_dir(&side, "diff", if old.is_some() { "old" } else { "new" })?;
    Ok(compare(
        &state_from_db(store)?,
        &state_from_dir(&side, settings)?,
    ))
}

fn py_type(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Number(number) => {
            if number.is_i64() || number.is_u64() {
                "int"
            } else {
                "float"
            }
        }
        Json::String(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

fn py_repr(value: &Json) -> String {
    match value {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::String(text) => format!("'{text}'"),
        Json::Number(number) => number.to_string(),
        Json::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Json::Object(_) => "{}".into(),
    }
}

pub(crate) fn relation_pairs(value: &Json) -> (Vec<(String, String)>, Vec<String>) {
    let mut pairs = Vec::new();
    let mut problems = Vec::new();
    let Some(map) = value.as_object() else {
        if !value.is_null() {
            problems.push(format!(
                "relations must be a mapping of kind -> targets, got {}",
                py_type(value)
            ));
        }
        return (pairs, problems);
    };
    let mut seen = BTreeSet::new();
    for (kind, targets) in map {
        if kind == "mention"
            || !kind
                .chars()
                .next()
                .is_some_and(|first| first.is_ascii_lowercase())
            || !kind
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "_-".contains(c))
            || kind.len() > 64
        {
            problems.push(format!("relation kind '{kind}' is not allowed"));
            continue;
        }
        let items: Vec<&Json> = match targets {
            Json::Array(items) => items.iter().collect(),
            Json::Null => Vec::new(),
            other => vec![other],
        };
        for target in items {
            if target.is_null() {
                continue;
            }
            let Some(text) = target
                .as_str()
                .map(str::trim)
                .filter(|text| !text.is_empty())
            else {
                problems.push(format!(
                    "relation {kind}: target {} is not a string",
                    py_repr(target)
                ));
                continue;
            };
            if seen.insert((kind.clone(), text.to_string())) {
                pairs.push((kind.clone(), text.to_string()));
            }
        }
    }
    (pairs, problems)
}

fn finding(file: &str, rule: &str, message: impl Into<String>) -> Json {
    serde_json::json!({
        "file": file,
        "rule": rule,
        "message": message.into(),
    })
}

fn finding_key(value: &Json) -> (String, String, String) {
    (
        value
            .get("file")
            .and_then(Json::as_str)
            .unwrap_or("")
            .into(),
        value
            .get("rule")
            .and_then(Json::as_str)
            .unwrap_or("")
            .into(),
        value
            .get("message")
            .and_then(Json::as_str)
            .unwrap_or("")
            .into(),
    )
}

fn relation_value(extra: &Map<String, Json>) -> Json {
    extra.get("relations").cloned().unwrap_or(Json::Null)
}

fn count_skipped(root: &Path) -> usize {
    fn walk(dir: &Path, root: &Path) -> usize {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        let mut count = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += walk(&path, root);
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            let source = matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("md" | "markdown" | "txt")
            );
            if !source || lower == "index.md" || lower == "log.md" {
                continue;
            }
            if path.strip_prefix(root).is_ok_and(|rel| {
                rel.components().any(|part| {
                    let text = part.as_os_str().to_string_lossy();
                    text.starts_with('.')
                        || matches!(
                            text.as_ref(),
                            "target" | "node_modules" | "__pycache__" | "venv" | "dist" | "build"
                        )
                })
            }) {
                count += 1;
            }
        }
        count
    }
    walk(root, root)
}

pub fn lint(dir: &str, settings: &Settings) -> Result<Json> {
    let root = require_dir(dir, "lint", "bundle_dir")?;
    let files = walk_source_files(&root)?;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut parsed: Vec<(String, crate::ingest::Parsed, Map<String, Json>)> = Vec::new();
    for path in &files {
        let rel = native_rel(path, &root).replace('\\', "/");
        let Ok(mut item) = parse_file(path, settings, Some(&root)) else {
            errors.push(finding(
                &rel,
                "parse",
                "ValueError: cannot parse source file",
            ));
            continue;
        };
        if settings.identity == "frontmatter" {
            if let Some(uid) = extra_string(&item.extra, "uid") {
                item.id = uid;
            }
        }
        let text = fs::read_to_string(path)?;
        let meta = parse_frontmatter(&text)
            .metadata
            .as_object()
            .cloned()
            .unwrap_or_default();
        let has = |key: &str| {
            meta.get(key)
                .and_then(Json::as_str)
                .is_some_and(|value| !value.is_empty())
        };
        if !has("type") {
            warnings.push(finding(
                &rel,
                "missing_type",
                "no 'type:' frontmatter; will import as 'note'",
            ));
        }
        if !has("title") {
            warnings.push(finding(
                &rel,
                "missing_title",
                "no 'title:' frontmatter; will use filename stem",
            ));
        }
        parsed.push((rel, item, meta));
    }

    let mut seen_ids: BTreeMap<String, String> = BTreeMap::new();
    for (rel, item, _) in &parsed {
        let key = if settings.identity == "frontmatter" {
            Some(item.id.clone())
        } else {
            extra_string(&item.extra, "uid")
        };
        let Some(key) = key else { continue };
        if let Some(other) = seen_ids.get(&key) {
            let suffix = if settings.identity == "frontmatter" {
                "; import refuses this file"
            } else {
                "; [[wikilinks]] to it are ambiguous"
            };
            errors.push(serde_json::json!({
                "file": rel,
                "rule": "duplicate_id",
                "message": format!("frontmatter id '{key}' is already used by {other}{suffix}"),
                "id": key,
                "other": other,
            }));
        } else {
            seen_ids.insert(key, rel.clone());
        }
    }

    let items: Vec<_> = parsed.iter().map(|(_, item, _)| item.clone()).collect();
    let index = LinkIndex::from_parsed(&items);
    let ids = path_ids(&items, &root);
    for (rel, item, meta) in &parsed {
        for raw in markdown_links(&item.body) {
            if is_external(&raw) {
                continue;
            }
            let (target, recorded) = link_target(&raw, &item.id, &index, Some(&ids));
            if target.is_none() {
                errors.push(serde_json::json!({
                    "file": rel,
                    "rule": "dangling_link",
                    "message": "would import as BrokenLink (target has no concept)",
                    "link": raw,
                    "target": if recorded.is_empty() { "(empty)" } else { &recorded },
                }));
            }
        }
        for raw in wikilinks(&item.body) {
            if raw.is_empty() || is_external(&raw) {
                continue;
            }
            let (target, reason, candidates) = index.resolve_wiki(&raw);
            if target.is_none() {
                let reason = reason.unwrap_or_else(|| "missing".into());
                let message = if reason == "ambiguous" {
                    format!(
                        "would import as BrokenLink (name is ambiguous: {})",
                        candidates.join(", ")
                    )
                } else {
                    "would import as BrokenLink (name resolves to nothing)".into()
                };
                errors.push(serde_json::json!({
                    "file": rel,
                    "rule": "dangling_wikilink",
                    "message": message,
                    "link": raw,
                    "reason": reason,
                    "candidates": candidates,
                }));
            }
        }
        let (pairs, problems) = relation_pairs(&relation_value(meta));
        for problem in problems {
            errors.push(finding(rel, "bad_relation", problem));
        }
        for (kind, raw) in pairs {
            let (target, reason, candidates, recorded) =
                resolve_relation_target(&raw, &item.id, &index, Some(&ids));
            if target.is_none() {
                let reason = reason.unwrap_or_else(|| "missing".into());
                errors.push(serde_json::json!({
                    "file": rel,
                    "rule": "dangling_relation",
                    "message": format!("relation {kind} would import as BrokenLink ({reason})"),
                    "link": raw,
                    "kind": kind,
                    "reason": reason,
                    "candidates": candidates,
                    "target": recorded,
                }));
            }
        }
    }
    errors.sort_by_key(finding_key);
    warnings.sort_by_key(finding_key);
    Ok(serde_json::json!({
        "dir": native_display(&root),
        "files": files.len(),
        "errors": errors,
        "warnings": warnings,
        "skipped_hidden": count_skipped(&root),
        "clean": errors.is_empty(),
    }))
}
