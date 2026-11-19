//! Bundle/concept export. Stage B mirrors the documented Python export API:
//! YAML frontmatter, graph-derived link enrichment, and generated indexes.

use crate::error::{OkfError, Result};
use crate::store::Store;
use serde_json::{Map, Value as Json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const CORE: &[&str] = &[
    "type",
    "title",
    "description",
    "resource",
    "tags",
    "timestamp",
];
const DROP: &[&str] = &[
    "id",
    "body",
    "embedding",
    "content_sig",
    "created_at",
    "updated_at",
    "effective_at",
    "source_path",
];

pub fn export_bundle(
    store: &Store,
    output_dir: &str,
    directory_id: Option<&str>,
    concept_type: Option<&str>,
    tags: Option<&[String]>,
    flavor: &str,
) -> Result<Json> {
    validate_flavor(flavor, "export_bundle")?;
    let root = Path::new(output_dir);
    fs_create_dir_all(root)?;
    let mut concepts: Vec<(String, Json, String, bool)> = Vec::new();
    for concept in store.export_concepts()? {
        let Some(id) = concept.get("id").and_then(Json::as_str).map(str::to_string) else {
            continue;
        };
        if let Some(want) = concept_type {
            if concept.get("type").and_then(Json::as_str) != Some(want) {
                continue;
            }
        }
        if let Some(want) = tags {
            let have: Vec<String> = concept
                .get("tags")
                .map(crate::model::normalize_tags)
                .unwrap_or_default();
            if !want.iter().all(|tag| have.contains(tag)) {
                continue;
            }
        }
        let in_subtree = match directory_id {
            Some(dir) if !dir.is_empty() => id.starts_with(&format!("{dir}/")),
            _ => true,
        };
        let body = concept
            .get("body")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        let uid = concept
            .get("uid")
            .and_then(Json::as_str)
            .map(str::to_string);
        let _ = uid;
        concepts.push((id, concept, body, in_subtree));
    }
    concepts.sort_by(|a, b| a.0.cmp(&b.0));
    let selected: Vec<_> = concepts
        .iter()
        .filter(|item| item.3)
        .map(|item| item.0.clone())
        .collect();
    let title_counts = if flavor == "obsidian" {
        title_counts(store)?
    } else {
        BTreeMap::new()
    };
    for (id, concept, body, include) in &concepts {
        if !include {
            continue;
        }
        let path = output_path(root, id);
        if let Some(parent) = path.parent() {
            fs_create_dir_all(parent)?;
        }
        let rendered = render_concept(store, concept, body, flavor, &title_counts)?;
        fs_write(&path, &rendered)?;
    }
    if flavor == "okf" {
        write_indexes(root, &concepts)?;
    }
    Ok(serde_json::json!({
        "output_dir": output_dir,
        "concept_ids": selected,
        "flavor": flavor,
    }))
}

pub fn export_concept(
    store: &Store,
    concept_id: &str,
    output_dir: &str,
    flavor: &str,
) -> Result<Json> {
    if output_dir.is_empty() {
        return Err(
            OkfError::new("MISSING_PARAM", "export_concept requires output_dir")
                .op("export_concept")
                .field("concept_id", Json::String(concept_id.into())),
        );
    }
    validate_flavor(flavor, "export_concept")?;
    let Some(concept) = store.get_by_id(concept_id)? else {
        return Err(OkfError::new(
            "UNKNOWN_CONCEPT",
            format!("concept '{concept_id}' does not exist"),
        )
        .op("export_concept")
        .field("concept_id", Json::String(concept_id.into())));
    };
    let body = concept
        .get("body")
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string();
    let title_counts = if flavor == "obsidian" {
        title_counts(store)?
    } else {
        BTreeMap::new()
    };
    let root = Path::new(output_dir);
    fs_create_dir_all(root)?;
    let path = root.join(format!("{concept_id}.md"));
    if let Some(parent) = path.parent() {
        fs_create_dir_all(parent)?;
    }
    fs_write(
        &path,
        &render_concept(store, &concept, &body, flavor, &title_counts)?,
    )?;
    Ok(serde_json::json!({
        "concept_id": concept_id,
        "path": path.to_string_lossy().into_owned(),
    }))
}

fn validate_flavor(flavor: &str, op: &str) -> Result<()> {
    if matches!(flavor, "okf" | "obsidian") {
        Ok(())
    } else {
        Err(OkfError::new(
            "BAD_VALUE",
            format!("flavor must be 'okf' or 'obsidian', got '{flavor}'"),
        )
        .op(op)
        .field("flavor", Json::String(flavor.into())))
    }
}

fn render_concept(
    store: &Store,
    concept: &Json,
    body: &str,
    flavor: &str,
    title_counts: &BTreeMap<String, usize>,
) -> Result<String> {
    let frontmatter = export_frontmatter(concept);
    let yaml = dump_yaml(&frontmatter, 0);
    let body = enrich_body(store, concept, body, flavor, title_counts)?;
    Ok(format!("---\n{yaml}---\n\n{body}"))
}

fn export_frontmatter(concept: &Json) -> Json {
    let mut data = Map::new();
    let Some(source) = concept.as_object() else {
        return Json::Object(data);
    };
    for key in CORE {
        if let Some(value) = source.get(*key) {
            let rendered = if *key == "timestamp" {
                value
                    .as_str()
                    .and_then(|text| crate::model::PyDateTime::parse(text).ok())
                    .map(|dt| Json::String(dt.isoformat()))
                    .unwrap_or_else(|| value.clone())
            } else {
                value.clone()
            };
            data.insert((*key).to_string(), rendered);
        }
    }
    for (key, value) in source {
        if CORE.contains(&key.as_str()) || DROP.contains(&key.as_str()) {
            continue;
        }
        data.insert(key.clone(), value.clone());
    }
    if let Some(uid) = source.get("uid") {
        data.remove("uid");
        data.insert("id".into(), uid.clone());
    }
    Json::Object(data)
}

fn enrich_body(
    store: &Store,
    concept: &Json,
    body: &str,
    flavor: &str,
    title_counts: &BTreeMap<String, usize>,
) -> Result<String> {
    let id = concept.get("id").and_then(Json::as_str).unwrap_or_default();
    let mut out = body.to_string();
    let wiki_refs = wikilinks(body);
    let outgoing = store.link_titles(id, true)?;
    if !outgoing.is_empty() {
        let mut lines = Vec::new();
        for (target, title) in &outgoing {
            let label = if title.is_empty() {
                target.rsplit('/').next().unwrap_or(target)
            } else {
                title
            };
            let exists = wiki_refs.iter().any(|value| value == target)
                || wiki_refs
                    .iter()
                    .any(|value| value.eq_ignore_ascii_case(label))
                || body.contains(&format!("]({target}"));
            if !exists {
                lines.push(render_link(target, label, flavor, title_counts));
            }
        }
        if !lines.is_empty() {
            out.push_str("\n## See Also\n");
            out.push_str(&lines.join("\n"));
        }
    }
    if flavor == "okf" {
        let incoming = store.link_titles(id, false)?;
        if !incoming.is_empty() {
            let mut lines = vec!["\n## Cited By\n".to_string()];
            for (source, title) in &incoming {
                let label = if title.is_empty() {
                    source.rsplit('/').next().unwrap_or(source)
                } else {
                    title
                };
                lines.push(render_link(source, label, flavor, title_counts));
            }
            out.push_str(&lines.join("\n"));
        }
    }
    Ok(out)
}

fn render_link(
    id: &str,
    label: &str,
    flavor: &str,
    title_counts: &BTreeMap<String, usize>,
) -> String {
    if flavor == "obsidian" {
        if title_counts
            .get(&label.to_ascii_lowercase())
            .copied()
            .unwrap_or(0)
            == 1
        {
            format!("- [[{label}]]")
        } else {
            format!("- [[{id}|{label}]]")
        }
    } else {
        format!("- [{label}]({id}.md)")
    }
}

fn title_counts(store: &Store) -> Result<BTreeMap<String, usize>> {
    let mut counts = BTreeMap::new();
    for concept in store.export_concepts()? {
        let title = concept
            .get("title")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        *counts.entry(title).or_insert(0) += 1;
    }
    Ok(counts)
}

fn write_indexes(root: &Path, concepts: &[(String, Json, String, bool)]) -> Result<()> {
    let mut children: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for (id, concept, _, _) in concepts {
        let parts: Vec<&str> = id.split('/').collect();
        for index in 1..parts.len() {
            let dir = parts[..index].join("/");
            let title = concept
                .get("title")
                .and_then(Json::as_str)
                .unwrap_or(parts[index])
                .to_string();
            children
                .entry(dir)
                .or_default()
                .push((title, format!("{}.md", id.replace('/', &native_sep()))));
        }
    }
    for (dir, mut entries) in children {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let heading = dir.rsplit('/').next().unwrap_or(&dir);
        let mut lines = vec![format!("# {heading}\n"), String::new()];
        for (title, rel) in entries {
            lines.push(format!("- [{title}]({rel})"));
        }
        lines.push(String::new());
        let path = root.join(dir.replace('/', &native_sep())).join("index.md");
        if let Some(parent) = path.parent() {
            fs_create_dir_all(parent)?;
        }
        fs_write(&path, &lines.join("\n"))?;
    }
    Ok(())
}

fn output_path(root: &Path, id: &str) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in id.split('/') {
        path.push(part);
    }
    path.set_extension("md");
    path
}

fn native_sep() -> String {
    std::path::MAIN_SEPARATOR.to_string()
}

fn wikilinks(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("[[") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find("]]") else { break };
        out.push(
            rest[..end]
                .split('|')
                .next()
                .unwrap_or_default()
                .trim()
                .to_string(),
        );
        rest = &rest[end + 2..];
    }
    out
}

fn dump_yaml(value: &Json, indent: usize) -> String {
    match value {
        Json::Object(map) => {
            let mut out = String::new();
            for (key, item) in map {
                match item {
                    Json::Object(inner) if !inner.is_empty() => {
                        out.push_str(&format!("{}{}:\n", " ".repeat(indent), key));
                        out.push_str(&dump_yaml(item, indent + 2));
                    }
                    Json::Array(items) if !items.is_empty() => {
                        out.push_str(&format!("{}{}:\n", " ".repeat(indent), key));
                        for entry in items {
                            out.push_str(&format!(
                                "{}- {}\n",
                                " ".repeat(indent),
                                yaml_scalar(entry)
                            ));
                        }
                    }
                    _ => out.push_str(&format!(
                        "{}{}: {}\n",
                        " ".repeat(indent),
                        key,
                        yaml_scalar(item)
                    )),
                }
            }
            out
        }
        other => format!("{}{}\n", " ".repeat(indent), yaml_scalar(other)),
    }
}

fn yaml_scalar(value: &Json) -> String {
    match value {
        Json::Null => "null".into(),
        Json::Bool(flag) => flag.to_string(),
        Json::Number(number) => number.to_string(),
        Json::String(text) => yaml_string(text),
        Json::Array(_) => "[]".into(),
        Json::Object(_) => "{}".into(),
    }
}

fn yaml_string(text: &str) -> String {
    let needs_quote = text.is_empty()
        || text.starts_with([
            '-', ':', '?', ',', '[', ']', '{', '}', '#', '&', '*', '!', '|', '>', '\'', '"', '%',
            '@', '`',
        ])
        || text.contains(": ")
        || text.contains(" #")
        || text.contains('\n')
        || looks_like_timestamp(text)
        || matches!(
            text,
            "null" | "true" | "false" | "yes" | "no" | "on" | "off" | "~"
        );
    if !needs_quote {
        return text.to_string();
    }
    format!("'{}'", text.replace('\'', "''"))
}

fn looks_like_timestamp(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 10
        && bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && bytes[..4].iter().all(u8::is_ascii_digit)
}

fn fs_create_dir_all(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).map_err(|err| {
        OkfError::new(
            "IO_ERROR",
            format!("cannot create {}: {err}", path.display()),
        )
    })
}

fn fs_write(path: &Path, content: &str) -> Result<()> {
    // Python's Path.write_text uses the platform newline on Windows.
    #[cfg(windows)]
    let content = content.replace("\r\n", "\n").replace('\n', "\r\n");
    std::fs::write(path, content).map_err(|err| {
        OkfError::new(
            "IO_ERROR",
            format!("cannot write {}: {err}", path.display()),
        )
    })
}
