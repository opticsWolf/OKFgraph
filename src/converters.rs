//! Stage B `ingest` dispatch: markdown, BibTeX, thoughts, and PDF refusal.

use crate::embed::{EmbedderPool, Pools};
use crate::error::{OkfError, Result};
use crate::export::fs_write;
use crate::ingest::{commit_parsed, finish_indexes, parse_file, Parsed};
use crate::model::{normalize_tags, utc_now_datetime};
use crate::pdf::{self, Plugin};
use crate::pycompat::{content_sig, PyValue};
use crate::settings::Settings;
use crate::store::Store;
use lbug::Value as LValue;
use serde_json::{Map, Value as Json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const MD_HONOURS: &[&str] = &[
    "md_path",
    "concept_id",
    "title",
    "description",
    "tags",
    "type",
    "extra",
    "materialize",
];
const BIB_HONOURS: &[&str] = &["bib_path", "tags"];
const THOUGHTS_HONOURS: &[&str] = &[
    "thoughts",
    "topic",
    "concept_id",
    "title",
    "description",
    "tags",
    "type",
    "extra",
    "materialize",
    "related",
    "folder",
];
const PDF_HONOURS: &[&str] = &["pdf_path", "output_dir"];
const RESERVED_EXTRA: &[&str] = &[
    "id",
    "type",
    "title",
    "description",
    "resource",
    "tags",
    "timestamp",
    "body",
    "embedding",
    "uid",
    "source_path",
];

fn honours(kind: &str) -> &'static [&'static str] {
    match kind {
        "md" => MD_HONOURS,
        "bib" => BIB_HONOURS,
        "thoughts" => THOUGHTS_HONOURS,
        "pdf" => PDF_HONOURS,
        _ => &[],
    }
}

fn validate(kind: &str, params: &Map<String, Json>) -> Result<()> {
    let allowed = honours(kind);
    if allowed.is_empty() {
        return Err(
            OkfError::new("BAD_VALUE", format!("unknown ingest kind '{kind}'"))
                .field("kind", Json::String(kind.into())),
        );
    }
    let mut ignored: Vec<String> = params
        .iter()
        .filter(|(key, value)| {
            *key != "kind"
                && !value.is_null()
                && !allowed.contains(&key.as_str())
                && !(kind == "pdf"
                    && matches!(
                        key.as_str(),
                        "auto_import"
                            | "routing_mode"
                            | "extract_images"
                            | "pdf_device"
                            | "pdf_precision"
                    ))
        })
        .map(|(key, _)| key.clone())
        .collect();
    ignored.sort();
    if !ignored.is_empty() {
        let mut err = OkfError::new(
            "BAD_VALUE",
            format!("kind='{kind}' ignores: {}", ignored.join(", ")),
        )
        .field("kind", Json::String(kind.into()))
        .field(
            "ignored",
            Json::Array(ignored.into_iter().map(Json::String).collect()),
        );
        err = err.remedy(if kind == "pdf" {
            "output_dir needs auto_import=False"
        } else {
            "drop the params of other kinds"
        });
        return Err(err);
    }
    Ok(())
}

fn check_extra(value: Option<&Json>) -> Result<BTreeMap<String, Json>> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let Some(object) = value.as_object() else {
        return Err(
            OkfError::new("BAD_VALUE", "extra must be a mapping of key -> value")
                .field("extra", Json::String("object".into())),
        );
    };
    let mut reserved: Vec<String> = object
        .keys()
        .filter(|key| RESERVED_EXTRA.contains(&key.as_str()))
        .cloned()
        .collect();
    if !reserved.is_empty() {
        reserved.sort();
        return Err(OkfError::new(
            "BAD_VALUE",
            format!(
                "extra may not set {} (use the dedicated parameters: concept_id, type, title, description, tags)",
                reserved.join(", ")
            ),
        )
        .field(
            "reserved",
            Json::Array(reserved.into_iter().map(Json::String).collect()),
        ));
    }
    Ok(object.clone().into_iter().collect())
}

fn require_file(path: Option<&str>, label: &str, field: &str) -> Result<String> {
    let Some(path) = path else {
        return Err(OkfError::new("MISSING_PARAM", format!("requires {field}"))
            .field("kind", Json::Null)
            .field(field, Json::Null));
    };
    if !Path::new(path).exists() {
        return Err(
            OkfError::new("FILE_NOT_FOUND", format!("{label} not found: {path}"))
                .field("path", Json::String(path.into())),
        );
    }
    Ok(path.to_string())
}

fn counts(store: &Store, id: &str) -> Result<(usize, usize)> {
    let chunks = store.get_chunks(id)?.len();
    let images = store
        .query("MATCH (i:ImageAsset) RETURN count(i) AS n", vec![])?
        .first()
        .and_then(|row| row.get("n"))
        .and_then(Json::as_i64)
        .unwrap_or(0) as usize;
    Ok((chunks, images))
}

fn commit(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    parsed: Parsed,
) -> Result<()> {
    let mut settings = settings.clone();
    settings.identity = "frontmatter".into();
    let engine = embedder.get(&settings)?;
    crate::embed::check_pins(&settings, &store.pin_info()?)?;
    commit_parsed(
        store,
        &settings,
        engine.as_ref(),
        pools,
        std::slice::from_ref(&parsed),
        None,
        true,
    )?;
    finish_indexes(store)
}

fn temp_markdown(text: &str) -> Result<PathBuf> {
    let name = format!("okf-ingest-{}.md", uuid::Uuid::new_v4());
    let path = std::env::temp_dir().join(name);
    fs_write(&path, text)?;
    Ok(path)
}

fn parse_markdown(text: &str, id: &str, settings: &Settings) -> Result<Parsed> {
    let path = temp_markdown(text)?;
    let mut settings = settings.clone();
    settings.identity = "frontmatter".into();
    let mut parsed = parse_file(&path, &settings, None)?;
    parsed.id = id.to_string();
    let authored: Vec<(String, PyValue)> = vec![
        ("id".into(), PyValue::Str(id.into())),
        ("type".into(), PyValue::Str(parsed.concept_type.clone())),
        ("title".into(), PyValue::Str(parsed.title.clone())),
        (
            "description".into(),
            parsed
                .description
                .clone()
                .map_or(PyValue::Null, PyValue::Str),
        ),
        (
            "resource".into(),
            parsed.resource.clone().map_or(PyValue::Null, PyValue::Str),
        ),
        (
            "tags".into(),
            PyValue::List(parsed.tags.iter().cloned().map(PyValue::Str).collect()),
        ),
        ("timestamp".into(), PyValue::Null),
    ];
    parsed.sig = content_sig(&PyValue::Dict(authored), &parsed.body);
    let _ = fs::remove_file(path);
    Ok(parsed)
}

fn slugify(value: &str) -> String {
    let safe: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.trim_matches(|ch| ch == '.' || ch == '_');
    if safe.is_empty() {
        "untitled".into()
    } else {
        safe.into()
    }
}

#[derive(Debug)]
struct BibEntry {
    entrytype: String,
    key: String,
    fields: BTreeMap<String, String>,
}

fn parse_bibtex(text: &str) -> Vec<BibEntry> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(rel) = text[i..].find('@') {
        let at = i + rel;
        let Some(etype_start) =
            text[at + 1..].find(|c: char| c.is_whitespace() || c == '{' || c == '(')
        else {
            break;
        };
        let entrytype = text[at + 1..at + 1 + etype_start].to_ascii_lowercase();
        let mut p = at + 1 + etype_start;
        while p < bytes.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        if p >= bytes.len() || !matches!(bytes[p], b'{' | b'(') {
            i = at + 1;
            continue;
        }
        let close = if bytes[p] == b'{' { b'}' } else { b')' };
        let start = p + 1;
        let mut depth = 1;
        let mut end = start;
        while end < bytes.len() {
            if bytes[end] == b'{' {
                depth += 1;
            }
            if bytes[end] == b'}' && close == b'}' {
                depth -= 1;
            }
            if bytes[end] == b')' && close == b')' {
                depth -= 1;
            }
            if depth == 0 {
                break;
            }
            end += 1;
        }
        if end >= bytes.len() {
            break;
        }
        let body = &text[start..end];
        let mut parts = Vec::new();
        let mut cur = String::new();
        let mut in_quote = false;
        let mut braces = 0;
        for ch in body.chars() {
            match ch {
                '"' => {
                    in_quote = !in_quote;
                    cur.push(ch);
                }
                '{' => {
                    braces += 1;
                    cur.push(ch);
                }
                '}' => {
                    braces -= 1;
                    cur.push(ch);
                }
                ',' if !in_quote && braces == 0 => {
                    parts.push(cur.trim().to_string());
                    cur.clear();
                }
                _ => cur.push(ch),
            }
        }
        if !cur.trim().is_empty() {
            parts.push(cur.trim().to_string());
        }
        let key = parts.first().cloned().unwrap_or_default();
        let mut fields = BTreeMap::new();
        for part in parts.iter().skip(1) {
            if let Some((name, value)) = part.split_once('=') {
                let value = value.trim().trim_matches(|c| c == '{' || c == '}').trim();
                fields.insert(name.trim().to_ascii_lowercase(), value.to_string());
            }
        }
        if !entrytype.is_empty() && !key.is_empty() {
            out.push(BibEntry {
                entrytype,
                key,
                fields,
            });
        }
        i = end + 1;
    }
    out
}

fn ingest_md(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    params: &Map<String, Json>,
) -> Result<Json> {
    let path = require_file(
        params.get("md_path").and_then(Json::as_str),
        "markdown file",
        "md_path",
    )?;
    let mut parsed = parse_file(Path::new(&path), settings, None)?;
    if parsed.description.is_none() {
        parsed.description = Some(String::new());
    }
    parsed.tags.sort();
    parsed.tags.dedup();
    if let Some(id) = params.get("concept_id").and_then(Json::as_str) {
        parsed.id = id.into();
    }
    if let Some(title) = params.get("title").and_then(Json::as_str) {
        parsed.title = title.into();
    }
    if let Some(description) = params.get("description").and_then(Json::as_str) {
        parsed.description = Some(description.into());
    }
    if let Some(kind) = params.get("type").and_then(Json::as_str) {
        parsed.concept_type = kind.into();
    }
    if let Some(tags) = params.get("tags") {
        let mut values = normalize_tags(tags);
        values.extend(parsed.tags);
        values.sort();
        values.dedup();
        parsed.tags = values;
    }
    let _extra = check_extra(params.get("extra"))?;
    commit(store, settings, embedder, pools, parsed.clone())?;
    let (chunk_count, image_count) = counts(store, &parsed.id)?;
    Ok(serde_json::json!({
        "concept_id": parsed.id,
        "title": parsed.title,
        "description": parsed.description,
        "tags": parsed.tags,
        "chunk_count": chunk_count,
        "image_count": image_count,
        "lint_issues": {
            "fixed_count": 0,
            "fixable_count": 0,
            "unfixable_count": 0,
            "error_count": 0,
        },
    }))
}

fn ingest_bib(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    params: &Map<String, Json>,
) -> Result<Json> {
    let path = require_file(
        params.get("bib_path").and_then(Json::as_str),
        "BibTeX file",
        "bib_path",
    )?;
    let text =
        fs::read_to_string(&path).map_err(|err| OkfError::new("IO_ERROR", err.to_string()))?;
    let file_tags = params.get("tags").map(normalize_tags).unwrap_or_default();
    let entries = parse_bibtex(&text);
    if entries.is_empty() {
        return Err(
            OkfError::new("BAD_VALUE", format!("no usable BibTeX entries in {path}"))
                .field("bib_path", Json::String(path))
                .field("skipped", Json::Array(Vec::new())),
        );
    }
    let mut ids = Vec::new();
    let mut seen = BTreeMap::new();
    for entry in entries {
        let slug = slugify(&entry.key);
        let count = seen.entry(slug.clone()).or_insert(0);
        *count += 1;
        let id = if *count == 1 {
            format!("refs/{slug}")
        } else {
            format!("refs/{slug}~{count}")
        };
        let title = entry
            .fields
            .get("title")
            .cloned()
            .unwrap_or_else(|| entry.key.clone());
        let authors = entry.fields.get("author").cloned().unwrap_or_default();
        let year = entry.fields.get("year").cloned().unwrap_or_default();
        let venue = ["journal", "booktitle", "school", "publisher"]
            .iter()
            .find_map(|key| {
                entry
                    .fields
                    .get(*key)
                    .filter(|value| !value.is_empty())
                    .cloned()
            })
            .unwrap_or_default();
        let head = [
            authors,
            if year.is_empty() {
                String::new()
            } else {
                format!("({year})")
            },
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
        let citation = [head, venue.clone()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(". ");
        let mut tags = vec!["reference".to_string(), entry.entrytype.clone()];
        if !year.is_empty() {
            tags.push(year.clone());
        }
        tags.extend(file_tags.iter().cloned());
        let mut meta = BTreeMap::new();
        meta.insert("title".to_string(), Json::String(title.clone()));
        meta.insert("type".to_string(), Json::String("reference".into()));
        meta.insert("citekey".to_string(), Json::String(entry.key.clone()));
        meta.insert(
            "entrytype".to_string(),
            Json::String(entry.entrytype.clone()),
        );
        for (key, bib_key) in [
            ("author", "author"),
            ("year", "year"),
            ("doi", "doi"),
            ("url", "url"),
        ] {
            if let Some(value) = entry.fields.get(bib_key).filter(|value| !value.is_empty()) {
                meta.insert(key.to_string(), Json::String(value.clone()));
            }
        }
        if !venue.is_empty() {
            meta.insert("venue".to_string(), Json::String(venue));
        }
        let yaml = crate::export::dump_yaml(&Json::Object(meta.into_iter().collect()), 0);
        let mut body = format!("## Citation\n\n{citation}\n\n");
        let mut links = Vec::new();
        if let Some(doi) = entry.fields.get("doi").filter(|value| !value.is_empty()) {
            links.push(format!("[DOI](https://doi.org/{doi})"));
        }
        if let Some(url) = entry.fields.get("url").filter(|value| !value.is_empty()) {
            links.push(format!("[URL]({url})"));
        }
        if !links.is_empty() {
            body.push_str(&format!("{}\n\n", links.join(" ")));
        }
        let abstract_text = entry
            .fields
            .get("abstract")
            .filter(|value| !value.is_empty())
            .cloned()
            .unwrap_or_else(|| "_No abstract provided._".into());
        body.push_str(&format!("## Abstract\n\n{abstract_text}\n"));
        let markdown = format!("---\n{yaml}---\n\n{body}");
        let mut parsed = parse_markdown(&markdown, &id, settings)?;
        parsed.body = markdown.clone();
        parsed.extra.clear();
        parsed.title = title;
        parsed.description = Some(citation);
        parsed.concept_type = "reference".into();
        parsed.tags = tags;
        commit(store, settings, embedder, pools, parsed)?;
        ids.push(Json::String(id));
    }
    Ok(serde_json::json!({
        "concept_ids": ids,
        "entry_count": ids.len(),
        "skipped_entries": Json::Array(Vec::new()),
    }))
}

fn json_string(value: Option<&Json>) -> Option<String> {
    value.and_then(Json::as_str).map(str::to_string)
}

fn thought_title(text: &str, topic: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let sentence = flat
        .split_once(['.', '!', '?'])
        .map(|(head, _)| format!("{head}."))
        .unwrap_or(flat);
    let sentence = sentence.trim_end_matches(['.', '!', '?', ' ']);
    if sentence.chars().filter(|ch| ch.is_alphanumeric()).count() < 3 {
        let now = utc_now_datetime();
        let iso = now.isoformat();
        format!("{topic} — {}", iso[..16].replace('T', " "))
    } else {
        let mut out = String::new();
        let mut last_space = 0usize;
        for (idx, ch) in sentence.char_indices() {
            if ch == ' ' && idx >= 60 {
                break;
            }
            if ch == ' ' {
                last_space = idx;
            }
            out.push(ch);
        }
        if out.chars().count() > 80 {
            out.truncate(last_space);
        }
        out
    }
}

fn ingest_thoughts(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    params: &Map<String, Json>,
) -> Result<Json> {
    let thoughts = params
        .get("thoughts")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            OkfError::new(
                "MISSING_PARAM",
                "kind='thoughts' requires thoughts and topic",
            )
            .field("thoughts", Json::Bool(false))
            .field(
                "topic",
                Json::Bool(params.get("topic").and_then(Json::as_str).is_some()),
            )
        })?;
    let topic = params
        .get("topic")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            OkfError::new(
                "MISSING_PARAM",
                "kind='thoughts' requires thoughts and topic",
            )
            .field("thoughts", Json::Bool(true))
            .field("topic", Json::Bool(false))
        })?;

    let now = utc_now_datetime();
    let created = now.isoformat();
    let concept_id = json_string(params.get("concept_id"))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            let slug: String = topic
                .to_lowercase()
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                        ch
                    } else {
                        '_'
                    }
                })
                .collect();
            let slug = slug.trim_matches(['.', '_']);
            let stamp = created[..19].replace(['-', ':', 'T'], "");
            format!(
                "thoughts/{slug}/{}_{}",
                stamp,
                &uuid::Uuid::new_v4().to_string()[..6]
            )
        });
    let title = json_string(params.get("title"))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| thought_title(thoughts, topic));
    let description = json_string(params.get("description"))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("Reasoning about {topic}"));
    let concept_type = json_string(params.get("type"))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "thought".into());
    let mut tags = vec![
        "thought".to_string(),
        "reasoning".to_string(),
        topic.to_string(),
    ];
    if let Some(value) = params.get("tags") {
        tags.extend(normalize_tags(value));
    }
    let mut unique: Vec<String> = Vec::new();
    for tag in tags.drain(..) {
        if !unique.contains(&tag) {
            unique.push(tag);
        }
    }
    tags = unique;
    if topic == "thought" && tags.len() == 2 {
        tags = vec!["thought".into(), "reasoning".into()];
    }

    let mut extra = BTreeMap::new();
    extra.insert("created".into(), created.clone());
    if let Some(Json::Object(values)) = params.get("extra") {
        for (key, value) in values {
            if RESERVED_EXTRA.contains(&key.as_str()) {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    format!(
                        "extra may not set {key} (use the dedicated parameters: concept_id, type, title, description, tags)"
                    ),
                )
                .field("reserved", Json::Array(vec![Json::String(key.clone())])));
            }
            extra.insert(
                key.clone(),
                crate::pycompat::extra_value(&PyValue::from_golden(value)),
            );
        }
    }
    let related: Vec<String> = match params.get("related") {
        Some(Json::Array(items)) => items
            .iter()
            .filter_map(Json::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    if !related.is_empty() {
        extra.insert(
            "relations".into(),
            serde_json::json!({"related": related}).to_string(),
        );
    }

    let mut meta: BTreeMap<String, Json> = BTreeMap::new();
    meta.insert("created".into(), Json::String(created.clone()));
    let meta_tags = vec![
        "thought".to_string(),
        "reasoning".to_string(),
        topic.to_string(),
    ];
    meta.insert(
        "tags".into(),
        serde_json::to_value(meta_tags).unwrap_or_default(),
    );
    meta.insert("thought_type".into(), Json::String("reasoning".into()));
    meta.insert("title".into(), Json::String(title.clone()));
    meta.insert("topic".into(), Json::String(topic.into()));
    meta.insert("type".into(), Json::String(concept_type.clone()));
    for (key, value) in params
        .get("extra")
        .and_then(Json::as_object)
        .into_iter()
        .flatten()
    {
        meta.insert(key.clone(), value.clone());
    }
    let yaml = crate::export::dump_yaml(&Json::Object(meta.into_iter().collect()), 0);
    let markdown = format!("---\n{yaml}---\n\n{thoughts}\n");

    let mut parsed = parse_markdown(&markdown, &concept_id, settings)?;
    parsed.body = markdown.clone();
    parsed.title = title.clone();
    parsed.concept_type = concept_type.clone();
    parsed.description = Some(description.clone());
    parsed.tags = tags.clone();
    parsed.timestamp = Some(now);
    parsed.extra = extra;
    let mut authored: Vec<(String, PyValue)> = vec![
        ("id".into(), PyValue::Str(concept_id.clone())),
        ("type".into(), PyValue::Str(concept_type.clone())),
        ("title".into(), PyValue::Str(title.clone())),
        ("description".into(), PyValue::Str(description.clone())),
        ("resource".into(), PyValue::Null),
        (
            "tags".into(),
            PyValue::List(tags.iter().cloned().map(PyValue::Str).collect()),
        ),
        ("timestamp".into(), PyValue::DateTime(now.isoformat())),
        ("thought_type".into(), PyValue::Str("reasoning".into())),
        ("topic".into(), PyValue::Str(topic.into())),
        ("created".into(), PyValue::Str(created)),
    ];
    for (key, value) in params
        .get("extra")
        .and_then(Json::as_object)
        .into_iter()
        .flatten()
    {
        authored.push((key.clone(), PyValue::from_golden(value)));
    }
    parsed.authored = BTreeMap::new();
    parsed.sig = content_sig(&PyValue::Dict(authored), &markdown);

    commit(store, settings, embedder, pools, parsed)?;
    // The single-concept path does not run the bundle's directory-hierarchy
    // pass; thoughts still get a virtual namespace, so materialize it here.
    let parts: Vec<&str> = concept_id.split('/').collect();
    for index in 1..parts.len() {
        let parent = parts[..index].join("/");
        let child = parts[..=index].join("/");
        store.write(
            "MERGE (:Directory {id: $id})",
            vec![("id", LValue::String(parent.clone()))],
        )?;
        let (from_label, to_label, from, to) = if index + 1 == parts.len() {
            ("Directory", "Concept", parent, child)
        } else {
            store.write(
                "MERGE (:Directory {id: $id})",
                vec![("id", LValue::String(child.clone()))],
            )?;
            ("Directory", "Directory", parent, child)
        };
        let _ = (from_label, to_label);
        store.write(
            &format!("MATCH (a:{from_label} {{id: $from}}), (b:{to_label} {{id: $to}}) MERGE (a)-[:CONTAINS]->(b)"),
            vec![
                ("from", LValue::String(from)),
                ("to", LValue::String(to)),
            ],
        )?;
    }
    let chunk_count = store.get_chunks(&concept_id)?.len();
    let mut out = serde_json::json!({
        "concept_id": concept_id.clone(),
        "title": title,
        "topic": topic,
        "tags": tags,
        "chunk_count": chunk_count,
        "markdown": markdown,
        "lint_issues": {
            "content": markdown,
            "fixed": true,
            "fixed_count": 1,
            "unfixable": [],
            "errors": [],
        },
    });
    if !related.is_empty() {
        let raw: Vec<String> = store
            .list_broken_links()?
            .into_iter()
            .filter(|link| link.get("source").and_then(Json::as_str) == Some(concept_id.as_str()))
            .filter_map(|link| link.get("raw").and_then(Json::as_str).map(str::to_string))
            .collect();
        out["unresolved_related"] = serde_json::to_value(raw).unwrap_or_default();
    }
    Ok(out)
}

fn ingest_pdf(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    params: &Map<String, Json>,
) -> Result<Json> {
    let pdf_path = json_string(params.get("pdf_path"))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            OkfError::new("MISSING_PARAM", "kind='pdf' requires pdf_path")
                .field("pdf_path", Json::Null)
        })?;
    let source = PathBuf::from(&pdf_path);
    if !source.is_file() {
        return Err(
            OkfError::new("FILE_NOT_FOUND", format!("PDF not found: {pdf_path}"))
                .field("path", Json::String(pdf_path.clone())),
        );
    }
    let auto_import = params
        .get("auto_import")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    let output_dir = if auto_import {
        std::env::temp_dir().join(format!("okf_ingest_{}", uuid::Uuid::new_v4()))
    } else {
        params
            .get("output_dir")
            .and_then(Json::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                source
                    .parent()
                    .map(Path::new)
                    .unwrap_or(Path::new("."))
                    .to_path_buf()
            })
    };
    fs::create_dir_all(&output_dir)?;
    let plugin = Plugin::load()?;
    let converted = plugin.convert(&source, &output_dir, &pdf::config(settings))?;
    if auto_import {
        let mut import_settings = settings.clone();
        import_settings.bundle_root = Some(output_dir.to_string_lossy().into_owned());
        let report = crate::ingest::import_bundle(
            store,
            &import_settings,
            embedder,
            pools,
            Some(&output_dir.to_string_lossy()),
        )?;
        let concept_ids = report
            .get("concept_ids")
            .cloned()
            .unwrap_or_else(|| Json::Array(Vec::new()));
        return Ok(serde_json::json!({
            "md_path": converted.md_path.to_string_lossy(),
            "concept_ids": concept_ids,
            "image_dir": converted.image_dir.to_string_lossy(),
            "page_count": converted.page_count,
        }));
    }
    Ok(serde_json::json!({
        "md_path": converted.md_path.to_string_lossy(),
        "concept_ids": Json::Array(Vec::new()),
        "image_dir": converted.image_dir.to_string_lossy(),
        "page_count": converted.page_count,
    }))
}

pub fn ingest(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    params: &Map<String, Json>,
) -> Result<Json> {
    let kind = params.get("kind").and_then(Json::as_str).ok_or_else(|| {
        OkfError::new("MISSING_PARAM", "kind is required").field("kind", Json::Null)
    })?;
    validate(kind, params)?;
    match kind {
        "md" => ingest_md(store, settings, embedder, pools, params),
        "bib" => ingest_bib(store, settings, embedder, pools, params),
        "pdf" => ingest_pdf(store, settings, embedder, pools, params),
        "thoughts" => ingest_thoughts(store, settings, embedder, pools, params),
        _ => Err(OkfError::new("BAD_VALUE", format!("unknown ingest kind '{kind}'")).op("ingest")),
    }
}

#[allow(dead_code)]
fn _unused_now() -> String {
    utc_now_datetime().isoformat()
}
