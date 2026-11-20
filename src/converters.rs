//! Stage B `ingest` dispatch: markdown, BibTeX, thoughts, and PDF refusal.

use crate::embed::{EmbedderPool, Pools};
use crate::error::{OkfError, Result};
use crate::export::fs_write;
use crate::ingest::{commit_parsed, finish_indexes, parse_file, refresh_content_sig, Parsed};
use crate::model::{local_now_datetime, normalize_tags, utc_now_datetime};
use crate::pdf::{self, Plugin};
use crate::pycompat::{content_sig, PyValue};
use crate::settings::Settings;
use crate::store::Store;
use lbug::Value as LValue;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

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
/// Params Python accepts for every kind but does not include in the
/// per-kind ignored-param check (they select the shared import pipeline).
const COMMON_INGEST_PARAMS: &[&str] = &[
    "mode",
    "routing_mode",
    "extract_images",
    "auto_import",
    "batch_size",
    "prune_missing",
    "force",
    "rebuild",
    "on_page",
    "converter",
];
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
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("kind must be 'md', 'pdf', 'thoughts' or 'bib', got '{kind}'"),
        )
        .field("kind", Json::String(kind.into())));
    }
    if let Some(rebuild) = params.get("rebuild").and_then(Json::as_str) {
        if !matches!(rebuild, "auto" | "defer" | "force") {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("rebuild must be 'auto', 'defer' or 'force', got {rebuild:?}"),
            )
            .op("ingest")
            .field("rebuild", Json::String(rebuild.into())));
        }
    }
    let auto_import = params
        .get("auto_import")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    // Python removes output_dir from PDF's honours when auto-importing.
    let allowed: Vec<&str> = if kind == "pdf" && auto_import {
        allowed
            .iter()
            .copied()
            .filter(|name| *name != "output_dir")
            .collect()
    } else {
        allowed.to_vec()
    };
    let mut ignored: Vec<String> = params
        .iter()
        .filter(|(key, value)| {
            *key != "kind"
                && !value.is_null()
                && !allowed.contains(&key.as_str())
                && !COMMON_INGEST_PARAMS.contains(&key.as_str())
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

/// Python's `_lint_converted_md` diagnostic categories.
///
/// Returns the (possibly fixed) content plus the report shape shared by
/// `_lint_converted_md`/`_lint_converted_md_str`. Markdown ingest only wants
/// the counts; converters that generate markdown (thoughts/BibTeX/PDF) fix it
/// first, exactly like the Python pipeline.
fn lint_markdown_full(content: &str, auto_fix: bool) -> (String, Json) {
    use mordant::linter::{
        parse_suppressions, run_fix_with_parser, run_lint, LintConfig, RuleParams,
    };
    use mordant::parser::{
        gfm_strikethrough, gfm_table, gfm_task_list_item, Parser, ParserExtension,
    };

    let parser = Parser::with_extensions(
        mordant::parser::Options::default(),
        gfm_table()
            .and(gfm_strikethrough())
            .and(gfm_task_list_item())
            .and(mordant::meta::meta_parser_extension(
                mordant::meta::MetaParserOptions::default(),
            )),
    );
    let config = LintConfig {
        disable: Vec::new(),
        enable: None,
        _enabled_when_default_false: None,
        suppressions: parse_suppressions(content),
        params: RuleParams::default(),
    };
    let mut reader = mordant::text::BasicReader::new(content);
    let (arena, root) = parser.parse(&mut reader);
    let diagnostics = run_lint(content, &arena, root, &config);
    let fixable_rules = ["MD009", "MD012", "MD047"];
    let error_rules = ["MD001", "MD031", "MD033"];
    let unfixable: Vec<&str> = diagnostics
        .iter()
        .filter(|item| !fixable_rules.contains(&item.rule))
        .map(|item| item.rule)
        .collect();
    let errors: Vec<&str> = diagnostics
        .iter()
        .filter(|item| error_rules.contains(&item.rule))
        .map(|item| item.rule)
        .collect();
    let fixable = diagnostics
        .iter()
        .filter(|item| fixable_rules.contains(&item.rule))
        .count();

    let (output, fixed_count) = if auto_fix && fixable > 0 {
        let outcome = run_fix_with_parser(content, &parser, &config, None);
        (outcome.output, outcome.fixed.len())
    } else {
        (content.to_string(), 0)
    };

    (
        output,
        serde_json::json!({
            "fixed": fixed_count > 0,
            "fixed_count": fixed_count,
            "fixable_count": fixable,
            "unfixable_count": unfixable.len(),
            "error_count": errors.len(),
            "unfixable": unfixable,
            "errors": errors,
        }),
    )
}

fn lint_markdown(content: &str) -> Json {
    let report = lint_markdown_full(content, false).1;
    serde_json::json!({
        "fixed_count": report.get("fixed_count").cloned().unwrap_or(Json::from(0)),
        "fixable_count": report.get("fixable_count").cloned().unwrap_or(Json::from(0)),
        "unfixable_count": report.get("unfixable_count").cloned().unwrap_or(Json::from(0)),
        "error_count": report.get("error_count").cloned().unwrap_or(Json::from(0)),
    })
}

/// `_lint_converted_md_str` report (thoughts and other generated markdown).
fn lint_markdown_str(report: Json, output: &str) -> Json {
    let mut map = Map::new();
    map.insert("content".into(), Json::String(output.to_string()));
    for key in ["fixed", "fixed_count", "unfixable", "errors"] {
        map.insert(key.into(), report.get(key).cloned().unwrap_or(Json::Null));
    }
    Json::Object(map)
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
        .query(
            "MATCH (c:Concept {id: $id})-[:INCLUDES_ASSET]->(i:ImageAsset)              RETURN count(i) AS n",
            vec![("id", lbug::Value::String(id.into()))],
        )?
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
    crate::ingest::pin_graph(store, &settings, engine.as_ref())?;
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
    // Remove the temp file on parse failure too; `path` remains as diagnostic
    // data but `synthetic` keeps it out of the FileHash ledger.
    let parsed = parse_file(&path, &settings, None);
    let _ = fs::remove_file(&path);
    let mut parsed = parsed?;
    parsed.id = id.to_string();
    parsed.synthetic = true;
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
    Ok(parsed)
}

fn slugify(value: &str) -> String {
    let mut out = String::new();
    let mut in_run = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
            out.push(ch);
            in_run = false;
        } else if !in_run {
            out.push('_');
            in_run = true;
        }
    }
    let out = out.trim_matches(['.', '_']);
    let out: String = out.chars().take(64).collect();
    if out.is_empty() {
        "untitled".into()
    } else {
        out
    }
}

fn unwrap_braces(value: &str) -> String {
    let mut current = value.to_string();
    loop {
        let mut out = String::new();
        let mut changed = false;
        let mut chars = current.char_indices().peekable();
        while let Some((_, ch)) = chars.next() {
            if ch == '{' {
                let mut inner = String::new();
                let mut closed = false;
                for (_, next) in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    if next == '{' {
                        changed = false;
                        break;
                    }
                    inner.push(next);
                }
                if closed && !inner.contains('{') {
                    out.push_str(&inner);
                    changed = true;
                } else {
                    out.push('{');
                    out.push_str(&inner);
                    if closed {
                        out.push('}');
                    }
                }
            } else {
                out.push(ch);
            }
        }
        if !changed || out == current {
            return out;
        }
        current = out;
    }
}

#[derive(Debug)]
struct BibEntry {
    entrytype: String,
    key: String,
    fields: BTreeMap<String, String>,
    line: usize,
}

fn split_bib_fields(body: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    let mut braces = 0usize;
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
                braces = braces.saturating_sub(1);
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
    parts
}

/// Evaluate a BibTeX value: strip braces/quotes, expand `@string` macros and
/// concatenate `#` parts. Unbalanced input is kept verbatim.
fn eval_bib_value(raw: &str, strings: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    for part in raw.split('#') {
        let mut part = part.trim();
        if part.len() >= 2 && part.starts_with('{') && part.ends_with('}') {
            part = &part[1..part.len() - 1];
            out.push_str(part);
        } else if part.len() >= 2 && part.starts_with('"') && part.ends_with('"') {
            out.push_str(&part[1..part.len() - 1]);
        } else if let Some(value) = strings.get(&part.to_ascii_lowercase()) {
            out.push_str(value);
        } else {
            out.push_str(part);
        }
    }
    out
}

fn collect_bib_strings(text: &str) -> BTreeMap<String, String> {
    let mut strings = BTreeMap::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(rel) = text[i..].find('@') {
        let at = i + rel;
        let rest = &text[at + 1..];
        let Some(name_end) = rest.find(|c: char| c.is_whitespace() || c == '{' || c == '(') else {
            break;
        };
        if rest[..name_end].eq_ignore_ascii_case("string") {
            let mut p = at + 1 + name_end;
            while p < bytes.len() && bytes[p].is_ascii_whitespace() {
                p += 1;
            }
            if p < bytes.len() && matches!(bytes[p], b'{' | b'(') {
                let close = if bytes[p] == b'{' { b'}' } else { b')' };
                let start = p + 1;
                let mut depth = 1;
                let mut end = start;
                while end < bytes.len() {
                    if bytes[end] == bytes[p] {
                        depth += 1;
                    } else if bytes[end] == close {
                        depth -= 1;
                    }
                    if depth == 0 {
                        break;
                    }
                    end += 1;
                }
                if end < bytes.len() {
                    for field in split_bib_fields(&text[start..end]) {
                        if let Some((name, value)) = field.split_once('=') {
                            let name = name.trim().to_ascii_lowercase();
                            if !name.is_empty() {
                                strings.insert(name, eval_bib_value(value.trim(), &strings));
                            }
                        }
                    }
                    i = end + 1;
                    continue;
                }
            }
        }
        i = at + 1;
    }
    strings
}

fn parse_bibtex(text: &str) -> Vec<BibEntry> {
    let strings = collect_bib_strings(text);
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
        if matches!(entrytype.as_str(), "comment" | "preamble" | "string") {
            // Opaque/macro definitions: Python's parser consumes them and
            // never emits a concept.
            i = end + 1;
            continue;
        }
        let parts = split_bib_fields(body);
        let key = parts.first().cloned().unwrap_or_default();
        let mut fields = BTreeMap::new();
        for part in parts.iter().skip(1) {
            if let Some((name, value)) = part.split_once('=') {
                fields.insert(
                    name.trim().to_ascii_lowercase(),
                    eval_bib_value(value.trim(), &strings),
                );
            }
        }
        if !entrytype.is_empty() && !key.is_empty() {
            out.push(BibEntry {
                entrytype,
                key,
                fields,
                line: text[..at].bytes().filter(|byte| *byte == b'\n').count() + 1,
            });
        }
        i = end + 1;
    }
    out
}

fn materialize_parsed(
    store: &Store,
    settings: &Settings,
    embedder: &EmbedderPool,
    pools: &Pools,
    parsed: &Parsed,
    op: &str,
) -> Result<Json> {
    let Some(root) = settings.bundle_root.as_ref() else {
        return Err(OkfError::new(
            "MISSING_PARAM",
            "materialize needs a bundle root: this router is file-free (bundle_root=None)",
        )
        .op(op)
        .field("materialize", Json::Bool(true))
        .remedy("open the router with bundle_root, or omit materialize"));
    };
    let (root, relative) = if let Some(rest) = parsed.id.strip_prefix('@') {
        let Some((alias, rest)) = rest.split_once('/') else {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("unknown root alias in concept_id {:?}", parsed.id),
            )
            .op(op)
            .field("concept_id", Json::String(parsed.id.clone())));
        };
        let Some(path) = settings.roots.get(alias) else {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("unknown root alias @{alias} in concept_id"),
            )
            .op(op)
            .field("concept_id", Json::String(parsed.id.clone())));
        };
        (path.as_str(), rest)
    } else {
        (root.as_str(), parsed.id.as_str())
    };
    if !crate::util::safe_bundle_id(relative) {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("concept id {:?} is not safe to materialize", parsed.id),
        )
        .op(op)
        .field("concept_id", Json::String(parsed.id.clone())));
    }
    let root = std::path::absolute(Path::new(root)).map_err(|err| {
        OkfError::internal(
            format!("cannot resolve bundle root {root:?}: {err}"),
            "std::io::Error",
        )
        .op(op)
    })?;
    let path = root.join(format!(
        "{}.md",
        relative.replace('/', std::path::MAIN_SEPARATOR_STR)
    ));
    let mut meta: BTreeMap<String, Json> = BTreeMap::new();
    for (key, value) in &parsed.authored {
        if matches!(
            key.as_str(),
            "id" | "type" | "title" | "description" | "resource" | "tags" | "timestamp"
        ) {
            continue;
        }
        meta.insert(key.clone(), crate::ingest::tag_to_json(value));
    }
    if settings.identity == "frontmatter" {
        meta.insert("id".into(), Json::String(relative.into()));
    }
    meta.insert("type".into(), Json::String(parsed.concept_type.clone()));
    meta.insert("title".into(), Json::String(parsed.title.clone()));
    if let Some(description) = &parsed.description {
        meta.insert("description".into(), Json::String(description.clone()));
    }
    meta.insert(
        "resource".into(),
        parsed
            .resource
            .as_ref()
            .map(|value| Json::String(value.clone()))
            .unwrap_or(Json::Null),
    );
    meta.insert(
        "tags".into(),
        Json::Array(parsed.tags.iter().cloned().map(Json::String).collect()),
    );
    meta.insert(
        "timestamp".into(),
        parsed
            .timestamp
            .map(|timestamp| Json::String(timestamp.isoformat()))
            .unwrap_or(Json::Null),
    );
    let yaml = crate::export::dump_yaml_sorted(&Json::Object(meta.into_iter().collect()), 0);
    let text = format!(
        "---
{yaml}---

{}
",
        parsed.body
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs_write(&path, &text)?;
    let path_text = path.to_string_lossy().into_owned();
    let mut import_settings = settings.clone();
    import_settings.bundle_root = Some(
        settings
            .bundle_root
            .as_ref()
            .map(std::path::absolute)
            .transpose()
            .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))?
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.to_string_lossy().into_owned()),
    );
    import_settings.roots = settings
        .roots
        .iter()
        .map(|(alias, value)| {
            std::path::absolute(value)
                .map(|value| (alias.clone(), value.to_string_lossy().into_owned()))
                .map_err(|err| OkfError::internal(err.to_string(), "std::io::Error"))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let report = crate::ingest::import_files(
        store,
        &import_settings,
        embedder,
        pools,
        std::slice::from_ref(&path_text),
    )?;
    let concept_id = report
        .get("concept_ids")
        .and_then(Json::as_array)
        .and_then(|ids| ids.first())
        .and_then(Json::as_str)
        .unwrap_or(&parsed.id)
        .to_string();
    Ok(serde_json::json!({
        "concept_id": concept_id,
        "path": path_text,
        "materialized": true,
        "title": parsed.title,
        "tags": parsed.tags,
    }))
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
    if let Some(title) = params
        .get("title")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
    {
        parsed.title = title.into();
    }
    if let Some(description) = params
        .get("description")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
    {
        parsed.description = Some(description.into());
    } else if parsed.description.as_deref().is_none_or(str::is_empty) {
        // Python falls back to the frontmatter `summary`, then "".
        let summary = parsed
            .authored
            .get("summary")
            .and_then(|value| match value {
                PyValue::Str(text) => Some(text.clone()),
                _ => None,
            });
        parsed.description = Some(summary.unwrap_or_default());
    }
    if let Some(kind) = params
        .get("type")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
    {
        parsed.concept_type = kind.into();
    }
    if let Some(tags) = params.get("tags") {
        let mut values = normalize_tags(tags);
        values.extend(parsed.tags);
        values.sort();
        values.dedup();
        parsed.tags = values;
    }
    for (key, value) in check_extra(params.get("extra"))? {
        let tagged = PyValue::from_golden(&value);
        parsed
            .extra
            .insert(key.clone(), crate::model::extra_value(&tagged));
        parsed.authored.insert(key, tagged);
    }
    refresh_content_sig(&mut parsed);
    let file_text = fs::read_to_string(&path).map_err(|err| {
        OkfError::internal(format!("cannot read {path}: {err}"), "std::io::Error")
    })?;
    let lint_issues = lint_markdown(&file_text);
    if params
        .get("materialize")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
        let mut out = materialize_parsed(store, settings, embedder, pools, &parsed, "ingest")?;
        out["lint_issues"] = lint_issues;
        return Ok(out);
    }
    commit(store, settings, embedder, pools, parsed.clone())?;
    let (chunk_count, image_count) = counts(store, &parsed.id)?;
    Ok(serde_json::json!({
        "concept_id": parsed.id,
        "title": parsed.title,
        "description": parsed.description,
        "tags": parsed.tags,
        "chunk_count": chunk_count,
        "image_count": image_count,
        "lint_issues": lint_issues,
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
    let bytes = fs::read(&path).map_err(|err| {
        OkfError::internal(format!("cannot read {path}: {err}"), "std::io::Error")
    })?;
    let text = match String::from_utf8(bytes.clone()) {
        Ok(text) => text,
        Err(_) => {
            // Python falls back to latin-1 for legacy BibTeX files.
            bytes.iter().map(|byte| *byte as char).collect()
        }
    };
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
    let mut skipped = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_key: BTreeMap<String, String> = BTreeMap::new();
    for entry in entries {
        let slug = slugify(&entry.key);
        let count = counts.entry(slug.clone()).or_insert(0);
        *count += 1;
        let id = if *count == 1 {
            first_key.insert(slug.clone(), entry.key.clone());
            format!("refs/{slug}")
        } else {
            let id = format!("refs/{slug}~{count}");
            skipped.push(serde_json::json!({
                "key": entry.key,
                "line": entry.line.to_string(),
                "concept_id": id,
                "reason": format!(
                    "slug collides with '{}'; disambiguated, nothing overwritten",
                    first_key.get(&slug).map(String::as_str).unwrap_or_default()
                ),
            }));
            id
        };
        let title = entry
            .fields
            .get("title")
            .map(|value| unwrap_braces(value))
            .filter(|value| !value.trim().is_empty())
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
        let raw_markdown = format!(
            "---
{yaml}---

{body}"
        )
        .trim_end_matches('\n')
        .to_string();
        let (markdown, _) = lint_markdown_full(&raw_markdown, true);
        let mut parsed = parse_markdown(&markdown, &id, settings)?;
        parsed.body = markdown.clone();
        // Python embeds `concept.body` (the pre-lint markdown) but stores the
        // lint-fixed body; keep both spellings.
        parsed.search_text_override = Some(format!("{title} {citation} {raw_markdown}"));
        // The bibliography metadata lives in the markdown frontmatter, like
        // Python's `_bib_entry_concept` ConceptModel (which has no extras).
        parsed.extra.clear();
        parsed.authored.clear();
        parsed.title = title;
        parsed.description = Some(citation);
        parsed.concept_type = "reference".into();
        parsed.tags = tags;
        refresh_content_sig(&mut parsed);
        commit(store, settings, embedder, pools, parsed)?;
        ids.push(Json::String(id));
    }
    Ok(serde_json::json!({
        "concept_ids": ids,
        "entry_count": ids.len(),
        "skipped_entries": skipped,
    }))
}

fn json_string(value: Option<&Json>) -> Option<String> {
    value.and_then(Json::as_str).map(str::to_string)
}

fn slugify_topic(topic: &str) -> String {
    let lowered = topic.to_lowercase().replace(' ', "_");
    let mut out = String::new();
    let mut in_run = false;
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
            out.push(ch);
            in_run = false;
        } else if !in_run {
            out.push('_');
            in_run = true;
        }
    }
    let out = out.trim_matches(['.', '_']);
    out.chars().take(30).collect()
}

fn auto_thought_title(text: &str, topic: &str, now: &crate::model::PyDateTime) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut sentence = flat.as_str();
    for (index, ch) in flat.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let after = &flat[index + ch.len_utf8()..];
            if after.is_empty() || after.starts_with(char::is_whitespace) {
                sentence = &flat[..index + ch.len_utf8()];
                break;
            }
        }
    }
    let sentence = sentence.trim_end_matches(['.', '!', '?', ' ']);
    if sentence.chars().filter(|ch| ch.is_alphanumeric()).count() < 3 {
        return format!(
            "{topic} — {}",
            now.isoformat()
                .get(..16)
                .unwrap_or_default()
                .replace('T', " ")
        );
    }
    let chars: Vec<char> = sentence.chars().collect();
    if chars.len() <= 72 {
        return sentence.to_string();
    }
    let cut: String = chars[..72].iter().collect();
    cut.rsplit_once(' ')
        .map(|(head, _)| head.to_string())
        .unwrap_or(cut)
}

fn safe_materialize_id(id: &str) -> bool {
    let Some(rest) = id.strip_prefix('@') else {
        return crate::util::safe_bundle_id(id);
    };
    let Some((alias, native)) = rest.split_once('/') else {
        return false;
    };
    !alias.is_empty()
        && alias
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        && crate::util::safe_bundle_id(native)
}

fn folder_thought_id(
    folder: &str,
    title: Option<&str>,
    concept_id: Option<&str>,
    materialize: Option<bool>,
) -> Result<String> {
    if concept_id.is_some() {
        return Err(OkfError::new(
            "BAD_VALUE",
            "folder and concept_id both set where the thought goes",
        )
        .field("folder", Json::String(folder.into()))
        .field(
            "concept_id",
            Json::String(concept_id.unwrap_or_default().into()),
        )
        .remedy("pass folder (file named from the title) or concept_id, not both"));
    }
    if materialize == Some(false) {
        return Err(OkfError::new(
            "BAD_VALUE",
            "folder writes a file: it cannot run with materialize=False",
        )
        .field("folder", Json::String(folder.into()))
        .field("materialize", Json::Bool(false))
        .remedy("drop materialize, or drop folder for a graph-only thought"));
    }
    let raw_folder = folder.trim();
    if raw_folder.is_empty() || raw_folder.starts_with(['/', '\\']) {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!(
                "folder {folder:?} is not a bundle sub-path (letters, digits, '_', '.', '-' and '/' only; no '..', not absolute; @alias/ names a root)"
            ),
        )
        .field("folder", Json::String(folder.into())));
    }
    let base = raw_folder.replace('\\', "/");
    let base = base.trim_matches('/');
    let lowered = title.unwrap_or_default().trim().to_lowercase();
    let mut name = String::new();
    let mut in_run = false;
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
            name.push(ch);
            in_run = false;
        } else if !in_run {
            name.push('-');
            in_run = true;
        }
    }
    let name = name.trim_matches(['-', '.', '_']);
    let name: String = name.chars().take(60).collect();
    let name = if name.is_empty() {
        let now = local_now_datetime();
        format!(
            "{}_{}",
            now.isoformat()
                .get(..19)
                .unwrap_or_default()
                .replace(['-', ':', 'T'], ""),
            &uuid::Uuid::new_v4().to_string()[..6]
        )
    } else {
        name
    };
    let cid = if base.is_empty() {
        name
    } else {
        format!("{base}/{name}")
    };
    if !safe_materialize_id(&cid) {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!(
                "folder {folder:?} is not a bundle sub-path (letters, digits, \
                 '_', '.', '-' and '/' only; no '..', not absolute; @alias/ names a root)"
            ),
        )
        .field("folder", Json::String(folder.into())));
    }
    Ok(cid)
}

fn thought_titles_in_topic(
    store: &Store,
    slug: &str,
    title: &str,
    exclude_id: &str,
) -> Result<Vec<String>> {
    let prefix = format!("thoughts/{slug}/");
    Ok(store
        .query(
            "MATCH (c:Concept) WHERE c.title = $title RETURN c.id AS id",
            vec![("title", LValue::String(title.into()))],
        )?
        .into_iter()
        .filter_map(|row| row.get("id").and_then(Json::as_str).map(str::to_string))
        .filter(|id| id != exclude_id && id.starts_with(&prefix))
        .collect())
}

fn py_str_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn py_value_repr(value: &Json) -> String {
    match value {
        Json::String(text) => py_str_repr(text),
        Json::Null => "None".into(),
        Json::Bool(flag) => {
            if *flag {
                "True".into()
            } else {
                "False".into()
            }
        }
        Json::Number(number) => number.to_string(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn with_related(
    mut extra: BTreeMap<String, Json>,
    related: Option<&Json>,
) -> Result<BTreeMap<String, Json>> {
    let Some(related) = related else {
        return Ok(extra);
    };
    let Some(items) = related.as_array() else {
        return Err(
            OkfError::new("BAD_VALUE", "related must be a list of concept ids").field(
                "related",
                Json::String(if related.is_string() { "str" } else { "object" }.into()),
            ),
        );
    };
    let mut values = Vec::new();
    for item in items {
        let Some(text) = item.as_str().map(str::trim).filter(|text| !text.is_empty()) else {
            return Err(
                OkfError::new("BAD_VALUE", "related entries must be non-empty strings").field(
                    "related",
                    Json::Array(vec![Json::String(py_value_repr(item))]),
                ),
            );
        };
        values.push(text.to_string());
    }
    let mut relations = match extra.get("relations") {
        None => Map::new(),
        Some(Json::String(_)) => {
            return Err(OkfError::new(
                "BAD_VALUE",
                "extra['relations'] must be a mapping of kind -> targets",
            )
            .field("relations", Json::String("str".into())))
        }
        Some(Json::Object(map)) => map.clone(),
        Some(other) => {
            return Err(OkfError::new(
                "BAD_VALUE",
                "extra['relations'] must be a mapping of kind -> targets",
            )
            .field("relations", Json::String(format!("{}", other))))
        }
    };
    let mut have: Vec<String> = match relations.get("related") {
        None => Vec::new(),
        Some(Json::String(text)) => vec![text.clone()],
        Some(Json::Array(items)) => items
            .iter()
            .filter_map(Json::as_str)
            .map(str::to_string)
            .collect(),
        Some(_) => Vec::new(),
    };
    for value in values {
        if !have.contains(&value) {
            have.push(value);
        }
    }
    relations.insert(
        "related".into(),
        Json::Array(have.into_iter().map(Json::String).collect()),
    );
    extra.insert(
        "relations".into(),
        Json::String(serde_json::to_string(&relations).unwrap_or_default()),
    );
    Ok(extra)
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

    let now = local_now_datetime();
    let created = now.isoformat();
    let timestamp = utc_now_datetime();
    let slug = {
        let value = slugify_topic(topic);
        if value.is_empty() {
            "untitled".to_string()
        } else {
            value
        }
    };
    let raw_title = params
        .get("title")
        .and_then(Json::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let mut materialize = params.get("materialize").and_then(Json::as_bool);
    let mut concept_id = json_string(params.get("concept_id")).filter(|value| !value.is_empty());
    if let Some(folder) = params
        .get("folder")
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
    {
        concept_id = Some(folder_thought_id(
            folder,
            raw_title.as_deref(),
            concept_id.as_deref(),
            materialize,
        )?);
        materialize = Some(true);
    }
    let concept_id = concept_id.unwrap_or_else(|| {
        let stamp = created[..19].replace(['-', ':', 'T'], "");
        format!(
            "thoughts/{slug}/{}_{}",
            stamp,
            &uuid::Uuid::new_v4().to_string()[..6]
        )
    });

    let title = if let Some(title) = raw_title {
        let taken = thought_titles_in_topic(store, &slug, &title, &concept_id)?;
        if let Some(existing) = taken.first() {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!(
                    "a thought titled {} already exists in topic {}: {existing}",
                    py_str_repr(&title),
                    py_str_repr(topic)
                ),
            )
            .field("title", Json::String(title))
            .field("topic", Json::String(topic.into()))
            .field("existing_id", Json::String(existing.clone()))
            .remedy(format!(
                "choose another title, or pass --concept-id {existing} to update that thought"
            )));
        }
        title
    } else {
        let base = auto_thought_title(thoughts, topic, &now);
        let mut title = base.clone();
        let mut n = 2;
        while !thought_titles_in_topic(store, &slug, &title, &concept_id)?.is_empty() {
            title = format!("{base} ({n})");
            n += 1;
        }
        title
    };

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

    let extra_json = with_related(check_extra(params.get("extra"))?, params.get("related"))?;
    let related_ids = related_ids_from_extra(&extra_json);
    let mut extra = BTreeMap::new();
    for (key, value) in &extra_json {
        extra.insert(
            key.clone(),
            crate::model::extra_value(&PyValue::from_golden(value)),
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
    for (key, value) in &extra_json {
        meta.insert(key.clone(), value.clone());
    }
    let yaml = crate::export::dump_yaml(&Json::Object(meta.into_iter().collect()), 0);
    let raw_markdown = format!(
        "---
{yaml}---

{thoughts}"
    );
    let (markdown, raw_report) = lint_markdown_full(&raw_markdown, true);
    let lint_report = lint_markdown_str(raw_report, &markdown);

    let mut parsed = parse_markdown(&markdown, &concept_id, settings)?;
    let materialized_body = parsed.body.clone();
    parsed.body = markdown.clone();
    parsed.title = title.clone();
    parsed.concept_type = concept_type.clone();
    parsed.description = Some(description.clone());
    parsed.tags = tags.clone();
    parsed.timestamp = Some(timestamp);
    parsed.extra = extra;

    // Graph-only concept data mirrors Python's ConceptModel: core fields +
    // `extra` (not the markdown-only thought_type/topic/created keys).
    let mut authored: BTreeMap<String, PyValue> = BTreeMap::new();
    for (key, value) in &extra_json {
        authored.insert(key.clone(), PyValue::from_golden(value));
    }
    authored.insert("id".into(), PyValue::Str(concept_id.clone()));
    authored.insert("type".into(), PyValue::Str(concept_type.clone()));
    authored.insert("title".into(), PyValue::Str(title.clone()));
    authored.insert("description".into(), PyValue::Str(description.clone()));
    authored.insert("resource".into(), PyValue::Null);
    authored.insert(
        "tags".into(),
        PyValue::List(tags.iter().cloned().map(PyValue::Str).collect()),
    );
    authored.insert("timestamp".into(), PyValue::DateTime(timestamp.isoformat()));
    parsed.authored = authored;

    if materialize == Some(true) {
        // The file keeps the thought metadata; the graph import then computes
        // the content signature from that file.
        parsed.body = materialized_body;
        for (key, value) in [
            ("thought_type", PyValue::Str("reasoning".into())),
            ("topic", PyValue::Str(topic.into())),
        ] {
            parsed.authored.insert(key.into(), value);
        }
        // `created` is the explicit extra value when supplied, otherwise the
        // timestamp the thought was first written (Python's `meta["created"]`).
        parsed
            .authored
            .entry("created".into())
            .or_insert_with(|| PyValue::Str(created));
        refresh_content_sig(&mut parsed);
        let mut out = materialize_parsed(store, settings, embedder, pools, &parsed, "ingest")?;
        out["topic"] = Json::String(topic.into());
        out["markdown"] = Json::String(markdown.clone());
        out["lint_issues"] = lint_report.clone();
        if !related_ids.is_empty() {
            out["unresolved_related"] = Json::Array(Vec::new());
        }
        return Ok(out);
    }

    refresh_content_sig(&mut parsed);
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
        "lint_issues": lint_report,
    });
    if !related_ids.is_empty() {
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

fn related_ids_from_extra(extra: &BTreeMap<String, Json>) -> Vec<String> {
    let Some(relations) = extra.get("relations").and_then(Json::as_str) else {
        return Vec::new();
    };
    let Ok(relations) = serde_json::from_str::<Map<String, Json>>(relations) else {
        return Vec::new();
    };
    relations
        .get("related")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(Json::as_str)
        .map(str::to_string)
        .collect()
}

/// Python `_ingest_namespace`: stable per-source PDF namespace. The source
/// bytes are the normal path; if they cannot be read, hash the converted work
/// tree so re-imports stay stable instead of minting random ids.
fn pdf_namespace(source: &Path, work_dir: &Path) -> String {
    if let Ok(bytes) = fs::read(source) {
        let digest = Sha256::digest(&bytes);
        return format!("pdf-{}", &hex_lower(&digest)[..12]);
    }
    let mut entries: Vec<String> = Vec::new();
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(items) = fs::read_dir(dir) else {
            return;
        };
        for item in items.flatten() {
            let path = item.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else if let Ok(meta) = path.metadata() {
                let rel = path.strip_prefix(root).unwrap_or(&path);
                out.push(format!("{}:{}", rel.to_string_lossy(), meta.len()));
            }
        }
    }
    walk(work_dir, work_dir, &mut entries);
    if !entries.is_empty() {
        entries.sort();
        let digest = Sha256::digest(
            entries
                .join(
                    "
",
                )
                .as_bytes(),
        );
        return format!("pdf-{}", &hex_lower(&digest)[..12]);
    }
    format!("pdf-{}", &uuid::Uuid::new_v4().simple().to_string()[..12])
}

/// Python suspends its delta detector for ephemeral PDF work dirs, so a
/// converted PDF leaves no FileHash/DirHash ledger behind. Drop the rows the
/// shared import pipeline wrote under the temporary namespace.
fn forget_pdf_ledger(store: &Store, namespace: &str) -> Result<()> {
    let prefix = format!("@{namespace}/");
    let files: Vec<String> = store
        .file_hashes()?
        .into_keys()
        .filter(|path| path.starts_with(&prefix))
        .collect();
    for path in files {
        store.forget_file_hash(&path)?;
    }
    let dirs: Vec<String> = store
        .dir_hashes()?
        .into_keys()
        .filter(|path| path == &format!("@{namespace}/.") || path.starts_with(&prefix))
        .collect();
    for dir in dirs {
        store.delete_dir_hash(&dir)?;
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
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
    let _temp_guard = auto_import.then(|| TempDir(output_dir.clone()));
    let plugin = Plugin::load(settings)?;
    let routing_mode = params
        .get("routing_mode")
        .and_then(Json::as_str)
        .unwrap_or("auto");
    let extract_images = params
        .get("extract_images")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    let converted = plugin.convert(
        &source,
        &output_dir,
        &pdf::config_with(settings, routing_mode, extract_images),
    )?;
    // Python lint-fixes converter output before importing it.
    let converted_text = fs::read_to_string(&converted.md_path).map_err(|err| {
        OkfError::internal(
            format!(
                "cannot read converted markdown {}: {err}",
                converted.md_path.display()
            ),
            "std::io::Error",
        )
    })?;
    let (fixed_text, _) = lint_markdown_full(&converted_text, true);
    if fixed_text != converted_text {
        fs_write(&converted.md_path, &fixed_text)?;
    }
    if auto_import {
        let namespace = pdf_namespace(&source, &output_dir);
        let mut import_settings = settings.clone();
        // PDF work dirs are an ephemeral, aliased root: Python passes
        // `alias=@pdf-<hash>` to import_bundle so same-stem pages from
        // different PDFs cannot overwrite each other.
        import_settings.bundle_root = None;
        import_settings.roots = std::collections::BTreeMap::from([(
            namespace.clone(),
            output_dir.to_string_lossy().into_owned(),
        )]);
        let report = crate::ingest::import_bundle(
            store,
            &import_settings,
            embedder,
            pools,
            None,
            crate::ingest::ImportOptions {
                prune_missing: params
                    .get("prune_missing")
                    .and_then(Json::as_bool)
                    .unwrap_or(false),
                force: params.get("force").and_then(Json::as_bool).unwrap_or(false),
            },
        )?;
        forget_pdf_ledger(store, &namespace)?;
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
    let mut settings = settings.clone();
    if let Some(mode) = params.get("mode").and_then(Json::as_str) {
        if !matches!(mode, "text" | "optional" | "omni") {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("mode must be text, optional, or omni, got '{mode}'"),
            )
            .op("ingest"));
        }
        settings.mode = mode.to_string();
    }
    match kind {
        "md" => ingest_md(store, &settings, embedder, pools, params),
        "bib" => ingest_bib(store, &settings, embedder, pools, params),
        "pdf" => ingest_pdf(store, &settings, embedder, pools, params),
        "thoughts" => ingest_thoughts(store, &settings, embedder, pools, params),
        _ => Err(OkfError::new(
            "BAD_VALUE",
            format!("kind must be 'md', 'pdf', 'thoughts' or 'bib', got '{kind}'"),
        )
        .field("kind", Json::String(kind.into()))
        .op("ingest")),
    }
}

#[allow(dead_code)]
fn _unused_now() -> String {
    utc_now_datetime().isoformat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bibtex_macros_and_control_entries_are_handled() {
        let text = r#"
@string{jml = "Journal of Machine Learning"}
@comment{ignore me}
@article{smith2020,
  title = {{A} Study},
  journal = jml,
  year = 2020
}
"#;
        let entries = parse_bibtex(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "smith2020");
        assert_eq!(
            entries[0].fields.get("journal").map(String::as_str),
            Some("Journal of Machine Learning")
        );
        assert_eq!(
            entries[0].fields.get("year").map(String::as_str),
            Some("2020")
        );
        assert_eq!(unwrap_braces(&entries[0].fields["title"]), "A Study");
    }

    #[test]
    fn bibtex_slug_collapses_and_truncates() {
        assert_eq!(slugify("a//b c"), "a_b_c");
        assert_eq!(slugify(&"x".repeat(80)).len(), 64);
        assert_eq!(slugify("..."), "untitled");
    }

    #[test]
    fn pdf_namespace_is_stable_and_source_specific() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.pdf");
        let second = dir.path().join("second.pdf");
        std::fs::write(&first, b"first source").unwrap();
        std::fs::write(&second, b"second source").unwrap();
        let one = pdf_namespace(&first, dir.path());
        assert_eq!(one, pdf_namespace(&first, dir.path()));
        assert!(one.starts_with("pdf-"));
        assert_eq!(one.len(), 16);
        assert_ne!(one, pdf_namespace(&second, dir.path()));
    }
}
