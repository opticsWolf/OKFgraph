//! Frontmatter, PyYAML-compatible scalars, mordant chunking, and lint.
//!
//! The chunk rules here are the on-disk contract: mordant block offsets,
//! exact-tiling token/word caps, and `+` continuation types.

use mordant::chunker::MarkdownChunker;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

pub const CONTINUATION_SUFFIX: &str = "+";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    pub parent_doc_id: String,
    pub chunk_text: String,
    pub block_type: String,
    pub start_offset: i64,
    pub end_offset: i64,
    pub chunk_index: i64,
}

pub fn base_block_type(block_type: &str) -> &str {
    block_type
        .strip_suffix(CONTINUATION_SUFFIX)
        .unwrap_or(block_type)
}

/// Python's `len(text.split())` fallback counter.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Split a markdown body into dense, byte-offset-tiled chunks.
///
/// `counter` is the token counter in production and the word-count fallback
/// in cold paths/tests. It must match the oracle's counter for exact offsets.
pub fn split_into_chunks(
    body: &str,
    document_id: &str,
    chunk_size: usize,
    counter: impl Fn(&str) -> usize,
) -> Vec<Chunk> {
    let mut chunker = MarkdownChunker::new(body.to_string());
    let mut chunks = Vec::new();
    for mordant in chunker.get_all_chunks() {
        let block_type = mordant.block_type.as_str().to_string();
        let base = Chunk {
            parent_doc_id: document_id.to_string(),
            chunk_text: mordant.text,
            block_type,
            start_offset: mordant.start_offset as i64,
            end_offset: mordant.end_offset as i64,
            chunk_index: -1,
        };
        chunks.extend(cap_chunk_budget(&base, chunk_size, &counter));
    }
    for (index, chunk) in chunks.iter_mut().enumerate() {
        chunk.chunk_index = index as i64;
    }
    chunks
}

/// Subdivide an oversized block. Pieces tile the parent exactly: text, byte
/// offsets and block type continuations all match the Python implementation.
fn cap_chunk_budget(
    chunk: &Chunk,
    max_units: usize,
    counter: &impl Fn(&str) -> usize,
) -> Vec<Chunk> {
    let text = &chunk.chunk_text;
    let limit = max_units.max(1);
    if counter(text) <= limit {
        return vec![chunk.clone()];
    }
    let spans = whitespace_spans(text);
    if spans.is_empty() {
        return vec![chunk.clone()];
    }

    let n = spans.len();
    let mut starts = vec![0usize];
    let mut start = 0usize;
    while start < n {
        let single = &text[spans[start].0..spans[start].1];
        let end = if counter(single) > limit {
            start + 1
        } else {
            let base = if start == 0 { 0 } else { spans[start].0 };
            let mut lo = start + 1;
            let mut hi = n;
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                let cand_end = if mid < n { spans[mid].0 } else { text.len() };
                if counter(&text[base..cand_end]) <= limit {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo
        };
        starts.push(end);
        start = end;
    }

    let mut bounds = vec![0usize];
    bounds.extend(starts[1..starts.len() - 1].iter().map(|&i| spans[i].0));
    bounds.push(text.len());

    let base_start = chunk.start_offset;
    let mut pieces = Vec::with_capacity(bounds.len() - 1);
    for i in 0..bounds.len() - 1 {
        let piece_text = text[bounds[i]..bounds[i + 1]].to_string();
        let start_byte = text[..bounds[i]].len();
        let end_byte = text[..bounds[i + 1]].len();
        pieces.push(Chunk {
            parent_doc_id: chunk.parent_doc_id.clone(),
            chunk_text: piece_text,
            block_type: if i == 0 {
                chunk.block_type.clone()
            } else {
                format!("{}{}", chunk.block_type, CONTINUATION_SUFFIX)
            },
            start_offset: base_start + start_byte as i64,
            end_offset: base_start + end_byte as i64,
            chunk_index: -1,
        });
    }
    pieces
}

/// Byte ranges of `\S+` runs (UTF-8 safe).
fn whitespace_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(begin) = start.take() {
                spans.push((begin, index));
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(begin) = start {
        spans.push((begin, text.len()));
    }
    spans
}

/// Rebuild the source markdown from stored chunks. Continuation pieces join
/// with an empty separator; other blocks use mordant's delimiter rules.
pub fn reconstruct_document(chunks: &[Chunk]) -> Option<String> {
    let (first, _) = chunks.split_first()?;
    let mut out = first.chunk_text.clone();
    for i in 1..chunks.len() {
        let prev = base_block_type(&chunks[i - 1].block_type);
        let curr_base = base_block_type(&chunks[i].block_type);
        let sep = if chunks[i].block_type.ends_with(CONTINUATION_SUFFIX) && prev == curr_base {
            String::new()
        } else {
            MarkdownChunker::get_delimiter(prev, curr_base)
        };
        out.push_str(&sep);
        out.push_str(&chunks[i].chunk_text);
    }
    Some(out)
}

/// Parsed python-frontmatter style document.
#[derive(Clone, Debug, PartialEq)]
pub struct Frontmatter {
    pub metadata: Json,
    pub content: String,
}

/// Parse `---` delimited frontmatter. Closing delimiter may be `---` or `...`.
pub fn parse_frontmatter(text: &str) -> Frontmatter {
    let mut lines = text.split_inclusive('\n');
    let Some(first) = lines.next() else {
        return Frontmatter {
            metadata: Json::Object(Map::new()),
            content: String::new(),
        };
    };
    if first.trim_end() != "---" {
        return Frontmatter {
            metadata: Json::Object(Map::new()),
            content: text.to_string(),
        };
    }
    let mut yaml = String::new();
    let mut rest = String::new();
    let mut in_front = true;
    for line in lines {
        if in_front && matches!(line.trim_end(), "---" | "...") {
            in_front = false;
            continue;
        }
        if in_front {
            yaml.push_str(line);
        } else {
            rest.push_str(line);
        }
    }
    let content = rest.trim_matches('\n').to_string();
    Frontmatter {
        metadata: parse_yaml(&yaml),
        content,
    }
}

/// A deliberately scoped PyYAML 1.1 resolver for the scalar forms OKF
/// frontmatter uses. Quoted scalars stay strings; duplicate mapping keys keep
/// the last value, like `yaml.safe_load`.
pub fn parse_yaml(text: &str) -> Json {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let (value, _) = parse_block(&lines, 0, 0);
    value
}

fn parse_block(lines: &[&str], index: usize, indent: usize) -> (Json, usize) {
    let mut map = Map::new();
    let mut i = index;
    while i < lines.len() {
        let line = lines[i];
        let leading = line.len() - line.trim_start().len();
        if leading < indent {
            break;
        }
        if leading > indent {
            i += 1;
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with('-') {
            let mut items = Vec::new();
            while i < lines.len() {
                let item = lines[i].trim();
                if !item.starts_with('-') || lines[i].len() - lines[i].trim_start().len() != indent
                {
                    break;
                }
                items.push(resolve_scalar(item[1..].trim()));
                i += 1;
            }
            return (Json::Array(items), i);
        }
        let Some((raw_key, raw_value)) = split_once_outside_quotes(trimmed, ':') else {
            i += 1;
            continue;
        };
        let key = unquote(raw_key.trim());
        let raw_value = raw_value.trim();
        i += 1;
        let value = if raw_value.is_empty() {
            if i < lines.len() && lines[i].len() - lines[i].trim_start().len() > indent {
                let (nested, next) = parse_block(lines, i, indent + 2);
                i = next;
                nested
            } else {
                Json::Null
            }
        } else {
            resolve_scalar(raw_value)
        };
        map.insert(key, value);
    }
    (Json::Object(map), i)
}

fn split_once_outside_quotes(text: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut quote = None;
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => quote = Some(ch),
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            c if c == delimiter && depth == 0 => return Some((&text[..index], &text[index + 1..])),
            _ => {}
        }
    }
    None
}

fn unquote(raw: &str) -> String {
    let raw = raw.trim();
    if raw.len() >= 2 && raw.starts_with('\'') && raw.ends_with('\'') {
        return raw[1..raw.len() - 1].replace("''", "'");
    }
    if raw.len() >= 2 && raw.starts_with('\"') && raw.ends_with('\"') {
        return raw[1..raw.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
    }
    raw.to_string()
}

fn resolve_scalar(raw: &str) -> Json {
    let text = raw.trim();
    if text.is_empty() {
        return Json::Null;
    }
    if (text.starts_with('\'') && text.ends_with('\''))
        || (text.starts_with('\"') && text.ends_with('\"'))
    {
        return Json::String(unquote(text));
    }
    if text.starts_with('[') && text.ends_with(']') {
        let inner = &text[1..text.len() - 1];
        return Json::Array(split_flow(inner).into_iter().map(resolve_scalar).collect());
    }
    if text.starts_with('{') && text.ends_with('}') {
        let inner = &text[1..text.len() - 1];
        let mut map = Map::new();
        for entry in split_flow(inner) {
            if let Some((key, value)) = split_once_outside_quotes(entry, ':') {
                map.insert(unquote(key), resolve_scalar(value));
            }
        }
        return Json::Object(map);
    }
    let lower = text.to_ascii_lowercase();
    if matches!(lower.as_str(), "yes" | "y" | "true" | "on") {
        return Json::Bool(true);
    }
    if matches!(lower.as_str(), "no" | "n" | "false" | "off") {
        return Json::Bool(false);
    }
    if matches!(lower.as_str(), "null" | "~") {
        return Json::Null;
    }
    if let Some(days) = tagged_date(text) {
        return days;
    }
    if let Some(number) = parse_int(text) {
        return number;
    }
    if let Some(number) = parse_float(text) {
        return number;
    }
    Json::String(text.to_string())
}

fn split_flow(inner: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut depth = 0usize;
    for (index, ch) in inner.char_indices() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '\"' => quote = Some(ch),
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(inner[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if !inner[start..].trim().is_empty() {
        parts.push(inner[start..].trim());
    }
    parts
}

fn tagged_date(text: &str) -> Option<Json> {
    let clean = text
        .strip_suffix('Z')
        .map(|s| format!("{s}+00:00"))
        .unwrap_or_else(|| text.to_string());
    let date_only = clean.len() == 10 && clean.as_bytes()[4] == b'-' && clean.as_bytes()[7] == b'-';
    if date_only && clean.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return Some(serde_json::json!({"__type__": "date", "value": clean}));
    }
    if clean.len() >= 19
        && clean.as_bytes().get(4) == Some(&b'-')
        && clean.as_bytes().get(7) == Some(&b'-')
        && matches!(clean.as_bytes().get(10), Some(b'T') | Some(b' '))
    {
        let iso = clean.replacen(' ', "T", 1);
        return Some(serde_json::json!({"__type__": "datetime", "value": iso}));
    }
    None
}

fn parse_int(text: &str) -> Option<Json> {
    let clean = text.replace('_', "");
    let (sign, body) = match clean.as_bytes().first() {
        Some(b'-') => (-1i64, &clean[1..]),
        Some(b'+') => (1i64, &clean[1..]),
        _ => (1i64, clean.as_str()),
    };
    let value = if let Some(hex) = body.strip_prefix("0x") {
        i64::from_str_radix(hex, 16).ok()?
    } else if let Some(binary) = body.strip_prefix("0b") {
        i64::from_str_radix(binary, 2).ok()?
    } else {
        if !body.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        body.parse::<i64>().ok()?
    };
    Some(Json::from(value.saturating_mul(sign)))
}

fn parse_float(text: &str) -> Option<Json> {
    let lower = text.to_ascii_lowercase().replace('_', "");
    let special = match lower.as_str() {
        ".inf" | "+.inf" => Some("inf"),
        "-.inf" => Some("-inf"),
        ".nan" => Some("nan"),
        _ => None,
    };
    if let Some(value) = special {
        return Some(serde_json::json!({"__type__": "float", "value": value}));
    }
    // PyYAML 1.1 only resolves floats with a decimal point; `1e3` is a string.
    if !lower.contains('.') {
        return None;
    }
    let value = lower.parse::<f64>().ok()?;
    serde_json::Number::from_f64(value).map(Json::Number)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_offsets_tile_the_source() {
        let body = "# Title\n\nOne two three four five.\n\n- a\n- b\n";
        let chunks = split_into_chunks(body, "doc", 1000, word_count);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_offset, 0);
        assert!(chunks.last().unwrap().end_offset <= body.len() as i64);
        assert_eq!(
            reconstruct_document(&chunks).as_deref(),
            Some(body.trim_end())
        );
    }

    #[test]
    fn continuation_pieces_join_without_delimiter() {
        let body = "one two three four five six seven eight nine ten";
        let chunks = split_into_chunks(body, "doc", 3, word_count);
        assert!(chunks
            .iter()
            .any(|c| c.block_type.ends_with(CONTINUATION_SUFFIX)));
        assert_eq!(reconstruct_document(&chunks).as_deref(), Some(body));
    }

    fn golden() -> Json {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("parity/golden/markdown.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("golden")).expect("json")
    }

    #[test]
    fn matches_python_chunk_golden() {
        let golden = golden();
        for (name, case) in golden["chunks"].as_object().expect("chunk cases") {
            let body = case["body"].as_str().expect("body");
            let size = case["chunk_size"].as_u64().expect("size") as usize;
            let actual = split_into_chunks(body, name, size, word_count);
            let actual = serde_json::to_value(actual).expect("serialize chunks");
            let expected: Vec<Json> = case["chunks"]
                .as_array()
                .expect("expected chunks")
                .iter()
                .map(|chunk| {
                    let mut chunk = chunk.clone();
                    if let Some(obj) = chunk.as_object_mut() {
                        obj.remove("heading_context");
                    }
                    chunk
                })
                .collect();
            assert_eq!(actual, Json::Array(expected), "chunk case {name}");
        }
    }

    #[test]
    fn matches_python_yaml_golden() {
        let golden = golden();
        for (name, case) in golden["yaml"].as_object().expect("yaml cases") {
            let input = case["input"].as_str().expect("input");
            assert_eq!(parse_yaml(input), case["value"], "yaml case {name}");
        }
    }

    #[test]
    fn matches_python_frontmatter_golden() {
        let golden = golden();
        for (name, case) in golden["frontmatter"]
            .as_object()
            .expect("frontmatter cases")
        {
            let input = case["input"].as_str().expect("input");
            let parsed = parse_frontmatter(input);
            assert_eq!(
                parsed.metadata, case["metadata"],
                "frontmatter metadata case {name}"
            );
            assert_eq!(
                parsed.content, case["content"],
                "frontmatter content case {name}"
            );
        }
    }
}
