//! Markdown image extraction and `ImageAsset` ingestion. The text route uses
//! the active text embedder; the vision route uses the paired native vision
//! lane. Asset IDs, captions, hashes, and MIME sniffing follow the Python 0.15
//! `okfgraph.images` contract.

use crate::embed::{Embedder, Pools, Task};
use crate::error::Result;
use crate::settings::Settings;
use crate::store::Store;
use base64::Engine;
use lbug::{LogicalType, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const VISION_CONTRACT: &str = "smart32/min262144/max1310720/bicubic-rgb";
const ASSET_STORE_DIRNAME: &str = "_assets";
const NAMESPACE: &str = "6f6b6661-0000-0000-0000-6f6b66617373";

#[derive(Debug)]
struct Extracted {
    asset_id: String,
    index: usize,
    alt_text: Option<String>,
    filename: String,
    mime_type: String,
    data: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Text,
    Vision,
}

fn sha256_hex(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = &text[i + 1..i + 3];
            if let Ok(value) = u8::from_str_radix(hex, 16) {
                out.push(value);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn extract_refs(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = body[cursor..].find("![") {
        let start = cursor + relative;
        let Some(close_alt) = body[start + 2..].find(']') else {
            break;
        };
        let close_alt = start + 2 + close_alt;
        let opening = close_alt + 1;
        if body.as_bytes().get(opening) != Some(&b'(') {
            cursor = close_alt + 1;
            continue;
        }
        let Some(close_src) = body[opening + 1..].find(')') else {
            break;
        };
        let src = body[opening + 1..opening + 1 + close_src].trim();
        let src = if src.starts_with('<') && src.ends_with('>') {
            src[1..src.len() - 1].trim()
        } else {
            src
        };
        if !src.is_empty() {
            out.push((
                body[start + 2..close_alt].trim().to_string(),
                src.to_string(),
            ));
        }
        cursor = opening + close_src + 1;
    }
    out
}

fn okf_asset_id(src: &str) -> Option<&str> {
    src.strip_prefix("okf-asset://")
        .map(|value| value.trim().trim_matches('/'))
        .filter(|value| !value.is_empty())
}

fn asset_id(concept_id: &str, index: usize, src: &str) -> String {
    if let Some(found) = okf_asset_id(src) {
        return found.to_string();
    }
    let namespace = uuid::Uuid::parse_str(NAMESPACE).expect("valid uuid namespace");
    let id = uuid::Uuid::new_v5(&namespace, format!("{concept_id}:{index}:{src}").as_bytes());
    format!("img_{id}")
}

fn filename_from_src(src: &str) -> String {
    if let Some(found) = okf_asset_id(src) {
        return found.to_string();
    }
    if let Some(header) = src.strip_prefix("data:") {
        let header = header.split(',').next().unwrap_or("image/png");
        let mime = header.split(';').next().unwrap_or("image/png");
        let extension = match mime {
            "image/gif" => ".gif",
            "image/webp" => ".webp",
            "image/jpeg" => ".jpg",
            _ => ".png",
        };
        return format!("inline{extension}");
    }
    let cleaned = src.split('#').next().unwrap_or(src);
    let cleaned = cleaned.split('?').next().unwrap_or(cleaned);
    let cleaned = cleaned.replace('\\', "/");
    let name = cleaned
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("");
    percent_decode(name).if_empty("image")
}

trait IfEmpty {
    fn if_empty(self, fallback: &str) -> Self;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> Self {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

fn sniff_mime(data: Option<&[u8]>, filename: &str) -> String {
    if let Some(head) = data {
        if head.starts_with(&[0x89, b'P', b'N', b'G']) {
            return "image/png".into();
        }
        if head.starts_with(&[0xff, 0xd8, 0xff]) {
            return "image/jpeg".into();
        }
        if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
            return "image/gif".into();
        }
        if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP" {
            return "image/webp".into();
        }
    }
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".png") {
        "image/png".into()
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg".into()
    } else if lower.ends_with(".gif") {
        "image/gif".into()
    } else if lower.ends_with(".webp") {
        "image/webp".into()
    } else {
        "application/octet-stream".into()
    }
}

fn remote_src(src: &str) -> bool {
    let lower = src.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn decode_data_uri(src: &str) -> Option<Vec<u8>> {
    let (header, payload) = src[5..].split_once(',')?;
    if header.to_ascii_lowercase().contains(";base64") {
        base64::engine::general_purpose::STANDARD
            .decode(payload)
            .ok()
    } else {
        Some(percent_decode(payload).into_bytes())
    }
}

fn within_root(path: &Path, root: Option<&Path>) -> bool {
    let Some(root) = root else {
        return true;
    };
    let Ok(path) = path.canonicalize() else {
        return false;
    };
    let Ok(root) = root.canonicalize() else {
        return false;
    };
    path.starts_with(root)
}

fn asset_store_bytes(asset_id: &str, search_dirs: &[PathBuf]) -> Option<Vec<u8>> {
    for base in search_dirs {
        for dir in [base.join(ASSET_STORE_DIRNAME), base.clone()] {
            let exact = dir.join(asset_id);
            if exact.is_file() {
                if let Ok(data) = fs::read(&exact) {
                    return Some(data);
                }
            }
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            let mut matches: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file()
                        && path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(|name| name.starts_with(&format!("{asset_id}.")))
                })
                .collect();
            matches.sort();
            for path in matches {
                if let Ok(data) = fs::read(path) {
                    return Some(data);
                }
            }
        }
    }
    None
}

fn load_bytes(src: &str, search_dirs: &[PathBuf], bundle_root: Option<&Path>) -> Option<Vec<u8>> {
    if src.starts_with("data:") {
        return decode_data_uri(src);
    }
    if let Some(id) = okf_asset_id(src) {
        return asset_store_bytes(id, search_dirs);
    }
    if remote_src(src) {
        return None; // remote fetching stays opt-in and out of Stage B parity
    }
    let lower = src.to_ascii_lowercase();
    if lower.starts_with("file:") || (lower.contains(':') && !lower.starts_with('/')) {
        return None;
    }
    let local = percent_decode(src);
    let candidate = PathBuf::from(&local);
    let candidates: Vec<PathBuf> = if candidate.is_absolute() {
        vec![candidate]
    } else {
        search_dirs
            .iter()
            .map(|base| base.join(&candidate))
            .collect()
    };
    for path in candidates {
        if path.is_file() && within_root(&path, bundle_root) {
            if let Ok(data) = fs::read(path) {
                return Some(data);
            }
        }
    }
    None
}

fn fallback_caption(filename: &str, index: usize, concept_id: &str) -> String {
    let where_ = concept_id.split('/').rfind(|part| !part.is_empty());
    format!(
        "{} (image {} in {})",
        if filename.is_empty() {
            "image"
        } else {
            filename
        },
        index,
        where_.unwrap_or(if concept_id.is_empty() {
            "document"
        } else {
            concept_id
        })
    )
}

fn extract_images(
    concept_id: &str,
    body: &str,
    base_dir: Option<&Path>,
    root: Option<&Path>,
) -> Vec<Extracted> {
    let mut search_dirs: Vec<PathBuf> = Vec::new();
    for dir in [base_dir, root].into_iter().flatten() {
        let dir = dir.to_path_buf();
        if !search_dirs.contains(&dir) {
            search_dirs.push(dir);
        }
    }
    extract_refs(body)
        .into_iter()
        .enumerate()
        .map(|(offset, (alt, src))| {
            let index = offset + 1;
            let filename = filename_from_src(&src);
            let data = load_bytes(&src, &search_dirs, root);
            Extracted {
                asset_id: asset_id(concept_id, index, &src),
                index,
                alt_text: if alt.is_empty() { None } else { Some(alt) },
                mime_type: sniff_mime(data.as_deref(), &filename),
                filename,
                data,
            }
        })
        .collect()
}

fn route_and_caption(img: &Extracted, mode: &str, concept_id: &str) -> (Route, String) {
    let fallback = fallback_caption(&img.filename, img.index, concept_id);
    match mode {
        "optional" if img.alt_text.is_some() => (Route::Text, img.alt_text.clone().unwrap()),
        "optional" | "omni" if img.data.is_some() => (Route::Vision, String::new()),
        _ => (Route::Text, img.alt_text.clone().unwrap_or(fallback)),
    }
}

fn text_hash(caption: &str) -> String {
    sha256_hex(&[b"text", b"|", caption.as_bytes()])
}

fn vision_hash(model_id: &str, precision: &str, data: &[u8]) -> String {
    sha256_hex(&[
        b"vision-onnx",
        b"|",
        model_id.as_bytes(),
        b"|",
        precision.as_bytes(),
        b"|",
        VISION_CONTRACT.as_bytes(),
        b"|",
        data,
    ])
}

fn configured_image_model(settings: &Settings) -> String {
    settings
        .image_model_id
        .clone()
        .unwrap_or_else(|| crate::embed::default_image_model_id(&settings.model_id))
}

fn precision_code(precision: &str) -> i64 {
    match precision {
        "fp16" => 16,
        "int8" => 8,
        _ => 32,
    }
}

#[cfg(feature = "vision")]
fn pin_image_graph(
    store: &Store,
    settings: &Settings,
    engine: &dyn Embedder,
    vision: &crate::embed::NativeVision,
) -> Result<()> {
    let model_id = configured_image_model(settings);
    let precision = vision.precision();
    if let Some(pinned) = store.meta_text("embedding_image_model")? {
        if pinned != model_id {
            return Err(crate::error::OkfError::new(
                "IMAGE_PIN_MISMATCH",
                format!("graph is pinned to image model '{pinned}', configured '{model_id}'"),
            ));
        }
    } else {
        store.set_meta_text("embedding_image_model", &model_id)?;
    }
    if store.meta_int("embedding_image_precision")?.is_none() {
        store.set_meta_int("embedding_image_precision", precision_code(precision))?;
    }
    if store.meta_text("embedding_image_space")?.is_none() {
        let revision = embroider::builtin_models()
            .iter()
            .find(|spec| spec.id == model_id)
            .and_then(|spec| match precision {
                "fp16" => spec.fp16.and_then(|artifact| artifact.revision),
                "int8" => spec.int8.and_then(|artifact| artifact.revision),
                _ => spec.fp32.revision,
            })
            .unwrap_or_default();
        let space = format!(
            "embroider/{model_id}@{revision}:{precision}:{}:0:vision:v1",
            settings.embedding_dim
        );
        store.set_meta_text("embedding_image_space", &space)?;
    }
    if store.meta_text("embedding_image_text_space")?.is_none() {
        if let Some(space) = engine.space() {
            store.set_meta_text("embedding_image_text_space", &space)?;
        }
    }
    Ok(())
}

fn vector_value(vector: &[f32]) -> Value {
    Value::List(
        LogicalType::Float,
        vector.iter().copied().map(Value::Float).collect(),
    )
}

/// Ingest the images referenced by one concept. Returns the number of linked
/// assets, matching Python's `stats["total"] - stats["skipped"]`.
#[allow(clippy::too_many_arguments)]
pub fn ingest_concept_images(
    store: &Store,
    settings: &Settings,
    engine: &dyn Embedder,
    pools: &Pools,
    concept_id: &str,
    body: &str,
    base_dir: Option<&Path>,
    root: Option<&Path>,
) -> Result<i64> {
    let images = extract_images(concept_id, body, base_dir, root);
    if images.is_empty() && store.list_images(concept_id)?.is_empty() {
        return Ok(0);
    }
    let total = images.len();
    #[cfg(feature = "vision")]
    if images.iter().any(|img| img.data.is_some())
        && matches!(settings.mode.as_str(), "optional" | "omni")
    {
        let vision = pools.vision.get(settings)?;
        pin_image_graph(store, settings, engine, &vision)?;
    }
    let mut planned = BTreeSet::new();
    let mut skip = 0;
    let mut pending: Vec<(Extracted, Route, String, String, Vec<f32>)> = Vec::new();
    for img in images {
        let known = store.image_asset_fingerprint(&img.asset_id)?;
        if img.data.is_none() && known.as_ref().is_none_or(|(_, has_data)| !has_data) {
            skip += 1;
            continue;
        }
        let (route, caption) = route_and_caption(&img, &settings.mode, concept_id);
        let data = img.data.clone().unwrap_or_default();
        let content_hash = match route {
            Route::Text => text_hash(&caption),
            Route::Vision => {
                #[cfg(feature = "vision")]
                {
                    let vision = pools.vision.get(settings)?;
                    vision_hash(vision.model_id(), vision.precision(), &data)
                }
                #[cfg(not(feature = "vision"))]
                {
                    return Err(crate::error::OkfError::new(
                        "BAD_VALUE",
                        "vision image ingestion requires a build with --features vision",
                    ));
                }
            }
        };
        planned.insert(img.asset_id.clone());
        if known
            .as_ref()
            .is_some_and(|(hash, _)| hash == &content_hash)
        {
            store.link_image_asset(concept_id, &img.asset_id)?;
            continue;
        }
        let embedding = match route {
            Route::Text => engine.encode(&caption, Task::Document)?,
            Route::Vision => {
                #[cfg(feature = "vision")]
                {
                    pools.vision.get(settings)?.encode_image(&data)?
                }
                #[cfg(not(feature = "vision"))]
                {
                    return Err(crate::error::OkfError::new(
                        "BAD_VALUE",
                        "vision image ingestion requires a build with --features vision",
                    ));
                }
            }
        };
        pending.push((img, route, caption, content_hash, embedding));
    }
    for (asset_id, _) in store.image_asset_fingerprints_for_concept(concept_id)? {
        if !planned.contains(&asset_id) {
            store.delete_image_asset(concept_id, &asset_id)?;
            store.bump_write_epoch()?;
        }
    }
    for (img, route, caption, content_hash, embedding) in pending {
        store.delete_image_asset(concept_id, &img.asset_id)?;
        store.bump_write_epoch()?;
        let route_name = if route == Route::Text {
            "text"
        } else {
            "vision-onnx"
        };
        store.write(
            "MATCH (c:Concept {id: $cid}) CREATE (i:ImageAsset { \
             id: $id, file_name: $file_name, mime_type: $mime_type, \
             alt_text: $alt_text, caption: $caption, embed_route: $route, \
             content_hash: $hash, data: $data, embedding: $embedding}) \
             MERGE (c)-[:INCLUDES_ASSET]->(i)",
            vec![
                ("cid", Value::String(concept_id.into())),
                ("id", Value::String(img.asset_id.clone())),
                ("file_name", Value::String(img.filename.clone())),
                ("mime_type", Value::String(img.mime_type.clone())),
                (
                    "alt_text",
                    img.alt_text
                        .as_ref()
                        .map(|text| Value::String(text.clone()))
                        .unwrap_or_else(|| Value::String(String::new())),
                ),
                ("caption", Value::String(caption)),
                ("route", Value::String(route_name.into())),
                ("hash", Value::String(content_hash)),
                ("data", Value::Blob(img.data.clone().unwrap_or_default())),
                ("embedding", vector_value(&embedding)),
            ],
        )?;
        store.bump_write_epoch()?;
    }
    Ok((total - skip) as i64)
}
