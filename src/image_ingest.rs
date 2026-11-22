//! Markdown image extraction and `ImageAsset` ingestion. The text route uses
//! the active text embedder; the vision route uses the paired native vision
//! lane. Asset IDs, captions, hashes, and MIME sniffing follow the Python 0.15
//! `okfgraph.images` contract.

#[cfg(any(feature = "vision", feature = "external"))]
use crate::embed::VisionEncoder;
use crate::embed::{Embedder, Pools, Task};
use crate::error::Result;
use crate::settings::Settings;
use crate::store::Store;
use base64::Engine;
use lbug::{LogicalType, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
#[cfg(feature = "external")]
use std::io::Read;
#[cfg(feature = "external")]
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

#[cfg(any(feature = "vision", feature = "external"))]
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
    remote: bool,
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

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn percent_decode(text: &str) -> String {
    // Decode bytes, never `str` slices: a malformed escape can otherwise split
    // a multi-byte UTF-8 character (for example `%€`).
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(high), Some(low)) = (hex_digit(bytes[i + 1]), hex_digit(bytes[i + 2])) {
                out.push((high << 4) | low);
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
        // Python's image regex is `src=<...>|[^)\s]+` followed by an
        // optional quoted title. A title is not part of the source and must
        // not leak into the asset id or the on-disk lookup.
        let mut p = opening + 1;
        while body.as_bytes().get(p).is_some_and(u8::is_ascii_whitespace) {
            p += 1;
        }
        let src = if body.as_bytes().get(p) == Some(&b'<') {
            let Some(end) = body[p + 1..].find('>') else {
                break;
            };
            let value = body[p + 1..p + 1 + end].trim().to_string();
            p += end + 2;
            value
        } else {
            let begin = p;
            while let Some(byte) = body.as_bytes().get(p) {
                if byte.is_ascii_whitespace() || *byte == b')' {
                    break;
                }
                p += 1;
            }
            body[begin..p].to_string()
        };
        if body.as_bytes().get(p).is_some_and(u8::is_ascii_whitespace) {
            while body.as_bytes().get(p).is_some_and(u8::is_ascii_whitespace) {
                p += 1;
            }
            if let Some(quote @ (b'\'' | b'"')) = body.as_bytes().get(p).copied() {
                p += 1;
                while let Some(byte) = body.as_bytes().get(p) {
                    if *byte == quote {
                        p += 1;
                        break;
                    }
                    p += 1;
                }
                while body.as_bytes().get(p).is_some_and(u8::is_ascii_whitespace) {
                    p += 1;
                }
            }
        }
        if body.as_bytes().get(p) != Some(&b')') {
            cursor = opening + 1;
            continue;
        }
        if !src.is_empty() {
            out.push((body[start + 2..close_alt].trim().to_string(), src));
        }
        cursor = p + 1;
    }
    out
}

/// Accept only a single stored asset name. Python 0.15 trusts whatever
/// follows `okf-asset://`; that permits `..`, absolute paths and Windows
/// drive/colon paths to escape the bundle and read arbitrary local bytes.
/// This is an intentional hardening divergence: those ids are refused here.
fn okf_asset_id(src: &str) -> Option<String> {
    let ident = src.strip_prefix("okf-asset://")?.trim().trim_matches('/');
    if ident.is_empty()
        || ident == "."
        || ident.contains("..")
        || ident.contains(['/', '\\', ':', '%', '\0'])
        || ident.chars().any(char::is_control)
    {
        return None;
    }
    Some(ident.to_string())
}

fn asset_id(concept_id: &str, index: usize, src: &str) -> String {
    if let Some(found) = okf_asset_id(src) {
        return found;
    }
    let namespace = uuid::Uuid::parse_str(NAMESPACE).expect("valid uuid namespace");
    let id = uuid::Uuid::new_v5(&namespace, format!("{concept_id}:{index}:{src}").as_bytes());
    format!("img_{id}")
}

fn filename_from_src(src: &str) -> String {
    if let Some(found) = okf_asset_id(src) {
        return found;
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
    // Defense in depth: every caller validates, but a stored asset id must
    // never become a path outside its own store directory.
    let asset_id = okf_asset_id(&format!("okf-asset://{asset_id}"))?;
    let prefix = format!("{asset_id}.");
    for base in search_dirs {
        for dir in [base.join(ASSET_STORE_DIRNAME), base.clone()] {
            let exact = dir.join(&asset_id);
            if exact.is_file() && within_root(&exact, Some(&dir)) {
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
                            .is_some_and(|name| name.starts_with(&prefix))
                })
                .collect();
            matches.sort();
            for path in matches {
                if within_root(&path, Some(&dir)) {
                    if let Ok(data) = fs::read(path) {
                        return Some(data);
                    }
                }
            }
        }
    }
    None
}

#[cfg(feature = "external")]
fn domain_allowed(host: &str, patterns: &[String]) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty()
        || host == "localhost"
        || host == "0.0.0.0"
        || host == "::1"
        || host.starts_with("127.")
        || host.starts_with("10.")
        || host.starts_with("192.168.")
        || host.starts_with("169.254.")
        || host
            .strip_prefix("172.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|octet| octet.parse::<u8>().ok())
            .is_some_and(|octet| (16..=31).contains(&octet))
    {
        return false;
    }
    if patterns.is_empty() {
        return true;
    }
    patterns.iter().any(|pattern| {
        let pattern = pattern.trim().to_ascii_lowercase();
        if let Some(base) = pattern.strip_prefix("*.") {
            host == base || host.ends_with(&format!(".{base}"))
        } else {
            host == pattern
        }
    })
}

#[cfg(feature = "external")]
const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;

#[cfg(feature = "external")]
fn v4_blocked(octets: [u8; 4]) -> bool {
    let [a, b, c, _] = octets;
    a == 0
        || a == 10
        || a == 127
        || a >= 224
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && (c == 0 || c == 2))
        || (a == 192 && b == 168)
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
}

#[cfg(feature = "external")]
fn ip_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_blocked(v4.octets()),
        IpAddr::V6(v6) => {
            let segments = v6.segments();
            v6.is_unspecified()
                || v6.is_loopback()
                || v6.is_multicast()
                || segments[0] & 0xfe00 == 0xfc00
                || segments[0] & 0xffc0 == 0xfe80
                || v6
                    .to_ipv4()
                    .is_some_and(|mapped| v4_blocked(mapped.octets()))
        }
    }
}

#[cfg(feature = "external")]
fn safe_resolution(host: &str, port: u16) -> Option<SocketAddr> {
    let addresses: Vec<SocketAddr> = (host, port).to_socket_addrs().ok()?.collect();
    let chosen = addresses.first()?.to_owned();
    addresses
        .iter()
        .all(|addr| !ip_blocked(addr.ip()))
        .then_some(chosen)
}

#[cfg(feature = "external")]
fn fetch_validated(url: reqwest::Url, host: &str, addr: SocketAddr) -> Option<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .resolve(host, addr)
        .build()
        .ok()?;
    let response = client.get(url).send().ok()?;
    if !response.status().is_success() {
        return None;
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_IMAGE_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() as u64 <= MAX_IMAGE_BYTES).then_some(bytes)
}

#[cfg(feature = "external")]
fn fetch_remote(url: &str, patterns: &[String]) -> Option<Vec<u8>> {
    let parsed = reqwest::Url::parse(url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    let host = parsed.host_str()?.to_string();
    if !domain_allowed(&host, patterns) {
        return None;
    }
    let port = parsed.port_or_known_default()?;
    let addr = safe_resolution(&host, port)?;
    fetch_validated(parsed, &host, addr)
}

#[cfg(not(feature = "external"))]
fn fetch_remote(_url: &str, _patterns: &[String]) -> Option<Vec<u8>> {
    None
}

fn load_bytes(
    src: &str,
    search_dirs: &[PathBuf],
    bundle_root: Option<&Path>,
    settings: &Settings,
) -> Option<Vec<u8>> {
    if src.starts_with("data:") {
        return decode_data_uri(src);
    }
    if let Some(id) = okf_asset_id(src) {
        return asset_store_bytes(&id, search_dirs);
    }
    if remote_src(src) {
        if !settings.allow_remote_images {
            return None;
        }
        return fetch_remote(src, &settings.allowed_image_domains);
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
    settings: &Settings,
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
            let data = load_bytes(&src, &search_dirs, root, settings);
            Extracted {
                asset_id: asset_id(concept_id, index, &src),
                index,
                alt_text: if alt.is_empty() { None } else { Some(alt) },
                mime_type: sniff_mime(data.as_deref(), &filename),
                filename,
                data,
                remote: remote_src(&src),
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

/// Python's `_caption_route`: the fallback for an image whose bytes the
/// vision lane refused to embed (other than a declared host outage).
fn caption_route(img: &Extracted, concept_id: &str) -> (Route, String) {
    let caption = img
        .alt_text
        .clone()
        .unwrap_or_else(|| fallback_caption(&img.filename, img.index, concept_id));
    (Route::Text, caption)
}

#[cfg(any(feature = "vision", feature = "external"))]
fn is_lane_outage(err: &crate::error::OkfError) -> bool {
    err.code == "EXTERNAL_UNAVAILABLE"
}

fn text_hash(caption: &str) -> String {
    sha256_hex(&[b"text", b"|", caption.as_bytes()])
}

#[cfg(any(feature = "vision", feature = "external"))]
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

#[cfg(any(feature = "vision", feature = "external"))]
fn configured_image_model(settings: &Settings) -> String {
    settings
        .image_model_id
        .clone()
        .unwrap_or_else(|| crate::embed::default_image_model_id(&settings.model_id))
}

#[cfg(any(feature = "vision", feature = "external"))]
fn precision_code(precision: &str) -> i64 {
    match precision {
        "fp16" => 16,
        "int8" => 8,
        _ => 32,
    }
}

#[cfg(any(feature = "vision", feature = "external"))]
fn pin_image_graph(
    store: &Store,
    settings: &Settings,
    engine: &dyn Embedder,
    vision: &dyn VisionEncoder,
) -> Result<()> {
    let model_id = if settings.image_backend == "external" {
        vision.model_id().to_string()
    } else {
        configured_image_model(settings)
    };
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
        if let Some(space) = vision.space() {
            store.set_meta_text("embedding_image_space", &space)?;
            if let Some(text_space) = vision.text_partner() {
                if text_space != settings.model_id {
                    return Err(crate::error::OkfError::new(
                        "IMAGE_PIN_MISMATCH",
                        format!("vision host serves text partner '{text_space}', but the graph uses '{}'", settings.model_id),
                    ));
                }
            }
            return Ok(());
        }
        #[cfg(feature = "vision")]
        let revision = embroider::builtin_models()
            .iter()
            .find(|spec| spec.id == model_id)
            .and_then(|spec| match precision {
                "fp16" => spec.fp16.and_then(|artifact| artifact.revision),
                "int8" => spec.int8.and_then(|artifact| artifact.revision),
                _ => spec.fp32.revision,
            })
            .unwrap_or_default();
        #[cfg(feature = "vision")]
        {
            let space = format!(
                "embroider/{model_id}@{revision}:{precision}:{}:0:vision:v1",
                settings.embedding_dim
            );
            store.set_meta_text("embedding_image_space", &space)?;
        }
        #[cfg(not(feature = "vision"))]
        {
            return Err(crate::error::OkfError::new(
                "IMAGE_PIN_MISMATCH",
                "an external vision host must name its vector space",
            ));
        }
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
#[cfg_attr(
    not(any(feature = "vision", feature = "external")),
    allow(unused_variables)
)]
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
    let images = extract_images(concept_id, body, base_dir, root, settings);
    if images.is_empty() && store.list_images(concept_id)?.is_empty() {
        return Ok(0);
    }
    let total = images.len();
    #[cfg_attr(
        not(any(feature = "vision", feature = "external")),
        allow(unused_variables, unused_mut)
    )]
    let mut vision_unavailable = false;
    #[cfg(any(feature = "vision", feature = "external"))]
    if images.iter().any(|img| img.data.is_some())
        && matches!(settings.mode.as_str(), "optional" | "omni")
    {
        match pools.vision.get(settings) {
            Ok(vision) => pin_image_graph(store, settings, engine, vision.as_ref())?,
            Err(err) if is_lane_outage(&err) => return Err(err),
            Err(err) => {
                // Missing/incompatible vision runtime is a per-image
                // problem in Python 0.15: the concept survives on captions.
                tracing::warn!(
                    "image ingestion: vision lane unavailable, falling back to captions: {}",
                    err.message
                );
                vision_unavailable = true;
            }
        }
    }
    let mut planned = BTreeSet::new();
    let mut skip = 0;
    let mut pending: Vec<(Extracted, Route, String, String, Vec<f32>)> = Vec::new();
    for img in images {
        let known = store.image_asset_fingerprint(&img.asset_id)?;
        // Like Python, a failed remote fetch still creates a caption-row
        // asset so the next import retries instead of forgetting the ref.
        if img.data.is_none() && !img.remote && known.as_ref().is_none_or(|(_, has_data)| !has_data)
        {
            skip += 1;
            continue;
        }
        let (mut route, mut caption) = route_and_caption(&img, &settings.mode, concept_id);
        if route == Route::Vision && vision_unavailable {
            (route, caption) = caption_route(&img, concept_id);
        }
        #[cfg_attr(
            not(any(feature = "vision", feature = "external")),
            allow(unused_variables)
        )]
        let data = img.data.clone().unwrap_or_default();
        let (content_hash, embedding) = match route {
            Route::Text => (
                text_hash(&caption),
                engine.encode(&caption, Task::Document)?,
            ),
            Route::Vision => {
                #[cfg(any(feature = "vision", feature = "external"))]
                {
                    let encoded = (|| -> crate::error::Result<(String, Vec<f32>)> {
                        let vision = pools.vision.get(settings)?;
                        let embedding = vision.encode_image(&data)?;
                        let hash = vision_hash(vision.model_id(), vision.precision(), &data);
                        Ok((hash, embedding))
                    })();
                    match encoded {
                        Ok(value) => value,
                        Err(err) if is_lane_outage(&err) => return Err(err),
                        Err(err) => {
                            // Undecodable bytes or a host refusing one image:
                            // keep the document with a caption row.
                            tracing::warn!(
                                "image ingestion: {} image(s) embedded as captions: {}::{}",
                                1,
                                err.code,
                                err.message.chars().take(160).collect::<String>()
                            );
                            (route, caption) = caption_route(&img, concept_id);
                            (
                                text_hash(&caption),
                                engine.encode(&caption, Task::Document)?,
                            )
                        }
                    }
                }
                #[cfg(not(any(feature = "vision", feature = "external")))]
                {
                    return Err(crate::error::OkfError::new(
                        "BAD_VALUE",
                        "vision image ingestion requires a build with --features vision or external",
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

#[cfg(test)]
mod tests {
    use super::percent_decode;

    #[test]
    fn percent_decode_is_byte_safe() {
        assert_eq!(percent_decode("%20"), " ");
        assert_eq!(percent_decode("%E2%82%AC"), "€");
        assert_eq!(percent_decode("%€"), "%€");
        assert_eq!(percent_decode("%é"), "%é");
        assert_eq!(percent_decode("%4"), "%4");
        assert_eq!(percent_decode("%GG"), "%GG");
        assert_eq!(percent_decode("100%25"), "100%");
    }

    #[test]
    fn image_refs_ignore_titles_and_keep_angle_sources() {
        let body = concat!(
            "![fox](pic.png \"A titled fox\")\n",
            "![other](<path with spaces.png> 'single title')\n",
            "![plain](plain.png)\n",
        );
        assert_eq!(
            super::extract_refs(body),
            vec![
                ("fox".to_string(), "pic.png".to_string()),
                ("other".to_string(), "path with spaces.png".to_string()),
                ("plain".to_string(), "plain.png".to_string()),
            ]
        );
    }

    #[test]
    fn asset_ids_refuse_traversal_and_absolute_paths() {
        let valid = "img_01234567-89ab-cdef-0123-456789abcdef";
        assert_eq!(
            super::okf_asset_id(&format!("okf-asset://{valid}")),
            Some(valid.to_string())
        );
        for bad in [
            "okf-asset://../../etc/passwd",
            "okf-asset:///etc/passwd",
            "okf-asset://C:/Users/secret.txt",
            "okf-asset://name:stream",
            "okf-asset://..",
            "okf-asset://.",
            "okf-asset://",
            "okf-asset://%2e%2e%2fsecret",
        ] {
            assert_eq!(super::okf_asset_id(bad), None, "{bad} should be refused");
        }
    }

    #[test]
    fn asset_store_never_reads_outside_the_store_directory() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("bundle");
        let assets = root.join("_assets");
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(temp.path().join("secret.txt"), b"top secret").unwrap();
        std::fs::write(assets.join("img_ok.png"), b"public").unwrap();
        assert_eq!(
            super::asset_store_bytes("img_ok", std::slice::from_ref(&root)),
            Some(b"public".to_vec())
        );
        assert_eq!(super::asset_store_bytes("../secret", &[root]), None);
    }

    #[cfg(feature = "external")]
    #[test]
    fn remote_domains_are_allowlisted_and_ssrf_safe() {
        let patterns = vec!["*.example.com".to_string(), "cdn.test".to_string()];
        assert!(super::domain_allowed("img.example.com", &patterns));
        assert!(super::domain_allowed("example.com", &patterns));
        assert!(super::domain_allowed("cdn.test", &patterns));
        assert!(!super::domain_allowed("localhost", &patterns));
        assert!(!super::domain_allowed("127.0.0.1", &patterns));
        assert!(!super::domain_allowed("192.168.1.2", &patterns));
        assert!(!super::domain_allowed("172.20.1.2", &patterns));
        assert!(!super::domain_allowed("10.0.0.1", &patterns));
        assert!(!super::domain_allowed("evil.test", &patterns));
        assert!(super::domain_allowed("anything.test", &[]));
    }

    #[cfg(feature = "external")]
    #[test]
    fn resolved_private_and_special_addresses_are_blocked() {
        use std::net::{IpAddr, Ipv4Addr};
        for ip in [
            "0.0.0.0",
            "10.1.2.3",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "172.16.0.1",
            "192.0.0.1",
            "192.0.2.1",
            "192.168.1.2",
            "198.51.100.7",
            "203.0.113.9",
            "224.0.0.1",
            "240.0.0.1",
        ] {
            let parsed: IpAddr = ip.parse().unwrap();
            assert!(super::ip_blocked(parsed), "{ip} should be blocked");
        }
        assert!(!super::ip_blocked(IpAddr::V4(Ipv4Addr::new(
            93, 184, 216, 34
        ))));
        assert!(super::ip_blocked("::1".parse().unwrap()));
        assert!(super::ip_blocked("fc00::1".parse().unwrap()));
        assert!(super::ip_blocked("fe80::1".parse().unwrap()));
        assert!(super::ip_blocked("ff02::1".parse().unwrap()));
        assert!(!super::ip_blocked("2606:4700:4700::1111".parse().unwrap()));
    }
}
