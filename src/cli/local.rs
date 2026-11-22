//! Local CLI helpers that are not public ops in Python 0.15.
//!
//! `model-info`, `setup` and the interactive shell are static/local verbs in
//! the oracle CLI. They live here so `src/cli/mod.rs` stays a dispatcher.

use crate::error::{OkfError, Result};
use crate::settings::Settings;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

const SKILLS: &[(&str, &str)] = &[
    (
        "okfgraph-cli",
        include_str!("../cli/skills/okfgraph-cli/SKILL.md"),
    ),
    (
        "okfgraph-ingest",
        include_str!("../cli/skills/okfgraph-ingest/SKILL.md"),
    ),
    (
        "okfgraph-mcp",
        include_str!("../cli/skills/okfgraph-mcp/SKILL.md"),
    ),
];

/// Offline cache report for the configured embedding model (no engine load).
#[cfg(feature = "embed")]
pub fn model_info(settings: &Settings, converters: bool) -> Result<Value> {
    let precision = match settings.precision.as_str() {
        "" | "auto" => {
            if settings.device == "cpu" || !embroider::cuda_available() {
                "fp32"
            } else {
                "fp16"
            }
        }
        other => other,
    };
    let parsed = embroider::Precision::parse(precision).map_err(|err| {
        OkfError::new("BAD_VALUE", err.to_string())
            .op("model_info")
            .field("precision", json!(precision))
    })?;
    let cache = settings.cache_dir.clone().map(PathBuf::from);
    let report = embroider::cache_info(&settings.model_id, None, cache, parsed)
        .map_err(|err| OkfError::new("BAD_VALUE", err.to_string()).op("model_info"))?;
    let mut files = Map::new();
    for (name, path) in &report.files {
        files.insert(
            name.clone(),
            path.as_ref()
                .map(|value| Value::String(normalize(value)))
                .unwrap_or(Value::Null),
        );
    }
    let mut info = json!({
        "model_id": report.model_id,
        "repo": report.repo,
        "precision": precision,
        "cache_dir": normalize(&report.cache_dir),
        "files": files,
        "cached": report.cached,
        "snapshot_path": report.snapshot_path.as_ref().map(|v| normalize(v)),
        "disk_usage_bytes": report.disk_usage_bytes,
    });
    if converters {
        // Converter cache status lives in the optional PDF add-on; the slim
        // wheel does not import bobine, so keep the field visible but inert.
        info["converters"] = json!({
            "available": false,
            "note": "converter status is provided by the okfgraph-pdf add-on",
            "reports": [],
        });
    }
    Ok(info)
}

#[cfg(feature = "embed")]
fn normalize(path: &std::path::Path) -> String {
    let mut out = path.to_path_buf();
    // Mirror os.path.normpath's removal of trailing separators.
    while !out.as_os_str().is_empty() && (out.ends_with("\\") || out.ends_with("/")) {
        out.pop();
    }
    out.to_string_lossy().into_owned()
}

#[cfg(not(feature = "embed"))]
pub fn model_info(_settings: &Settings, _converters: bool) -> crate::error::Result<Value> {
    Err(OkfError::new(
        "CONFIG_INVALID",
        "model-info requires the native embed feature",
    )
    .op("model_info"))
}

/// Copy the three packaged agent skills into a harness skills directory.
pub fn setup_skills(target: Option<&str>, names: Option<Vec<String>>) -> Result<Value> {
    let available: BTreeMap<&str, &str> = SKILLS.iter().copied().collect();
    let chosen = names.unwrap_or_else(|| available.keys().map(|value| value.to_string()).collect());
    let unknown: Vec<String> = chosen
        .iter()
        .filter(|name| !available.contains_key(name.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        let known: Vec<&str> = available.keys().copied().collect();
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("unknown skill(s) {unknown:?}; packaged: {known:?}"),
        )
        .op("setup_skills"));
    }
    let root = PathBuf::from(target.unwrap_or(".claude/skills"));
    let mut installed = Vec::new();
    let mut updated = Vec::new();
    let mut unchanged = Vec::new();
    for name in chosen {
        let bytes = available[name.as_str()].as_bytes();
        let dest = root.join(&name).join("SKILL.md");
        let bucket = if dest.is_file() {
            if std::fs::read(&dest)
                .map(|old| old == bytes)
                .unwrap_or(false)
            {
                "unchanged"
            } else {
                "updated"
            }
        } else {
            "installed"
        };
        if bucket != "unchanged" {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    OkfError::internal(err.to_string(), "std::fs::create_dir_all")
                })?;
            }
            std::fs::write(&dest, bytes)
                .map_err(|err| OkfError::internal(err.to_string(), "std::fs::write"))?;
        }
        match bucket {
            "installed" => installed.push(name),
            "updated" => updated.push(name),
            _ => unchanged.push(name),
        }
    }
    Ok(json!({
        "target": root.to_string_lossy(),
        "installed": installed,
        "updated": updated,
        "unchanged": unchanged,
    }))
}

/// Fill an offline Ladybug extension directory from `--from` or the cache.
pub fn setup_extensions(target: Option<&str>, from: Option<&str>, download: bool) -> Result<Value> {
    let root = target.map(PathBuf::from).unwrap_or_else(default_target);
    std::fs::create_dir_all(&root)
        .map_err(|err| OkfError::internal(err.to_string(), "std::fs::create_dir_all"))?;
    let mut copied = Map::new();
    let mut missing = Vec::new();
    for ext in ["vector", "fts"] {
        let name = format!("lib{ext}.lbug_extension");
        let source = from.map(PathBuf::from).map(|path| path.join(&name));
        let cached = ladybug_cache().map(|path| path.join(&name));
        let found = source
            .filter(|path| path.is_file())
            .or_else(|| cached.filter(|path| path.is_file()));
        match found {
            Some(path) => {
                let dest = root.join(&name);
                if path != dest {
                    std::fs::copy(&path, &dest)
                        .map_err(|err| OkfError::internal(err.to_string(), "std::fs::copy"))?;
                }
                copied.insert(
                    ext.into(),
                    Value::String(path.to_string_lossy().into_owned()),
                );
            }
            None => missing.push(ext),
        }
    }
    if !missing.is_empty() {
        let where_ = match from {
            Some(path) => format!("--from {path}"),
            None if !download => {
                format!("Ladybug's cache {} (download disabled)", ladybug_display())
            }
            None => format!("Ladybug's cache {}", ladybug_display()),
        };
        return Err(OkfError::new(
            "FILE_NOT_FOUND",
            format!("extension(s) {} not found in {where_}", missing.join(", ")),
        )
        .op("setup_extensions"));
    }
    // Presence is not integrity: load both native files into a scratch DB
    // exactly as `okfgraph.components.extensions.verify` does.
    let checks = crate::util::extensions::verify(&root)?;
    let verified: Map<String, Value> = checks
        .into_iter()
        .map(|(ext, status)| (ext, Value::String(status)))
        .collect();
    let ok = verified.values().all(|value| value.as_str() == Some("ok"));
    let toml = format!(
        "[database]\nextension_dir = \"{}\"",
        root.to_string_lossy().replace('\\', "/")
    );
    Ok(json!({
        "target": root.to_string_lossy(),
        "copied": copied,
        "verified": verified,
        "ok": ok,
        "toml": toml,
    }))
}

fn default_target() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(".local").join("share"));
    base.join("okfgraph")
        .join("extensions")
        .join(abi_version())
        .join(platform_tag())
}

fn ladybug_cache() -> Option<PathBuf> {
    Some(
        home()
            .join(".lbdb")
            .join("extension")
            .join(abi_version())
            .join(platform_tag()),
    )
}

fn ladybug_display() -> String {
    ladybug_cache()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn home() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn abi_version() -> &'static str {
    // Pinned to the oracle's lbug 0.22.1 extension ABI.
    "0.22.0"
}

fn platform_tag() -> String {
    let system = if cfg!(target_os = "windows") {
        "win"
    } else if cfg!(target_os = "macos") {
        "osx"
    } else {
        "linux"
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "amd64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        std::env::consts::ARCH
    };
    format!("{system}_{arch}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_names_are_stable() {
        let names: Vec<_> = SKILLS.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["okfgraph-cli", "okfgraph-ingest", "okfgraph-mcp"]);
    }

    #[test]
    fn unknown_skill_is_a_typed_usage_error() {
        let err = setup_skills(None, Some(vec!["nope".into()])).unwrap_err();
        assert_eq!(err.code, "BAD_VALUE");
    }
}
