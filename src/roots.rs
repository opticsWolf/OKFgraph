//! Named-bundle-root helpers shared by import, writes, delete and detach.
//!
//! The primary root (`Settings::bundle_root`) owns bare FileHash keys and
//! bare concept ids. Each entry in `Settings::roots` owns the `@alias/...`
//! namespace. Path resolution is longest-prefix (Python's
//! `resolve_alias_for_path`), so a file in a subdirectory of a named root
//! still mints a namespaced id.

use crate::error::{OkfError, Result};
use crate::settings::Settings;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        }
    })
}

/// Resolve `.`/`..` without `canonicalize()`: Python `Path.resolve()` keeps
/// the ordinary Windows drive spelling, while Rust's canonicalize() emits a
/// `\\?\` verbatim prefix that would leak into envelope paths. An absent or
/// unmounted root still gets a plain absolute path (a legal liveness state).
pub fn canonicalish(path: &Path) -> PathBuf {
    let resolved = absolute(path);
    // `std::path::absolute` can add a `\\?\` prefix on Windows depending on
    // the input spelling. Strip it so root-relative comparisons and envelope
    // paths keep Python's ordinary drive-letter form.
    #[cfg(windows)]
    {
        let text = resolved.to_string_lossy();
        for prefix in ["\\\\?\\", "\\\\.\\"] {
            if let Some(rest) = text.strip_prefix(prefix) {
                return PathBuf::from(rest);
            }
        }
    }
    resolved
}

fn valid_alias(alias: &str) -> bool {
    let mut chars = alias.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphanumeric()
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Validate aliases and reject roots that overlap the primary tree: one file
/// in two roots would mint two identities.
pub fn validate(settings: &Settings) -> Result<()> {
    let mut paths: BTreeMap<String, PathBuf> = BTreeMap::new();
    for (alias, path) in &settings.roots {
        if !valid_alias(alias) {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!(
                    "root alias {alias:?} must match letters/digits/_/- and must not start with '@'"
                ),
            ));
        }
        if path.trim().is_empty() {
            return Err(OkfError::new(
                "CONFIG_INVALID",
                format!("root {alias:?} needs a non-empty path"),
            ));
        }
        paths.insert(alias.clone(), canonicalish(Path::new(path)));
    }
    if let Some(primary) = settings.bundle_root.as_ref() {
        if !primary.trim().is_empty() {
            paths.insert(String::new(), canonicalish(Path::new(primary)));
        }
    }
    let names: Vec<String> = paths.keys().cloned().collect();
    for (i, left) in names.iter().enumerate() {
        for right in names.iter().skip(i + 1) {
            let a = &paths[left];
            let b = &paths[right];
            if a == b || a.starts_with(b) || b.starts_with(a) {
                return Err(OkfError::new(
                    "CONFIG_INVALID",
                    format!(
                        "roots {left:?} ({}) and {right:?} ({}) overlap: a file in two roots would mint two identities",
                        a.display(),
                        b.display()
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// `(alias, native_id)` -> `@alias/native_id`, or the bare id for the primary
/// root.
pub fn namespaced_id(alias: &str, native_id: &str) -> String {
    if alias.is_empty() {
        native_id.to_string()
    } else {
        format!("@{alias}/{native_id}")
    }
}

/// `@alias/rest` -> `(alias, rest)`; anything else -> `("", id)`.
pub fn split_namespaced(id: &str) -> (String, String) {
    if let Some(rest) = id.strip_prefix('@') {
        if let Some((alias, native)) = rest.split_once('/') {
            if !alias.is_empty() && !native.is_empty() && valid_alias(alias) {
                return (alias.to_string(), native.to_string());
            }
        }
    }
    (String::new(), id.to_string())
}

/// Prefix a delta key with its root's namespace, idempotently.
pub fn prefix_key(alias: &str, native_rel: &str) -> String {
    if alias.is_empty() || native_rel.starts_with(&format!("@{alias}/")) {
        native_rel.to_string()
    } else {
        format!("@{alias}/{native_rel}")
    }
}

/// Strip another root's namespace from a delta key.
pub fn strip_prefix<'a>(alias: &str, key: &'a str) -> Option<&'a str> {
    if alias.is_empty() {
        return Some(key);
    }
    key.strip_prefix(&format!("@{alias}/"))
}

#[derive(Debug, Clone, Default)]
pub struct Roots {
    primary: Option<PathBuf>,
    named: BTreeMap<String, PathBuf>,
}

impl Roots {
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            primary: settings
                .bundle_root
                .as_ref()
                .filter(|path| !path.trim().is_empty())
                .map(|path| canonicalish(Path::new(path))),
            named: settings
                .roots
                .iter()
                .map(|(alias, path)| (alias.clone(), canonicalish(Path::new(path))))
                .collect(),
        }
    }

    /// Absolute candidates for a FileHash key, in Python's priority order.
    pub fn resolve_key(&self, key: &str) -> Vec<PathBuf> {
        let native = key.replace('\\', "/");
        let (alias, relative) = match native
            .strip_prefix('@')
            .and_then(|rest| rest.split_once('/'))
        {
            Some((alias, rest)) if !alias.is_empty() && !rest.is_empty() => {
                (Some(alias.to_string()), rest.to_string())
            }
            _ => (None, native.clone()),
        };
        let mut out = Vec::new();
        if let Some(root) = alias.as_ref().and_then(|alias| self.named.get(alias)) {
            out.push(root.join(&relative));
        }
        if let Some(primary) = &self.primary {
            // Unknown aliases keep the documented primary-root fallback with
            // the full `@alias/...` key; known aliases keep it as a fallback
            // for graphs imported before the alias was configured.
            out.push(primary.join(if alias.is_some() { &native } else { &relative }));
        }
        out
    }

    /// Longest-prefix match of a path against the named roots. The primary
    /// root deliberately stays the empty alias: files there keep bare ids.
    pub fn alias_for_path(&self, path: &Path) -> String {
        let resolved = canonicalish(path);
        let mut best: Option<(&str, usize)> = None;
        for (alias, root) in &self.named {
            if resolved == *root || resolved.starts_with(root) {
                let len = root.components().count();
                if best.is_none_or(|(_, best_len)| len > best_len) {
                    best = Some((alias.as_str(), len));
                }
            }
        }
        best.map(|(alias, _)| alias.to_string()).unwrap_or_default()
    }

    /// Exact alias of an explicit identity root, if it is a configured root.
    pub fn alias_for_root(&self, root: &Path) -> String {
        let resolved = canonicalish(root);
        self.named
            .iter()
            .find(|(_, path)| **path == resolved)
            .map(|(alias, _)| alias.clone())
            .unwrap_or_default()
    }

    pub fn root_for_alias(&self, alias: &str) -> Option<&Path> {
        if alias.is_empty() {
            self.primary.as_deref()
        } else {
            self.named.get(alias).map(PathBuf::as_path)
        }
    }

    pub fn all_roots(&self) -> Vec<(&str, &Path)> {
        let mut out = Vec::new();
        if let Some(primary) = &self.primary {
            out.push(("", primary.as_path()));
        }
        out.extend(
            self.named
                .iter()
                .map(|(alias, path)| (alias.as_str(), path.as_path())),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, Value};

    fn settings() -> Settings {
        Settings::from_map(&Map::from_iter([
            (
                "bundle_root".to_string(),
                Value::String("parity/corpus/multi-root/primary".into()),
            ),
            (
                "roots".to_string(),
                serde_json::json!({"other": "parity/corpus/multi-root/other"}),
            ),
        ]))
        .expect("settings")
    }

    #[test]
    fn named_root_resolves_before_primary_fallback() {
        let roots = Roots::from_settings(&settings());
        let candidates = roots.resolve_key("@other/alpha.md");
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].ends_with("multi-root/other/alpha.md"));
        assert!(candidates[1].ends_with("multi-root/primary/@other/alpha.md"));
    }

    #[test]
    fn alias_for_path_uses_longest_prefix_and_splits_ids() {
        let roots = Roots::from_settings(&settings());
        assert_eq!(
            roots.alias_for_path(Path::new("parity/corpus/multi-root/other/alpha.md")),
            "other"
        );
        assert_eq!(
            roots.alias_for_path(Path::new("parity/corpus/minimal/a.md")),
            ""
        );
        assert_eq!(
            split_namespaced("@other/sub/doc"),
            ("other".into(), "sub/doc".into())
        );
        assert_eq!(split_namespaced("alpha"), (String::new(), "alpha".into()));
    }

    #[test]
    fn overlapping_roots_are_refused() {
        let map = Map::from_iter([
            (
                "bundle_root".to_string(),
                Value::String("parity/corpus".into()),
            ),
            (
                "roots".to_string(),
                serde_json::json!({"other": "parity/corpus/multi-root/other"}),
            ),
        ]);
        let err = Settings::from_map(&map).expect_err("overlap");
        assert_eq!(err.code, "CONFIG_INVALID");
        assert!(err.message.contains("overlap"));
    }
}
