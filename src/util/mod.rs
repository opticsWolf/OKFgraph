//! Paths, hashing, clock, network/SSRF, filesystem, and crash reports.

use std::path::Path;

pub mod crash;
pub mod extensions;

/// Python 0.15 `is_concept_file`: one choke point for import, detach, lint,
/// diff and sync so all runners agree on source discovery. Extensions are
/// case-insensitive and reserved filenames are always skipped.
pub fn is_concept_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(ext.as_deref(), Some("md" | "markdown" | "txt")) {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    name != "index.md" && name != "log.md"
}

/// Format a caller-supplied path the way Python's `str(Path(value))` does.
/// On Windows that means normalizing `/` to `\`; elsewhere the string is
/// unchanged (a backslash is a legal filename character on POSIX).
pub fn python_path_str(path: &str) -> String {
    if cfg!(windows) {
        path.replace('/', "\\")
    } else {
        path.to_string()
    }
}

/// Reject concept ids that would escape a bundle/output directory when joined.
/// Path-derived ids may contain spaces/unicode, but never `..`, a leading
/// slash, a backslash, an empty component, or a Windows drive/colon.
pub fn safe_bundle_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('/')
        && !id.contains('\\')
        && !id.contains(':')
        && id
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}
