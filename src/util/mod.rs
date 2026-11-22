//! Paths, hashing, clock, network/SSRF, filesystem, and crash reports.

pub mod crash;
pub mod extensions;

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
