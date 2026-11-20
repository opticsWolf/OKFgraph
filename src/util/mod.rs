//! Paths, hashing, clock, network/SSRF, filesystem, and crash reports.

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
