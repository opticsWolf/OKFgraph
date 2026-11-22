//! Offline Ladybug extension integrity checks.
//!
//! `okf setup extensions` copies native extension files into a directory.
//! Presence alone is not integrity: the files must load into a scratch Ladybug
//! database built by this exact binary. This mirrors the Python 0.15 verifier.

use crate::error::{OkfError, Result};
use lbug::{Connection, Database, SystemConfig};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const EXTENSIONS: [&str; 2] = ["vector", "fts"];

fn find_extension(dir: &Path, name: &str) -> Option<PathBuf> {
    let file = format!("lib{name}.lbug_extension");
    [dir.join(&file), dir.join(name).join(&file)]
        .into_iter()
        .find(|candidate| candidate.is_file())
}

fn scratch_dir() -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "okfgraph-ext-verify-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Load both extensions into a scratch DB. Returns `{ext: "ok" | error}`.
pub fn verify(dir: &Path) -> Result<BTreeMap<String, String>> {
    let scratch = scratch_dir()?;
    let result = verify_in(&scratch, dir);
    let _ = fs::remove_dir_all(&scratch);
    result
}

fn verify_in(scratch: &Path, dir: &Path) -> Result<BTreeMap<String, String>> {
    let db_path = scratch.join("verify.db");
    let db = Database::new(&db_path, SystemConfig::default().read_only(false))
        .map_err(|err| OkfError::internal(err.to_string(), "lbug::Database::new"))?;
    let conn = Connection::new(&db)
        .map_err(|err| OkfError::internal(err.to_string(), "lbug::Connection::new"))?;
    let mut out = BTreeMap::new();
    for ext in EXTENSIONS {
        let Some(path) = find_extension(dir, ext) else {
            out.insert(ext.to_string(), "missing".to_string());
            continue;
        };
        let escaped = path.to_string_lossy().replace('\'', "''");
        match conn.query(&format!("LOAD EXTENSION '{escaped}'")) {
            Ok(_) => {
                out.insert(ext.to_string(), "ok".to_string());
            }
            Err(err) => {
                out.insert(ext.to_string(), err.to_string());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_extensions_are_reported() {
        let temp = tempfile::tempdir().unwrap();
        let report = verify(temp.path()).unwrap();
        assert_eq!(report.get("vector").map(String::as_str), Some("missing"));
        assert_eq!(report.get("fts").map(String::as_str), Some("missing"));
    }
}
