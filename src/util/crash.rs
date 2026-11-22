//! Crash reports for unexpected native failures.
//!
//! The CLI turns `INTERNAL` errors into a one-line message; this module keeps
//! the full error context on disk so a truncated terminal capture still has
//! the diagnostic detail. Reports are redacted before they are written: home
//! directories, bearer tokens and key/token/password values are replaced.

use crate::error::OkfError;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use time::macros::format_description;
use time::OffsetDateTime;

/// Keep the newest 20 reports, matching the Python 0.15 writer.
const KEEP: usize = 20;

/// Resolve the crash directory: `OKF_CRASH_DIR`, then the platform state dir.
pub fn crash_dir() -> PathBuf {
    if let Ok(explicit) = env::var("OKF_CRASH_DIR") {
        if !explicit.is_empty() {
            return PathBuf::from(explicit);
        }
    }
    #[cfg(windows)]
    {
        if let Ok(base) = env::var("LOCALAPPDATA") {
            if !base.is_empty() {
                return PathBuf::from(base).join("okfgraph").join("crash");
            }
        }
    }
    let base = env::var("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".local").join("state"));
    base.join("okfgraph").join("crash")
}

fn home_dir() -> PathBuf {
    env::var("USERPROFILE")
        .or_else(|_| env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

/// Write a redacted report and return its path. Never propagates I/O errors.
pub fn write_report(error: &OkfError, command: &str, argv: &[String]) -> Option<PathBuf> {
    write_report_text(&format!("{error:#?}"), command, argv)
}

fn write_report_text(text: &str, command: &str, argv: &[String]) -> Option<PathBuf> {
    let dir = crash_dir();
    fs::create_dir_all(&dir).ok()?;
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let stamp = now
        .format(format_description!(
            "[year][month][day]-[hour][minute][second]-[subsecond digits:6]"
        ))
        .unwrap_or_else(|_| "00000000-000000-000000".to_string());
    let path = dir.join(format!("okf-rust-{stamp}-{}.log", std::process::id()));
    let body = format!(
        "time:     {}\ncommand:  {}\nargv:     {}\nokfgraph: {}\nrust:     {}\nplatform: {}\ncwd:      {}\n\n{}\n",
        now.format(format_description!(
            "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:6]"
        ))
        .unwrap_or_else(|_| "unknown".to_string()),
        if command.is_empty() {
            "(start-up)"
        } else {
            command
        },
        redact(&format!("{:?}", redact_args(argv))),
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_RUST_VERSION"),
        std::env::consts::OS,
        redact(&env::current_dir().unwrap_or_default().to_string_lossy()),
        redact(text),
    );
    fs::write(&path, body).ok()?;
    prune(&dir);
    Some(path)
}

fn redact_args(argv: &[String]) -> Vec<String> {
    argv.iter()
        .map(|arg| {
            if arg.contains("token") || arg.contains("key") || arg.contains("password") {
                "<redacted-arg>".to_string()
            } else {
                arg.clone()
            }
        })
        .collect()
}

fn prune(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("okf-") && name.ends_with(".log"))
        })
        .filter_map(|entry| {
            entry
                .metadata()
                .ok()
                .and_then(|meta| meta.modified().ok())
                .map(|time| (time, entry.path()))
        })
        .collect();
    files.sort_by_key(|(time, path)| (*time, path.clone()));
    for (_, old) in files.into_iter().rev().skip(KEEP) {
        let _ = fs::remove_file(old);
    }
}

/// Redact common secret and local-path patterns. This is deliberately simple
/// and dependency-free: it must work even when the rest of the core is down.
pub fn redact(text: &str) -> String {
    let mut out = redact_secrets(text);
    out = redact_home_paths(&out);
    out
}

fn redact_secrets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &text[i..];
        let lower = rest.to_ascii_lowercase();
        if lower.starts_with("bearer ") {
            out.push_str("Bearer <redacted>");
            i += 7;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b',' {
                i += 1;
            }
            continue;
        }
        let lower_rest = lower.trim_start();
        let skip = rest.len() - lower_rest.len();
        let key = [
            "authorization: ",
            "api_key: ",
            "api-key: ",
            "apikey: ",
            "token: ",
            "password: ",
            "secret: ",
            "api_key=",
            "api-key=",
            "apikey=",
            "token=",
            "password=",
            "secret=",
        ]
        .into_iter()
        .find(|key| lower_rest.starts_with(key));
        if let Some(key) = key {
            out.push_str(&rest[..skip + key.len()]);
            out.push_str("<redacted>");
            i += skip + key.len();
            while i < bytes.len()
                && !bytes[i].is_ascii_whitespace()
                && bytes[i] != b','
                && bytes[i] != b';'
                && bytes[i] != b'}'
            {
                i += 1;
            }
            continue;
        }
        let ch = rest.chars().next().expect("non-empty rest");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn redact_home_paths(text: &str) -> String {
    let mut out = text.to_string();
    for prefix in ["C:\\Users\\", "c:\\Users\\", "/Users/", "/home/"] {
        while let Some(start) = find_ci(&out, prefix) {
            let value_start = start + prefix.len();
            let rest = &out[value_start..];
            let end = rest
                .find(['\\', '/', '"', '\'', ' ', ',', ';'])
                .map(|offset| value_start + offset)
                .unwrap_or(out.len());
            if end == value_start {
                break;
            }
            out.replace_range(start..end, "<HOME>");
        }
    }
    out
}

fn find_ci(text: &str, needle: &str) -> Option<usize> {
    let lower = text.to_ascii_lowercase();
    lower.find(&needle.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn redacts_tokens_and_home_paths() {
        let text = "C:\\Users\\Main\\secret.db bearer abc123 token=xyz api_key: s3cr3t";
        let redacted = redact(text);
        assert!(!redacted.contains("Main"));
        assert!(!redacted.contains("abc123"));
        assert!(!redacted.contains("xyz"));
        assert!(!redacted.contains("s3cr3t"));
        assert!(redacted.contains("<HOME>"));
        assert!(redacted.contains("Bearer <redacted>"));
    }

    #[test]
    fn writes_and_prunes_redacted_reports() {
        let _guard = env_lock().lock().unwrap();
        let dir = std::env::temp_dir().join(format!(
            "okfgraph-crash-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        env::set_var("OKF_CRASH_DIR", &dir);
        let err = OkfError::internal("bearer abc123", "read");
        let path =
            write_report(&err, "read", &["okf".into(), "read".into()]).expect("report written");
        let body = fs::read_to_string(path).unwrap();
        assert!(body.contains("INTERNAL"));
        assert!(!body.contains("abc123"));
        assert!(body.contains("Bearer <redacted>"));
        for n in 0..25 {
            fs::write(
                dir.join(format!("okf-rust-{n:04}.log")),
                format!("stale {n}"),
            )
            .unwrap();
        }
        env::remove_var("OKF_CRASH_DIR");
        // A second report prunes the directory back to KEEP.
        env::set_var("OKF_CRASH_DIR", &dir);
        let _ = write_report(&err, "read", &[]);
        let count = fs::read_dir(&dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".log"))
            })
            .count();
        env::remove_var("OKF_CRASH_DIR");
        fs::remove_dir_all(&dir).ok();
        assert_eq!(count, KEEP);
    }
}
