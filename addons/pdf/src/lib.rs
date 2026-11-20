//! Optional native PDF/Office plugin for OKFgraph.
//!
//! The C ABI is deliberately tiny so the slim core wheel never links bobine:
//!
//! - `okf_pdf_abi_version() -> u32`
//! - `okf_pdf_convert(input, workdir, config_json) -> JSON string`
//! - `okf_pdf_free(ptr)`
//!
//! The JSON result is either `{"md_path", "image_dir", "page_count"}` or
//! `{"error": "..."}`. Strings returned by `convert` must be released with
//! `okf_pdf_free`.

use bobine::config::ConverterConfig;
use bobine::converter::{HybridConverter, ProgressHooks};
use serde_json::{json, Value as Json};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const ABI_VERSION: c_uint = 1;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    /// Warm converter: the ONNX sessions are expensive, so repeated PDF
    /// ingests on the same thread reuse them until the idle window expires
    /// or the conversion configuration changes.
    static WARM: RefCell<Option<Warm>> = const { RefCell::new(None) };
}

struct Warm {
    key: String,
    cache: PathBuf,
    last_used: Instant,
    converter: HybridConverter,
}

impl Warm {
    fn close(mut self) {
        self.converter.close();
    }
}

fn take_warm(key: &str, cache: &Path, idle: Duration) -> Option<Warm> {
    WARM.with(|slot| {
        let mut slot = slot.borrow_mut();
        let reusable = slot.as_ref().is_some_and(|warm| {
            warm.key == key && warm.cache == cache && warm.last_used.elapsed() <= idle
        });
        if reusable {
            return slot.take();
        }
        if let Some(stale) = slot.take() {
            stale.close();
        }
        None
    })
}

fn restore_warm(warm: Option<Warm>) {
    WARM.with(|slot| {
        if let Some(previous) = slot.replace(warm) {
            previous.close();
        }
    });
}

#[no_mangle]
pub extern "C" fn okf_pdf_abi_version() -> c_uint {
    ABI_VERSION
}

unsafe fn str_arg(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("null plugin argument".into());
    }
    Ok(CStr::from_ptr(ptr).to_string_lossy().into_owned())
}

fn error_json(message: impl Into<String>) -> CString {
    CString::new(json!({"error": message.into()}).to_string())
        .unwrap_or_else(|_| CString::new("{\"error\":\"plugin error\"}").expect("literal"))
}

/// A bobine failure about one file is a conversion outcome, not a missing
/// add-on. Mirror Python 0.15's typed `CONVERSION_FAILED` envelope so both
/// runners report the same code/fields/remedy for corrupt or unsupported
/// inputs.
fn conversion_error_json(input: &Path, message: impl Into<String>) -> CString {
    let source = input.to_string_lossy();
    let value = json!({
        "error": format!("bobine could not convert {source}: {}", message.into()),
        "fields": {"path": source, "converter": "bobine"},
        "remedy": "check the file opens in another reader, or route it through a different converter",
    });
    CString::new(value.to_string()).unwrap_or_else(|_| error_json("invalid path"))
}

fn default_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(cache);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(cache).join("huggingface").join("hub");
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn result_json(md_path: &Path, image_dir: &Path, page_count: usize) -> CString {
    let value = json!({
        "md_path": md_path.to_string_lossy(),
        "image_dir": image_dir.to_string_lossy(),
        "page_count": page_count as i64,
    });
    CString::new(value.to_string()).unwrap_or_else(|_| error_json("invalid path"))
}

/// # Safety
///
/// `input`, `workdir`, and `config_json` must point at NUL-terminated C
/// strings valid for the duration of the call. The returned pointer must be
/// released with [`okf_pdf_free`].
#[no_mangle]
pub unsafe extern "C" fn okf_pdf_convert(
    input: *const c_char,
    workdir: *const c_char,
    config_json: *const c_char,
) -> *mut c_char {
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<CString, String> {
        let input = str_arg(input)?;
        let workdir = str_arg(workdir)?;
        let config = str_arg(config_json)?;
        let config_value: Json =
            serde_json::from_str(&config).map_err(|err| format!("bad config: {err}"))?;
        let idle = config_value
            .get("idle_seconds")
            .and_then(Json::as_u64)
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(300));
        // Core sends only the knobs it owns; fill every other required field
        // from bobine's defaults before deserializing. `idle_seconds` is ours,
        // not bobine's.
        let mut core_value = config_value.clone();
        if let Some(object) = core_value.as_object_mut() {
            object.remove("idle_seconds");
        }
        let mut merged = serde_json::to_value(ConverterConfig::default())
            .map_err(|err| format!("bad config defaults: {err}"))?;
        if let (Some(dst), Some(src)) = (merged.as_object_mut(), core_value.as_object()) {
            for (key, value) in src {
                dst.insert(key.clone(), value.clone());
            }
        }
        let config: ConverterConfig =
            serde_json::from_value(merged).map_err(|err| format!("bad config: {err}"))?;
        let cache = core_value
            .get("cache_dir")
            .and_then(Json::as_str)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_cache_dir);
        let input = PathBuf::from(input);
        let workdir = PathBuf::from(workdir);
        std::fs::create_dir_all(&workdir).map_err(|err| err.to_string())?;
        let hooks = ProgressHooks::default();
        let key = config_value.to_string();
        let mut warm = take_warm(&key, &cache, idle);
        if warm.is_none() {
            warm = Some(Warm {
                key,
                cache: cache.clone(),
                last_used: Instant::now(),
                converter: HybridConverter::new(config, &cache),
            });
        }
        let outcome = {
            let current = warm.as_mut().expect("warm converter initialized");
            current
                .converter
                .ingest_document(&input, &workdir, None, &hooks)
        };
        if let Some(current) = warm.as_mut() {
            current.last_used = Instant::now();
        }
        restore_warm(warm);
        let doc = match outcome {
            Ok(doc) => doc,
            Err(err) => return Ok(conversion_error_json(&input, err.to_string())),
        };
        Ok(result_json(&doc.md_path, &doc.image_dir, doc.page_count))
    }));
    match outcome {
        Ok(Ok(value)) => value.into_raw(),
        Ok(Err(message)) => {
            LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(message.clone()));
            error_json(message).into_raw()
        }
        Err(_) => error_json("PDF converter panicked").into_raw(),
    }
}

/// # Safety
///
/// `ptr` must be null or a pointer previously returned by
/// [`okf_pdf_convert`] that has not been freed yet.
#[no_mangle]
pub unsafe extern "C" fn okf_pdf_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
