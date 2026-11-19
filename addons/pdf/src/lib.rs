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

const ABI_VERSION: c_uint = 1;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
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

fn default_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(home);
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
        let mut config: ConverterConfig = serde_json::from_value(config_value.clone())
            .map_err(|err| format!("bad config: {err}"))?;
        if config_value.get("routing_mode").is_none() {
            config.routing.routing_mode = bobine::config::RoutingMode::Auto;
        }
        let cache = config_value
            .get("cache_dir")
            .and_then(Json::as_str)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_cache_dir);
        let input = PathBuf::from(input);
        let workdir = PathBuf::from(workdir);
        std::fs::create_dir_all(&workdir).map_err(|err| err.to_string())?;
        let hooks = ProgressHooks::default();
        let mut converter = HybridConverter::new(config, &cache);
        let doc = converter
            .ingest_document(&input, &workdir, None, &hooks)
            .map_err(|err| err.to_string())?;
        converter.close();
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
