//! Lazy loader for the optional native PDF/Office conversion plugin.
//!
//! The core never links `bobine` directly. `okfgraph-pdf` exposes a tiny C
//! ABI; missing or incompatible plugins become `CONVERSION_FAILED` with the
//! documented install hint instead of a crash-report path.

use crate::error::{OkfError, Result};
use libloading::{Library, Symbol};
use serde_json::{json, Value as Json};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

const ABI_VERSION: u32 = 1;

type AbiVersion = unsafe extern "C" fn() -> c_uint;
type ConvertFn = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> *mut c_char;
type FreeFn = unsafe extern "C" fn(*mut c_char);

/// Result returned by the plugin's `convert` entry point.
#[derive(Debug, Clone)]
pub struct Converted {
    pub md_path: PathBuf,
    pub image_dir: PathBuf,
    pub page_count: i64,
}

/// A loaded PDF conversion plugin.
pub struct Plugin {
    _library: Library,
    convert: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> *mut c_char,
    free: unsafe extern "C" fn(*mut c_char),
}

fn hint() -> &'static str {
    "pip install \"okfgraph[cpu,pdf]\" or \"okfgraph[gpu,pdf]\""
}

fn failed(message: impl Into<String>) -> OkfError {
    OkfError::new("CONVERSION_FAILED", message).remedy(hint())
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(path) = std::env::var("OKFGRAPH_PDF_PLUGIN") {
        if !path.is_empty() {
            // An explicit override is authoritative: a wrong path is a clear
            // load error, not a silent fallback to a stale dev build.
            return vec![PathBuf::from(path)];
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in [
                "okfgraph_pdf.dll",
                "libokfgraph_pdf.so",
                "libokfgraph_pdf.dylib",
            ] {
                out.push(dir.join(name));
                // Wheels put the add-on beside `okfgraph/`; native CLI use
                // outside the Python facade finds it through these layouts.
                out.push(dir.join("okfgraph_pdf").join(name));
                out.push(dir.join("okfgraph.libs").join(name));
                if let Some(parent) = dir.parent() {
                    out.push(parent.join("lib").join(name));
                    out.push(parent.join("okfgraph_pdf").join(name));
                }
            }
        }
    }
    out
}

thread_local! {
    /// Warm plugin handle for `pdf_idle_seconds`; the converter itself still
    /// owns its one-shot ONNX sessions inside the add-on process.
    static WARM: RefCell<Option<(Rc<Plugin>, Instant)>> = const { RefCell::new(None) };
}

impl Plugin {
    /// Find and load the plugin, reusing the thread-local handle for
    /// `settings.pdf_idle_seconds`. No plugin means a typed, actionable error.
    pub fn load(settings: &crate::settings::Settings) -> Result<Rc<Self>> {
        let idle = settings.pdf_idle_seconds.max(0) as u64;
        let cached = WARM.with(|slot| {
            let mut slot = slot.borrow_mut();
            match slot.as_ref() {
                Some((plugin, used)) if used.elapsed().as_secs() <= idle => Some(Rc::clone(plugin)),
                _ => {
                    slot.take();
                    None
                }
            }
        });
        if let Some(plugin) = cached {
            WARM.with(|slot| {
                if let Some((_, used)) = slot.borrow_mut().as_mut() {
                    *used = Instant::now();
                }
            });
            return Ok(plugin);
        }
        let plugin = Rc::new(Self::load_uncached()?);
        WARM.with(|slot| {
            *slot.borrow_mut() = Some((Rc::clone(&plugin), Instant::now()));
        });
        Ok(plugin)
    }

    fn load_uncached() -> Result<Self> {
        let candidates = candidate_paths();
        let selected = candidates
            .iter()
            .find(|path| path.is_file())
            .ok_or_else(|| failed("PDF conversion requires the native add-on"))?;
        unsafe {
            let library = Library::new(selected).map_err(|err| {
                failed(format!(
                    "could not load PDF plugin {}: {err}",
                    selected.display()
                ))
            })?;
            let version: Symbol<AbiVersion> = library
                .get(b"okf_pdf_abi_version\0")
                .map_err(|_| failed("PDF plugin does not implement the OKFgraph ABI"))?;
            if version() != ABI_VERSION {
                return Err(failed(format!(
                    "PDF plugin ABI {} is incompatible with expected {}",
                    version(),
                    ABI_VERSION
                )));
            }
            let convert_fn = {
                let convert: Symbol<ConvertFn> = library
                    .get(b"okf_pdf_convert\0")
                    .map_err(|_| failed("PDF plugin does not implement okf_pdf_convert"))?;
                *convert
            };
            let free_fn = {
                let free: Symbol<FreeFn> = library
                    .get(b"okf_pdf_free\0")
                    .map_err(|_| failed("PDF plugin does not implement okf_pdf_free"))?;
                *free
            };
            Ok(Self {
                _library: library,
                convert: convert_fn,
                free: free_fn,
            })
        }
    }

    /// Convert `input` into `work_dir`. `config` is the bobine config JSON.
    pub fn convert(&self, input: &Path, work_dir: &Path, config: &Json) -> Result<Converted> {
        let input = CString::new(input.to_string_lossy().as_bytes())
            .map_err(|_| failed("PDF input path contains a NUL byte"))?;
        let work = CString::new(work_dir.to_string_lossy().as_bytes())
            .map_err(|_| failed("PDF work directory contains a NUL byte"))?;
        let config = CString::new(config.to_string())
            .map_err(|_| failed("PDF config contains a NUL byte"))?;
        let raw = unsafe { (self.convert)(input.as_ptr(), work.as_ptr(), config.as_ptr()) };
        if raw.is_null() {
            return Err(failed("PDF plugin returned no result"));
        }
        let text = unsafe {
            let value = CStr::from_ptr(raw).to_string_lossy().into_owned();
            (self.free)(raw);
            value
        };
        let value: Json = serde_json::from_str(&text)
            .map_err(|err| failed(format!("PDF plugin returned invalid JSON: {err}")))?;
        if let Some(error) = value.get("error").and_then(Json::as_str) {
            return Err(failed(error));
        }
        let md_path = value
            .get("md_path")
            .and_then(Json::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| failed("PDF plugin did not return md_path"))?;
        let image_dir = value
            .get("image_dir")
            .and_then(Json::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| work_dir.to_path_buf());
        let page_count = value.get("page_count").and_then(Json::as_i64).unwrap_or(0);
        Ok(Converted {
            md_path,
            image_dir,
            page_count,
        })
    }
}

/// Build the config JSON passed to the plugin.
pub fn config(settings: &crate::settings::Settings) -> Json {
    config_with(settings, "auto", true)
}

pub fn config_with(
    settings: &crate::settings::Settings,
    routing_mode: &str,
    extract_images: bool,
) -> Json {
    json!({
        "routing_mode": routing_mode,
        "extract_images": extract_images,
        "device": settings.pdf_device,
        "precision": settings.pdf_precision,
        "cache_dir": settings.cache_dir,
        // The add-on keeps its HybridConverter/ONNX sessions warm between
        // calls; this is the same idle window used for the plugin handle.
        "idle_seconds": settings.pdf_idle_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_forwards_ingest_knobs_and_idle_window() {
        let settings = crate::settings::Settings {
            pdf_device: "cpu".into(),
            pdf_precision: "int8".into(),
            pdf_idle_seconds: 42,
            cache_dir: Some("cache".into()),
            ..Default::default()
        };
        let config = config_with(&settings, "ocr", false);
        assert_eq!(config["routing_mode"], "ocr");
        assert_eq!(config["extract_images"], false);
        assert_eq!(config["device"], "cpu");
        assert_eq!(config["precision"], "int8");
        assert_eq!(config["idle_seconds"], 42);
        assert_eq!(config["cache_dir"], "cache");
    }
}
