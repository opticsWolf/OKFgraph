//! Embedder trait and the real embroider/ONNX text backend.
//!
//! The model is `google/embeddinggemma-2` by default (mean-pooled, Matryoshka
//! truncated to the graph's stored `embedding_dim`). `device: "auto"` follows
//! embroider's CUDA probe, and `ORT_DYLIB_PATH` selects the ONNX Runtime
//! binary — set it to a GPU build with CUDA/cuDNN sibling DLLs on PATH for
//! GPU embedding. No Python, no model downloads in normal parity runs (the
//! pinned HF revisions are expected to be cached).

use crate::error::{OkfError, Result};
use crate::settings::Settings;
#[cfg(feature = "embed")]
use embroider::{diag, DeviceReq, JinaV5, Precision};
#[cfg(feature = "vision")]
use embroider::{GemmaVision, JinaV5Vision};
use serde_json::{Map, Value as Json};
use std::fmt;
#[cfg(feature = "embed")]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The two task prompts used by the oracle. Gemma text uses mean pooling;
/// the wrapper is intentionally stringly-typed to match the Python API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Query,
    Document,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Query => "Query",
            Self::Document => "Document",
        }
    }
}

/// Minimal embedding contract used by search and ingest.
pub trait Embedder: Send + Sync + fmt::Debug {
    fn encode(&self, text: &str, task: Task) -> Result<Vec<f32>>;
    fn encode_many(&self, texts: &[String], task: Task) -> Result<Vec<Vec<f32>>>;
    fn count_tokens(&self, text: &str) -> Result<usize>;
    fn dim(&self) -> usize;
    fn model_id(&self) -> &str;
    fn used_cuda(&self) -> bool;
    fn precision(&self) -> &'static str;
    fn max_len(&self) -> usize;
    fn space(&self) -> Option<String> {
        None
    }
}

/// Lazy, shared embedder pool. Opening ONNX is expensive, so `Router::open`
/// never touches the model; the first search/embed call opens it once.
#[derive(Clone, Default)]
pub struct EmbedderPool {
    inner: Arc<Mutex<Option<Arc<dyn Embedder>>>>,
}

impl fmt::Debug for EmbedderPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loaded = self
            .inner
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        f.debug_struct("EmbedderPool")
            .field("loaded", &loaded)
            .finish()
    }
}

impl EmbedderPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, settings: &Settings) -> Result<Arc<dyn Embedder>> {
        if settings.backend == "external" {
            return Err(OkfError::new(
                "EXTERNAL_UNAVAILABLE",
                "external embedding lanes are not implemented in this build",
            )
            .field("backend", Json::String("external".into()))
            .field("endpoint", Json::String(settings.endpoint_url.clone()))
            .remedy("set backend='embroider' or install a build with external lanes"));
        }
        if let Some(loaded) = self
            .inner
            .lock()
            .map_err(|_| OkfError::internal("embedder lock poisoned", "mutex"))?
            .as_ref()
        {
            return Ok(Arc::clone(loaded));
        }
        #[cfg(feature = "embed")]
        {
            let embedder: Arc<dyn Embedder> = Arc::new(NativeEmbedder::open(settings)?);
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| OkfError::internal("embedder lock poisoned", "mutex"))?;
            Ok(Arc::clone(guard.get_or_insert(embedder)))
        }
        #[cfg(not(feature = "embed"))]
        {
            let _ = settings;
            Err(OkfError::new(
                "NO_ORT_RUNTIME",
                "this build has no embedder; rebuild with --features embed",
            ))
        }
    }

    pub fn loaded(&self) -> Option<Arc<dyn Embedder>> {
        self.inner
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(Arc::clone))
    }

    pub fn is_loaded(&self) -> bool {
        self.inner
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }
}

/// The real native backend. One session per router process.
#[cfg(feature = "embed")]
pub struct NativeEmbedder {
    engine: JinaV5,
    dim: usize,
    model_id: String,
    used_cuda: bool,
    precision: &'static str,
    max_len: usize,
    space: Option<String>,
}

#[cfg(feature = "embed")]
impl fmt::Debug for NativeEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeEmbedder")
            .field("dim", &self.dim)
            .field("model_id", &self.model_id)
            .field("used_cuda", &self.used_cuda)
            .field("precision", &self.precision)
            .field("space", &self.space)
            .finish()
    }
}

#[cfg(feature = "embed")]
impl NativeEmbedder {
    /// Configure ORT from the environment before its first initialization.
    /// `ORT_DYLIB_PATH` names the GPU dylib; `OKFGRAPH_ORT_DLL_DIRS` supplies
    /// additional `;`/`:`-separated directories for CUDA/cuDNN siblings.
    pub fn configure_runtime() -> Result<Json> {
        let dylib = std::env::var("ORT_DYLIB_PATH").ok();
        let raw_dirs = std::env::var("OKFGRAPH_ORT_DLL_DIRS").unwrap_or_default();
        let dirs: Vec<String> = raw_dirs
            .split([';', ':'])
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect();
        diag::configure_runtime(dylib.as_deref(), &dirs)
            .map_err(|err| OkfError::internal(err.to_string(), "ort runtime"))
    }

    pub fn open(settings: &Settings) -> Result<Self> {
        Self::configure_runtime()?;
        let device = DeviceReq::parse(&settings.device)
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        let precision = Precision::parse(&settings.precision)
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        let cache_dir = settings.cache_dir.as_ref().map(PathBuf::from);
        let max_length = usize::try_from(settings.max_length.unwrap_or(0))
            .ok()
            .filter(|n| *n > 0);
        let engine = JinaV5::open(
            &settings.model_id,
            None,
            cache_dir,
            settings.embedding_dim as usize,
            device,
            max_length,
            precision,
            settings.cpu_arena,
        )
        .map_err(|err| OkfError::internal(err.to_string(), "embroider::JinaV5"))?;
        let used_cuda = engine.used_cuda();
        let precision = engine.precision().as_str();
        let model_id = engine.model_id().to_string();
        let dim = engine.dim();
        let max_len = engine.max_len();
        let space = {
            let precision_enum = Precision::parse(precision).ok();
            precision_enum
                .and_then(|p| embroider::artifact_for(&model_id, p).revision)
                .map(|revision| {
                    format!(
                        "embroider/{model_id}@{revision}:{precision}:{dim}:{max_len}:text_embed:v1"
                    )
                })
        };
        Ok(Self {
            engine,
            dim,
            model_id,
            used_cuda,
            precision,
            max_len,
            space,
        })
    }

    pub fn runtime_report() -> Result<Json> {
        Self::configure_runtime()?;
        Ok(diag::runtime_report())
    }
}

#[cfg(feature = "embed")]
impl Embedder for NativeEmbedder {
    fn encode(&self, text: &str, task: Task) -> Result<Vec<f32>> {
        self.engine
            .encode_one(text, task.as_str())
            .map_err(|err| OkfError::internal(err.to_string(), "embroider::encode"))
    }

    fn encode_many(&self, texts: &[String], task: Task) -> Result<Vec<Vec<f32>>> {
        self.engine
            .encode_many(texts, task.as_str())
            .map_err(|err| OkfError::internal(err.to_string(), "embroider::encode_many"))
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.engine
            .count_tokens(text)
            .map_err(|err| OkfError::internal(err.to_string(), "embroider::count_tokens"))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    fn precision(&self) -> &'static str {
        self.precision
    }

    fn max_len(&self) -> usize {
        self.max_len
    }

    fn space(&self) -> Option<String> {
        self.space.clone()
    }
}

/// Lazy cross-encoder pool. The rerank model only opens on first use.
#[cfg(feature = "embed")]
#[derive(Clone, Default)]
pub struct RerankerPool {
    inner: Arc<Mutex<Option<Arc<embroider::Reranker>>>>,
}

#[cfg(feature = "embed")]
impl fmt::Debug for RerankerPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loaded = self.inner.lock().map(|g| g.is_some()).unwrap_or(false);
        f.debug_struct("RerankerPool")
            .field("loaded", &loaded)
            .finish()
    }
}

#[cfg(feature = "embed")]
impl RerankerPool {
    pub fn new() -> Self {
        Self::default()
    }

    fn get(&self, settings: &Settings) -> Result<Arc<embroider::Reranker>> {
        if settings.rerank_backend == "external" {
            return Err(OkfError::new(
                "EXTERNAL_UNAVAILABLE",
                "external rerank lanes are not implemented in this build",
            )
            .field("backend", Json::String("external".into()))
            .field(
                "endpoint",
                Json::String(settings.rerank_endpoint_url.clone()),
            )
            .remedy("set rerank_backend='embroider' or install a build with external lanes"));
        }
        if let Some(loaded) = self
            .inner
            .lock()
            .map_err(|_| OkfError::internal("reranker lock poisoned", "mutex"))?
            .as_ref()
        {
            return Ok(Arc::clone(loaded));
        }
        NativeEmbedder::configure_runtime()?;
        let device = DeviceReq::parse(&settings.rerank_device)
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        let precision = Precision::parse(&settings.rerank_precision)
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        let reranker = embroider::Reranker::open(
            &settings.rerank_model_id,
            None,
            settings.cache_dir.as_ref().map(PathBuf::from),
            device,
            settings
                .rerank_window
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v > 0),
            precision,
            settings.cpu_arena,
        )
        .map_err(|err| OkfError::internal(err.to_string(), "embroider::Reranker"))?;
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| OkfError::internal("reranker lock poisoned", "mutex"))?;
        Ok(Arc::clone(guard.get_or_insert_with(|| Arc::new(reranker))))
    }

    pub fn precision(&self) -> Option<&'static str> {
        self.inner
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|r| r.precision().as_str()))
    }

    pub fn score(&self, settings: &Settings, query: &str, docs: &[String]) -> Result<Vec<f64>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let reranker = self.get(settings)?;
        let pairs: Vec<(&str, &str)> = docs.iter().map(|doc| (query, doc.as_str())).collect();
        Ok(reranker
            .score_many(&pairs)
            .map_err(|err| OkfError::internal(err.to_string(), "embroider::Reranker::score_many"))?
            .into_iter()
            .map(f64::from)
            .collect())
    }
}

/// Validate the graph pins before an embedder is used against stored vectors.
/// Native counterpart of `embroider.prepare_image`: decode to RGB and
/// bicubic-resize to the slot's contract grid. The Jina resize kernel is
/// Catmull-Rom, which is Pillow's `BICUBIC` filter.
#[cfg(feature = "vision")]
pub fn prepare_image(data: &[u8], model_id: &str) -> Result<(Vec<u8>, usize, usize)> {
    let decoded = image::load_from_memory(data)
        .map_err(|err| OkfError::internal(err.to_string(), "image decode"))?;
    let rgb = decoded.to_rgb8();
    let (height, width) = (rgb.height(), rgb.width());
    let (target_h, target_w) = if model_id.starts_with("google/") {
        embroider::gemma_default_target_size(height, width)
    } else {
        embroider::vision_target_size(height, width)
    }
    .map_err(|err| OkfError::internal(err.to_string(), "vision_target_size"))?;
    let prepared = if (target_h, target_w) == (height, width) {
        rgb
    } else {
        image::imageops::resize(
            &rgb,
            target_w,
            target_h,
            image::imageops::FilterType::CatmullRom,
        )
    };
    Ok((prepared.into_raw(), target_h as usize, target_w as usize))
}

#[cfg(feature = "embed")]
pub fn default_image_model_id(text_model_id: &str) -> String {
    embroider::builtin_models()
        .iter()
        .find(|spec| spec.text_partner == Some(text_model_id))
        .map(|spec| spec.id.to_string())
        .unwrap_or_else(|| embroider::VISION_NANO_MODEL.to_string())
}

#[cfg(not(feature = "embed"))]
pub fn default_image_model_id(text_model_id: &str) -> String {
    format!("{text_model_id}-vision")
}

#[cfg(feature = "vision")]
enum VisionEngine {
    Jina(Box<JinaV5Vision>),
    Gemma(GemmaVision),
}

#[cfg(feature = "vision")]
pub struct NativeVision {
    engine: VisionEngine,
    model_id: String,
    dim: usize,
    used_cuda: bool,
    precision: &'static str,
}

#[cfg(feature = "vision")]
impl fmt::Debug for NativeVision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeVision")
            .field("model_id", &self.model_id)
            .field("dim", &self.dim)
            .field("used_cuda", &self.used_cuda)
            .field("precision", &self.precision)
            .finish()
    }
}

#[cfg(feature = "vision")]
impl NativeVision {
    pub fn open(settings: &Settings) -> Result<Self> {
        NativeEmbedder::configure_runtime()?;
        let model_id = settings
            .image_model_id
            .clone()
            .unwrap_or_else(|| default_image_model_id(&settings.model_id));
        let device = DeviceReq::parse(&settings.device)
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        let precision = Precision::parse(&settings.image_precision)
            .map_err(|err| OkfError::new("CONFIG_INVALID", err.to_string()))?;
        let cache_dir = settings.cache_dir.as_ref().map(PathBuf::from);
        let dim = settings.embedding_dim as usize;
        let (engine, used_cuda, precision) = if model_id.starts_with("google/") {
            let engine =
                GemmaVision::open(&model_id, None, cache_dir, dim, device, precision, None)
                    .map_err(|err| OkfError::internal(err.to_string(), "embroider::GemmaVision"))?;
            let used_cuda = engine.used_cuda();
            let precision = engine.precision().as_str();
            (VisionEngine::Gemma(engine), used_cuda, precision)
        } else {
            let engine =
                JinaV5Vision::open(&model_id, None, cache_dir, dim, device, precision, None)
                    .map_err(|err| {
                        OkfError::internal(err.to_string(), "embroider::JinaV5Vision")
                    })?;
            let used_cuda = engine.used_cuda();
            let precision = engine.precision().as_str();
            (VisionEngine::Jina(Box::new(engine)), used_cuda, precision)
        };
        Ok(Self {
            engine,
            model_id,
            dim,
            used_cuda,
            precision,
        })
    }

    pub fn encode_image(&self, data: &[u8]) -> Result<Vec<f32>> {
        let (rgb, height, width) = prepare_image(data, &self.model_id)?;
        let result = match &self.engine {
            VisionEngine::Jina(engine) => engine.encode_image(&rgb, height, width),
            VisionEngine::Gemma(engine) => engine.encode_image(&rgb, height, width),
        };
        result.map_err(|err| OkfError::internal(err.to_string(), "embroider::encode_image"))
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    pub fn precision(&self) -> &'static str {
        self.precision
    }
}

#[cfg(feature = "vision")]
#[derive(Clone, Default)]
pub struct VisionPool {
    inner: Arc<Mutex<Option<Arc<NativeVision>>>>,
}

#[cfg(feature = "vision")]
impl fmt::Debug for VisionPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VisionPool")
            .field("loaded", &self.is_loaded())
            .finish()
    }
}

#[cfg(feature = "vision")]
impl VisionPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, settings: &Settings) -> Result<Arc<NativeVision>> {
        if settings.image_backend == "external" {
            return Err(OkfError::new(
                "EXTERNAL_UNAVAILABLE",
                "external vision lanes are not implemented in this build",
            )
            .field("backend", Json::String("external".into()))
            .field(
                "endpoint",
                Json::String(settings.image_endpoint_url.clone()),
            )
            .remedy("set image_backend='embroider' or install a build with external lanes"));
        }
        if let Some(loaded) = self
            .inner
            .lock()
            .map_err(|_| OkfError::internal("vision lock poisoned", "mutex"))?
            .as_ref()
        {
            return Ok(Arc::clone(loaded));
        }
        let vision = Arc::new(NativeVision::open(settings)?);
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| OkfError::internal("vision lock poisoned", "mutex"))?;
        Ok(Arc::clone(guard.get_or_insert(vision)))
    }

    pub fn loaded(&self) -> Option<Arc<NativeVision>> {
        self.inner
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(Arc::clone))
    }

    pub fn is_loaded(&self) -> bool {
        self.inner
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Pools {
    #[cfg(feature = "vision")]
    pub vision: VisionPool,
}

impl Pools {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn check_pins(settings: &Settings, pins: &Json) -> Result<()> {
    let empty = pins.get("empty").and_then(Json::as_bool).unwrap_or(false);
    let pinned_model = pins
        .get("pinned")
        .and_then(|p| p.get("model_id"))
        .and_then(Json::as_str);
    if !empty {
        if let Some(model) = pinned_model {
            if model != settings.model_id {
                return Err(OkfError::new(
                    "MODEL_PIN_MISMATCH",
                    format!(
                        "graph is pinned to model '{model}', configured '{}'",
                        settings.model_id
                    ),
                )
                .field("pinned", Json::String(model.into()))
                .field("configured", Json::String(settings.model_id.clone())));
            }
        }
    }
    let pinned_dim = pins
        .get("pinned")
        .and_then(|p| p.get("embedding_dim"))
        .and_then(Json::as_i64);
    if let Some(dim) = pinned_dim {
        if dim != settings.embedding_dim && !empty {
            return Err(OkfError::new(
                "DIM_MISMATCH",
                format!(
                    "graph stores {dim}-dim vectors, configured {}",
                    settings.embedding_dim
                ),
            )
            .field("pinned", Json::from(dim))
            .field("configured", Json::from(settings.embedding_dim)));
        }
    }
    let pinned_precision = pins
        .get("pinned")
        .and_then(|p| p.get("precision"))
        .and_then(Json::as_str);
    if let (Some(pinned), Some(configured)) = (
        pinned_precision,
        (settings.precision != "auto").then_some(settings.precision.as_str()),
    ) {
        if pinned != configured && !empty {
            return Err(OkfError::new(
                "PRECISION_PIN_MISMATCH",
                format!("graph is pinned to precision '{pinned}', configured '{configured}'"),
            )
            .field("pinned", Json::String(pinned.into()))
            .field("configured", Json::String(configured.into())));
        }
    }
    Ok(())
}

/// JSON-safe model info for `okf model-info` and `lanes`.
pub fn model_info(embedder: &dyn Embedder) -> Json {
    let mut out = Map::new();
    out.insert("model_id".into(), Json::String(embedder.model_id().into()));
    out.insert("embedding_dim".into(), Json::from(embedder.dim() as i64));
    out.insert(
        "device".into(),
        Json::String(if embedder.used_cuda() { "cuda" } else { "cpu" }.into()),
    );
    out.insert(
        "precision".into(),
        Json::String(embedder.precision().into()),
    );
    Json::Object(out)
}
