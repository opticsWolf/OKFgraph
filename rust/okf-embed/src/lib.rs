//! Jina v5 text embeddings through ONNX Runtime — Rust core, PyO3 bindings.
//!
//! Exact port of `okfgraph.components.embedding.EmbeddingEngine._encode`:
//! task prefix → tokenize (8192) → ONNX forward → **last-token pooling** → L2
//! → Matryoshka truncate → re-normalise. Any deviation here silently moves the
//! unified text/omni vector space, so the parity harness pins this against the
//! optimum/numpy path at ≤1e-5.
//!
//! Session/threading choices are stolen from EmbedAnything's `ort_jina.rs`
//! (CUDA→CPU probe, Level3, intra=phys/2, inter=1); the device fallback
//! semantics mirror bobine and the Python router: CUDA is opportunistic,
//! never fatal.

use std::path::PathBuf;
use std::sync::RwLock;

use anyhow::{anyhow, Result};
use ndarray::Array2;
use ort::ep::{ExecutionProvider, CUDA};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use pyo3::prelude::*;

/// Hard truncation limit, mirroring `tokenizer(..., max_length=8192)`.
const MAX_LENGTH: usize = 8192;
/// Native width of jina-embeddings-v5 outputs.
const NATIVE_DIM: usize = 1024;
/// Official Matryoshka levels (warning-only outside these, error outside 32..=1024).
const ALLOWED_DIMS: &[usize] = &[32, 64, 128, 256, 512, 768, 1024];

/// ort errors carry occurrences of non-Send/Sync payloads, so they cannot
/// travel in anyhow::Error directly — stringify at every boundary.
/// (ort::Error is generic over the failing stage; accept any Display.)
fn oe<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T> {
    r.map_err(|e| anyhow!("{e}"))
}

#[derive(Clone, Copy, PartialEq)]
enum DeviceReq {
    Auto,
    Cpu,
    Cuda,
}

impl DeviceReq {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(anyhow!("device must be 'auto', 'cpu' or 'cuda', got '{other}'")),
        }
    }
}

/// Probe the loaded ORT library for a usable CUDA execution provider,
/// mirroring bobine's auto-enable and the router's CPU fallback.
fn cuda_available() -> bool {
    CUDA::default().is_available().unwrap_or(false)
}

pub struct JinaV5 {
    session: RwLock<Session>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    model_id: String,
    used_cuda: bool,
    feed_token_type_ids: bool,
    output_name: String,
}

impl JinaV5 {
    pub fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
        truncate_dim: usize,
        device: DeviceReq,
    ) -> Result<Self> {
        if truncate_dim == 0 || truncate_dim > NATIVE_DIM {
            return Err(anyhow!(
                "truncate_dim must be within 1..={NATIVE_DIM}, got {truncate_dim}"
            ));
        }
        if truncate_dim < 32 {
            return Err(anyhow!("truncate_dim must be >= 32, got {truncate_dim}"));
        }
        if !ALLOWED_DIMS.contains(&truncate_dim) {
            eprintln!(
                "okf-embed: truncate_dim={truncate_dim} is not an official Matryoshka level \
                 {ALLOWED_DIMS:?}; retrieval quality may be suboptimal."
            );
        }

        // ---- model acquisition (optimum `subfolder="onnx"` layout) ----
        let (owner, name) = model_id
            .split_once('/')
            .ok_or_else(|| anyhow!("model_id must be 'owner/name', got '{model_id}'"))?;
        let client = if let Some(dir) = cache_dir {
            hf_hub::HFClientBuilder::new()
                .cache_dir(dir)
                .build()
                .map(hf_hub::HFClientSync::from_inner)
                .map_err(|e| anyhow!("{e}"))?
                .map_err(|e| anyhow!("{e}"))?
        } else {
            hf_hub::HFClientSync::new().map_err(|e| anyhow!("{e}"))?
        };
        let repo = client.model(owner.to_string(), name.to_string());
        let rev = revision.map(str::to_string);
        let onnx_path = repo
            .download_file()
            .filename("onnx/model.onnx".to_string())
            .maybe_revision(rev.clone())
            .send()
            .map_err(|e| anyhow!("onnx/model.onnx missing for '{model_id}': {e}"))?;
        // External-data sidecar must sit next to model.onnx; the hub cache
        // layout preserves that, so a plain fetch into the same dir suffices.
        if repo
            .download_file()
            .filename("onnx/model.onnx_data".to_string())
            .maybe_revision(rev)
            .send()
            .is_err()
        {
            eprintln!("okf-embed: no onnx/model.onnx_data sidecar; assuming inline weights");
        }
        let tok_path = repo
            .download_file()
            .filename("tokenizer.json".to_string())
            .maybe_revision(revision.map(str::to_string))
            .send()
            .map_err(|e| anyhow!("tokenizer.json missing for '{model_id}': {e}"))?;

        // ---- tokenizer (no padding here; single-doc encodes need none) ----
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("tokenizer.json failed to load: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("truncation setup failed: {e}"))?;

        // ---- device resolution: CUDA opportunistic, never fatal ----
        let cuda = cuda_available();
        let (want_cuda, used_cuda) = match device {
            DeviceReq::Cpu => (false, false),
            DeviceReq::Auto => (cuda, cuda),
            DeviceReq::Cuda if cuda => (true, true),
            DeviceReq::Cuda => {
                eprintln!(
                    "okf-embed: CUDA requested but no CUDA execution provider in the loaded \
                     ONNX Runtime — falling back to CPU. Install onnxruntime-gpu for acceleration."
                );
                (false, false)
            }
        };

        // ---- session (threading stolen from EmbedAnything's ort_jina.rs) ----
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let intra = std::cmp::max(1, threads / 2); // physical cores over logical
        let mut builder = oe(Session::builder())?;
        builder = oe(builder
            .with_optimization_level(GraphOptimizationLevel::Level3))?
        ;
        builder = oe(builder.with_intra_threads(intra))?;
        builder = oe(builder.with_inter_threads(1))?;
        if want_cuda {
            builder = oe(builder.with_execution_providers([CUDA::default().build()]))?;
        }
        let session = oe(builder.commit_from_file(&onnx_path))
            .map_err(|e| anyhow!("loading {}: {e:#}", onnx_path.display()))?;

        // ---- contract discovery (robust to export variants) ----
        // Jina v5's optimum export declares only input_ids + attention_mask;
        // token_type_ids is fed solely when the graph asks for it (v2-style).
        let has_input = |want: &str| session.inputs().iter().any(|o| o.name() == want);
        for need in ["input_ids", "attention_mask"] {
            if !has_input(need) {
                let have: Vec<&str> =
                    session.inputs().iter().map(|o| o.name()).collect();
                return Err(anyhow!(
                    "ONNX export lacks required input '{need}' (has {have:?})"
                ));
            }
        }
        let feed_token_type_ids = has_input("token_type_ids");
        let output_name = if session.outputs().iter().any(|o| o.name() == "last_hidden_state") {
            "last_hidden_state".to_string()
        } else {
            session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .ok_or_else(|| anyhow!("ONNX export declares no outputs"))?
        };

        Ok(Self {
            session: RwLock::new(session),
            tokenizer,
            dim: truncate_dim,
            model_id: model_id.to_string(),
            used_cuda,
            feed_token_type_ids,
            output_name,
        })
    }

    /// Exact port of `EmbeddingEngine._encode` (prefix → last-token → L2 →
    /// truncate → re-normalise).
    pub fn encode_one(&self, text: &str, task: &str) -> Result<Vec<f32>> {
        let prefixed;
        let text = if text.starts_with("Query:") || text.starts_with("Document:") {
            text
        } else {
            prefixed = format!("{task}: {text}");
            &prefixed
        };

        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> =
            enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        let t = ids.len();
        if t == 0 {
            return Err(anyhow!("tokenizer returned zero tokens"));
        }
        let mask_sum: i64 = mask.iter().sum();
        let ids = Array2::from_shape_vec((1, t), ids)
            .map_err(|e| anyhow!("input shape: {e}"))?;
        let mask_arr =
            Array2::from_shape_vec((1, t), mask).map_err(|e| anyhow!("mask shape: {e}"))?;

        let hidden = {
            let mut guard =
                self.session.write().map_err(|_| anyhow!("session lock poisoned"))?;
            let ids_t = oe(ort::value::TensorRef::from_array_view(&ids))?;
            let mask_t = oe(ort::value::TensorRef::from_array_view(&mask_arr))?;
            let outputs = if self.feed_token_type_ids {
                let zeros = Array2::<i64>::zeros((1, t));
                let zeros_t = oe(ort::value::TensorRef::from_array_view(&zeros))?;
                oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "token_type_ids" => zeros_t
                }))?
            } else {
                oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t
                }))?
            };
            oe(outputs[self.output_name.as_str()].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| anyhow!("expected [B,T,H] output: {e}"))?
        };

        // Last-token pooling: index of final real token, clamped ≥ 0.
        let last_idx = (mask_sum as usize).saturating_sub(1);
        let pooled = hidden.slice(ndarray::s![0, last_idx, ..]).to_owned();

        // L2 normalise full width, truncate, re-normalise (matches Python).
        let norm = pooled.mapv(|x| x * x).sum().sqrt();
        let normed =
            if norm > 0.0 { pooled.mapv(|x| x / norm) } else { pooled };
        let mut v: Vec<f32> = normed.iter().take(self.dim).copied().collect();
        v.resize(self.dim, 0.0);
        let n2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n2 > 0.0 {
            for x in &mut v {
                *x /= n2;
            }
        }
        Ok(v)
    }

    /// Sequential by design: padded batching wastes attention compute on
    /// variable-length documents (matches `_encode_batch`).
    pub fn encode_many(&self, texts: &[String], task: &str) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.encode_one(t, task)).collect()
    }

    /// Token count without special tokens — replaces
    /// `len(tokenizer.encode(t, add_special_tokens=False))` for the
    /// context-window guard, so the transformers dependency can go.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?
            .len())
    }
}

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

#[pyclass(name = "JinaV5")]
struct PyJinaV5 {
    inner: JinaV5,
}

#[pymethods]
impl PyJinaV5 {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None, truncate_dim=512, device="auto"))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
        truncate_dim: usize,
        device: &str,
    ) -> PyResult<Self> {
        let inner = JinaV5::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
            truncate_dim,
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (text, task="Document"))]
    fn encode(&self, py: Python<'_>, text: &str, task: &str) -> PyResult<Vec<f32>> {
        py.detach(|| {
            self.inner
                .encode_one(text, task)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    #[pyo3(signature = (texts, task="Document"))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        task: &str,
    ) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| {
            self.inner
                .encode_many(&texts, task)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim
    }

    #[getter]
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    #[getter]
    fn used_cuda(&self) -> bool {
        self.inner.used_cuda
    }

    #[getter]
    fn max_length(&self) -> usize {
        MAX_LENGTH
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner
            .count_tokens(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    }
}

#[pymodule]
fn okf_embed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJinaV5>()?;
    m.add("NATIVE_DIM", NATIVE_DIM)?;
    m.add("MAX_LENGTH", MAX_LENGTH)?;
    Ok(())
}
