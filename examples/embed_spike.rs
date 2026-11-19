//! M2 spike: open the real EmbeddingGemma 2 model and encode one text.
//! Set ORT_DYLIB_PATH to the GPU ONNX Runtime dylib; device="auto" uses CUDA
//! when the CUDA execution provider is actually available.
use okfgraph::embed::{Embedder, NativeEmbedder, Task};
use okfgraph::settings::Settings;
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "badger diet".to_string());
    let settings = Settings::default()
        .with("model_id", Value::String("google/embeddinggemma-2".into()))?
        .with("embedding_dim", Value::from(512))?
        .with("device", Value::String("auto".into()))?
        .with("precision", Value::String("auto".into()))?;
    let embedder = NativeEmbedder::open(&settings)?;
    let vector = embedder.encode(&text, Task::Query)?;
    let report = NativeEmbedder::runtime_report()?;
    let out = json!({
        "text": text,
        "model_id": embedder.model_id(),
        "dim": embedder.dim(),
        "used_cuda": embedder.used_cuda(),
        "precision": embedder.precision(),
        "vector": vector,
        "ort": report,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
