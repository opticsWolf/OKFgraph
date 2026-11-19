//! M2 spike: run the native vision lane on images and print the vectors.
//! Pair the Jina text-nano graph model with its cached omni vision slot:
//!   cargo run --example vision_spike -- parity/corpus/images/*.png
use okfgraph::embed::{prepare_image, NativeVision};
use okfgraph::settings::Settings;
use serde_json::{json, Map, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        paths.push("parity/corpus/images/red-fox.png".to_string());
    }
    let settings = Settings::default()
        .with(
            "model_id",
            Value::String("jinaai/jina-embeddings-v5-text-nano-retrieval".into()),
        )?
        .with("embedding_dim", Value::from(768))?
        .with("device", Value::String("auto".into()))?
        .with("image_precision", Value::String("auto".into()))?;
    let vision = NativeVision::open(&settings)?;
    let mut cases = Map::new();
    for path in paths {
        let data = std::fs::read(&path)?;
        let (rgb, height, width) = prepare_image(&data, vision.model_id())?;
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(&rgb);
        let pixels_sha256: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        let vector = vision.encode_image(&data)?;
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(&path)
            .to_string();
        cases.insert(
            name,
            json!({
                "height": height,
                "width": width,
                "pixels_sha256": pixels_sha256,
                "vector": vector,
            }),
        );
    }
    let out = json!({
        "model_id": vision.model_id(),
        "dim": vision.dim(),
        "used_cuda": vision.used_cuda(),
        "precision": vision.precision(),
        "cases": cases,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
