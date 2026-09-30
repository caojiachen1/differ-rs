//! ONNX backend test: compare 3 images from test/ folder.
//!
//! Run with:
//!   cargo run -p dinov3-core --features onnx --example onnx_demo
//!
//! Uses test/1, test/2, test/3 (.jpg preferred, .png fallback) and reports:
//!   similarity(1, 2), similarity(1, 3), similarity(2, 3)

use std::path::PathBuf;
use std::time::Instant;

use differ_tauri_lib::dinov3::backend::onnx::OnnxBackend;
use differ_tauri_lib::dinov3::{cosine_similarity, InferenceBackend};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
}

fn main() {
    println!("=== DINOv3 ONNX Similarity Test ===\n");

    // 1. Resolve model path:
    //    ONNX_MODEL_PATH env var > unquantized model.onnx > Q4 model (download if needed)
    let model_dir = workspace_root().join("models");
    let model_path = match std::env::var("ONNX_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let fp32 = model_dir.join("model.onnx");
            if fp32.exists() {
                fp32
            } else {
                let q4 = model_dir.join("model_q4.onnx");
                if !q4.exists() {
                    println!("Model not found, downloading...");
                    match OnnxBackend::download_model(&model_dir) {
                        Ok(p) => println!("Downloaded to: {}", p.display()),
                        Err(e) => {
                            eprintln!("ERROR: Failed to download model: {e:#}");
                            std::process::exit(1);
                        }
                    }
                }
                q4
            }
        }
    };
    println!("Model: {}", model_path.display());

    let mut backend = OnnxBackend::new();
    let t = Instant::now();
    if let Err(e) = backend.load_model(&model_path) {
        eprintln!("ERROR: Failed to load model: {e:#}");
        std::process::exit(1);
    }
    println!("Model loaded in {:.2?} (EP: {})", t.elapsed(), backend.execution_provider());

    // 2. Load 3 test images (prefer .jpg, fall back to .png)
    let test_dir = workspace_root().join("test");
    let image_bases = ["1", "2", "3"];

    let mut features: Vec<(String, Vec<f32>)> = Vec::new();

    for base in &image_bases {
        let jpg_path = test_dir.join(format!("{base}.jpg"));
        let png_path = test_dir.join(format!("{base}.png"));
        let path = if jpg_path.exists() { jpg_path } else { png_path };
        if !path.exists() {
            eprintln!("ERROR: Test image not found: {base}.jpg / {base}.png under {}", test_dir.display());
            std::process::exit(1);
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        let img_data = std::fs::read(&path).expect("Failed to read image");
        let t = Instant::now();
        match backend.extract_features(&img_data) {
            Ok(f) => {
                println!("  {} : {:.1}ms, {} features", name, t.elapsed().as_secs_f64() * 1000.0, f.len());
                features.push((name.to_string(), f));
            }
            Err(e) => {
                eprintln!("  {} : FAILED - {e:#}", name);
                std::process::exit(1);
            }
        }
    }

    // 3. Compute similarities
    let sim_1_2 = cosine_similarity(&features[0].1, &features[1].1);
    let sim_1_3 = cosine_similarity(&features[0].1, &features[2].1);
    let sim_2_3 = cosine_similarity(&features[1].1, &features[2].1);

    println!("\n--- Results ---");
    println!("  sim({}, {}) = {:.4}", features[0].0, features[1].0, sim_1_2);
    println!("  sim({}, {}) = {:.4}", features[0].0, features[2].0, sim_1_3);
    println!("  sim({}, {}) = {:.4}", features[1].0, features[2].0, sim_2_3);
    println!("\n=== Done ===");
}
