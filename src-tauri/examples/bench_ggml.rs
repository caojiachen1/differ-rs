//! GGML performance benchmark: pure inference latency + batch throughput.
//!
//! Run with:
//!   cargo run --release -p dinov3-core --features ggml-cuda --example bench_ggml [N_IMAGES]
//!
//! Reports:
//!   - pure infer latency (preprocessing excluded, same input reused)
//!   - end-to-end batch_extract throughput over N images (test/ images repeated)

use std::path::PathBuf;
use std::time::Instant;

use differ_tauri_lib::dinov3::backend::ggml::GgmlBackend;
use differ_tauri_lib::dinov3::InferenceBackend;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn main() {
    let n_images: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    println!("=== GGML benchmark (target: {} images) ===\n", n_images);
    let root = workspace_root();

    let model_path = std::env::var("GGML_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("models").join("dinov3_vits16.bin"));
    let mut backend = GgmlBackend::new();
    let t = Instant::now();
    backend.load_model(&model_path).expect("failed to load model");
    println!("Model loaded in {:.2?}", t.elapsed());

    // Collect test images
    let test_dir = root.join("test");
    let mut images: Vec<Vec<u8>> = std::fs::read_dir(&test_dir)
        .expect("cannot read test dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
                Some(ref e) if e == "jpg" || e == "jpeg" || e == "png"
            )
        })
        .map(|p| std::fs::read(&p).expect("read image"))
        .collect();
    assert!(!images.is_empty(), "no test images found");
    images.sort_by_key(|d| d.len());

    // --- 1. Pure inference latency: run one image repeatedly ---
    // First call warms up (CUDA graph capture etc.)
    let warmup = backend.extract_features(&images[0]).expect("warmup failed");
    println!("Warmup OK ({} features)", warmup.len());

    const LAT_ITERS: usize = 30;
    let t = Instant::now();
    for _ in 0..LAT_ITERS {
        backend.extract_features(&images[0]).expect("inference failed");
    }
    let per_img_e2e = t.elapsed().as_secs_f64() * 1000.0 / LAT_ITERS as f64;
    println!("\n--- Single-image extract (incl. decode+resize), {} iters ---", LAT_ITERS);
    println!("  {:.2} ms/image", per_img_e2e);

    // --- 2. Pure GPU inference (preprocessing excluded) ---
    {
        use dinov3_ggml::{GgmlVitModel, VitConfig, preprocess_image};
        let cfg = VitConfig::vit_small_16();
        let mut model = GgmlVitModel::new(cfg.clone()).expect("create model");
        model.load_weights(&model_path).expect("load weights");
        let input = preprocess_image(&images[0], &cfg).expect("preprocess");
        let h = cfg.input_height as i32;
        let w = cfg.input_width as i32;

        // single
        model.infer(&input, h, w).expect("warmup");
        let t = Instant::now();
        for _ in 0..LAT_ITERS {
            model.infer(&input, h, w).expect("infer");
        }
        println!("\n--- Pure GPU single infer, {} iters ---", LAT_ITERS);
        println!("  {:.2} ms/image", t.elapsed().as_secs_f64() * 1000.0 / LAT_ITERS as f64);

        // batched
        let b = model.max_batch();
        let mut flat = Vec::with_capacity(b * input.len());
        for _ in 0..b {
            flat.extend_from_slice(&input);
        }
        model.infer_batch(&flat, b).expect("batch warmup");
        let iters = 10;
        let t = Instant::now();
        for _ in 0..iters {
            model.infer_batch(&flat, b).expect("batch infer");
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / (iters * b) as f64;
        println!("\n--- Pure GPU batched infer (B={}), {} iters ---", b, iters);
        println!("  {:.2} ms/image ({:.1} img/s)", ms, 1000.0 / ms);
    }

    // --- 3. Batch throughput over n_images (repeat test set) ---
    let datas: Vec<&[u8]> = (0..n_images).map(|i| images[i % images.len()].as_slice()).collect();
    let t = Instant::now();
    let feats = backend.batch_extract(&datas).expect("batch failed");
    let secs = t.elapsed().as_secs_f64();
    println!("\n--- batch_extract, {} images ---", feats.len());
    println!("  total {:.2} s -> {:.1} img/s ({:.2} ms/img)",
        secs, n_images as f64 / secs, secs * 1000.0 / n_images as f64);
}
