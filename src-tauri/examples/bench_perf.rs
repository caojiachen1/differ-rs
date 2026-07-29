//! Unified backend performance benchmark (same metrics for ONNX and GGML).
//!
//! Usage:
//!   cargo run --release -p dinov3-core --features "ggml-cuda,onnx" --example bench_perf -- <onnx|ggml> [N_IMAGES]
//!
//! Reports:
//!   - model load time
//!   - single-image extract latency (avg / min / p95 over 30 iters, incl. decode+resize)
//!   - sequential throughput over the test set
//!   - batch_extract throughput over N images (test/ images repeated)

use std::path::PathBuf;
use std::time::Instant;

use differ_tauri_lib::dinov3::backend::ggml::GgmlBackend;
use differ_tauri_lib::dinov3::backend::onnx::OnnxBackend;
use differ_tauri_lib::dinov3::InferenceBackend;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn main() {
    let backend_name = std::env::args().nth(1).unwrap_or_default().to_ascii_lowercase();
    let n_images: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    if backend_name != "onnx" && backend_name != "ggml" {
        eprintln!("usage: bench_perf <onnx|ggml> [N_IMAGES]");
        std::process::exit(2);
    }
    let root = workspace_root();

    println!("=== bench_perf: backend={} target={} images ===\n", backend_name, n_images);

    // Load backend
    let t = Instant::now();
    let backend: Box<dyn InferenceBackend> = if backend_name == "onnx" {
        let mut b = OnnxBackend::new();
        let model = std::env::var("ONNX_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("models").join("model.onnx"));
        b.load_model(&model).expect("failed to load ONNX model");
        println!("[load] ONNX model loaded in {:.2?} (EP: {})", t.elapsed(), b.execution_provider());
        Box::new(b)
    } else {
        let mut b = GgmlBackend::new();
        let model = std::env::var("GGML_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("models").join("dinov3_vits16.bin"));
        b.load_model(&model).expect("failed to load GGML model");
        println!("[load] GGML model loaded in {:.2?}", t.elapsed());
        Box::new(b)
    };

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
    assert!(!images.is_empty(), "no test images found in test/");
    images.sort_by_key(|d| d.len());
    println!("[data] {} test images", images.len());

    // Warmup (graph capture, cuDNN autotune, etc.)
    for _ in 0..3 {
        backend.extract_features(&images[0]).expect("warmup failed");
    }

    // 1. Single-image latency, 30 iters on the same image
    const ITERS: usize = 30;
    let mut lat = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        backend.extract_features(&images[0]).expect("inference failed");
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg: f64 = lat.iter().sum::<f64>() / ITERS as f64;
    let p95 = lat[(ITERS as f64 * 0.95) as usize];
    println!("\n[latency] single extract, {} iters (incl. decode+resize)", ITERS);
    println!("  avg {:.2} ms | min {:.2} ms | p95 {:.2} ms", avg, lat[0], p95);

    // 2. Sequential throughput over the whole test set
    let t = Instant::now();
    for img in &images {
        backend.extract_features(img).expect("inference failed");
    }
    let secs = t.elapsed().as_secs_f64();
    println!("\n[sequential] {} images: {:.2} s -> {:.1} img/s",
        images.len(), secs, images.len() as f64 / secs);

    // 3. batch_extract throughput over n_images (repeat test set)
    let datas: Vec<&[u8]> = (0..n_images).map(|i| images[i % images.len()].as_slice()).collect();
    let t = Instant::now();
    let feats = backend.batch_extract(&datas).expect("batch failed");
    let secs = t.elapsed().as_secs_f64();
    println!("\n[batch] batch_extract, {} images", feats.len());
    println!("  total {:.2} s -> {:.1} img/s ({:.2} ms/img)",
        secs, n_images as f64 / secs, secs * 1000.0 / n_images as f64);

    println!("\n=== bench_perf {} done ===", backend_name);
}
