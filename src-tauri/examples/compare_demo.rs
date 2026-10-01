//! Cross-backend comparison: ONNX (CUDA) vs GGML (CUDA) vs Candle (CUDA).
//!
//! Runs all images in test/ through every enabled backend and reports:
//!   - per-image extraction time for each backend
//!   - cross-backend feature agreement (cosine between backends on the same image)
//!   - pairwise image similarities per backend and their spread across backends
//!   - batch (batch_extract) throughput per backend
//!
//! Backends are selected by cargo features (onnx / ggml / candle), so this
//! example compiles and runs with whatever subset is enabled.
//!
//! Run with:
//!   cargo run --release -p differ-tauri --features "candle-cuda" --example compare_demo

use std::path::PathBuf;
use std::time::Instant;

use differ_tauri_lib::dinov3::{cosine_similarity, l2_normalize, InferenceBackend};

#[cfg(feature = "onnx")]
use differ_tauri_lib::dinov3::backend::onnx::OnnxBackend;
#[cfg(feature = "ggml")]
use differ_tauri_lib::dinov3::backend::ggml::GgmlBackend;
#[cfg(feature = "candle")]
use differ_tauri_lib::dinov3::backend::candle::CandleBackend;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn main() {
    println!("=== DINOv3 Multi-Backend Comparison ===\n");
    // Cross-backend comparison is only meaningful when every backend runs
    // the same input resolution. The GGML backend defaults to the fast
    // throughput tier (256x256), so pin it to the accuracy tier here.
    std::env::set_var("GGML_VIT_TIER", "high");
    println!("Note: GGML_VIT_TIER=high forced so all backends compare at 518x518.\n");
    let root = workspace_root();

    // Collect test images (sorted by name)
    let test_dir = root.join("test");
    let mut images: Vec<PathBuf> = std::fs::read_dir(&test_dir)
        .expect("cannot read test dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
                Some(ref e) if e == "jpg" || e == "jpeg" || e == "png"
            )
        })
        .collect();
    images.sort();
    println!("Found {} test images", images.len());

    // Assemble the enabled backends into a uniform list.
    let mut names: Vec<String> = Vec::new();
    let mut backends: Vec<Box<dyn InferenceBackend>> = Vec::new();

    #[cfg(feature = "onnx")]
    {
        let mut b = OnnxBackend::new();
        let m = std::env::var("ONNX_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("models").join("model.onnx"));
        let t = Instant::now();
        b.load_model(&m).expect("failed to load ONNX model");
        println!("ONNX   loaded in {:.2?} (EP: {})", t.elapsed(), b.execution_provider());
        names.push("onnx".to_string());
        backends.push(Box::new(b));
    }
    #[cfg(feature = "ggml")]
    {
        let mut b = GgmlBackend::new();
        let m = std::env::var("GGML_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("models").join("dinov3_vits16.bin"));
        let t = Instant::now();
        b.load_model(&m).expect("failed to load GGML model");
        println!("GGML   loaded in {:.2?}", t.elapsed());
        names.push("ggml".to_string());
        backends.push(Box::new(b));
    }
    #[cfg(feature = "candle")]
    {
        let mut b = CandleBackend::new();
        let m = std::env::var("CANDLE_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("models").join("dinov3_vits16.safetensors"));
        let t = Instant::now();
        match b.load_model(&m) {
            Ok(()) => {
                println!("Candle loaded in {:.2?}", t.elapsed());
                names.push("candle".to_string());
                backends.push(Box::new(b));
            }
            Err(e) => println!(
                "Candle SKIPPED ({}: {e:#}); place a safetensors model to include it",
                m.display()
            ),
        }
    }

    let nb = backends.len();
    assert!(nb > 0, "no backend enabled; build with a feature like candle-cuda/onnx/ggml-cuda");
    println!();

    // Per-image extraction across all backends (features stored L2-normalized).
    struct Entry {
        name: String,
        times_ms: Vec<f64>,   // per backend
        feats: Vec<Vec<f32>>, // per backend, normalized
    }
    let mut entries: Vec<Entry> = Vec::new();

    print!("{:<8}", "image");
    for n in &names {
        print!(" {:>13}", format!("{} (ms)", n));
    }
    println!();

    for path in &images {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let data = std::fs::read(path).expect("failed to read image");

        let mut times = Vec::with_capacity(nb);
        let mut feats = Vec::with_capacity(nb);
        for b in &backends {
            let t = Instant::now();
            let mut f = b.extract_features(&data).expect("inference failed");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            l2_normalize(&mut f);
            feats.push(f);
        }

        print!("{:<8}", name);
        for tms in &times {
            print!(" {:>13.1}", tms);
        }
        println!();

        entries.push(Entry { name, times_ms: times, feats });
    }

    // Timing summary (skip first image as warmup)
    let skip = if entries.len() > 1 { 1 } else { 0 };
    let denom = (entries.len() - skip).max(1) as f64;
    println!("\n--- Speed (avg, excl. warmup) ---");
    for (bi, name) in names.iter().enumerate() {
        let avg: f64 = entries[skip..].iter().map(|e| e.times_ms[bi]).sum::<f64>() / denom;
        println!("  {:<7}: {:.1} ms/image", name, avg);
    }

    // Cross-backend feature agreement per image (all backend pairs).
    if nb > 1 {
        println!("\n--- Cross-backend feature agreement (cosine per image) ---");
        print!("{:<8}", "image");
        for bi in 0..nb {
            for bj in (bi + 1)..nb {
                print!(" {:>16}", format!("{}~{}", names[bi], names[bj]));
            }
        }
        println!();
        for e in &entries {
            print!("{:<8}", e.name);
            for bi in 0..nb {
                for bj in (bi + 1)..nb {
                    print!(" {:>16.5}", cosine_similarity(&e.feats[bi], &e.feats[bj]));
                }
            }
            println!();
        }
    }

    // Pairwise image similarities per backend, plus spread across backends.
    println!("\n--- Pairwise image similarities (per backend) ---");
    print!("{:<16}", "pair");
    for n in &names {
        print!(" {:>10}", n);
    }
    if nb > 1 {
        print!(" {:>10}", "spread");
    }
    println!();

    let mut global_max_spread = 0.0f32;
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let sims: Vec<f32> = (0..nb)
                .map(|bi| cosine_similarity(&entries[i].feats[bi], &entries[j].feats[bi]))
                .collect();
            print!("{:<16}", format!("{} vs {}", entries[i].name, entries[j].name));
            for s in &sims {
                print!(" {:>10.4}", s);
            }
            if nb > 1 {
                let mn = sims.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let spread = mx - mn;
                global_max_spread = global_max_spread.max(spread);
                print!(" {:>10.4}", spread);
            }
            println!();
        }
    }
    if nb > 1 {
        println!("\n  max sim spread across backends = {:.4}", global_max_spread);
    }

    // Batch (pipelined) throughput per backend.
    println!("\n--- Batch throughput (batch_extract, {} images) ---", images.len());
    let datas: Vec<Vec<u8>> = images
        .iter()
        .map(|p| std::fs::read(p).expect("failed to read image"))
        .collect();
    let refs: Vec<&[u8]> = datas.iter().map(|d| d.as_slice()).collect();

    for (bi, name) in names.iter().enumerate() {
        let t = Instant::now();
        let batch = backends[bi].batch_extract(&refs).expect("batch failed");
        let batch_ms = t.elapsed().as_secs_f64() * 1000.0;
        let seq_total: f64 = entries.iter().map(|e| e.times_ms[bi]).sum();

        // Sanity: batch results must match single-image results.
        let mut max_dev = 0.0f32;
        for (i, e) in entries.iter().enumerate() {
            let mut bf = batch[i].clone();
            l2_normalize(&mut bf);
            max_dev = max_dev.max(1.0 - cosine_similarity(&bf, &e.feats[bi]));
        }

        println!(
            "  {:<7}: sequential {:>7.1} ms -> batch {:>7.1} ms  ({:.1} img/s, {:.2}x)  batch-dev {:.6}",
            name,
            seq_total,
            batch_ms,
            images.len() as f64 / (batch_ms / 1000.0),
            seq_total / batch_ms,
            max_dev
        );
    }

    println!("\n=== Done ===");
}
