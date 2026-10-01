//! End-to-end application-path benchmark: the exact code path of the
//! `extract_features` Tauri command (folder scan -> file reads -> backend
//! batch_extract -> L2 normalize -> SQLite cache write), minus IPC.
//!
//! Usage:
//!   cargo run --release -p differ-tauri --example app_bench -- [folder] [rounds]
//!
//! Default folder: test/. The first (cold) round extracts everything and
//! measures extraction throughput; a second (warm) round measures the
//! cache-hit path. Reports img/s the same way the UI will display it.

use std::time::Instant;

use differ_tauri_lib::commands::inference::resolve_model_path;
use differ_tauri_lib::dinov3::InferenceBackend;
use differ_tauri_lib::services::{cache_service, similarity_service};
use differ_tauri_lib::state::ProgressInfo;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let folder = args.get(1).cloned().unwrap_or_else(|| "test".to_string());
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    println!("=== app_bench: folder {} ===\n", folder);

    // Load the GGML backend the same way the app does
    let model_path = resolve_model_path("ggml", None).expect("model not found");
    let mut backend =
        differ_tauri_lib::dinov3::backend::ggml::GgmlBackend::new();
    backend
        .load_model(&model_path)
        .expect("failed to load model");
    let backend: Box<dyn differ_tauri_lib::dinov3::InferenceBackend> = Box::new(backend);
    let info = backend.model_info();
    println!(
        "[backend] {} (feature dim {})",
        info.name,
        info.config.output_dim()
    );

    // Diagnostics: how many images does the scanner actually see?
    let scanned = differ_tauri_lib::services::image_service::scan_folder(&folder, true)
        .expect("scan failed");
    println!("[scan] {} entries on disk", scanned.len());

    // Cold: wipe the cache so every image is a miss
    cache_service::clear_cache(&folder).expect("clear cache failed");

    for round in 0..rounds {
        let t = Instant::now();
        let mut last_print = Instant::now();
        let (entries, hits, misses) =
            similarity_service::extract_features_for_folder_with_progress(
                backend.as_ref(),
                &folder,
                true,
                |p: ProgressInfo| {
                    // Show rolling speed like the UI would
                    if last_print.elapsed().as_millis() > 500 {
                        last_print = Instant::now();
                        println!(
                            "  ... {}/{} ({:.0} img/s)",
                            p.processed, p.total, p.images_per_second
                        );
                    }
                },
            )
            .expect("extraction failed");
        let secs = t.elapsed().as_secs_f64();
        println!(
            "[round {}] {} images ({} hits / {} misses) in {:.2} s -> {:.0} img/s ({:.2} ms/img)\n",
            round,
            entries.len(),
            hits,
            misses,
            secs,
            entries.len() as f64 / secs,
            secs * 1000.0 / entries.len().max(1) as f64
        );
    }
}
