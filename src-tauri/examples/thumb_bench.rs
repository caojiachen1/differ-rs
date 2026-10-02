//! Thumbnail throughput benchmark: times `get_thumbnails_batch` over a
//! folder of real photos. Run with:
//!   cargo run --release --manifest-path src-tauri/Cargo.toml --example thumb_bench -- <folder> [size]

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let folder = args
        .get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "../bench_real_base".to_string());
    let size: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(150);

    let mut paths = Vec::new();
    for entry in std::fs::read_dir(&folder).unwrap() {
        let entry = entry.unwrap();
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()).map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp")).unwrap_or(false) {
            paths.push(p.to_string_lossy().to_string());
        }
    }
    paths.sort();
    println!("{} images in {} (size {}px)", paths.len(), folder, size);

    let t = Instant::now();
    let results = differ_tauri_lib::services::image_service::get_thumbnails_batch(&paths, size, false);
    let elapsed = t.elapsed().as_secs_f64();
    let ok = results.iter().filter(|r| r.is_some()).count();
    println!(
        "{} ok / {} total in {:.2} s -> {:.0} thumbs/s (cold, includes OS page cache)",
        ok,
        paths.len(),
        elapsed,
        ok as f64 / elapsed.max(1e-9)
    );
}
