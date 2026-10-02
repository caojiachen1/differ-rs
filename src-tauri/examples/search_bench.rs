//! Search-latency benchmark: cold load vs in-memory snapshot searches.
//!
//!   cargo run --release --manifest-path src-tauri/Cargo.toml \
//!     --example search_bench -- <folder> [expected_dim]

use std::time::Instant;

use differ_tauri_lib::services::similarity_service::{
    build_folder_snapshot, load_folder_features_cache_only, search_snapshot,
};

fn main() {
    let folder = std::env::args()
        .nth(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "../bench_scan_50k".to_string());
    let expected_len: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);

    // Cold: scan + full cache table load (what the first search pays)
    let t = Instant::now();
    let loaded = load_folder_features_cache_only(&folder, true, expected_len, |_| {}).unwrap();
    let cold = t.elapsed().as_secs_f64();
    println!(
        "cold load : {:.0} ms ({} images, {} hits, {} misses)",
        cold * 1e3,
        loaded.items.len() + loaded.misses.len(),
        loaded.cache_hits,
        loaded.misses.len()
    );

    let snapshot = build_folder_snapshot(&folder, expected_len, loaded.items);

    // Warm: pure in-memory snapshot searches (what repeat searches pay)
    let sources: Vec<String> = snapshot
        .items
        .iter()
        .step_by(snapshot.items.len() / 5)
        .take(5)
        .map(|(e, _)| e.path.clone())
        .collect();
    for src in &sources {
        let t = Instant::now();
        let results = search_snapshot(&snapshot, src, 0.9).unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!(
            "snapshot  : {:>6.1} ms -> {} matches (50k-image folder)",
            dt * 1e3,
            results.len()
        );
    }
}
