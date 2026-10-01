//! Folder-scan benchmark: metadata enumeration speed over a large tree.
//!
//! Usage: cargo run --release -p differ-tauri --example scan_bench -- [folder] [rounds]

use std::time::Instant;

fn main() {
    let folder = std::env::args().nth(1).unwrap_or_else(|| "test".to_string());
    let rounds: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    for r in 0..rounds {
        let t = Instant::now();
        let mut last_emit = Instant::now();
        let entries = differ_tauri_lib::services::image_service::scan_folder_with_progress(
            &folder,
            true,
            |p| {
                if last_emit.elapsed().as_millis() > 500 {
                    last_emit = Instant::now();
                    let rate = p.files_found as f64 / p.elapsed_secs.max(1e-9);
                    println!("  ... {} files, {:.0} files/s", p.files_found, rate);
                }
            },
        )
        .expect("scan failed");
        let secs = t.elapsed().as_secs_f64();
        println!(
            "[round {r}] {} images in {:.2} s -> {:.0} files/s",
            entries.len(),
            secs,
            entries.len() as f64 / secs
        );
    }
}
