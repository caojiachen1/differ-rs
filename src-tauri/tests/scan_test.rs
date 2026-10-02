//! Scan-folder performance smoke test against a large real image folder.
//! Skipped when the folder does not exist.

use differ_tauri_lib::services::image_service;
use std::time::Instant;

#[test]
fn test_scan_folder_speed() {
    let folder = "D:/ComfyUI/ComfyUI/output";
    if !std::path::Path::new(folder).exists() {
        eprintln!("SKIP: {} not available", folder);
        return;
    }

    let t = Instant::now();
    let entries = image_service::scan_folder(folder, true).unwrap();
    let elapsed = t.elapsed();

    let with_thumbs = entries.iter().filter(|e| e.thumbnail.is_some()).count();
    eprintln!(
        "Scanned {} images in {:.2?} ({:.1} img/s), {} thumbnails OK, {} failed",
        entries.len(),
        elapsed,
        entries.len() as f64 / elapsed.as_secs_f64(),
        with_thumbs,
        entries.len() - with_thumbs,
    );

    assert!(!entries.is_empty());
    // Virtually all thumbnails should now succeed (RGBA fix)
    assert!(with_thumbs as f64 >= entries.len() as f64 * 0.95);
}
