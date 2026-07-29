//! End-to-end test of the extraction pipeline: parallel extraction,
//! progress reporting, and .NET-compatible folder cache reuse.
//!
//! Requires the GGML model at ../models/dinov3_vits16.bin and test images
//! in ../test (skipped otherwise).

use differ_tauri_lib::services::{cache_service, similarity_service};
use differ_tauri_lib::state::ProgressInfo;
use differ_tauri_lib::dinov3::backend::ggml::GgmlBackend;
use differ_tauri_lib::dinov3::InferenceBackend;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn test_extraction_progress_and_cache_reuse() {
    let root = workspace_root();
    let model_path = root.join("models").join("dinov3_vits16.bin");
    let test_dir = root.join("test");
    if !model_path.exists() || !test_dir.exists() {
        eprintln!("SKIP: model or test images not available");
        return;
    }

    // Fresh folder with copies of a few test images (clean cache)
    let tmp = tempfile::tempdir().unwrap();
    for name in ["1.jpg", "2.png", "3.jpg"] {
        std::fs::copy(test_dir.join(name), tmp.path().join(name)).unwrap();
    }
    let folder = tmp.path().to_str().unwrap();

    let mut backend = GgmlBackend::new();
    backend.load_model(&model_path).unwrap();

    // --- First run: all misses, extracted in parallel chunks ---
    let mut events: Vec<ProgressInfo> = Vec::new();
    let (results, hits, misses) = similarity_service::extract_features_for_folder_with_progress(
        &backend,
        folder,
        true,
        |p| events.push(p),
    )
    .unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(hits, 0);
    assert_eq!(misses, 3);
    assert!(!events.is_empty(), "progress events must be emitted");
    assert_eq!(events.last().unwrap().processed, 3);
    assert_eq!(events.last().unwrap().total, 3);
    // Features are L2-normalized full sequences
    for (_, f) in &results {
        assert_eq!(f.len(), 395136);
        let norm: f32 = f.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "features must be L2-normalized, norm={norm}");
    }

    // Cache database created with the .NET-compatible layout
    let db_path = tmp.path().join(".differ_cache.db");
    assert!(db_path.exists(), ".differ_cache.db must exist");
    let stats = cache_service::get_cache_stats(folder).unwrap();
    assert_eq!(stats.cached_count, 3);

    // LastModified must be a Windows FILETIME (like .NET ToFileTimeUtc)
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let lm: i64 = conn
            .query_row("SELECT LastModified FROM ImageFeatures LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert!(
            lm > 116_444_736_000_000_000,
            "LastModified must be FILETIME ticks, got {lm}"
        );
        let len: i64 = conn
            .query_row("SELECT DinoFeatureLength FROM ImageFeatures LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(len, 395136);
    }

    // --- Second run: everything served from cache ---
    let (results2, hits2, misses2) = similarity_service::extract_features_for_folder_with_progress(
        &backend,
        folder,
        true,
        |_| {},
    )
    .unwrap();
    assert_eq!(results2.len(), 3);
    assert_eq!(hits2, 3, "second run must be fully cache-hit");
    assert_eq!(misses2, 0);

    // Cached features identical to freshly extracted ones
    for ((e1, f1), (e2, f2)) in results.iter().zip(results2.iter()) {
        assert_eq!(e1.file_name, e2.file_name);
        assert_eq!(f1, f2, "cached features must round-trip exactly");
    }

    // --- Similarity sanity on top of the cache (1.jpg vs 2.png ~ 0.92) ---
    let source = tmp.path().join("1.jpg");
    let matches = similarity_service::search_similar_in_folder(
        &backend,
        source.to_str().unwrap(),
        folder,
        0.9,
        |_| {},
    )
    .unwrap();
    assert_eq!(matches.len(), 1, "only 2.png should match at 0.9 threshold");
    assert_eq!(matches[0].file_name, "2.png");
    assert!((matches[0].similarity - 0.9244).abs() < 0.01);
}
