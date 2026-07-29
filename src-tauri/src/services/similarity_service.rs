//! Similarity service for orchestrating feature extraction and similarity search.
//!
//! Feature extraction mirrors the original .NET flow (scan -> per-image cache
//! check -> extract -> cache) but processes cache misses in parallel chunks:
//! image files are read and preprocessed on all CPU cores (via the backend's
//! pipelined `batch_extract`), while GPU inference stays serialized.

use rayon::prelude::*;

use crate::dinov3::{cosine_similarity, l2_normalize, InferenceBackend};
use crate::services::{cache_service, image_service};
use crate::state::{ImageEntry, ProgressInfo, SimilarityResult};

/// Number of cache misses processed per batch_extract call.
/// Also the granularity of progress events and cache write transactions.
const EXTRACT_CHUNK_SIZE: usize = 8;

/// Emit a progress update every N cache hits.
const HIT_PROGRESS_INTERVAL: usize = 50;

/// Extract features for all images in a folder, reporting progress.
///
/// Cache semantics match the .NET original: per-folder `.differ_cache.db`,
/// entries keyed by path and validated by size + mtime, misses are extracted
/// and written back. Failed images are skipped (logged), like the original.
///
/// Returns (entries with features, cache hits, cache misses).
pub fn extract_features_for_folder_with_progress(
    backend: &dyn InferenceBackend,
    folder_path: &str,
    recursive: bool,
    mut on_progress: impl FnMut(ProgressInfo),
) -> Result<(Vec<(ImageEntry, Vec<f32>)>, usize, usize), String> {
    // Scan folder (no thumbnails: not needed for feature extraction)
    let entries = image_service::scan_folder(folder_path, recursive, false)?;
    let total = entries.len();

    let mut cache_conn = cache_service::open_cache(folder_path)?;

    let mut results: Vec<(ImageEntry, Vec<f32>)> = Vec::with_capacity(total);
    let mut misses: Vec<ImageEntry> = Vec::new();
    let mut cache_hits = 0usize;

    // Phase 1: resolve cache hits
    for entry in entries {
        match cache_service::get_features(&cache_conn, &entry.path) {
            Ok(Some(features)) => {
                cache_hits += 1;
                results.push((entry, features));
                if cache_hits % HIT_PROGRESS_INTERVAL == 0 {
                    on_progress(ProgressInfo {
                        total,
                        processed: results.len(),
                        cache_hits,
                        cache_misses: 0,
                        message: format!(
                            "Loading DINOv3 features: {}/{} images (Cache: {} hits)",
                            results.len(), total, cache_hits
                        ),
                    });
                }
            }
            Ok(None) => misses.push(entry),
            Err(e) => {
                log::warn!("Cache lookup failed for {}: {}", entry.path, e);
                misses.push(entry);
            }
        }
    }

    let mut cache_misses = 0usize;
    let mut processed = results.len();
    on_progress(ProgressInfo {
        total,
        processed,
        cache_hits,
        cache_misses,
        message: format!(
            "Extracting DINOv3 features: {}/{} images (Cache: {} hits, {} to extract)",
            processed, total, cache_hits, misses.len()
        ),
    });

    // Phase 2: extract misses in chunks.
    // File reads run in parallel here; decode+resize runs in parallel inside
    // the backend's batch_extract; inference itself is serialized on the GPU.
    for chunk in misses.chunks(EXTRACT_CHUNK_SIZE) {
        let datas: Vec<Result<Vec<u8>, String>> = chunk
            .par_iter()
            .map(|entry| image_service::load_image_bytes(&entry.path))
            .collect();

        // Pair up readable images; skip unreadable ones like the .NET original
        let mut batch_entries: Vec<&ImageEntry> = Vec::with_capacity(chunk.len());
        let mut batch_bytes: Vec<Vec<u8>> = Vec::with_capacity(chunk.len());
        for (entry, data) in chunk.iter().zip(datas) {
            match data {
                Ok(bytes) => {
                    batch_entries.push(entry);
                    batch_bytes.push(bytes);
                }
                Err(e) => {
                    log::error!("Skipping unreadable image {}: {}", entry.path, e);
                    processed += 1;
                }
            }
        }

        let byte_refs: Vec<&[u8]> = batch_bytes.iter().map(|b| b.as_slice()).collect();
        let mut extracted: Vec<(usize, Vec<f32>)> = Vec::with_capacity(batch_entries.len());

        match backend.batch_extract(&byte_refs) {
            Ok(feature_sets) => {
                for (i, features) in feature_sets.into_iter().enumerate() {
                    extracted.push((i, features));
                }
            }
            Err(batch_err) => {
                // One bad image fails the whole batch: retry individually so
                // a single corrupt file doesn't abort the folder scan.
                log::warn!("Batch extraction failed ({}), retrying individually", batch_err);
                for (i, bytes) in byte_refs.iter().enumerate() {
                    match backend.extract_features(bytes) {
                        Ok(features) => extracted.push((i, features)),
                        Err(e) => {
                            log::error!(
                                "Feature extraction failed for {}: {}",
                                batch_entries[i].path, e
                            );
                            processed += 1;
                        }
                    }
                }
            }
        }

        // L2-normalize (idempotent for backends that already normalize)
        // and persist the chunk in one cache transaction.
        let mut chunk_results: Vec<(ImageEntry, Vec<f32>)> = Vec::with_capacity(extracted.len());
        for (i, mut features) in extracted {
            l2_normalize(&mut features);
            chunk_results.push((batch_entries[i].clone(), features));
        }

        let cache_items: Vec<(&str, &Vec<f32>)> = chunk_results
            .iter()
            .map(|(entry, features)| (entry.path.as_str(), features))
            .collect();
        if let Err(e) = cache_service::store_features_batch(&mut cache_conn, &cache_items) {
            log::warn!("Failed to write cache batch: {}", e);
        }

        cache_misses += chunk_results.len();
        processed += chunk_results.len();
        results.extend(chunk_results);

        on_progress(ProgressInfo {
            total,
            processed,
            cache_hits,
            cache_misses,
            message: format!(
                "Extracting DINOv3 features: {}/{} images (Cache: {} hits, {} misses)",
                processed, total, cache_hits, cache_misses
            ),
        });
    }

    // Keep deterministic file-name order like the original
    results.sort_by(|a, b| a.0.file_name.cmp(&b.0.file_name));

    Ok((results, cache_hits, cache_misses))
}

/// Extract features for all images in a folder (no progress reporting).
pub fn extract_features_for_folder(
    backend: &dyn InferenceBackend,
    folder_path: &str,
    recursive: bool,
) -> Result<Vec<(ImageEntry, Vec<f32>)>, String> {
    extract_features_for_folder_with_progress(backend, folder_path, recursive, |_| {})
        .map(|(results, _, _)| results)
}

/// Extract features for a single image, using the folder cache when possible.
fn extract_single_with_cache(
    backend: &dyn InferenceBackend,
    image_path: &str,
) -> Result<Vec<f32>, String> {
    // Try the cache of the folder containing the image
    let folder = std::path::Path::new(image_path)
        .parent()
        .map(|p| p.to_string_lossy().to_string());

    if let Some(folder) = &folder {
        if let Ok(conn) = cache_service::open_cache(folder) {
            if let Ok(Some(features)) = cache_service::get_features(&conn, image_path) {
                return Ok(features);
            }
        }
    }

    let bytes = image_service::load_image_bytes(image_path)?;
    let mut features = backend
        .extract_features(&bytes)
        .map_err(|e| format!("Failed to extract source features: {}", e))?;
    l2_normalize(&mut features);

    if let Some(folder) = &folder {
        if let Ok(conn) = cache_service::open_cache(folder) {
            let _ = cache_service::store_features(&conn, image_path, &features);
        }
    }

    Ok(features)
}

/// Search for similar images within a folder.
///
/// # Returns
/// A vector of SimilarityResult sorted by similarity descending.
pub fn search_similar_in_folder(
    backend: &dyn InferenceBackend,
    source_path: &str,
    folder_path: &str,
    threshold: f32,
    on_progress: impl FnMut(ProgressInfo),
) -> Result<Vec<SimilarityResult>, String> {
    // Source features (cache-aware)
    let source_features = extract_single_with_cache(backend, source_path)?;

    // Features for all images in the folder (cache + parallel extraction)
    let (folder_features, _, _) =
        extract_features_for_folder_with_progress(backend, folder_path, true, on_progress)?;

    let mut results: Vec<SimilarityResult> = folder_features
        .into_iter()
        .filter(|(entry, _)| entry.path != source_path)
        .filter_map(|(entry, features)| {
            if features.len() != source_features.len() {
                log::warn!(
                    "Skipping {}: cached feature length {} != source {}",
                    entry.path, features.len(), source_features.len()
                );
                return None;
            }
            let similarity = cosine_similarity(&source_features, &features);
            (similarity >= threshold).then(|| SimilarityResult {
                path: entry.path,
                file_name: entry.file_name,
                similarity,
                thumbnail: entry.thumbnail,
            })
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    attach_thumbnails(&mut results);
    Ok(results)
}

/// Fill in thumbnails for search results (in parallel).
///
/// Feature extraction scans without thumbnails for speed, so results only
/// need them generated here, for the (few) matching images.
fn attach_thumbnails(results: &mut [SimilarityResult]) {
    results
        .par_iter_mut()
        .filter(|r| r.thumbnail.is_none())
        .for_each(|r| {
            r.thumbnail = image_service::thumbnail_base64(&r.path, 150);
        });
}

/// Compare images between two folders.
///
/// # Returns
/// A vector of SimilarityResult containing matches from the target folder.
pub fn compare_folders(
    backend: &dyn InferenceBackend,
    source_folder: &str,
    target_folder: &str,
    threshold: f32,
    mut on_progress: impl FnMut(ProgressInfo),
) -> Result<Vec<SimilarityResult>, String> {
    let (source_features, _, _) = extract_features_for_folder_with_progress(
        backend, source_folder, true, &mut on_progress,
    )?;
    let (target_features, _, _) = extract_features_for_folder_with_progress(
        backend, target_folder, true, &mut on_progress,
    )?;

    // Pairwise comparison parallelized over target images
    let mut results: Vec<SimilarityResult> = target_features
        .par_iter()
        .filter_map(|(target_entry, target_vec)| {
            let best = source_features
                .iter()
                .filter(|(_, source_vec)| source_vec.len() == target_vec.len())
                .map(|(_, source_vec)| cosine_similarity(source_vec, target_vec))
                .fold(f32::MIN, f32::max);
            (best >= threshold).then(|| SimilarityResult {
                path: target_entry.path.clone(),
                file_name: target_entry.file_name.clone(),
                similarity: best,
                thumbnail: target_entry.thumbnail.clone(),
            })
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    attach_thumbnails(&mut results);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_calculation() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }
}
