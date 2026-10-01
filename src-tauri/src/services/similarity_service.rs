//! Similarity service for orchestrating feature extraction and similarity search.
//!
//! Feature extraction mirrors the original .NET flow (scan -> per-image cache
//! check -> extract -> cache) but processes cache misses in parallel chunks:
//! image files are read and preprocessed on all CPU cores (via the backend's
//! pipelined `batch_extract`), while GPU inference stays serialized.

use rayon::prelude::*;
use std::time::Instant;

use crate::dinov3::{cosine_similarity, l2_normalize, InferenceBackend};
use crate::services::{cache_service, image_service};
use crate::state::{ImageEntry, ProgressInfo, SimilarityResult};

/// Number of cache misses processed per batch_extract call. Must stay well
/// above the backend's GPU batch size so batched forwards run at full width
/// and preprocessing has enough images to saturate all cores. Scaled by
/// input resolution to bound preprocessed-buffer memory (518px: ~3.2 MB of
/// input floats per image). Also the granularity of progress events and
/// cache write transactions.
const EXTRACT_CHUNK_SIZE: usize = 64;
const EXTRACT_CHUNK_SIZE_SMALL_INPUT: usize = 256;

fn extract_chunk_size(input_width: usize) -> usize {
    if input_width <= 256 {
        EXTRACT_CHUNK_SIZE_SMALL_INPUT
    } else {
        EXTRACT_CHUNK_SIZE
    }
}

/// Emit a progress update every N cache hits.
const HIT_PROGRESS_INTERVAL: usize = 50;

/// Rolling extraction-speed tracker: smoothed instantaneous rate between
/// progress updates, plus total elapsed seconds.
struct SpeedTracker {
    start: Instant,
    last: Instant,
    last_processed: usize,
    rate: f64,
}

impl SpeedTracker {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            last: Instant::now(),
            last_processed: 0,
            rate: 0.0,
        }
    }

    /// Feed the new processed count; returns (recent img/s, elapsed secs).
    fn update(&mut self, processed: usize) -> (f64, f64) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        if processed > self.last_processed && dt > 0.001 {
            let inst = (processed - self.last_processed) as f64 / dt;
            self.rate = if self.rate > 0.0 { self.rate * 0.6 + inst * 0.4 } else { inst };
            self.last = now;
            self.last_processed = processed;
        }
        (self.rate, now.duration_since(self.start).as_secs_f64())
    }
}

/// Feature representation stored and compared by the app.
///
/// Default is the CLS token (the first `hidden_size` floats of the
/// flattened [CLS, registers, patches] output — DINOv3's global image
/// descriptor). This shrinks per-image features ~1029x (1.58 MB -> 1.5 KB
/// at 518px), which removes SQLite persistence as an extraction
/// bottleneck and makes folder-wide similarity search ~1000x cheaper.
/// Ranking structure is preserved but the similarity SCALE differs from
/// the legacy all-token cosine (near-duplicates score high in both;
/// dissimilar images score lower with CLS), so thresholds may need
/// retuning. Set `DIFFER_FEATURE_MODE=full` to restore legacy vectors.
pub(crate) fn feature_pooling_enabled() -> bool {
    !std::env::var("DIFFER_FEATURE_MODE")
        .map(|v| v.eq_ignore_ascii_case("full"))
        .unwrap_or(false)
}

/// Expected cached-vector dimension for the active feature mode.
fn expected_feature_len(config: &crate::dinov3::config::ModelConfig) -> usize {
    if feature_pooling_enabled() {
        config.hidden_size
    } else {
        config.output_dim()
    }
}

/// Apply the feature mode to a freshly extracted all-token vector.
fn apply_feature_mode(mut features: Vec<f32>, hidden_size: usize) -> Vec<f32> {
    if feature_pooling_enabled() && features.len() > hidden_size {
        features.truncate(hidden_size);
    }
    features
}

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

    // Feature dimension of the active model configuration + feature mode:
    // cached vectors with any other dimension were produced by a different
    // configuration and must be re-extracted.
    let model_config = backend.model_info().config;
    let expected_len = expected_feature_len(&model_config);
    let hidden_size = model_config.hidden_size;

    let mut tracker = SpeedTracker::new();

    // Read side of the cache (writes go through the writer thread below)
    let cache_conn = cache_service::open_cache(folder_path)?;

    // Cache writes run on a dedicated thread: chunk vectors are handed off
    // through a bounded channel so SQLite latency never sits on the
    // extraction critical path (WAL mode lets the reads below run
    // concurrently with the writer).
    let (cache_tx, cache_rx) = std::sync::mpsc::sync_channel::<Vec<(String, Vec<f32>)>>(4);
    let writer_folder = folder_path.to_string();
    let cache_writer = std::thread::spawn(move || {
        let mut conn = match cache_service::open_cache(&writer_folder) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("Cache writer: failed to open cache: {}", e);
                return;
            }
        };
        for items in cache_rx {
            let refs: Vec<(&str, &Vec<f32>)> =
                items.iter().map(|(p, f)| (p.as_str(), f)).collect();
            if let Err(e) = cache_service::store_features_batch(&mut conn, &refs) {
                log::warn!("Cache writer: failed to store batch: {}", e);
            }
        }
    });

    let mut results: Vec<(ImageEntry, Vec<f32>)> = Vec::with_capacity(total);
    let mut misses: Vec<ImageEntry> = Vec::new();
    let mut cache_hits = 0usize;

    // Phase 1: resolve cache hits
    for entry in entries {
        match cache_service::get_features(&cache_conn, &entry.path, expected_len) {
            Ok(Some(features)) => {
                cache_hits += 1;
                results.push((entry, features));
                if cache_hits % HIT_PROGRESS_INTERVAL == 0 {
                    let (ips, elapsed) = tracker.update(results.len());
                    on_progress(ProgressInfo {
                        total,
                        processed: results.len(),
                        cache_hits,
                        cache_misses: 0,
                        images_per_second: ips,
                        elapsed_secs: elapsed,
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
    {
        let (ips, elapsed) = tracker.update(processed);
        on_progress(ProgressInfo {
            total,
            processed,
            cache_hits,
            cache_misses,
            images_per_second: ips,
            elapsed_secs: elapsed,
            message: format!(
                "Extracting DINOv3 features: {}/{} images (Cache: {} hits, {} to extract)",
                processed, total, cache_hits, misses.len()
            ),
        });
    }

    // Phase 2: extract misses in chunks.
    // File reads run 2 chunks ahead on helper threads while the backend
    // pipeline processes the current chunk (reads would otherwise leave the
    // GPU idle between chunks). Within a chunk, reads run in parallel and
    // decode+resize overlaps inference inside the backend's batch_extract.
    let read_chunk = |entries: &[ImageEntry]| -> (Vec<ImageEntry>, Vec<Result<Vec<u8>, String>>) {
        let datas: Vec<Result<Vec<u8>, String>> = entries
            .par_iter()
            .map(|entry| image_service::load_image_bytes(&entry.path))
            .collect();
        (entries.to_vec(), datas)
    };
    let chunk_size = extract_chunk_size(model_config.input_width);
    let n_chunks = misses.chunks(chunk_size).len();
    let spawn_read = |ci: usize| {
        let next: Vec<ImageEntry> = misses
            .chunks(chunk_size)
            .nth(ci)
            .unwrap()
            .to_vec();
        std::thread::spawn(move || read_chunk(&next))
    };
    // Rolling queue of in-flight reads, always kept PREFETCH_DEPTH deep
    const PREFETCH_DEPTH: usize = 2;
    let mut in_flight: std::collections::VecDeque<
        std::thread::JoinHandle<(Vec<ImageEntry>, Vec<Result<Vec<u8>, String>>)>,
    > = std::collections::VecDeque::new();
    for ci in 0..n_chunks.min(PREFETCH_DEPTH) {
        in_flight.push_back(spawn_read(ci));
    }

    for (ci, chunk) in misses.chunks(chunk_size).enumerate() {
        let (owned_entries, datas) = match in_flight.pop_front() {
            Some(handle) => handle
                .join()
                .map_err(|_| "File-read prefetch thread panicked".to_string())?,
            None => read_chunk(chunk),
        };
        if ci + PREFETCH_DEPTH < n_chunks {
            in_flight.push_back(spawn_read(ci + PREFETCH_DEPTH));
        }

        // Pair up readable images; skip unreadable ones like the .NET original
        let mut batch_entries: Vec<ImageEntry> = Vec::with_capacity(owned_entries.len());
        let mut batch_bytes: Vec<Vec<u8>> = Vec::with_capacity(owned_entries.len());
        for (entry, data) in owned_entries.into_iter().zip(datas) {
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

        // Pool to the active feature mode (CLS by default), then
        // L2-normalize (idempotent for backends that already normalize)
        let mut chunk_results: Vec<(ImageEntry, Vec<f32>)> = Vec::with_capacity(extracted.len());
        for (i, features) in extracted {
            let mut features = apply_feature_mode(features, hidden_size);
            l2_normalize(&mut features);
            chunk_results.push((batch_entries[i].clone(), features));
        }

        // Hand the chunk to the cache writer thread (off the critical path)
        let cache_items: Vec<(String, Vec<f32>)> = chunk_results
            .iter()
            .map(|(entry, features)| (entry.path.clone(), features.clone()))
            .collect();
        if cache_tx.send(cache_items).is_err() {
            log::warn!("Cache writer terminated; chunk features not persisted");
        }

        cache_misses += chunk_results.len();
        processed += chunk_results.len();
        results.extend(chunk_results);

        {
            let (ips, elapsed) = tracker.update(processed);
            on_progress(ProgressInfo {
                total,
                processed,
                cache_hits,
                cache_misses,
                images_per_second: ips,
                elapsed_secs: elapsed,
                message: format!(
                    "Extracting DINOv3 features: {}/{} images (Cache: {} hits, {} misses)",
                    processed, total, cache_hits, cache_misses
                ),
            });
        }
    }

    // Wait for the cache writer to drain its backlog before returning
    drop(cache_tx);
    if let Err(e) = cache_writer.join() {
        log::warn!("Cache writer panicked: {:?}", e);
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
            let expected_len = expected_feature_len(&backend.model_info().config);
            if let Ok(Some(features)) = cache_service::get_features(&conn, image_path, expected_len) {
                return Ok(features);
            }
        }
    }

    let bytes = image_service::load_image_bytes(image_path)?;
    let extracted = backend
        .extract_features(&bytes)
        .map_err(|e| format!("Failed to extract source features: {}", e))?;
    let mut features = apply_feature_mode(extracted, backend.model_info().config.hidden_size);
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
