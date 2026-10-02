//! Similarity service for orchestrating feature extraction and similarity search.
//!
//! Feature extraction mirrors the original .NET flow (scan -> per-image cache
//! check -> extract -> cache) but processes cache misses in parallel chunks:
//! image files are read and preprocessed on all CPU cores (via the backend's
//! pipelined `batch_extract`), while GPU inference stays serialized.
//!
//! Cache resolution reads the whole per-folder cache DB in one table scan and
//! validates file identities in parallel, so a fully-cached folder loads in
//! well under a second. `compare` additionally runs its pairwise best-match
//! step through blocked GEMM kernels (the `gemm` crate): plain Rust float-sum
//! loops don't auto-vectorize without fast-math, which made large-folder
//! compares orders of magnitude slower. Cache-only entry points
//! (`compare_folders_cache_only`, `search_similar_cache_only`) never touch a
//! backend, letting the CLI skip model load entirely when everything is
//! cached.

use rayon::prelude::*;
use std::time::Instant;

use gemm::Parallelism;

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
    if input_width <= 224 {
        512
    } else if input_width <= 256 {
        EXTRACT_CHUNK_SIZE_SMALL_INPUT
    } else {
        EXTRACT_CHUNK_SIZE
    }
}

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

/// Backend-facing variant of [`expected_feature_len`], used by command-layer
/// snapshot validation without reaching into model internals.
pub fn expected_feature_len_for_backend(backend: &dyn InferenceBackend) -> usize {
    expected_feature_len(&backend.model_info().config)
}

/// Apply the feature mode to a freshly extracted all-token vector.
fn apply_feature_mode(mut features: Vec<f32>, hidden_size: usize) -> Vec<f32> {
    if feature_pooling_enabled() && features.len() > hidden_size {
        features.truncate(hidden_size);
    }
    features
}

/// A folder's features resolved against its per-folder cache.
pub struct FolderFeatures {
    /// Images with features in hand (cache hits, plus extracted misses when
    /// a backend was available to resolve them).
    pub items: Vec<(ImageEntry, Vec<f32>)>,
    /// Images that need extraction; empty when a backend resolved them.
    pub misses: Vec<ImageEntry>,
    /// Number of images served from the folder cache.
    pub cache_hits: usize,
}

/// Outcome of a cache-only operation that may still need the model.
#[derive(Debug)]
pub enum CacheOnlyOutcome<T> {
    /// Everything was served from cache; the model was never touched.
    Ready(T),
    /// `misses` images require extraction, so a loaded model is required.
    NeedsModel { misses: usize },
}

/// Load a folder's features from its cache without any model or backend.
///
/// Scans the folder, streams the whole cache table in one query, then
/// validates every file's identity (size + mtime) in parallel. Emits
/// progress for the resolution phase.
pub fn load_folder_features_cache_only(
    folder_path: &str,
    recursive: bool,
    expected_len: usize,
    on_progress: impl FnMut(ProgressInfo),
) -> Result<FolderFeatures, String> {
    let entries = image_service::scan_folder(folder_path, recursive)?;
    resolve_cache_only(folder_path, entries, expected_len, on_progress)
}

/// Features for `path` from a preloaded cache map, if the file on disk still
/// matches the identity the cache was written with (size + mtime).
fn cached_features_if_valid(
    map: &std::collections::HashMap<String, cache_service::CachedFeature>,
    path: &str,
) -> Option<Vec<f32>> {
    let (file_size, last_modified) = match cache_service::file_identity(path) {
        Ok(id) => id,
        Err(e) => {
            log::warn!("Cache lookup failed for {}: {}", path, e);
            return None;
        }
    };
    let cached = map.get(path)?;
    (cached.file_size == file_size && cached.last_modified == last_modified)
        .then(|| cached.features.clone())
}

/// Split scanned entries into (hits with features, misses, hit count) using a
/// preloaded cache map; file identities are re-checked in parallel. rayon's
/// collect preserves input order, keeping the result deterministic.
fn split_entries_by_cache(
    entries: Vec<ImageEntry>,
    map: &std::collections::HashMap<String, cache_service::CachedFeature>,
) -> (Vec<(ImageEntry, Vec<f32>)>, Vec<ImageEntry>, usize) {
    let resolved: Vec<Option<Vec<f32>>> = entries
        .par_iter()
        .map(|entry| cached_features_if_valid(map, &entry.path))
        .collect();

    let mut items = Vec::with_capacity(entries.len());
    let mut misses = Vec::new();
    let mut cache_hits = 0usize;
    for (entry, features) in entries.into_iter().zip(resolved) {
        match features {
            Some(f) => {
                cache_hits += 1;
                items.push((entry, f));
            }
            None => misses.push(entry),
        }
    }
    (items, misses, cache_hits)
}

/// Resolve a scanned folder's images against its cache (no backend needed).
///
/// One table scan + parallel stat of every file replaces the old serial
/// per-image stat + SQLite point-query loop.
fn resolve_cache_only(
    folder_path: &str,
    entries: Vec<ImageEntry>,
    expected_len: usize,
    mut on_progress: impl FnMut(ProgressInfo),
) -> Result<FolderFeatures, String> {
    let total = entries.len();
    let progress = |processed: usize, hits: usize, message: String| ProgressInfo {
        total,
        processed,
        cache_hits: hits,
        cache_misses: 0,
        images_per_second: 0.0,
        elapsed_secs: 0.0,
        message,
    };

    if total == 0 {
        return Ok(FolderFeatures { items: Vec::new(), misses: Vec::new(), cache_hits: 0 });
    }

    on_progress(progress(
        0,
        0,
        format!("Loading DINOv3 features: 0/{} images", total),
    ));

    let map = {
        let conn = cache_service::open_cache(folder_path)?;
        cache_service::load_all_features(&conn, expected_len)?
    };
    let (mut items, misses, cache_hits) = split_entries_by_cache(entries, &map);

    on_progress(progress(
        cache_hits,
        cache_hits,
        format!(
            "Loaded DINOv3 features: {}/{} images from cache ({} to extract)",
            cache_hits,
            total,
            misses.len()
        ),
    ));

    items.sort_by(|a, b| a.0.file_name.cmp(&b.0.file_name));
    Ok(FolderFeatures { items, misses, cache_hits })
}

/// Extract features for the given cache misses with the backend pipeline.
///
/// Reads files 2 chunks ahead on helper threads while the backend pipeline
/// processes the current chunk; cache writes run on a dedicated thread so
/// SQLite latency never sits on the extraction critical path.
fn extract_misses_with_backend(
    backend: &dyn InferenceBackend,
    misses: Vec<ImageEntry>,
    cache_folder: &str,
    total: usize,
    cache_hits: usize,
    on_progress: &mut dyn FnMut(ProgressInfo),
) -> Result<Vec<(ImageEntry, Vec<f32>)>, String> {
    let model_config = backend.model_info().config;
    let mut tracker = SpeedTracker::new();
    let mut processed = cache_hits;
    let mut cache_misses = 0usize;

    // Cache writes run on a dedicated thread: chunk vectors are handed off
    // through a bounded channel (WAL mode lets the reads below run
    // concurrently with the writer).
    let (cache_tx, cache_rx) = std::sync::mpsc::sync_channel::<Vec<(String, Vec<f32>)>>(4);
    let writer_folder = cache_folder.to_string();
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

    let mut results: Vec<(ImageEntry, Vec<f32>)> = Vec::with_capacity(misses.len());

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
            let mut features = apply_feature_mode(features, model_config.hidden_size);
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

    Ok(results)
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
    let entries = image_service::scan_folder(folder_path, recursive)?;
    let total = entries.len();
    let expected_len = expected_feature_len(&backend.model_info().config);

    let mut resolved = resolve_cache_only(folder_path, entries, expected_len, &mut on_progress)?;
    let cache_hits = resolved.cache_hits;
    let mut cache_misses = 0usize;

    if !resolved.misses.is_empty() {
        let extracted = extract_misses_with_backend(
            backend,
            std::mem::take(&mut resolved.misses),
            folder_path,
            total,
            cache_hits,
            &mut on_progress,
        )?;
        cache_misses = extracted.len();
        resolved.items.extend(extracted);
    }

    // Keep deterministic file-name order like the original
    resolved.items.sort_by(|a, b| a.0.file_name.cmp(&b.0.file_name));

    Ok((resolved.items, cache_hits, cache_misses))
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

/// Extract one image and persist it into the given folder's cache.
///
/// The search flow stores the query image's features in the *search root*
/// cache (recursive scans key every nested path there), so a repeated search
/// with the same source hits the cache instead of re-running the GPU.
fn extract_and_store_single(
    backend: &dyn InferenceBackend,
    image_path: &str,
    conn: &rusqlite::Connection,
) -> Result<Vec<f32>, String> {
    let bytes = image_service::load_image_bytes(image_path)?;
    let extracted = backend
        .extract_features(&bytes)
        .map_err(|e| format!("Failed to extract source features: {}", e))?;
    let mut features = apply_feature_mode(extracted, backend.model_info().config.hidden_size);
    l2_normalize(&mut features);
    let _ = cache_service::store_features(conn, image_path, &features);
    Ok(features)
}

/// For every target vector, the best cosine similarity against any source.
///
/// Inputs must be L2-normalized rows of equal dimension (dot == cosine).
/// Uses the gemm crate's hand-written SIMD kernels through double blocking:
/// output tiles stay L2-resident while each tile is big enough for the
/// threaded kernels to pay off. Falls back to a parallel scalar loop when
/// dimensions don't line up (should not happen; guarded anyway).
fn pairwise_best_similarity(sources: &[&[f32]], targets: &[&[f32]]) -> Vec<f32> {
    let mut best = vec![f32::MIN; targets.len()];
    if sources.is_empty() || targets.is_empty() {
        return best;
    }
    let dim = sources[0].len();
    if dim == 0 || sources.iter().any(|v| v.len() != dim) || targets.iter().any(|v| v.len() != dim) {
        // Mixed dimensions: compare each pair the slow but correct way
        return targets
            .par_iter()
            .map(|t| {
                sources
                    .iter()
                    .filter(|s| s.len() == t.len())
                    .map(|s| cosine_similarity(s, t))
                    .fold(f32::MIN, f32::max)
            })
            .collect();
    }

    // Pack rows contiguously for the GEMM kernels.
    let (s, t) = (sources.len(), targets.len());
    let mut a: Vec<f32> = Vec::with_capacity(s * dim);
    for v in sources {
        a.extend_from_slice(v);
    }
    let mut b: Vec<f32> = Vec::with_capacity(t * dim);
    for v in targets {
        b.extend_from_slice(v);
    }

    // Tile sizes: a tile output (512×2048 f32 = 4 MB) stays L2-resident while
    // 512·2048·384·2 ≈ 0.8 GFLOP gives the threaded kernels enough work.
    const TB: usize = 512; // targets per tile (rows of C)
    const SB: usize = 2048; // sources per tile (cols of C)

    for tb in (0..t).step_by(TB) {
        let tn = (t - tb).min(TB);
        for sb in (0..s).step_by(SB) {
            let sn = (s - sb).min(SB);
            // C (tn×sn, row-major) = B_block × A_blockᵀ, i.e. C[j][i] is the
            // dot product of target (tb+j) with source (sb+i). Row-major
            // output keeps the row-max reduction below cache-friendly.
            // Stride convention: element (i,j) lives at base + i*rs + j*cs.
            let mut c = vec![0f32; tn * sn];
            unsafe {
                gemm::gemm(
                    tn, sn, dim,
                    c.as_mut_ptr(), 1, sn as isize,
                    false,
                    b.as_ptr().add(tb * dim), 1, dim as isize,
                    a.as_ptr().add(sb * dim), dim as isize, 1,
                    // NB gemm's alpha/beta are BLAS-reversed (dst = beta·A·B +
                    // alpha·dst), so the product multiplier goes in beta's slot.
                    0.0, 1.0,
                    false, false, false,
                    Parallelism::Rayon(0), // 0 = let the rayon pool decide
                );
            }
            for (j, row) in c.chunks_exact(sn).enumerate() {
                let row_best = row.iter().copied().fold(f32::MIN, f32::max);
                let idx = tb + j;
                if row_best > best[idx] {
                    best[idx] = row_best;
                }
            }
        }
    }
    best
}

/// Compare two resolved feature sets and build the sorted result list.
fn finish_compare(
    source_features: &mut [(ImageEntry, Vec<f32>)],
    target_features: &mut [(ImageEntry, Vec<f32>)],
    threshold: f32,
) -> Vec<SimilarityResult> {
    // All features share the expected dimension by construction; use the
    // first source's length as reference and drop stragglers defensively.
    let Some(dim) = source_features.first().map(|(_, f)| f.len()) else {
        return Vec::new();
    };

    // Dot == cosine only for unit vectors; every writer normalizes, but
    // caches produced by other tools may not be. Idempotent and cheap.
    for (_, f) in source_features.iter_mut().chain(target_features.iter_mut()) {
        if f.len() == dim {
            l2_normalize(f);
        }
    }

    let sources: Vec<&[f32]> = source_features
        .iter()
        .filter(|(_, f)| f.len() == dim)
        .map(|(_, f)| f.as_slice())
        .collect();
    let target_idx: Vec<usize> = (0..target_features.len())
        .filter(|&i| target_features[i].1.len() == dim)
        .collect();

    let best = pairwise_best_similarity(
        &sources,
        &target_idx
            .iter()
            .map(|&i| target_features[i].1.as_slice())
            .collect::<Vec<_>>(),
    );

    let mut results: Vec<SimilarityResult> = target_idx
        .iter()
        .zip(&best)
        .filter(|(_, sim)| **sim >= threshold)
        .map(|(&i, sim)| {
            let (entry, _) = &target_features[i];
            SimilarityResult {
                path: entry.path.clone(),
                file_name: entry.file_name.clone(),
                similarity: *sim,
                thumbnail: entry.thumbnail.clone(),
            }
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    // No thumbnails here: generating them decodes every match at full
    // resolution, which made result counts in the hundreds take seconds.
    // Results come back immediately; the UI (or nothing, in the CLI) fetches
    // thumbnails lazily for visible items via get_thumbnails_batch.
    results
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
    search_similar_in_folder_with_snapshot(backend, source_path, folder_path, threshold, on_progress)
        .map(|(results, _)| results)
}

/// Like [`search_similar_in_folder`], but also returns the folder's resolved
/// feature snapshot (built after any extraction has drained to the cache), so
/// callers can serve subsequent searches from memory.
pub fn search_similar_in_folder_with_snapshot(
    backend: &dyn InferenceBackend,
    source_path: &str,
    folder_path: &str,
    threshold: f32,
    on_progress: impl FnMut(ProgressInfo),
) -> Result<(Vec<SimilarityResult>, Option<crate::state::FolderSnapshot>), String> {
    match search_similar_impl(Some(backend), None, source_path, folder_path, threshold, on_progress)
    {
        Ok((CacheOnlyOutcome::Ready(results), snapshot)) => Ok((results, snapshot)),
        Ok((CacheOnlyOutcome::NeedsModel { .. }, _)) => {
            unreachable!("backend provided, so NeedsModel cannot be returned")
        }
        Err(e) => Err(e),
    }
}

/// Cache-only variant of [`search_similar_in_folder`].
///
/// Returns [`CacheOnlyOutcome::NeedsModel`] as soon as any involved image
/// (source or folder member) is missing from its cache.
pub fn search_similar_cache_only(
    expected_len: usize,
    source_path: &str,
    folder_path: &str,
    threshold: f32,
    on_progress: impl FnMut(ProgressInfo),
) -> Result<CacheOnlyOutcome<Vec<SimilarityResult>>, String> {
    search_similar_impl(None, Some(expected_len), source_path, folder_path, threshold, on_progress)
        .map(|(outcome, _)| outcome)
}

/// Build a snapshot from resolved folder items, capturing the cache-db
/// identity as it is NOW (call after extraction writes have drained).
pub fn build_folder_snapshot(
    folder_path: &str,
    expected_len: usize,
    items: Vec<(ImageEntry, Vec<f32>)>,
) -> crate::state::FolderSnapshot {
    let (db_modified, db_len) = cache_service::cache_db_identity(folder_path)
        .unwrap_or((std::time::SystemTime::UNIX_EPOCH, 0));
    crate::state::FolderSnapshot {
        db_modified,
        db_len,
        expected_len,
        loaded_at: std::time::Instant::now(),
        items,
    }
}

/// Pure in-memory search against a [`crate::state::FolderSnapshot`]: no
/// scan, no database, one stat of the query image.
///
/// Returns None when the source image isn't part of the snapshot or its file
/// changed since the snapshot was built — the caller falls back to the full
/// cache-aware path in that case.
pub fn search_snapshot(
    snapshot: &crate::state::FolderSnapshot,
    source_path: &str,
    threshold: f32,
) -> Option<Vec<SimilarityResult>> {
    let source_idx = snapshot.items.iter().position(|(entry, _)| {
        std::path::Path::new(&entry.path) == std::path::Path::new(source_path)
    })?;
    let (source_entry, source_features) = &snapshot.items[source_idx];

    // The query file must be unchanged since the snapshot was built
    let meta = std::fs::metadata(source_path).ok()?;
    if meta.len() != source_entry.file_size {
        return None;
    }
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if modified != source_entry.modified {
        return None;
    }

    let source_path_norm = std::path::Path::new(source_path);
    let mut results: Vec<SimilarityResult> = snapshot
        .items
        .par_iter()
        .filter(|(entry, _)| std::path::Path::new(&entry.path) != source_path_norm)
        .filter_map(|(entry, features)| {
            if features.len() != source_features.len() {
                return None;
            }
            let similarity = cosine_similarity(source_features, features);
            (similarity >= threshold).then(|| SimilarityResult {
                path: entry.path.clone(),
                file_name: entry.file_name.clone(),
                similarity,
                thumbnail: None,
            })
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    Some(results)
}

fn search_similar_impl(
    backend: Option<&dyn InferenceBackend>,
    expected_len_override: Option<usize>,
    source_path: &str,
    folder_path: &str,
    threshold: f32,
    mut on_progress: impl FnMut(ProgressInfo),
) -> Result<(CacheOnlyOutcome<Vec<SimilarityResult>>, Option<crate::state::FolderSnapshot>), String> {
    let expected_len = match (backend, expected_len_override) {
        (Some(b), _) => expected_feature_len(&b.model_info().config),
        (None, Some(len)) => len,
        (None, None) => {
            return Err("Feature dimension unknown: no backend and no expected length".to_string())
        }
    };

    // Scan the folder once and load its cache; the same map also answers the
    // query image (recursive scans key every nested path in the root cache).
    let entries = image_service::scan_folder(folder_path, true)?;
    let total = entries.len();
    let conn = cache_service::open_cache(folder_path)?;
    let map = cache_service::load_all_features(&conn, expected_len)?;

    on_progress(ProgressInfo {
        total,
        processed: 0,
        cache_hits: 0,
        cache_misses: 0,
        images_per_second: 0.0,
        elapsed_secs: 0.0,
        message: format!("Loading DINOv3 features: 0/{} images", total),
    });
    let (mut folder_items, folder_misses, cache_hits) = split_entries_by_cache(entries, &map);
    on_progress(ProgressInfo {
        total,
        processed: cache_hits,
        cache_hits,
        cache_misses: 0,
        images_per_second: 0.0,
        elapsed_secs: 0.0,
        message: format!(
            "Loaded DINOv3 features: {}/{} images from cache ({} to extract)",
            cache_hits,
            total,
            folder_misses.len()
        ),
    });

    // Query image: match it among the scanned entries first — WalkDir keys
    // the cache with its own separator style, so a Path-component comparison
    // is the only string-form-agnostic lookup. Fall back to the raw map
    // (covers a source stored under this exact string before), then to
    // extraction with the backend.
    let source_entry_idx = folder_items
        .iter()
        .position(|(entry, _)| std::path::Path::new(&entry.path) == std::path::Path::new(source_path));

    let (source_features, source_was_hit) = if let Some(idx) = source_entry_idx {
        (folder_items[idx].1.clone(), true)
    } else if let Some(f) = cached_features_if_valid(&map, source_path) {
        (f, true)
    } else {
        let Some(b) = backend else {
            return Ok((CacheOnlyOutcome::NeedsModel {
                misses: folder_misses.len() + 1,
            }, None));
        };
        (extract_and_store_single(b, source_path, &conn)?, false)
    };
    let total_misses = folder_misses.len() + usize::from(!source_was_hit);

    if !folder_misses.is_empty() {
        let Some(b) = backend else {
            return Ok((CacheOnlyOutcome::NeedsModel { misses: total_misses }, None));
        };
        let extracted = extract_misses_with_backend(
            b,
            folder_misses,
            folder_path,
            total,
            cache_hits,
            &mut on_progress,
        )?;
        folder_items.extend(extracted);
    }

    let mut results: Vec<SimilarityResult> = folder_items
        .iter()
        .filter(|(entry, _)| {
            std::path::Path::new(&entry.path) != std::path::Path::new(source_path)
        })
        .filter_map(|(entry, features)| {
            if features.len() != source_features.len() {
                log::warn!(
                    "Skipping {}: cached feature length {} != source {}",
                    entry.path, features.len(), source_features.len()
                );
                return None;
            }
            let similarity = cosine_similarity(&source_features, features);
            (similarity >= threshold).then(|| SimilarityResult {
                path: entry.path.clone(),
                file_name: entry.file_name.clone(),
                similarity,
                thumbnail: entry.thumbnail.clone(),
            })
        })
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    // Thumbnails are fetched lazily by the UI, not blocking the result set

    // Snapshot the resolved state for instant repeat searches. Built after
    // extraction writes drained (the writer thread joins before returning),
    // so the db identity captured now matches the in-memory items.
    let snapshot = build_folder_snapshot(folder_path, expected_len, folder_items);
    Ok((CacheOnlyOutcome::Ready(results), Some(snapshot)))
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
    on_progress: impl FnMut(ProgressInfo),
) -> Result<Vec<SimilarityResult>, String> {
    match compare_folders_impl(Some(backend), None, source_folder, target_folder, threshold, on_progress) {
        Ok(CacheOnlyOutcome::Ready(results)) => Ok(results),
        Ok(CacheOnlyOutcome::NeedsModel { .. }) => {
            unreachable!("backend provided, so NeedsModel cannot be returned")
        }
        Err(e) => Err(e),
    }
}

/// Cache-only variant of [`compare_folders`].
///
/// Returns [`CacheOnlyOutcome::NeedsModel`] as soon as either folder has
/// images missing from its cache.
pub fn compare_folders_cache_only(
    expected_len: usize,
    source_folder: &str,
    target_folder: &str,
    threshold: f32,
    on_progress: impl FnMut(ProgressInfo),
) -> Result<CacheOnlyOutcome<Vec<SimilarityResult>>, String> {
    compare_folders_impl(None, Some(expected_len), source_folder, target_folder, threshold, on_progress)
}

fn compare_folders_impl(
    backend: Option<&dyn InferenceBackend>,
    expected_len_override: Option<usize>,
    source_folder: &str,
    target_folder: &str,
    threshold: f32,
    mut on_progress: impl FnMut(ProgressInfo),
) -> Result<CacheOnlyOutcome<Vec<SimilarityResult>>, String> {
    let expected_len = match (backend, expected_len_override) {
        (Some(b), _) => expected_feature_len(&b.model_info().config),
        (None, Some(len)) => len,
        (None, None) => {
            return Err("Feature dimension unknown: no backend and no expected length".to_string())
        }
    };

    // Source folder
    let src_entries = image_service::scan_folder(source_folder, true)?;
    let mut source = resolve_cache_only(source_folder, src_entries, expected_len, &mut on_progress)?;

    // Target folder
    let tgt_entries = image_service::scan_folder(target_folder, true)?;
    let mut target = resolve_cache_only(target_folder, tgt_entries, expected_len, &mut on_progress)?;

    let total_misses = source.misses.len() + target.misses.len();

    match backend {
        Some(backend) => {
            if !source.misses.is_empty() {
                let miss_count = source.misses.len();
                let extracted = extract_misses_with_backend(
                    backend,
                    std::mem::take(&mut source.misses),
                    source_folder,
                    source.cache_hits + miss_count,
                    source.cache_hits,
                    &mut on_progress,
                )?;
                source.items.extend(extracted);
            }
            if !target.misses.is_empty() {
                let miss_count = target.misses.len();
                let extracted = extract_misses_with_backend(
                    backend,
                    std::mem::take(&mut target.misses),
                    target_folder,
                    target.cache_hits + miss_count,
                    target.cache_hits,
                    &mut on_progress,
                )?;
                target.items.extend(extracted);
            }
        }
        None if total_misses > 0 => {
            return Ok(CacheOnlyOutcome::NeedsModel { misses: total_misses });
        }
        None => {}
    }

    Ok(CacheOnlyOutcome::Ready(finish_compare(
        &mut source.items,
        &mut target.items,
        threshold,
    )))
}

/// (Thumbnails for search/compare results are generated on demand by the UI
/// through `image_service::get_thumbnails_batch`; they used to be attached
/// here, which decoded every match at full resolution before any result
/// could be shown.)

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

    /// Tiny deterministic LCG so the GEMM test needs no rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        }
    }

    fn normalized_random_vec(dim: usize, rng: &mut Lcg) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn test_pairwise_best_matches_naive() {
        let dim = 96;
        let mut rng = Lcg(42);
        let sources: Vec<Vec<f32>> = (0..70).map(|_| normalized_random_vec(dim, &mut rng)).collect();
        let targets: Vec<Vec<f32>> = (0..130).map(|_| normalized_random_vec(dim, &mut rng)).collect();

        let src_refs: Vec<&[f32]> = sources.iter().map(|v| v.as_slice()).collect();
        let tgt_refs: Vec<&[f32]> = targets.iter().map(|v| v.as_slice()).collect();
        let got = pairwise_best_similarity(&src_refs, &tgt_refs);

        for (t, target) in targets.iter().enumerate() {
            let naive = sources
                .iter()
                .map(|s| cosine_similarity(s, target))
                .fold(f32::MIN, f32::max);
            assert!(
                (got[t] - naive).abs() < 1e-4,
                "target {}: gemm {} vs naive {}",
                t, got[t], naive
            );
        }
    }

    #[test]
    fn test_pairwise_best_with_identical_vector() {
        let v = vec![1.0f32, 0.0, 0.0];
        let w = vec![0.0f32, 1.0, 0.0];
        let best = pairwise_best_similarity(&[v.as_slice(), w.as_slice()], &[v.as_slice()]);
        assert!((best[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_best_mixed_dims_fallback() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        // One target has a different dim than the sources: the fallback must
        // skip mismatched pairs instead of misindexing the GEMM.
        let best = pairwise_best_similarity(&[a.as_slice()], &[b.as_slice()]);
        assert_eq!(best[0], f32::MIN);
    }

    #[test]
    fn test_search_snapshot_ranks_and_invalidates() {
        use crate::state::ImageEntry;

        let tmp = tempfile::tempdir().unwrap();
        let mk_entry = |name: &str, data: &[u8]| -> (ImageEntry, String) {
            let path = tmp.path().join(name);
            std::fs::write(&path, data).unwrap();
            let meta = std::fs::metadata(&path).unwrap();
            let modified = meta
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            (
                ImageEntry {
                    path: path.to_string_lossy().to_string(),
                    file_name: name.to_string(),
                    file_size: meta.len(),
                    modified,
                    thumbnail: None,
                },
                path.to_string_lossy().to_string(),
            )
        };

        let (source_entry, source_path) = mk_entry("a.jpg", b"source-image-bytes");
        let (dup_entry, _) = mk_entry("b.jpg", b"dup-image-bytes!!");
        let (far_entry, _) = mk_entry("c.jpg", b"far-image-bytes!!");

        let norm = |v: Vec<f32>| {
            let mut v = v;
            l2_normalize(&mut v);
            v
        };
        let source_features = norm(vec![1.0, 0.0, 0.0]);
        let dup_features = norm(vec![0.99, 0.1, 0.0]);
        let far_features = norm(vec![0.0, 0.9, 0.1]);

        let snapshot = crate::state::FolderSnapshot {
            db_modified: std::time::SystemTime::UNIX_EPOCH,
            db_len: 0,
            expected_len: 3,
            loaded_at: std::time::Instant::now(),
            items: vec![
                (source_entry.clone(), source_features.clone()),
                (dup_entry, dup_features),
                (far_entry, far_features),
            ],
        };

        // Source excluded, near-duplicate ranked first at 0.9 threshold,
        // far vector filtered out
        let results = search_snapshot(&snapshot, &source_path, 0.9).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.9);

        // A source not in the snapshot declines (caller falls back)
        let (_, stranger) = mk_entry("stranger.jpg", b"stranger-bytes");
        assert!(search_snapshot(&snapshot, &stranger, 0.9).is_none());

        // A modified source file invalidates the hit
        std::fs::write(&source_path, b"source-image-bytes-CHANGED").unwrap();
        assert!(search_snapshot(&snapshot, &source_path, 0.9).is_none());
    }
}
