//! Application state management for the Tauri backend.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};
use crate::dinov3::InferenceBackend;
use serde::{Deserialize, Serialize};

/// How many folder snapshots to keep resident. Each costs roughly
/// `images x feature_dim x 4` bytes (~77 MB for 50k images at 384 dims).
const MAX_FOLDER_SNAPSHOTS: usize = 4;

/// Main application state shared across Tauri commands.
pub struct AppState {
    /// The inference backend (DINOv3 model), wrapped in Arc for cloning into spawn_blocking.
    pub backend: Arc<Mutex<Option<Box<dyn InferenceBackend>>>>,
    /// Which backend is currently loaded ("ggml", "onnx" or "none").
    /// Arc so the startup auto-load thread can update it after `manage()`.
    pub backend_name: Arc<Mutex<String>>,
    /// In-memory feature snapshots per folder, keyed by folder path. Repeated
    /// searches against a snapshot skip the folder scan and the full cache
    /// table load and run pure in-memory ranking (~ms for 50k images).
    pub folder_snapshots: Arc<Mutex<HashMap<String, FolderSnapshot>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            backend: Arc::new(Mutex::new(None)),
            backend_name: Arc::new(Mutex::new("none".to_string())),
            folder_snapshots: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// A folder's features loaded into memory for instant repeated searches.
///
/// Validated against the per-folder cache db's (mtime, size): any write to
/// the db (extraction, cache clear) invalidates the snapshot, and the
/// `expected_len` guard drops it when the active model configuration
/// changes.
pub struct FolderSnapshot {
    /// Cache-db mtime + size at load time; mismatch means stale.
    pub db_modified: SystemTime,
    pub db_len: u64,
    /// Feature dimension the items were produced with.
    pub expected_len: usize,
    /// When this snapshot was (re)built; bounds memory via eviction.
    pub loaded_at: Instant,
    /// Every image whose features resolved (cache hit or extracted).
    pub items: Vec<(ImageEntry, Vec<f32>)>,
}

impl FolderSnapshot {
    /// Whether this snapshot still matches the folder's cache db and the
    /// active feature dimension.
    pub fn is_valid(&self, folder_path: &str, expected_len: usize) -> bool {
        if self.expected_len != expected_len {
            return false;
        }
        match crate::services::cache_service::cache_db_identity(folder_path) {
            Some((modified, len)) => modified == self.db_modified && len == self.db_len,
            None => false,
        }
    }
}

/// Insert/replace a snapshot, evicting the oldest when over capacity.
pub fn store_snapshot(
    snapshots: &mut HashMap<String, FolderSnapshot>,
    folder: String,
    snapshot: FolderSnapshot,
) {
    while snapshots.len() >= MAX_FOLDER_SNAPSHOTS {
        let oldest = snapshots
            .iter()
            .max_by_key(|(_, s)| s.loaded_at)
            .map(|(k, _)| k.clone());
        match oldest {
            Some(k) => {
                snapshots.remove(&k);
            }
            None => break,
        }
    }
    snapshots.insert(folder, snapshot);
}

/// Represents a single image entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEntry {
    /// Full path to the image file.
    pub path: String,
    /// File name without path.
    pub file_name: String,
    /// File size in bytes.
    pub file_size: u64,
    /// Last modified timestamp (Unix epoch seconds).
    pub modified: u64,
    /// Thumbnail data (JPEG encoded, base64).
    pub thumbnail: Option<String>,
}

/// Progress information for long-running operations.
///
/// Also used as the payload of the `extraction-progress` event emitted
/// while DINOv3 features are being extracted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// Total number of items to process.
    pub total: usize,
    /// Number of items processed so far.
    pub processed: usize,
    /// Number of images served from the folder cache.
    #[serde(default)]
    pub cache_hits: usize,
    /// Number of images that required fresh extraction.
    #[serde(default)]
    pub cache_misses: usize,
    /// Recent processing speed in images/second (0 when not applicable).
    #[serde(default)]
    pub images_per_second: f64,
    /// Seconds elapsed since the operation started.
    #[serde(default)]
    pub elapsed_secs: f64,
    /// Current status message.
    pub message: String,
}

/// Result of a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Path to the matching image.
    pub path: String,
    /// File name of the matching image.
    pub file_name: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Thumbnail data (base64 encoded JPEG).
    pub thumbnail: Option<String>,
}

/// Cache statistics for a folder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cached features.
    pub cached_count: usize,
    /// Total cache size in bytes.
    pub cache_size_bytes: u64,
    /// Whether the cache is valid.
    pub is_valid: bool,
}
