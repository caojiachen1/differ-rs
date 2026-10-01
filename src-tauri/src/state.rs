//! Application state management for the Tauri backend.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::dinov3::InferenceBackend;
use serde::{Deserialize, Serialize};

/// Main application state shared across Tauri commands.
pub struct AppState {
    /// The inference backend (DINOv3 model), wrapped in Arc for cloning into spawn_blocking.
    pub backend: Arc<Mutex<Option<Box<dyn InferenceBackend>>>>,
    /// Which backend is currently loaded ("ggml", "onnx" or "none").
    /// Arc so the startup auto-load thread can update it after `manage()`.
    pub backend_name: Arc<Mutex<String>>,
    /// Cached image entries per folder.
    pub image_cache: Mutex<HashMap<String, Vec<ImageEntry>>>,
    /// Feature cache for extracted features.
    pub feature_cache: Mutex<FeatureCache>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            backend: Arc::new(Mutex::new(None)),
            backend_name: Arc::new(Mutex::new("none".to_string())),
            image_cache: Mutex::new(HashMap::new()),
            feature_cache: Mutex::new(FeatureCache::new()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
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

/// Cache for storing extracted feature vectors.
pub struct FeatureCache {
    /// Map from image path to feature vector.
    pub features: HashMap<String, Vec<f32>>,
}

impl FeatureCache {
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    pub fn get(&self, path: &str) -> Option<&Vec<f32>> {
        self.features.get(path)
    }

    pub fn insert(&mut self, path: String, features: Vec<f32>) {
        self.features.insert(path, features);
    }

    pub fn clear(&mut self) {
        self.features.clear();
    }
}

impl Default for FeatureCache {
    fn default() -> Self {
        Self::new()
    }
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
