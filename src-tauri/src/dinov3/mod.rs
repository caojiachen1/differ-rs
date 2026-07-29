//! dinov3: DINOv3-based image feature extraction (merged from the former
//! `dinov3-core` crate).
//!
//! Provides:
//! - Model configuration presets for various ViT architectures
//! - Image preprocessing pipeline (resize, normalize, NCHW format)
//! - Post-processing utilities (L2 normalization, cosine similarity)
//! - Backend abstraction for multiple inference engines (GGML, ONNX, Candle)

pub mod backend;
pub mod config;
pub mod model;
pub mod postprocessing;
pub mod preprocessing;

// Re-export key types for convenience
pub use config::ModelConfig;
pub use model::{ModelInfo, TransformerLayer, VitWeights};
pub use postprocessing::{cosine_similarity, find_similar, l2_normalize};
pub use preprocessing::preprocess_image;

/// Path to the test images directory (relative to the workspace root).
pub const TEST_IMAGES_DIR: &str = "../test";

/// Trait defining the interface for all inference backends.
///
/// Each backend (GGML, ONNX, Candle) must implement this trait to provide
/// model loading and feature extraction capabilities.
pub trait InferenceBackend: Send + Sync {
    /// Load a model from the specified path.
    fn load_model(&mut self, model_path: &std::path::Path) -> anyhow::Result<()>;

    /// Extract feature vector from raw image bytes.
    fn extract_features(&self, image_data: &[u8]) -> anyhow::Result<Vec<f32>>;

    /// Extract features from multiple images.
    ///
    /// Default implementation processes images sequentially.
    /// Backends may override this with optimized batch processing.
    fn batch_extract(&self, images: &[&[u8]]) -> anyhow::Result<Vec<Vec<f32>>> {
        images.iter().map(|img| self.extract_features(img)).collect()
    }

    /// Return information about the currently loaded model.
    fn model_info(&self) -> ModelInfo;
}
