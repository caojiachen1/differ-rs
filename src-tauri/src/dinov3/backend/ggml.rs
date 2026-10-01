//! GGML backend for DINOv3 inference.
//!
//! Thin adapter over the standalone `dinov3-ggml` crate, which implements
//! the full ViT forward pass with GGML (CPU or CUDA via the `ggml-cuda`
//! feature) and the pipelined batch extraction.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::dinov3::config::ModelConfig;
use crate::dinov3::{InferenceBackend, ModelInfo};

/// Input resolution of the default max-throughput tier. 256x256 (261 tokens)
/// cuts compute ~5x vs 518x518; DINOv3's RoPE position encoding is
/// resolution-agnostic, so the same weights run at both sizes.
const FAST_TIER_INPUT: usize = 256;
/// Batched-graph size for the fast tier (measured optimum on RTX 5080).
const FAST_TIER_MAX_BATCH: usize = 32;
/// Batched-graph size for the high tier (518x518, measured optimum).
const HIGH_TIER_MAX_BATCH: usize = 4;

/// GGML-based ViT inference backend.
pub struct GgmlBackend {
    extractor: Option<dinov3_ggml::FeatureExtractor>,
    config: ModelConfig,
    /// Return CLS-token features from batch extraction (the app-level
    /// pooling mode): same forward pass, ~1029x less GPU->host transfer.
    pooled_features: bool,
}

impl GgmlBackend {
    pub fn new() -> Self {
        Self {
            extractor: None,
            config: ModelConfig::vit_small_16(),
            pooled_features: false,
        }
    }

    /// Batch extraction returns only the CLS-token feature per image
    /// (the app's default pooled mode). The caller persists/compares
    /// `hidden_size`-dim vectors instead of full all-token output.
    pub fn with_feature_pool(mut self, pooled: bool) -> Self {
        self.pooled_features = pooled;
        self
    }

    /// Create a backend with a specific model configuration.
    pub fn with_config(config: ModelConfig) -> Self {
        Self {
            extractor: None,
            config,
            pooled_features: false,
        }
    }

    /// Download the converted DINOv3 GGML weights from ModelScope.
    ///
    /// Downloads `dinov3_vits16.bin` to the specified directory and returns
    /// the path to the model file.
    pub fn download_model(output_dir: &Path) -> Result<PathBuf> {
        const MODEL_URL: &str = "https://modelscope.cn/models/cjc1887415157/dinov3-ggml/resolve/master/dinov3_vits16.bin";

        std::fs::create_dir_all(output_dir)
            .context("Failed to create model output directory")?;

        let model_path = output_dir.join("dinov3_vits16.bin");

        if !model_path.exists() {
            log::info!("Downloading dinov3_vits16.bin ...");
            Self::download_file(MODEL_URL, &model_path)?;
            log::info!("Downloaded dinov3_vits16.bin to {:?}", model_path);
        } else {
            log::info!("dinov3_vits16.bin already exists at {:?}", model_path);
        }

        Ok(model_path)
    }

    /// Download a file from a URL using curl.
    fn download_file(url: &str, dest: &Path) -> Result<()> {
        let status = std::process::Command::new("curl")
            .args(["-L", "-o", &dest.to_string_lossy(), "--progress-bar", url])
            .status()
            .context("Failed to execute curl - is curl installed?")?;

        if !status.success() {
            bail!("curl failed with exit code: {:?}", status.code());
        }

        Ok(())
    }

    /// Default to the max-throughput tier (256x256 + batched graph): batch
    /// folder extraction is ~4x faster than 518x518 there. Set
    /// `GGML_VIT_TIER=high` for the accuracy-first 518x518 tier.
    fn fast_tier() -> bool {
        !std::env::var("GGML_VIT_TIER")
            .map(|v| v.eq_ignore_ascii_case("high"))
            .unwrap_or(false)
    }

    /// Input resolution the backend actually runs at.
    fn effective_input(&self) -> (usize, usize) {
        if Self::fast_tier() {
            (FAST_TIER_INPUT, FAST_TIER_INPUT)
        } else {
            (self.config.input_height, self.config.input_width)
        }
    }

    /// Map the dinov3-core ModelConfig onto the dinov3-ggml VitConfig.
    fn vit_config(&self) -> dinov3_ggml::VitConfig {
        let (input_height, input_width) = self.effective_input();
        dinov3_ggml::VitConfig {
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_heads,
            intermediate_size: self.config.intermediate_size,
            patch_size: self.config.patch_size,
            num_register_tokens: self.config.num_register_tokens,
            input_height,
            input_width,
            rope_freq_base: 100.0, // DINOv3 rope_theta
            image_mean: self.config.image_mean,
            image_std: self.config.image_std,
            max_batch: if Self::fast_tier() {
                FAST_TIER_MAX_BATCH
            } else {
                HIGH_TIER_MAX_BATCH
            },
            // Approximation stack of the fast tier only: the accuracy-first
            // high tier decodes at full resolution.
            scaled_decode: Self::fast_tier(),
            jpeg_scan_limit: 0,
        }
    }
}

impl Default for GgmlBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for GgmlBackend {
    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        let extractor = dinov3_ggml::FeatureExtractor::load(model_path, self.vit_config())
            .context("Failed to load GGML ViT model")?;
        self.extractor = Some(extractor);
        log::info!("GGML backend: loaded model from {:?}", model_path);
        Ok(())
    }

    fn extract_features(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        let extractor = self.extractor.as_ref()
            .context("Model not loaded. Call load_model() first.")?;
        extractor.extract(image_data)
    }

    /// Batch extraction with pipelined parallelism (parallel preprocessing,
    /// serialized backend inference) provided by dinov3-ggml.
    fn batch_extract(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        let extractor = self.extractor.as_ref()
            .context("Model not loaded. Call load_model() first.")?;
        if self.pooled_features {
            extractor.extract_batch_cls(images)
        } else {
            extractor.extract_batch(images)
        }
    }

    fn model_info(&self) -> ModelInfo {
        // Report the effective input size (fast tier overrides the config
        // dims): callers derive the expected feature dimension from this.
        let mut config = self.config.clone();
        let (input_height, input_width) = self.effective_input();
        config.input_height = input_height;
        config.input_width = input_width;
        ModelInfo {
            name: format!(
                "GGML ViT-S/{} @{}x{} ({} layers)",
                self.config.patch_size,
                input_width,
                input_height,
                self.config.num_layers
            ),
            backend: "ggml".to_string(),
            config,
            quantization: None,
        }
    }
}
