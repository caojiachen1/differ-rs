//! GGML backend for DINOv3 inference.
//!
//! Thin adapter over the standalone `dinov3-ggml` crate, which implements
//! the full ViT forward pass with GGML (CPU or CUDA via the `ggml-cuda`
//! feature) and the pipelined batch extraction.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::dinov3::config::ModelConfig;
use crate::dinov3::{InferenceBackend, ModelInfo};

/// GGML-based ViT inference backend.
pub struct GgmlBackend {
    extractor: Option<dinov3_ggml::FeatureExtractor>,
    config: ModelConfig,
}

impl GgmlBackend {
    pub fn new() -> Self {
        Self {
            extractor: None,
            config: ModelConfig::vit_small_16(),
        }
    }

    /// Create a backend with a specific model configuration.
    pub fn with_config(config: ModelConfig) -> Self {
        Self {
            extractor: None,
            config,
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

    /// Map the dinov3-core ModelConfig onto the dinov3-ggml VitConfig.
    fn vit_config(&self) -> dinov3_ggml::VitConfig {
        dinov3_ggml::VitConfig {
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_heads,
            intermediate_size: self.config.intermediate_size,
            patch_size: self.config.patch_size,
            num_register_tokens: self.config.num_register_tokens,
            input_height: self.config.input_height,
            input_width: self.config.input_width,
            rope_freq_base: 100.0, // DINOv3 rope_theta
            image_mean: self.config.image_mean,
            image_std: self.config.image_std,
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
        extractor.extract_batch(images)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: format!("GGML ViT-S/{} ({} layers)", self.config.patch_size, self.config.num_layers),
            backend: "ggml".to_string(),
            config: self.config.clone(),
            quantization: None,
        }
    }
}
