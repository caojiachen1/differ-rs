//! ONNX Runtime backend for DINOv3 inference.
//!
//! Uses the `ort` crate to load and run ONNX models.
//! Execution provider priority: TensorRT > CUDA > DirectML > CPU.
//! Each failed provider logs its error before falling back to the next.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use log::{info, warn};
use rayon::prelude::*;

use crate::dinov3::config::ModelConfig;
use crate::dinov3::model::ModelInfo;
use crate::dinov3::postprocessing::l2_normalize;
use crate::dinov3::preprocessing::preprocess_image;
use crate::dinov3::InferenceBackend;

/// Convert an `ort::Error` to `anyhow::Error`.
fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("ORT error: {}", e)
}

/// ONNX Runtime inference backend for DINOv3.
///
/// Loads a DINOv3 ONNX model and performs feature extraction.
/// Tries TensorRT, then CUDA, then DirectML, then CPU.
pub struct OnnxBackend {
    session: Option<Mutex<ort::session::Session>>,
    config: ModelConfig,
    model_path: Option<PathBuf>,
    ep_name: &'static str,
}

impl OnnxBackend {
    /// Create a new OnnxBackend with default ViT-S/16 config.
    pub fn new() -> Self {
        Self {
            session: None,
            config: ModelConfig::vit_small_16(),
            model_path: None,
            ep_name: "none",
        }
    }

    /// Create a new OnnxBackend with a custom model config.
    pub fn with_config(config: ModelConfig) -> Self {
        Self {
            session: None,
            config,
            model_path: None,
            ep_name: "none",
        }
    }

    /// Get the model config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Whether the backend is using GPU acceleration.
    pub fn is_using_gpu(&self) -> bool {
        matches!(self.ep_name, "TensorRT" | "CUDA" | "DirectML")
    }

    /// Name of the active execution provider ("TensorRT", "CUDA", "DirectML", "CPU").
    pub fn execution_provider(&self) -> &'static str {
        self.ep_name
    }

    /// Download the DINOv3 Q4 ONNX model from ModelScope.
    ///
    /// Downloads `model_q4.onnx` and `model_q4.onnx_data` to the specified directory.
    /// Returns the path to the main model file.
    pub fn download_model(output_dir: &Path) -> Result<PathBuf> {
        const MODEL_URL: &str = "https://modelscope.cn/models/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX-MHA/resolve/master/onnx/model_q4.onnx";
        const MODEL_DATA_URL: &str = "https://modelscope.cn/models/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX-MHA/resolve/master/onnx/model_q4.onnx_data";

        std::fs::create_dir_all(output_dir)
            .context("Failed to create model output directory")?;

        let model_path = output_dir.join("model_q4.onnx");
        let model_data_path = output_dir.join("model_q4.onnx_data");

        // Download main model file
        if !model_path.exists() {
            info!("Downloading model_q4.onnx ...");
            Self::download_file(MODEL_URL, &model_path)?;
            info!("Downloaded model_q4.onnx to {:?}", model_path);
        } else {
            info!("model_q4.onnx already exists at {:?}", model_path);
        }

        // Download model data file (external weights for Q4)
        if !model_data_path.exists() {
            info!("Downloading model_q4.onnx_data ...");
            Self::download_file(MODEL_DATA_URL, &model_data_path)?;
            info!("Downloaded model_q4.onnx_data to {:?}", model_data_path);
        } else {
            info!("model_q4.onnx_data already exists at {:?}", model_data_path);
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

    /// Try to create an ONNX session with the specified execution provider.
    ///
    /// `ep` is one of "TensorRT", "CUDA", "DirectML", "CPU". Provider
    /// registration uses `error_on_failure` so an unavailable provider fails
    /// here instead of silently falling back to CPU inside ONNX Runtime.
    fn try_create_session(model_path: &Path, ep: &'static str) -> Result<ort::session::Session> {
        let mut builder = ort::session::Session::builder().map_err(ort_err)?;

        // Enable graph optimizations
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
            .map_err(|e| ort_err(e))?;

        match ep {
            "TensorRT" => {
                // Cache built engines next to the model; without this TensorRT
                // rebuilds the engine on every process start (minutes).
                let cache_dir = model_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("trt_cache");
                std::fs::create_dir_all(&cache_dir)
                    .with_context(|| format!("Failed to create TensorRT cache dir {:?}", cache_dir))?;
                let trt = ort::ep::TensorRT::default()
                    .with_engine_cache(true)
                    .with_engine_cache_path(cache_dir.to_string_lossy())
                    .with_timing_cache(true)
                    .build()
                    .error_on_failure();
                builder = builder.with_execution_providers([trt]).map_err(|e| ort_err(e))?;
            }
            "CUDA" => {
                let cuda = ort::ep::CUDA::default().build().error_on_failure();
                builder = builder.with_execution_providers([cuda]).map_err(|e| ort_err(e))?;
            }
            "DirectML" => {
                // DirectML requires memory pattern disabled.
                builder = builder.with_memory_pattern(false).map_err(|e| ort_err(e))?;
                let dml = ort::ep::DirectML::default().build().error_on_failure();
                builder = builder.with_execution_providers([dml]).map_err(|e| ort_err(e))?;
            }
            _ => {
                // CPU with multi-threading
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get().max(1) / 2)
                    .unwrap_or(2);
                builder = builder
                    .with_intra_threads(num_threads)
                    .map_err(|e| ort_err(e))?;

                let cpu_ep = ort::ep::CPU::default().build();
                builder = builder
                    .with_execution_providers([cpu_ep])
                    .map_err(|e| ort_err(e))?;
            }
        }

        let session = builder
            .commit_from_file(model_path.to_string_lossy().as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to create ONNX session ({} EP) from {:?}: {}", ep, model_path, e))?;

        Ok(session)
    }
    /// Run inference on an already-preprocessed NCHW tensor and return
    /// L2-normalized features. Session access is serialized by the mutex.
    fn infer_preprocessed(&self, input_tensor: Vec<f32>) -> Result<Vec<f32>> {
        let session_guard = self
            .session
            .as_ref()
            .context("ONNX model not loaded. Call load_model() first.")?
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire session lock"))?;

        // Get input/output names from the session
        if session_guard.inputs().is_empty() {
            bail!("ONNX model has no inputs");
        }
        if session_guard.outputs().is_empty() {
            bail!("ONNX model has no outputs");
        }
        let input_name = session_guard.inputs()[0].name().to_string();
        let output_name = session_guard.outputs()[0].name().to_string();

        // Create ort Tensor from the preprocessed NCHW data
        let h = self.config.input_height;
        let w = self.config.input_width;
        let input_value = ort::value::Tensor::from_array(([1usize, 3, h, w], input_tensor))
            .map_err(|e| ort_err(e))?;

        // Run inference (need mutable session)
        let mut session = session_guard;
        let outputs = session
            .run(ort::inputs![input_name.as_str() => input_value])
            .context("ONNX inference failed")?;

        // Extract output tensor by name
        let output_value = outputs
            .get(&output_name)
            .context("Failed to get output tensor from ONNX session")?;

        // Extract float data from the output tensor
        let (_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(ort_err)
            .context("Failed to extract f32 tensor from ONNX output")?;

        let mut features: Vec<f32> = data.to_vec();
        if features.is_empty() {
            bail!("ONNX model produced empty output tensor");
        }

        // L2 normalize
        l2_normalize(&mut features);
        Ok(features)
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for OnnxBackend {
    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!(
                "ONNX model file not found: {:?}. Use OnnxBackend::download_model() to download it.",
                model_path
            );
        }

        info!("Loading ONNX model from {:?}", model_path);

        // Try execution providers in priority order, logging each failure.
        let mut session = None;
        for ep in ["TensorRT", "CUDA", "DirectML", "CPU"] {
            match Self::try_create_session(model_path, ep) {
                Ok(s) => {
                    info!("ONNX model loaded with {} execution provider", ep);
                    session = Some((s, ep));
                    break;
                }
                Err(e) => {
                    warn!("ONNX {} EP failed: {:#}. Falling back to next provider.", ep, e);
                }
            }
        }
        let (session, ep_name) = session
            .context("All ONNX execution providers failed (TensorRT, CUDA, DirectML, CPU)")?;

        self.session = Some(Mutex::new(session));
        self.model_path = Some(model_path.to_path_buf());
        self.ep_name = ep_name;

        info!("ONNX model loaded successfully from {:?}", model_path);
        Ok(())
    }

    fn extract_features(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        // 1. Preprocess image to NCHW tensor [1, 3, H, W]
        let input_tensor = preprocess_image(image_data, &self.config)
            .context("Failed to preprocess image")?;

        // 2. Run inference + L2 normalize
        let features = self.infer_preprocessed(input_tensor)?;

        info!(
            "Extracted {} features from image (EP: {})",
            features.len(),
            self.ep_name
        );

        Ok(features)
    }

    /// Batch extraction with pipelined parallelism:
    /// image decode + resize (CPU-bound) runs on all cores via rayon, while
    /// session runs are serialized (ort session requires &mut for run()).
    fn batch_extract(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        // Stage 1: parallel preprocessing
        let inputs: Vec<Result<Vec<f32>>> = images
            .par_iter()
            .map(|img| preprocess_image(img, &self.config).context("Failed to preprocess image"))
            .collect();

        // Stage 2: sequential inference
        inputs
            .into_iter()
            .enumerate()
            .map(|(i, input)| {
                let input = input.with_context(|| format!("image {}/{}", i + 1, images.len()))?;
                self.infer_preprocessed(input)
                    .with_context(|| format!("ONNX inference failed for image {}/{}", i + 1, images.len()))
            })
            .collect()
    }

    fn model_info(&self) -> ModelInfo {
        let quantization = if self.session.is_some() {
            Some("Q4".to_string())
        } else {
            None
        };

        ModelInfo {
            name: if self.session.is_some() {
                format!("ONNX DINOv3 ViT-S/16 Q4 ({})", self.ep_name)
            } else {
                "ONNX DINOv3 (not loaded)".to_string()
            },
            backend: "onnx".to_string(),
            config: self.config.clone(),
            quantization,
        }
    }
}
