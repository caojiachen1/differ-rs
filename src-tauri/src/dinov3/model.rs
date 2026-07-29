//! ViT model structure definitions for weight storage and model info.

use serde::{Deserialize, Serialize};
use crate::dinov3::config::ModelConfig;

/// Information about the currently loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub backend: String,
    pub config: ModelConfig,
    pub quantization: Option<String>,
}

/// Complete set of weights for a DINOv3 Vision Transformer model.
pub struct VitWeights {
    /// Patch embedding convolution weights: [hidden_size, 3, patch_size, patch_size]
    pub patch_embed_weight: Vec<f32>,
    /// Patch embedding bias: [hidden_size]
    pub patch_embed_bias: Vec<f32>,
    /// CLS token: [1, hidden_size]
    pub cls_token: Vec<f32>,
    /// Register tokens: [num_register_tokens, hidden_size]
    pub register_tokens: Vec<f32>,
    /// Absolute position embedding (for Candle backend): [seq_len, hidden_size]
    pub pos_embed: Option<Vec<f32>>,
    /// Transformer encoder layers
    pub layers: Vec<TransformerLayer>,
    /// Final layer norm weight: [hidden_size]
    pub norm_weight: Vec<f32>,
    /// Final layer norm bias: [hidden_size]
    pub norm_bias: Vec<f32>,
}

/// Weights for a single transformer encoder layer.
pub struct TransformerLayer {
    /// LayerNorm1 weight (before attention): [hidden_size]
    pub norm1_weight: Vec<f32>,
    /// LayerNorm1 bias: [hidden_size]
    pub norm1_bias: Vec<f32>,
    /// QKV projection weight: [3 * hidden_size, hidden_size]
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias: [3 * hidden_size]
    pub qkv_bias: Vec<f32>,
    /// Attention output projection weight: [hidden_size, hidden_size]
    pub proj_weight: Vec<f32>,
    /// Attention output projection bias: [hidden_size]
    pub proj_bias: Vec<f32>,
    /// LayerNorm2 weight (before MLP): [hidden_size]
    pub norm2_weight: Vec<f32>,
    /// LayerNorm2 bias: [hidden_size]
    pub norm2_bias: Vec<f32>,
    /// MLP fc1 weight: [intermediate_size, hidden_size]
    pub mlp_fc1_weight: Vec<f32>,
    /// MLP fc1 bias: [intermediate_size]
    pub mlp_fc1_bias: Vec<f32>,
    /// MLP fc2 weight: [hidden_size, intermediate_size]
    pub mlp_fc2_weight: Vec<f32>,
    /// MLP fc2 bias: [hidden_size]
    pub mlp_fc2_bias: Vec<f32>,
    /// MLP fc3 weight for SwiGLU (larger models only): [intermediate_size, hidden_size]
    pub mlp_fc3_weight: Option<Vec<f32>>,
    /// MLP fc3 bias for SwiGLU: [intermediate_size]
    pub mlp_fc3_bias: Option<Vec<f32>>,
    /// LayerScale gamma for attention (DINOv3): [hidden_size]
    pub ls1_gamma: Option<Vec<f32>>,
    /// LayerScale gamma for MLP (DINOv3): [hidden_size]
    pub ls2_gamma: Option<Vec<f32>>,
}
