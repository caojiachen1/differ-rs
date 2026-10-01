//! Model configuration for DINOv3 Vision Transformer variants.
//!
//! Provides [`ModelConfig`] with preset configurations for ViT-S/16, ViT-B/16,
//! ViT-L/16, and ViT-H/16 architectures used in DINOv3.

use serde::{Deserialize, Serialize};

/// Configuration parameters for a DINOv3 Vision Transformer model.
///
/// Contains all hyperparameters needed for model initialization and
/// image preprocessing (normalization constants, input dimensions, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Dimensionality of the hidden representations.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of the feed-forward intermediate layer.
    pub intermediate_size: usize,
    /// Size of each image patch (e.g., 16 for ViT-16).
    pub patch_size: usize,
    /// Number of register tokens appended to the sequence (DINOv3 feature).
    pub num_register_tokens: usize,
    /// Expected input image height in pixels.
    pub input_height: usize,
    /// Expected input image width in pixels.
    pub input_width: usize,
    /// ImageNet normalization mean for RGB channels.
    pub image_mean: [f32; 3],
    /// ImageNet normalization standard deviation for RGB channels.
    pub image_std: [f32; 3],
}

impl ModelConfig {
    /// ViT-S/16 configuration (Small).
    ///
    /// - 384 hidden dims, 12 layers, 6 heads, 1536 intermediate size
    pub fn vit_small_16() -> Self {
        Self {
            hidden_size: 384,
            num_layers: 12,
            num_heads: 6,
            intermediate_size: 1536,
            patch_size: 16,
            num_register_tokens: 4,
            input_height: 518,
            input_width: 518,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    /// Feature dimension of this configuration: sequence_length * hidden_size,
    /// where sequence_length = 1 CLS + register tokens + patch grid.
    /// Used to validate cached feature vectors against the active model.
    pub fn output_dim(&self) -> usize {
        (1 + self.num_register_tokens
            + (self.input_height / self.patch_size) * (self.input_width / self.patch_size))
            * self.hidden_size
    }

    /// ViT-B/16 configuration (Base).
    ///
    /// - 768 hidden dims, 12 layers, 12 heads, 3072 intermediate size
    pub fn vit_base_16() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            patch_size: 16,
            num_register_tokens: 4,
            input_height: 518,
            input_width: 518,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    /// ViT-L/16 configuration (Large).
    ///
    /// - 1024 hidden dims, 24 layers, 16 heads, 4096 intermediate size
    pub fn vit_large_16() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
            patch_size: 16,
            num_register_tokens: 4,
            input_height: 518,
            input_width: 518,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    /// ViT-H/16 configuration (Huge).
    ///
    /// - 1280 hidden dims, 32 layers, 16 heads, 5120 intermediate size
    ///   (uses SwiGLU activation with larger intermediate dimension)
    pub fn vit_huge_16() -> Self {
        Self {
            hidden_size: 1280,
            num_layers: 32,
            num_heads: 16,
            intermediate_size: 5120,
            patch_size: 16,
            num_register_tokens: 4,
            input_height: 518,
            input_width: 518,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    /// Calculate the number of patches per dimension.
    pub fn num_patches(&self) -> usize {
        let h = self.input_height / self.patch_size;
        let w = self.input_width / self.patch_size;
        h * w
    }

    /// Calculate the total sequence length (CLS + patches + registers).
    pub fn sequence_length(&self) -> usize {
        1 + self.num_patches() + self.num_register_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_base_16_patches() {
        let config = ModelConfig::vit_base_16();
        // 518 / 16 = 32 patches per dim, 32*32 = 1024 patches
        assert_eq!(config.num_patches(), 1024);
        // 1 (CLS) + 1024 (patches) + 4 (registers) = 1029
        assert_eq!(config.sequence_length(), 1029);
    }

    #[test]
    fn test_presets_have_correct_dims() {
        assert_eq!(ModelConfig::vit_small_16().hidden_size, 384);
        assert_eq!(ModelConfig::vit_base_16().hidden_size, 768);
        assert_eq!(ModelConfig::vit_large_16().hidden_size, 1024);
        assert_eq!(ModelConfig::vit_huge_16().hidden_size, 1280);
    }
}
