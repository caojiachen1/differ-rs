//! Image preprocessing pipeline for DINOv3 inference.
//!
//! Converts raw image bytes into a normalized NCHW tensor suitable for
//! Vision Transformer input. The pipeline:
//! 1. Decodes the image from bytes
//! 2. Resizes to the model's expected input dimensions (e.g., 518x518 or
//!    256x256) with a SIMD Lanczos3 kernel (`fast_image_resize`)
//! 3. Converts to RGB float32
//! 4. Normalizes pixels to [0,1] then applies ImageNet standardization
//!
//! Must stay in lockstep with the identical pipeline inside the
//! `dinov3-ggml` crate: all backends must see the same pixel input for
//! cross-backend feature comparisons to be meaningful. The SIMD kernel is
//! not bit-identical to the previous `image`-crate implementation (same
//! Lanczos3 filter; differences are a few hundredths of a u8 level on
//! average, verified by the oracle example in dinov3-ggml).

use anyhow::{Context, Result};
use fast_image_resize::{
    images::Image as FirImage, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::DynamicImage;

use crate::dinov3::config::ModelConfig;

/// Preprocess raw image bytes into a normalized NCHW float tensor.
///
/// The output tensor has shape [1, 3, height, width] flattened into a Vec<f32>.
/// Pixel values are normalized to [0, 1] then standardized with ImageNet mean/std.
pub fn preprocess_image(image_data: &[u8], config: &ModelConfig) -> Result<Vec<f32>> {
    let img = image::load_from_memory(image_data)
        .context("Failed to decode image from bytes")?;
    preprocess_dynamic_image(&img, config)
}

/// Preprocess a DynamicImage into a normalized NCHW float tensor.
pub fn preprocess_dynamic_image(img: &DynamicImage, config: &ModelConfig) -> Result<Vec<f32>> {
    let rgb = img.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let pixels = rgb.into_raw();

    // SIMD resize (Lanczos3, same filter as before)
    let src_image = FirImage::from_vec_u8(width as u32, height as u32, pixels, PixelType::U8x3)
        .context("Invalid source image buffer")?;
    let mut dst_image = FirImage::new(config.input_width as u32, config.input_height as u32, PixelType::U8x3);
    let mut resizer = Resizer::new();
    let options =
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .context("Resize failed")?;

    let width = dst_image.width() as usize;
    let height = dst_image.height() as usize;
    let pixels = dst_image.buffer();

    // Convert HWC -> NCHW with normalization
    let num_pixels = width * height;
    let mut tensor = vec![0.0f32; 3 * num_pixels];

    for y in 0..height {
        for x in 0..width {
            let src_idx = (y * width + x) * 3;
            let r = pixels[src_idx] as f32 / 255.0;
            let g = pixels[src_idx + 1] as f32 / 255.0;
            let b = pixels[src_idx + 2] as f32 / 255.0;

            // ImageNet standardization
            let dst_idx = y * width + x;
            tensor[dst_idx] = (r - config.image_mean[0]) / config.image_std[0];           // Channel 0 (R)
            tensor[num_pixels + dst_idx] = (g - config.image_mean[1]) / config.image_std[1]; // Channel 1 (G)
            tensor[2 * num_pixels + dst_idx] = (b - config.image_mean[2]) / config.image_std[2]; // Channel 2 (B)
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_output_shape() {
        let config = ModelConfig::vit_small_16();
        // Create a simple 100x100 red image
        let img = DynamicImage::new_rgb8(100, 100);
        let result = preprocess_dynamic_image(&img, &config).unwrap();
        // Should be 3 * 518 * 518
        assert_eq!(result.len(), 3 * 518 * 518);
    }

    #[test]
    fn test_preprocess_normalization() {
        let config = ModelConfig::vit_small_16();
        // Create a white image (255,255,255) -> normalized should be (1.0 - mean) / std
        let mut img = DynamicImage::new_rgb8(10, 10);
        // Fill with white
        for pixel in img.as_mut_rgb8().unwrap().pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let result = preprocess_dynamic_image(&img, &config).unwrap();
        // Check R channel (first pixel): (1.0 - 0.485) / 0.229 ≈ 2.249
        let expected_r = (1.0 - 0.485) / 0.229;
        assert!((result[0] - expected_r).abs() < 0.01);
    }
}
