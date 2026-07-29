//! Inference backend implementations for DINOv3.
//!
//! Each backend provides a different inference engine:
//! - `ggml`: Hand-written ViT inference using GGML C library (supports quantization)
//! - `onnx`: ONNX Runtime inference via the `ort` crate
//! - `candle`: Native Rust ViT implementation using Hugging Face Candle

#[cfg(feature = "ggml")]
pub mod ggml;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "candle")]
pub mod candle;
