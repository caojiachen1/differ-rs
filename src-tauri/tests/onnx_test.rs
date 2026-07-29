//! Integration tests for the ONNX Runtime backend.
//!
//! These tests require the ONNX model files to be present.
//! Run with: `cargo test -p dinov3-core --features onnx --test onnx_test`

#[cfg(feature = "onnx")]
mod onnx_tests {
    use std::path::PathBuf;
    use differ_tauri_lib::dinov3::InferenceBackend;
    use differ_tauri_lib::dinov3::backend::onnx::OnnxBackend;

    /// Path to the C# project's model directory (contains model_q4.onnx).
    fn model_path() -> PathBuf {
        // The model is in the C# project's Model directory
        // From crates/dinov3-core -> differ-rust -> differ.NET (workspace root) -> differ.NET/Model
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("..").join("differ.NET").join("Model").join("model_q4.onnx")
    }

    /// A test image from the test directory.
    fn test_image_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("test").join("001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg")
    }

    #[test]
    fn test_model_path_exists() {
        let path = model_path();
        assert!(
            path.exists(),
            "ONNX model not found at {:?}. The C# project's Model directory should contain model_q4.onnx.",
            path
        );
    }

    #[test]
    fn test_load_model_and_extract_features() {
        let model_path = model_path();
        if !model_path.exists() {
            eprintln!("Skipping test: model not found at {:?}", model_path);
            return;
        }

        let mut backend = OnnxBackend::new();
        backend.load_model(&model_path).expect("Failed to load ONNX model");

        // Verify model info
        let info = backend.model_info();
        assert_eq!(info.backend, "onnx");
        assert!(info.name.contains("ONNX"));
        assert!(info.quantization.is_some());

        // Load a test image
        let image_path = test_image_path();
        if !image_path.exists() {
            eprintln!("Skipping feature extraction: test image not found at {:?}", image_path);
            return;
        }

        let image_data = std::fs::read(&image_path).expect("Failed to read test image");
        let features = backend
            .extract_features(&image_data)
            .expect("Failed to extract features");

        // DINOv3 ViT-S/16 output:
        // seq_len = 1 (CLS) + 1024 (patches) + 4 (registers) = 1029
        // hidden_size = 384
        // Total features = 1029 * 384 = 395136
        let expected_len = 1029 * 384;
        assert_eq!(
            features.len(),
            expected_len,
            "Expected {} features, got {}",
            expected_len,
            features.len()
        );

        // Verify L2 normalization (norm should be ~1.0)
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Features should be L2-normalized, but norm = {}",
            norm
        );

        println!("ONNX backend using GPU: {}", backend.is_using_gpu());
        println!("Extracted {} features, norm = {:.6}", features.len(), norm);
        println!("Model info: {:?}", info);
    }

    #[test]
    fn test_extract_features_from_two_images() {
        let model_path = model_path();
        if !model_path.exists() {
            eprintln!("Skipping test: model not found");
            return;
        }

        let mut backend = OnnxBackend::new();
        backend.load_model(&model_path).expect("Failed to load model");

        let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..").join("test");
        if !test_dir.exists() {
            eprintln!("Skipping test: test directory not found");
            return;
        }

        // Read two different images
        let img1_path = test_dir.join("001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg");
        let img2_path = test_dir.join("001tqDw3gy1i2aa4kkvmkj62bc3jse8302.jpg");

        if !img1_path.exists() || !img2_path.exists() {
            eprintln!("Skipping test: test images not found");
            return;
        }

        let img1_data = std::fs::read(&img1_path).unwrap();
        let img2_data = std::fs::read(&img2_path).unwrap();

        let features1 = backend.extract_features(&img1_data).expect("Failed on image 1");
        let features2 = backend.extract_features(&img2_data).expect("Failed on image 2");

        // Features should have the same dimension
        assert_eq!(features1.len(), features2.len());

        // Cosine similarity of L2-normalized vectors
        let similarity: f32 = features1.iter().zip(features2.iter()).map(|(a, b)| a * b).sum();
        println!("Cosine similarity between two different images: {:.4}", similarity);

        // Different images should have similarity < 1.0
        assert!(similarity < 1.0, "Different images should not be identical");
    }

    #[test]
    fn test_load_nonexistent_model() {
        let mut backend = OnnxBackend::new();
        let result = backend.load_model(&PathBuf::from("/nonexistent/model.onnx"));
        assert!(result.is_err(), "Should fail when model file doesn't exist");
    }

    #[test]
    fn test_extract_without_loading() {
        let backend = OnnxBackend::new();
        let result = backend.extract_features(&[0u8; 100]);
        assert!(result.is_err(), "Should fail when model is not loaded");
    }
}
