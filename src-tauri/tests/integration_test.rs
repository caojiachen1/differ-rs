//! Integration tests for ONNX backend with real images.
//!
//! Run with: `cargo test -p dinov3-core --features onnx --test integration_test -- --nocapture`

#[cfg(feature = "onnx")]
mod onnx_tests {
    use differ_tauri_lib::dinov3::backend::onnx::OnnxBackend;
    use differ_tauri_lib::dinov3::{cosine_similarity, InferenceBackend};
    use std::path::PathBuf;
    use std::time::Instant;

    /// Path to the ONNX model.
    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("differ.NET")
            .join("Model")
            .join("model_q4.onnx")
    }

    /// Path to the test images directory.
    fn test_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("test")
    }

    /// Helper: skip test if model not found.
    fn skip_if_no_model() -> bool {
        let path = model_path();
        if !path.exists() {
            println!("ONNX model not found at {:?}, skipping", path);
            true
        } else {
            false
        }
    }

    /// Helper: load model and return backend.
    fn load_backend() -> OnnxBackend {
        let mut backend = OnnxBackend::new();
        backend
            .load_model(&model_path())
            .expect("Failed to load ONNX model");
        backend
    }

    // =====================================================================
    // Basic feature extraction tests
    // =====================================================================

    #[test]
    fn test_onnx_extract_features() {
        if skip_if_no_model() {
            return;
        }

        let backend = load_backend();
        let img_path = test_dir().join("001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg");
        if !img_path.exists() {
            println!("Test image not found, skipping");
            return;
        }

        let img_data = std::fs::read(&img_path).expect("Failed to read image");
        let features = backend
            .extract_features(&img_data)
            .expect("Failed to extract features");

        // ViT-S/16: seq_len=1029, hidden=384, total=395136
        assert!(!features.is_empty(), "Features should not be empty");
        assert_eq!(
            features.len(),
            1029 * 384,
            "Expected {} features, got {}",
            1029 * 384,
            features.len()
        );
        println!("Feature dimension: {}", features.len());

        // Verify L2 norm ≈ 1.0
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "Features should be L2-normalized, norm={}",
            norm
        );
        println!("L2 norm: {:.6}", norm);
    }

    #[test]
    fn test_onnx_similarity() {
        if skip_if_no_model() {
            return;
        }

        let backend = load_backend();
        let td = test_dir();

        let img1_path = td.join("001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg");
        let img2_path = td.join("001tqDw3gy1i2aa4kkvmkj62bc3jse8302.jpg");

        if !img1_path.exists() || !img2_path.exists() {
            println!("Test images not found, skipping");
            return;
        }

        let img1_data = std::fs::read(&img1_path).unwrap();
        let img2_data = std::fs::read(&img2_path).unwrap();

        let features1 = backend
            .extract_features(&img1_data)
            .expect("Failed on image 1");
        let features2 = backend
            .extract_features(&img2_data)
            .expect("Failed on image 2");

        // Already L2-normalized by the backend
        let similarity = cosine_similarity(&features1, &features2);
        println!("Cosine similarity (img1 vs img2): {:.4}", similarity);

        // Similarity should be between -1 and 1
        assert!(
            (-1.0..=1.0).contains(&similarity),
            "Similarity {} out of range [-1, 1]",
            similarity
        );

        // Different images should not be identical
        assert!(
            similarity < 1.0,
            "Different images should not have similarity 1.0"
        );

        // Self-similarity should be ~1.0
        let self_sim = cosine_similarity(&features1, &features1);
        assert!(
            (self_sim - 1.0).abs() < 1e-4,
            "Self-similarity should be ~1.0, got {}",
            self_sim
        );
        println!("Self-similarity: {:.6}", self_sim);
    }

    #[test]
    fn test_onnx_batch_extract() {
        if skip_if_no_model() {
            return;
        }

        let backend = load_backend();
        let td = test_dir();

        // Collect a few test image paths
        let image_names = [
            "001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg",
            "001tqDw3gy1i2aa4kkvmkj62bc3jse8302.jpg",
            "001tqDw3gy1i3wcjqeuo8j60u011ib2902.jpg",
            "001tqDw3gy1i3wcjt7kndj61uo2bcnhh02.jpg",
            "001tqDw3gy1i3wcjwivh1j60u011ihdt02.jpg",
        ];

        let mut images: Vec<Vec<u8>> = Vec::new();
        for name in &image_names {
            let path = td.join(name);
            if path.exists() {
                images.push(std::fs::read(&path).expect("Failed to read image"));
            }
        }

        if images.len() < 2 {
            println!("Not enough test images found, skipping");
            return;
        }

        let image_refs: Vec<&[u8]> = images.iter().map(|img| img.as_slice()).collect();
        let all_features = backend
            .batch_extract(&image_refs)
            .expect("Batch extract failed");

        assert_eq!(
            all_features.len(),
            images.len(),
            "Should extract features for all images"
        );

        // All features should have the same dimension
        let dim = all_features[0].len();
        for (i, f) in all_features.iter().enumerate() {
            assert_eq!(
                f.len(),
                dim,
                "Image {} has different feature dimension: {} vs {}",
                i,
                f.len(),
                dim
            );
        }

        println!(
            "Batch extracted {} images, each with {} features",
            all_features.len(),
            dim
        );

        // Compute pairwise similarities
        for i in 0..all_features.len() {
            for j in (i + 1)..all_features.len() {
                let sim = cosine_similarity(&all_features[i], &all_features[j]);
                println!("  Similarity(img{} vs img{}): {:.4}", i, j, sim);
                assert!(
                    (-1.0..=1.0).contains(&sim),
                    "Similarity out of range"
                );
            }
        }
    }

    // =====================================================================
    // Determinism test: same image → same features
    // =====================================================================

    #[test]
    fn test_onnx_determinism() {
        if skip_if_no_model() {
            return;
        }

        let backend = load_backend();
        let img_path = test_dir().join("001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg");
        if !img_path.exists() {
            println!("Test image not found, skipping");
            return;
        }

        let img_data = std::fs::read(&img_path).unwrap();

        let features_a = backend
            .extract_features(&img_data)
            .expect("First extraction failed");
        let features_b = backend
            .extract_features(&img_data)
            .expect("Second extraction failed");

        assert_eq!(features_a.len(), features_b.len());

        // Features should be identical (deterministic inference)
        let max_diff: f32 = features_a
            .iter()
            .zip(features_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, |a, b| a.max(b));

        println!("Max feature difference between two runs: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-5,
            "Features should be deterministic, max diff = {}",
            max_diff
        );
    }

    // =====================================================================
    // Performance benchmark
    // =====================================================================

    #[test]
    fn test_onnx_performance_benchmark() {
        if skip_if_no_model() {
            return;
        }

        let td = test_dir();
        if !td.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        // Collect up to 10 test images
        let image_names = [
            "001P0DUIgy1i5ycucsqe5j645n68dx6u02.jpg",
            "001tqDw3gy1i2aa4kkvmkj62bc3jse8302.jpg",
            "001tqDw3gy1i3wcjqeuo8j60u011ib2902.jpg",
            "001tqDw3gy1i3wcjt7kndj61uo2bcnhh02.jpg",
            "001tqDw3gy1i3wcjwivh1j60u011ihdt02.jpg",
            "001tqDw3gy1i58t9i8es6j63dc4hsb2g02.jpg",
            "001tqDw3ly1hwrek3lhsxj61jk1jk1kx02.jpg",
            "001tqDw3ly1hx4hhaqq8cj61zz2nzqv502.jpg",
            "001tqDw3ly1i2yxelhd43j63y84xsu1202.jpg",
            "001tqDw3ly1i5a0ntk4rtj6223222npd02.jpg",
        ];

        let mut images: Vec<(String, Vec<u8>)> = Vec::new();
        for name in &image_names {
            let path = td.join(name);
            if path.exists() {
                let data = std::fs::read(&path).expect("Failed to read image");
                images.push((name.to_string(), data));
            }
            if images.len() >= 10 {
                break;
            }
        }

        if images.is_empty() {
            println!("No test images found, skipping benchmark");
            return;
        }

        println!("\n=== ONNX Performance Benchmark ===");
        println!("Using {} images", images.len());

        // Load model (timed)
        let model_start = Instant::now();
        let backend = load_backend();
        let model_load_time = model_start.elapsed();
        println!("Model load time: {:?}", model_load_time);
        println!("GPU acceleration: {}", backend.is_using_gpu());

        // Warmup run
        let _ = backend.extract_features(&images[0].1);

        // Benchmark each image
        let mut total_inference_ms = 0.0;
        let mut feature_dim = 0;

        for (name, data) in &images {
            let start = Instant::now();
            let features = backend
                .extract_features(data)
                .expect("Feature extraction failed");
            let elapsed = start.elapsed();

            let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
            total_inference_ms += elapsed_ms;
            feature_dim = features.len();

            println!("  {} : {:.1}ms ({} features)", name, elapsed_ms, features.len());
        }

        let avg_ms = total_inference_ms / images.len() as f64;
        println!("\n--- Summary ---");
        println!("Feature dimension: {}", feature_dim);
        println!("Total inference time: {:.1}ms", total_inference_ms);
        println!("Average inference time: {:.1}ms per image", avg_ms);
        println!(
            "Throughput: {:.1} images/sec",
            1000.0 / avg_ms * images.len() as f64 / images.len() as f64
        );

        // Verify all features have the same dimension
        assert_eq!(feature_dim, 1029 * 384, "Unexpected feature dimension");

        // Basic sanity: inference should complete in reasonable time (< 30s per image on CPU)
        assert!(
            avg_ms < 30000.0,
            "Average inference time too high: {:.1}ms",
            avg_ms
        );
    }
}
