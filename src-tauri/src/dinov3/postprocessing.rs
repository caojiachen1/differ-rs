//! Post-processing utilities for DINOv3 feature vectors.
//!
//! Provides L2 normalization and cosine similarity computation for
//! comparing image feature vectors extracted by DINOv3.

/// L2-normalize a feature vector in-place.
///
/// Divides each element by the L2 (Euclidean) norm of the vector.
/// If the norm is zero or near-zero, the vector is left unchanged.
pub fn l2_normalize(features: &mut [f32]) {
    let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in features.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute cosine similarity between two L2-normalized vectors.
///
/// For L2-normalized vectors, this is equivalent to the dot product.
/// Returns a value in [-1, 1], where 1 means identical direction.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Find all candidates above a similarity threshold.
///
/// Returns a vector of (index, similarity) pairs, sorted by similarity descending.
/// Both query and candidates should be L2-normalized.
pub fn find_similar(query: &[f32], candidates: &[Vec<f32>], threshold: f32) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, candidate)| (i, cosine_similarity(query, candidate)))
        .filter(|(_, sim)| *sim >= threshold)
        .collect();
    
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Map cosine similarity from [-1, 1] to [0, 100] percentage scale.
pub fn similarity_to_percentage(similarity: f32) -> f32 {
    ((similarity + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        // norm = 5.0, so [0.6, 0.8]
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![0.6f32, 0.8];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
    
    #[test]
    fn test_find_similar() {
        let query = vec![1.0f32, 0.0];
        let candidates = vec![
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.99, 0.01],
        ];
        let results = find_similar(&query, &candidates, 0.5);
        assert_eq!(results.len(), 2); // first and third
        assert!(results[0].1 >= results[1].1); // sorted descending
    }
}
