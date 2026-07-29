//! Image service for folder scanning and thumbnail generation.

use std::fs;
use std::path::Path;
use std::time::SystemTime;
use image::codecs::jpeg::JpegEncoder;
use image::DynamicImage;
use rayon::prelude::*;
use walkdir::WalkDir;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use crate::state::ImageEntry;

/// Supported image file extensions.
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "gif", "webp"];

/// Check if a file is a supported image based on extension.
fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Scan a folder for images.
///
/// # Arguments
/// * `folder_path` - Path to the folder to scan
/// * `recursive` - Whether to scan subfolders recursively
/// * `generate_thumbnails` - Whether to generate thumbnails for each image
///
/// # Returns
/// A vector of ImageEntry containing metadata and optional thumbnails.
pub fn scan_folder(folder_path: &str, recursive: bool, generate_thumbnails: bool) -> Result<Vec<ImageEntry>, String> {
    let path = Path::new(folder_path);
    if !path.exists() {
        return Err(format!("Folder does not exist: {}", folder_path));
    }
    if !path.is_dir() {
        return Err(format!("Path is not a directory: {}", folder_path));
    }

    let mut entries = Vec::new();

    let walker = if recursive {
        WalkDir::new(path).into_iter()
    } else {
        WalkDir::new(path).max_depth(1).into_iter()
    };

    for entry in walker.filter_entry(|e| {
        // Skip hidden directories, but never filter out the scan root itself
        e.depth() == 0 || !e.file_name().to_string_lossy().starts_with('.')
    }) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                log::warn!("Error walking directory: {}", e);
                continue;
            }
        };

        if !entry.file_type().is_file() {
            continue;
        }

        if !is_image_file(entry.path()) {
            continue;
        }

        let file_path = entry.path();
        let metadata = match fs::metadata(file_path) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Error reading metadata for {:?}: {}", file_path, e);
                continue;
            }
        };

        let modified = metadata.modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        entries.push(ImageEntry {
            path: file_path.to_string_lossy().to_string(),
            file_name: file_path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            file_size: metadata.len(),
            modified,
            thumbnail: None,
        });
    }

    // Generate thumbnails in parallel across all cores (decode is the bottleneck)
    if generate_thumbnails {
        entries.par_iter_mut().for_each(|entry| {
            match generate_thumbnail(&entry.path, 150) {
                Ok(data) => entry.thumbnail = Some(BASE64.encode(data)),
                Err(e) => log::warn!("Error generating thumbnail for {}: {}", entry.path, e),
            }
        });
    }

    // Sort by file name
    entries.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    Ok(entries)
}

/// Generate a thumbnail for an image.
///
/// # Arguments
/// * `image_path` - Path to the image file
/// * `size` - Maximum dimension (width or height) of the thumbnail
///
/// # Returns
/// JPEG-encoded thumbnail data as bytes.
pub fn generate_thumbnail(image_path: &str, size: u32) -> Result<Vec<u8>, String> {
    let img = image::open(image_path)
        .map_err(|e| format!("Failed to open image: {}", e))?;

    // Calculate new dimensions maintaining aspect ratio (never 0)
    let (width, height) = (img.width().max(1), img.height().max(1));
    let (new_width, new_height) = if width > height {
        (size.max(1), ((height as f32 / width as f32 * size as f32) as u32).max(1))
    } else {
        (((width as f32 / height as f32 * size as f32) as u32).max(1), size.max(1))
    };

    // Fast integer downsampling (much faster than Lanczos3, fine for thumbnails)
    let resized = img.thumbnail_exact(new_width, new_height);

    // JPEG cannot encode alpha: always convert to RGB8 first
    // (this used to fail for RGBA PNG/WebP images)
    let rgb = DynamicImage::ImageRgb8(resized.to_rgb8());

    let mut buffer = Vec::new();
    let encoder = JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buffer), 80);
    rgb.write_with_encoder(encoder)
        .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;

    Ok(buffer)
}

/// Generate a base64-encoded JPEG thumbnail; None on failure.
pub fn thumbnail_base64(image_path: &str, size: u32) -> Option<String> {
    match generate_thumbnail(image_path, size) {
        Ok(data) => Some(BASE64.encode(data)),
        Err(e) => {
            log::warn!("Error generating thumbnail for {}: {}", image_path, e);
            None
        }
    }
}

/// Load an image file and return its raw bytes.
pub fn load_image_bytes(image_path: &str) -> Result<Vec<u8>, String> {
    fs::read(image_path)
        .map_err(|e| format!("Failed to read image file: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file(Path::new("test.jpg")));
        assert!(is_image_file(Path::new("test.JPEG")));
        assert!(is_image_file(Path::new("test.png")));
        assert!(is_image_file(Path::new("test.bmp")));
        assert!(is_image_file(Path::new("test.gif")));
        assert!(is_image_file(Path::new("test.webp")));
        assert!(!is_image_file(Path::new("test.txt")));
        assert!(!is_image_file(Path::new("test.pdf")));
        assert!(!is_image_file(Path::new("test")));
    }

    #[test]
    fn test_thumbnail_rgba_and_extreme_aspect() {
        let tmp = tempfile::tempdir().unwrap();

        // RGBA PNG (used to fail: JPEG cannot encode alpha)
        let rgba_path = tmp.path().join("rgba.png");
        let rgba = image::RgbaImage::from_pixel(64, 48, image::Rgba([255, 0, 0, 128]));
        rgba.save(&rgba_path).unwrap();
        let thumb = generate_thumbnail(rgba_path.to_str().unwrap(), 150).unwrap();
        assert!(!thumb.is_empty());

        // Extreme aspect ratio (used to compute a 0-height thumbnail)
        let wide_path = tmp.path().join("wide.png");
        let wide = image::RgbImage::new(2000, 3);
        wide.save(&wide_path).unwrap();
        let thumb = generate_thumbnail(wide_path.to_str().unwrap(), 150).unwrap();
        assert!(!thumb.is_empty());
    }
}
