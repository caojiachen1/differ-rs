//! Image service for folder scanning and thumbnail generation.

use std::fs;
use std::path::Path;
use std::time::SystemTime;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::codecs::jpeg::JpegEncoder;
use image::DynamicImage;
use walkdir::WalkDir;
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

/// Progress update emitted during a folder scan.
pub struct ScanProgress {
    /// Image files found so far.
    pub files_found: usize,
    /// Seconds elapsed since the scan started.
    pub elapsed_secs: f64,
}

/// Scan a folder for images.
///
/// Metadata comes straight from the directory enumeration (walkdir's
/// DirEntry metadata on Windows is the FindFirstFile data — no extra stat
/// syscall per file), which is what makes 100k-file folders scan in about
/// a second. Thumbnails are NOT generated here: the UI requests them
/// lazily per visible window via [`get_thumbnails_batch`].
pub fn scan_folder(folder_path: &str, recursive: bool) -> Result<Vec<ImageEntry>, String> {
    scan_folder_with_progress(folder_path, recursive, |_| {})
}

/// Scan a folder, reporting progress roughly every 250 ms
/// (`files_found`, `files_per_second`).
pub fn scan_folder_with_progress(
    folder_path: &str,
    recursive: bool,
    mut on_progress: impl FnMut(ScanProgress),
) -> Result<Vec<ImageEntry>, String> {
    let start = std::time::Instant::now();
    let mut last_emit = std::time::Instant::now();

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

        // Metadata from the enumeration itself — no per-file stat syscall
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Error reading metadata for {:?}: {}", entry.path(), e);
                continue;
            }
        };

        let file_path = entry.path();
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

        if last_emit.elapsed().as_millis() >= 250 {
            last_emit = std::time::Instant::now();
            on_progress(ScanProgress {
                files_found: entries.len(),
                elapsed_secs: start.elapsed().as_secs_f64(),
            });
        }
    }

    on_progress(ScanProgress {
        files_found: entries.len(),
        elapsed_secs: start.elapsed().as_secs_f64(),
    });

    // Sort by file name
    entries.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    Ok(entries)
}

/// In-memory thumbnail cache: paths -> base64 JPEG, bounded so a long
/// browsing session on a 100k-image folder cannot grow without limit.
const THUMB_CACHE_CAP: usize = 4096;

fn thumb_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, String>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, String>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Generate base64 thumbnails for a batch of images in parallel, served
/// from the bounded in-memory cache where possible (`force` bypasses the
/// cache read — used by the UI's "reload thumbnail" action; successful
/// results are written back either way). Returns one entry per input path
/// (None when the thumbnail could not be generated).
pub fn get_thumbnails_batch(paths: &[String], size: u32, force: bool) -> Vec<Option<String>> {
    use rayon::prelude::*;

    let mut cache = thumb_cache().lock().unwrap_or_else(|e| e.into_inner());
    let results: Vec<Option<String>> = paths
        .par_iter()
        .map(|p| {
            if !force {
                if let Some(hit) = cache.get(p) {
                    return Some(hit.clone());
                }
            }
            match thumbnail_base64(p, size) {
                Some(b64) => Some(b64),
                None => None, // failed lookups are not cached; retryable
            }
        })
        .collect();

    // Insert misses into the cache under a capacity bound
    for (p, r) in paths.iter().zip(&results) {
        if let Some(b64) = r {
            if !cache.contains_key(p) {
                if cache.len() >= THUMB_CACHE_CAP {
                    // Drop ~256 pseudo-oldest entries: cheap, avoids a full
                    // LRU bookkeeping structure on the hot path
                    let victims: Vec<String> = cache
                        .keys()
                        .take(256)
                        .cloned()
                        .collect();
                    for v in victims {
                        cache.remove(&v);
                    }
                }
                cache.insert(p.clone(), b64.clone());
            }
        }
    }
    results
}

/// Thumbnail dimensions preserving aspect ratio (never 0).
fn thumbnail_dims(width: u32, height: u32, size: u32) -> (u32, u32) {
    let (width, height) = (width.max(1), height.max(1));
    if width > height {
        (size.max(1), ((height as f32 / width as f32 * size as f32) as u32).max(1))
    } else {
        (((width as f32 / height as f32 * size as f32) as u32).max(1), size.max(1))
    }
}

/// Encode an RGB8 buffer as a JPEG, resizing to `tw x th` first.
fn rgb_to_jpeg_thumbnail(pixels: Vec<u8>, w: u32, h: u32, tw: u32, th: u32) -> Result<Vec<u8>, String> {
    let img = image::RgbImage::from_raw(w, h, pixels)
        .ok_or_else(|| "decoded buffer does not match dimensions".to_string())?;
    let resized = DynamicImage::ImageRgb8(img).thumbnail_exact(tw, th);
    let mut buffer = Vec::new();
    let encoder = JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buffer), 80);
    resized
        .write_with_encoder(encoder)
        .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;
    Ok(buffer)
}

/// JPEG fast path: libjpeg-turbo scaled DCT decode.
///
/// Decodes straight at 1/8, 1/4 or 1/2 of the source resolution (the
/// largest scale factor that still covers the thumbnail), so a 24MP photo
/// never materializes full-resolution pixels — roughly an order of
/// magnitude faster than decode-then-resize. `None` when the content is
/// not baseline JPEG or scaled decoding doesn't apply (lossless JPEG,
/// source too small).
fn jpeg_thumbnail_fast(bytes: &[u8], size: u32) -> Option<Vec<u8>> {
    // Sniff JPEG magic from content (extension may lie, as elsewhere here)
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    use turbojpeg::{Decompressor, PixelFormat, ScalingFactor};

    let mut decompressor = Decompressor::new().ok()?;
    let header = decompressor.read_header(bytes).ok()?;
    if header.is_lossless {
        return None;
    }
    let (tw, th) = thumbnail_dims(header.width as u32, header.height as u32, size);
    // Ascending scale: the first factor that covers the thumbnail wins
    // (cheapest decode that keeps enough resolution).
    for factor in [
        ScalingFactor::ONE_EIGHTH,
        ScalingFactor::ONE_QUARTER,
        ScalingFactor::ONE_HALF,
    ] {
        let w = factor.scale(header.width);
        let h = factor.scale(header.height);
        if (w as u32) < tw || (h as u32) < th {
            continue;
        }
        decompressor.set_scaling_factor(factor).ok()?;
        decompressor.set_fast_upsample(true).ok()?;
        let mut img = turbojpeg::Image {
            pixels: vec![0u8; w * h * 3],
            width: w,
            pitch: w * 3,
            height: h,
            format: PixelFormat::RGB,
        };
        decompressor.decompress(bytes, img.as_deref_mut()).ok()?;
        return rgb_to_jpeg_thumbnail(img.pixels, w as u32, h as u32, tw, th).ok();
    }
    None
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
    // Detect the format from the file CONTENT, not the extension: web-saved
    // files often carry mismatched extensions (png/avif/webp bytes named
    // .jpg), and image::open trusts the extension and would fail to decode.
    // Feature extraction already sniffs contents (load_from_memory), so this
    // also fixes thumbnails missing for images that extract just fine.
    let bytes = fs::read(image_path).map_err(|e| format!("Failed to read image: {}", e))?;

    // JPEG (by content): scaled-DCT decode path, ~5-10x faster
    if let Some(thumb) = jpeg_thumbnail_fast(&bytes, size) {
        return Ok(thumb);
    }

    let img = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| format!("Failed to detect image format: {}", e))?
        .decode()
        .map_err(|e| format!("Failed to decode image: {}", e))?;

    let (new_width, new_height) = thumbnail_dims(img.width(), img.height(), size);

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
    fn test_thumbnail_sniffs_mismatched_extension() {
        let tmp = tempfile::tempdir().unwrap();

        // PNG/WebP/GIF content named .jpg (web-saved files): thumbnails must
        // sniff the format from the content, not trust the extension
        let cases = [
            ("png_as_jpg.jpg", image::ImageFormat::Png),
            ("webp_as_jpg.jpg", image::ImageFormat::WebP),
            ("gif_as_jpg.jpg", image::ImageFormat::Gif),
        ];
        for (name, format) in cases {
            let path = tmp.path().join(name);
            let img = image::RgbImage::from_pixel(64, 48, image::Rgb([10, 200, 30]));
            img.save_with_format(&path, format).unwrap();
            let thumb = generate_thumbnail(path.to_str().unwrap(), 150)
                .unwrap_or_else(|e| panic!("thumbnail failed for {name}: {e}"));
            assert!(!thumb.is_empty());
        }
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

    #[test]
    fn test_jpeg_thumbnail_fast_path() {
        let tmp = tempfile::tempdir().unwrap();

        // Large JPEG: the scaled-DCT fast path must produce a decodable
        // thumbnail no larger than `size` on either axis
        let big_path = tmp.path().join("big.jpg");
        let big = image::RgbImage::from_fn(1200, 900, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        big.save(&big_path).unwrap();
        let bytes = fs::read(&big_path).unwrap();
        let fast = jpeg_thumbnail_fast(&bytes, 150)
            .expect("scaled-DCT path must handle a plain 1200x900 JPEG");
        let decoded = image::load_from_memory(&fast).unwrap();
        assert!(decoded.width() <= 150 && decoded.height() <= 150);
        assert_eq!((decoded.width(), decoded.height()), (150, 112));

        // Small JPEG: no scale factor covers the target, fast path declines
        let small_path = tmp.path().join("small.jpg");
        let small = image::RgbImage::from_pixel(100, 80, image::Rgb([1, 2, 3]));
        small.save(&small_path).unwrap();
        let small_bytes = fs::read(&small_path).unwrap();
        assert!(jpeg_thumbnail_fast(&small_bytes, 150).is_none());
        // ...and generate_thumbnail still serves it through the generic path
        let thumb = generate_thumbnail(small_path.to_str().unwrap(), 150).unwrap();
        assert!(!thumb.is_empty());

        // Non-JPEG content is declined by the fast path
        let png_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert!(jpeg_thumbnail_fast(&png_bytes, 150).is_none());
    }
}
