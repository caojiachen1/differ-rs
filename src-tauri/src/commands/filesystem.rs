//! Filesystem-related Tauri commands.

use crate::state::ImageEntry;
use crate::services::image_service;

/// Scan a folder for images.
#[tauri::command]
pub async fn scan_folder(
    path: String,
    recursive: bool,
) -> Result<Vec<ImageEntry>, String> {
    // Run blocking operation in a separate thread
    tokio::task::spawn_blocking(move || {
        image_service::scan_folder(&path, recursive, true)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Get a thumbnail for an image.
#[tauri::command]
pub async fn get_thumbnail(
    path: String,
    size: u32,
) -> Result<Vec<u8>, String> {
    tokio::task::spawn_blocking(move || {
        image_service::generate_thumbnail(&path, size)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Read a full image file and return its bytes as base64.
#[tauri::command]
pub async fn read_image(path: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        use base64::Engine;
        let bytes = std::fs::read(&path).map_err(|e| format!("Failed to read image: {}", e))?;
        Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Reveal a file in the system file manager (Explorer / Finder / xdg-open).
#[tauri::command]
pub async fn show_in_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        // Explorer wants backslashes; "/select,<path>" highlights the file
        let win_path = path.replace('/', "\\");
        std::process::Command::new("explorer")
            .arg(format!("/select,{}", win_path))
            .spawn()
            .map_err(|e| format!("Failed to open Explorer: {}", e))?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .args(["-R", &path])
            .spawn()
            .map_err(|e| format!("Failed to open Finder: {}", e))?;
    }
    #[cfg(target_os = "linux")]
    {
        let dir = std::path::Path::new(&path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or(path.clone());
        std::process::Command::new("xdg-open")
            .arg(dir)
            .spawn()
            .map_err(|e| format!("Failed to open file manager: {}", e))?;
    }
    Ok(())
}
