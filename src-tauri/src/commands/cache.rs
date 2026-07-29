//! Cache-related Tauri commands.

use crate::state::CacheStats;
use crate::services::cache_service;

/// Get cache statistics for a folder.
#[tauri::command]
pub async fn get_cache_stats(
    folder_path: String,
) -> Result<CacheStats, String> {
    tokio::task::spawn_blocking(move || {
        cache_service::get_cache_stats(&folder_path)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}

/// Clear the cache for a folder.
#[tauri::command]
pub async fn clear_cache(
    folder_path: String,
) -> Result<(), String> {
    tokio::task::spawn_blocking(move || {
        cache_service::clear_cache(&folder_path)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}
