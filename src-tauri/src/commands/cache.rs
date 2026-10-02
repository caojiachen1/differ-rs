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
///
/// Also drops the folder's in-memory feature snapshot — its db identity no
/// longer matches after deletion, but removing it here avoids one stale
/// lookup attempt.
#[tauri::command]
pub async fn clear_cache(
    folder_path: String,
    state: tauri::State<'_, crate::state::AppState>,
) -> Result<(), String> {
    let snapshots_arc = std::sync::Arc::clone(&state.folder_snapshots);
    tokio::task::spawn_blocking(move || {
        let result = cache_service::clear_cache(&folder_path);
        if let Ok(mut snapshots) = snapshots_arc.lock() {
            snapshots.remove(&folder_path);
        }
        result
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
}
