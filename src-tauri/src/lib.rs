//! differ-tauri: Tauri 2.x application for image similarity search.
//!
//! This is the main library crate that sets up the Tauri application,
//! registers commands, and manages application state.

pub mod commands;
pub mod dinov3;
pub mod services;
pub mod state;

use std::sync::Arc;
use state::AppState;
use commands::{
    scan_folder, get_thumbnail, show_in_folder,
    load_model, get_backend_info, extract_features, search_similar, compare_folders,
    get_cache_stats, clear_cache,
};
use commands::inference::{create_backend, resolve_model_path};

/// Run the Tauri application.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    log::info!("Starting Image Similarity Finder...");

    let app_state = AppState::new();

    // Auto-load a model in the background on startup:
    // prefer the GGML backend (CUDA), fall back to ONNX.
    let backend_arc = Arc::clone(&app_state.backend);
    let name_for_thread = Arc::clone(&app_state.backend_name);
    std::thread::spawn(move || {
        for kind in ["ggml", "onnx"] {
            let path = match resolve_model_path(kind, None) {
                Ok(p) => p,
                Err(e) => {
                    log::warn!("{}", e);
                    continue;
                }
            };
            log::info!("Auto-loading {} model from {}...", kind, path.display());
            match create_backend(kind, &path) {
                Ok(backend) => {
                    match backend_arc.lock() {
                        Ok(mut guard) => {
                            *guard = Some(backend);
                            if let Ok(mut n) = name_for_thread.lock() {
                                *n = kind.to_string();
                            }
                            log::info!("{} model auto-loaded successfully.", kind);
                        }
                        Err(e) => log::error!("Failed to lock backend after model load: {}", e),
                    }
                    return;
                }
                Err(e) => log::error!("Failed to auto-load {} model: {}", kind, e),
            }
        }
        log::warn!("No model auto-loaded. Load one manually via Settings.");
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        // The main window starts hidden ("visible": false) to avoid the white
        // flash on startup: maximize + show it only once the page has loaded.
        .on_page_load(|webview, _payload| {
            let window = webview.window();
            let _ = window.maximize();
            let _ = window.show();
            let _ = window.set_focus();
        })
        .invoke_handler(tauri::generate_handler![
            // Filesystem commands
            scan_folder,
            get_thumbnail,
            show_in_folder,
            // Inference commands
            load_model,
            get_backend_info,
            extract_features,
            search_similar,
            compare_folders,
            // Cache commands
            get_cache_stats,
            clear_cache,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
