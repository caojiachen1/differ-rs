//! Inference-related Tauri commands.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use crate::dinov3::InferenceBackend;
use crate::state::{AppState, ProgressInfo, SimilarityResult};
use crate::services::similarity_service;

/// Event name for extraction progress updates.
pub const PROGRESS_EVENT: &str = "extraction-progress";

/// Resolve the model file for a backend by probing well-known locations.
///
/// Order: explicit path > env var > models/ next to the executable >
/// the workspace models/ directory (development).
pub fn resolve_model_path(backend: &str, explicit: Option<&str>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        return if path.exists() {
            Ok(path)
        } else {
            Err(format!("Model file not found: {}", p))
        };
    }

    let (env_var, file_names): (&str, &[&str]) = match backend {
        "ggml" => ("GGML_MODEL_PATH", &["dinov3_vits16.bin"]),
        // Prefer the unquantized F32 model, fall back to Q4
        _ => ("ONNX_MODEL_PATH", &["model.onnx", "model_q4.onnx"]),
    };

    if let Ok(p) = std::env::var(env_var) {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Ok(path);
        }
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("models"));
        }
    }
    // Development fallback: differ-rust/models relative to this crate
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models"));

    for root in &roots {
        for name in file_names {
            let candidate = root.join(name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    Err(format!(
        "No {} model found. Searched {:?} for {:?}. Set {} or configure the path in Settings.",
        backend, roots, file_names, env_var
    ))
}

/// Create and load an inference backend ("ggml" or "onnx").
pub fn create_backend(backend: &str, model_path: &Path) -> Result<Box<dyn InferenceBackend>, String> {
    let mut b: Box<dyn InferenceBackend> = match backend {
        "ggml" => Box::new(crate::dinov3::backend::ggml::GgmlBackend::new()),
        "onnx" => Box::new(crate::dinov3::backend::onnx::OnnxBackend::new()),
        other => return Err(format!("Unknown backend: {} (expected \"ggml\" or \"onnx\")", other)),
    };
    b.load_model(model_path)
        .map_err(|e| format!("Failed to load {} model: {:#}", backend, e))?;
    Ok(b)
}

/// Load the DINOv3 model with the requested backend ("ggml" by default).
#[tauri::command]
pub async fn load_model(
    backend: Option<String>,
    model_path: Option<String>,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let backend_kind = backend.unwrap_or_else(|| "ggml".to_string());
    let resolved = resolve_model_path(&backend_kind, model_path.as_deref())?;

    let backend_arc = Arc::clone(&state.backend);
    let kind = backend_kind.clone();
    let resolved_clone = resolved.clone();

    tokio::task::spawn_blocking(move || {
        let loaded = create_backend(&kind, &resolved_clone)?;
        let mut guard = backend_arc
            .lock()
            .map_err(|e| format!("Failed to lock backend: {}", e))?;
        *guard = Some(loaded);
        Ok::<_, String>(())
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))??;

    *state.backend_name.lock().map_err(|e| e.to_string())? = backend_kind.clone();
    Ok(format!(
        "{} model loaded: {}",
        backend_kind,
        resolved.display()
    ))
}

/// Get information about the currently loaded backend.
#[tauri::command]
pub async fn get_backend_info(state: State<'_, AppState>) -> Result<String, String> {
    let name = state.backend_name.lock().map_err(|e| e.to_string())?.clone();
    let guard = state.backend.lock().map_err(|e| e.to_string())?;
    match guard.as_ref() {
        Some(backend) => Ok(format!("{} - {}", name, backend.model_info().name)),
        None => Ok("No model loaded".to_string()),
    }
}

/// Extract features for all images in a folder.
///
/// Emits `extraction-progress` events while running; reuses the per-folder
/// cache and extracts misses with the parallel pipeline.
#[tauri::command]
pub async fn extract_features(
    folder_path: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<ProgressInfo, String> {
    let backend_arc = Arc::clone(&state.backend);

    let result = tokio::task::spawn_blocking(move || {
        let guard = backend_arc
            .lock()
            .map_err(|e| format!("Failed to lock backend: {}", e))?;
        let backend = guard
            .as_ref()
            .ok_or_else(|| "Model not loaded. Please load a model first.".to_string())?;

        let (entries, cache_hits, cache_misses) =
            similarity_service::extract_features_for_folder_with_progress(
                backend.as_ref(),
                &folder_path,
                true, // recursive
                |progress| {
                    let _ = app.emit(PROGRESS_EVENT, &progress);
                },
            )?;

        let total = entries.len();
        Ok::<_, String>(ProgressInfo {
            total,
            processed: total,
            cache_hits,
            cache_misses,
            message: format!(
                "Extracted features from {} images (Cache: {} hits, {} misses).",
                total, cache_hits, cache_misses
            ),
        })
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    result
}

/// Search for similar images.
#[tauri::command]
pub async fn search_similar(
    source_path: String,
    folder_path: String,
    threshold: f32,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<SimilarityResult>, String> {
    let backend_arc = Arc::clone(&state.backend);

    let result = tokio::task::spawn_blocking(move || {
        let guard = backend_arc
            .lock()
            .map_err(|e| format!("Failed to lock backend: {}", e))?;
        let backend = guard
            .as_ref()
            .ok_or_else(|| "Model not loaded. Please load a model first.".to_string())?;

        similarity_service::search_similar_in_folder(
            backend.as_ref(),
            &source_path,
            &folder_path,
            threshold,
            |progress| {
                let _ = app.emit(PROGRESS_EVENT, &progress);
            },
        )
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    result
}

/// Compare images between two folders.
#[tauri::command]
pub async fn compare_folders(
    source_folder: String,
    target_folder: String,
    threshold: f32,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<SimilarityResult>, String> {
    let backend_arc = Arc::clone(&state.backend);

    let result = tokio::task::spawn_blocking(move || {
        let guard = backend_arc
            .lock()
            .map_err(|e| format!("Failed to lock backend: {}", e))?;
        let backend = guard
            .as_ref()
            .ok_or_else(|| "Model not loaded. Please load a model first.".to_string())?;

        similarity_service::compare_folders(
            backend.as_ref(),
            &source_folder,
            &target_folder,
            threshold,
            |progress| {
                let _ = app.emit(PROGRESS_EVENT, &progress);
            },
        )
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    result
}
