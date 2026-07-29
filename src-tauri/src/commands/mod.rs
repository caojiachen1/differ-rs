//! Tauri command handlers.

pub mod filesystem;
pub mod inference;
pub mod cache;

// Re-export commands for easy registration
pub use filesystem::{scan_folder, get_thumbnail, show_in_folder};
pub use inference::{load_model, get_backend_info, extract_features, search_similar, compare_folders};
pub use cache::{get_cache_stats, clear_cache};
