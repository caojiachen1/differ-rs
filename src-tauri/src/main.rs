//! Entry point for the differ-tauri application.

// Hide the console window in release builds on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    differ_tauri_lib::run();
}
