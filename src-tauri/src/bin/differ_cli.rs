//! differ-cli — command-line interface for the DINOv3 image-similarity
//! engine. Same engine, services and cache as the desktop app.
//!
//! Built to be script/AI-friendly:
//! - one JSON document on stdout with `--json` (progress goes to stderr)
//! - deterministic, stable field names; results sorted by similarity
//! - non-zero exit code on failure, 2 on usage errors (clap)
//! - no interactive prompts; models auto-resolve and auto-download
//!
//! Examples:
//!   differ-cli scan D:\photos --json
//!   differ-cli extract D:\photos
//!   differ-cli similar D:\photos\ref.jpg D:\photos --threshold 0.9 --json
//!   differ-cli compare D:\libA D:\libB --threshold 0.9
//!   differ-cli cache stats D:\photos

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use differ_tauri_lib::commands::inference::{
    create_backend, download_model_for_backend, resolve_model_path,
};
use differ_tauri_lib::dinov3::InferenceBackend;
use differ_tauri_lib::services::{cache_service, similarity_service};
use differ_tauri_lib::state::ProgressInfo;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    /// GGML (CUDA when available; recommended, fastest)
    Ggml,
    /// ONNX Runtime (DirectML/CUDA/TensorRT)
    Onnx,
    /// Candle (CUDA)
    Candle,
}

impl BackendKind {
    fn as_str(self) -> &'static str {
        match self {
            BackendKind::Ggml => "ggml",
            BackendKind::Onnx => "onnx",
            BackendKind::Candle => "candle",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Tier {
    /// 256x256 input (261 tokens) — default, best speed/quality balance
    Fast,
    /// 224x224 input (201 tokens) — max GPU throughput
    Ultra,
    /// 518x518 input (1029 tokens) — accuracy-first
    High,
}

#[derive(Subcommand)]
enum CacheOp {
    /// Show feature-cache statistics for a folder
    Stats { folder: PathBuf },
    /// Delete the feature cache of a folder
    Clear { folder: PathBuf },
}

#[derive(Subcommand)]
enum Command {
    /// List images in a folder (metadata only, never decodes pixels)
    Scan {
        path: PathBuf,
        /// Scan subfolders recursively (default: true)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        recursive: bool,
    },
    /// Extract features for a folder into its per-folder cache
    Extract {
        path: PathBuf,
        /// Scan subfolders recursively (default: true)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        recursive: bool,
        /// Also write all pooled features (path -> vector) to this JSON file
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// Find images in a folder similar to a source image
    Similar {
        source: PathBuf,
        folder: PathBuf,
        /// Minimum cosine similarity (0.0-1.0; app default 0.90)
        #[arg(long, default_value_t = 0.9)]
        threshold: f32,
        /// Keep only the first N results (0 = all)
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Compare two folders: best match of every target image against the
    /// source folder
    Compare {
        source_folder: PathBuf,
        target_folder: PathBuf,
        /// Minimum cosine similarity (0.0-1.0; app default 0.90)
        #[arg(long, default_value_t = 0.9)]
        threshold: f32,
    },
    /// Feature-cache operations
    Cache {
        #[command(subcommand)]
        op: CacheOp,
    },
}

#[derive(Parser)]
#[command(
    name = "differ-cli",
    version,
    about = "DINOv3 image similarity: scan folders, extract features, find near-duplicates",
    after_help = "Exit codes: 0 success, 1 runtime error, 2 usage error.\n\
                  Add --json for machine-readable stdout (progress goes to stderr).\n\
                  Tier/model flags apply to the ggml backend; other backends ignore them."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Inference backend
    #[arg(long, value_enum, default_value_t = BackendKind::Ggml, global = true)]
    backend: BackendKind,

    /// Explicit model file path (default: auto-resolve then auto-download)
    #[arg(long, global = true)]
    model: Option<PathBuf>,

    /// GGML input resolution tier (ggml backend only)
    #[arg(long, value_enum, default_value_t = Tier::Fast, global = true)]
    tier: Tier,

    /// Machine-readable JSON on stdout
    #[arg(long, global = true)]
    json: bool,

    /// No progress output on stderr
    #[arg(long, global = true)]
    quiet: bool,
}

/// Rate-limited progress reporter on stderr.
struct Progress {
    label: &'static str,
    start: Instant,
    last: Instant,
    quiet: bool,
}

impl Progress {
    fn new(label: &'static str, quiet: bool) -> Self {
        Self { label, start: Instant::now(), last: Instant::now(), quiet }
    }

    fn tick(&mut self, p: &ProgressInfo) {
        if self.quiet || self.last.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.last = Instant::now();
        let rate = p.images_per_second;
        eprintln!(
            "[{}] {}/{} ({:.0} img/s, {:.0}s)",
            self.label, p.processed, p.total, rate,
            self.start.elapsed().as_secs_f64()
        );
    }
}

fn build_backend(backend: BackendKind, model: &Option<PathBuf>, tier: Tier) -> Result<Box<dyn InferenceBackend>> {
    let kind = backend.as_str();
    // Tier selection for the ggml backend is read from the environment at
    // model creation; set it before constructing the backend.
    if backend == BackendKind::Ggml {
        let value = match tier {
            Tier::Fast => "fast",
            Tier::Ultra => "ultra",
            Tier::High => "high",
        };
        std::env::set_var("GGML_VIT_TIER", value);
    }

    let resolved = match resolve_model_path(kind, model.as_ref().and_then(|p| p.to_str())) {
        Ok(p) => p,
        Err(_) if model.is_none() => download_model_for_backend(kind)
            .map_err(anyhow::Error::msg)
            .context("Model download failed")?,
        Err(e) => return Err(anyhow::Error::msg(e)),
    };
    let backend = create_backend(kind, &resolved)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("Failed to load {} model from {}", kind, resolved.display()))?;
    Ok(backend)
}

fn scan_progress_line(elapsed: f64, files: usize) -> String {
    format!("{files} images, {elapsed:.2} s ({:.0} files/s)", files as f64 / elapsed.max(1e-9))
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Scan { path, recursive } => {
            let folder = path.to_string_lossy().to_string();
            let t = Instant::now();
            let entries = differ_tauri_lib::services::image_service::scan_folder(&folder, recursive)
                .map_err(anyhow::Error::msg)?;
            let elapsed = t.elapsed().as_secs_f64();

            if cli.json {
                #[derive(Serialize)]
                struct Entry<'a> {
                    path: &'a str,
                    file_name: &'a str,
                    file_size: u64,
                    modified: u64,
                }
                let images: Vec<Entry> = entries
                    .iter()
                    .map(|e| Entry {
                        path: &e.path,
                        file_name: &e.file_name,
                        file_size: e.file_size,
                        modified: e.modified,
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "folder": folder,
                        "count": entries.len(),
                        "elapsed_secs": elapsed,
                        "images": images,
                    })
                );
            } else {
                println!("{}", scan_progress_line(elapsed, entries.len()));
            }
        }

        Command::Extract { path, recursive, dump } => {
            let backend = build_backend(cli.backend, &cli.model, cli.tier)?;
            let backend = backend.as_ref();
            let folder = path.to_string_lossy().to_string();
            let mut progress = Progress::new("extract", cli.quiet);
            let t = Instant::now();
            let (results, hits, misses) =
                similarity_service::extract_features_for_folder_with_progress(
                    backend,
                    &folder,
                    recursive,
                    |p| progress.tick(&p),
                )
                .map_err(anyhow::Error::msg)?;
            let elapsed = t.elapsed().as_secs_f64();
            // Report the dimension of the stored (pooled) vectors, not the
            // raw model output
            let dim = results.first().map(|(_, f)| f.len()).unwrap_or(0);

            if let Some(dump_path) = &dump {
                #[derive(Serialize)]
                struct Features<'a> {
                    feature_dim: usize,
                    images: Vec<(&'a str, &'a [f32])>,
                }
                let images: Vec<(&str, &[f32])> = results
                    .iter()
                    .map(|(e, f)| (e.path.as_str(), f.as_slice()))
                    .collect();
                let doc = Features { feature_dim: dim, images };
                let json = serde_json::to_string(&doc)?;
                std::fs::write(dump_path, json)
                    .with_context(|| format!("Failed to write {:?}", dump_path))?;
                eprintln!("[dump] wrote {} feature vectors to {}", results.len(), dump_path.display());
            }

            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "folder": folder,
                        "count": results.len(),
                        "cache_hits": hits,
                        "cache_misses": misses,
                        "images_per_second": results.len() as f64 / elapsed.max(1e-9),
                        "elapsed_secs": elapsed,
                        "feature_dim": dim,
                    })
                );
            } else {
                println!(
                    "Extracted {} images in {:.2} s ({:.0} img/s) | cache: {} hits, {} misses | feature dim {}",
                    results.len(),
                    elapsed,
                    results.len() as f64 / elapsed.max(1e-9),
                    hits,
                    misses,
                    dim
                );
            }
        }

        Command::Similar { source, folder, threshold, limit } => {
            let backend = build_backend(cli.backend, &cli.model, cli.tier)?;
            let backend = backend.as_ref();
            let source_str = source.to_string_lossy().to_string();
            let folder_str = folder.to_string_lossy().to_string();
            let mut progress = Progress::new("search", cli.quiet);
            let t = Instant::now();
            let mut results = similarity_service::search_similar_in_folder(
                backend,
                &source_str,
                &folder_str,
                threshold,
                |p| progress.tick(&p),
            )
            .map_err(anyhow::Error::msg)?;
            if limit > 0 && results.len() > limit {
                results.truncate(limit);
            }
            let elapsed = t.elapsed().as_secs_f64();

            if cli.json {
                #[derive(Serialize)]
                struct Match<'a> {
                    path: &'a str,
                    file_name: &'a str,
                    similarity: f32,
                }
                let items: Vec<Match> = results
                    .iter()
                    .map(|r| Match { path: &r.path, file_name: &r.file_name, similarity: r.similarity })
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "source": source_str,
                        "folder": folder_str,
                        "threshold": threshold,
                        "count": items.len(),
                        "elapsed_secs": elapsed,
                        "results": items,
                    })
                );
            } else if results.is_empty() {
                println!("No matches above threshold {} in {}", threshold, folder_str);
            } else {
                println!(
                    "{} matches for {} (threshold {}):",
                    results.len(), source_str, threshold
                );
                for (i, r) in results.iter().enumerate() {
                    println!("  {:>3}. {:.4}  {}", i + 1, r.similarity, r.path);
                }
            }
        }

        Command::Compare { source_folder, target_folder, threshold } => {
            let backend = build_backend(cli.backend, &cli.model, cli.tier)?;
            let backend = backend.as_ref();
            let src = source_folder.to_string_lossy().to_string();
            let tgt = target_folder.to_string_lossy().to_string();
            let mut progress = Progress::new("compare", cli.quiet);
            let t = Instant::now();
            let results = similarity_service::compare_folders(
                backend,
                &src,
                &tgt,
                threshold,
                |p| progress.tick(&p),
            )
            .map_err(anyhow::Error::msg)?;
            let elapsed = t.elapsed().as_secs_f64();

            if cli.json {
                #[derive(Serialize)]
                struct Match<'a> {
                    target_path: &'a str,
                    target_file_name: &'a str,
                    best_similarity: f32,
                }
                let items: Vec<Match> = results
                    .iter()
                    .map(|r| Match {
                        target_path: &r.path,
                        target_file_name: &r.file_name,
                        best_similarity: r.similarity,
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "source_folder": src,
                        "target_folder": tgt,
                        "threshold": threshold,
                        "count": items.len(),
                        "elapsed_secs": elapsed,
                        "matches": items,
                    })
                );
            } else if results.is_empty() {
                println!("No matches above threshold {} between the folders", threshold);
            } else {
                println!(
                    "{} target images matched (threshold {}):",
                    results.len(), threshold
                );
                for r in &results {
                    println!("  {:.4}  {}", r.similarity, r.path);
                }
            }
        }

        Command::Cache { op } => match op {
            CacheOp::Stats { folder } => {
                let folder = folder.to_string_lossy().to_string();
                let stats = cache_service::get_cache_stats(&folder).map_err(anyhow::Error::msg)?;
                if cli.json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "folder": folder,
                            "cached_count": stats.cached_count,
                            "cache_size_bytes": stats.cache_size_bytes,
                            "is_valid": stats.is_valid,
                        })
                    );
                } else {
                    println!(
                        "Cache for {}: {} entries, {:.1} MB",
                        folder,
                        stats.cached_count,
                        stats.cache_size_bytes as f64 / (1024.0 * 1024.0)
                    );
                }
            }
            CacheOp::Clear { folder } => {
                let folder = folder.to_string_lossy().to_string();
                cache_service::clear_cache(&folder).map_err(anyhow::Error::msg)?;
                if cli.json {
                    println!("{}", serde_json::json!({ "folder": folder, "cleared": true }));
                } else {
                    println!("Cleared cache for {}", folder);
                }
            }
        },
    }
    Ok(())
}
