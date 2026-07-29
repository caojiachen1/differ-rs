/**
 * Tauri API wrapper - provides typed access to backend commands
 */
import { invoke } from '@tauri-apps/api/core';

// Types matching Rust backend structures
export interface ImageEntry {
  path: string;
  file_name: string;
  file_size: number;
  modified: number;
  thumbnail: string | null;
}

export interface ProgressInfo {
  total: number;
  processed: number;
  cache_hits: number;
  cache_misses: number;
  message: string;
}

export interface SimilarityResult {
  path: string;
  file_name: string;
  similarity: number;
  thumbnail: string | null;
}

export interface CacheStats {
  cached_count: number;
  cache_size_bytes: number;
  is_valid: boolean;
}

export const api = {
  /**
   * Scan a folder for images
   */
  scanFolder: (path: string, recursive: boolean): Promise<ImageEntry[]> =>
    invoke<ImageEntry[]>('scan_folder', { path, recursive }),

  /**
   * Get a thumbnail for an image (returns raw bytes)
   */
  getThumbnail: (path: string, size: number): Promise<number[]> =>
    invoke<number[]>('get_thumbnail', { path, size }),

  /**
   * Read the full image file as base64
   */
  readImage: (path: string): Promise<string> =>
    invoke<string>('read_image', { path }),

  /**
   * Reveal a file in the system file manager (Explorer)
   */
  showInFolder: (path: string): Promise<void> =>
    invoke<void>('show_in_folder', { path }),

  /**
   * Load the DINOv3 model with the given backend ('ggml' | 'onnx' | 'candle')
   */
  loadModel: (backend?: 'ggml' | 'onnx' | 'candle', modelPath?: string): Promise<string> =>
    invoke<string>('load_model', { backend: backend || null, modelPath: modelPath || null }),

  /**
   * Get info about the currently loaded backend/model
   */
  getBackendInfo: (): Promise<string> =>
    invoke<string>('get_backend_info'),

  /**
   * Extract features for all images in a folder
   */
  extractFeatures: (folderPath: string): Promise<ProgressInfo> =>
    invoke<ProgressInfo>('extract_features', { folderPath }),

  /**
   * Search for similar images
   */
  searchSimilar: (sourcePath: string, folderPath: string, threshold: number): Promise<SimilarityResult[]> =>
    invoke<SimilarityResult[]>('search_similar', { sourcePath, folderPath, threshold }),

  /**
   * Compare images between two folders
   */
  compareFolders: (source: string, target: string, threshold: number): Promise<SimilarityResult[]> =>
    invoke<SimilarityResult[]>('compare_folders', { sourceFolder: source, targetFolder: target, threshold }),

  /**
   * Get cache statistics for a folder
   */
  getCacheStats: (folderPath: string): Promise<CacheStats> =>
    invoke<CacheStats>('get_cache_stats', { folderPath }),

  /**
   * Clear the cache for a folder
   */
  clearCache: (folderPath: string): Promise<void> =>
    invoke<void>('clear_cache', { folderPath }),
};
