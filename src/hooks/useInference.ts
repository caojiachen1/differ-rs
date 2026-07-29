/**
 * useInference - Manages inference state and operations
 */
import { useState, useCallback, useEffect } from 'react';
import { listen } from '@tauri-apps/api/event';
import { api, type ImageEntry, type SimilarityResult, type ProgressInfo } from '../services/tauriApi';

export type AppStatus = 'idle' | 'scanning' | 'extracting' | 'searching' | 'comparing' | 'error';

export interface InferenceState {
  status: AppStatus;
  statusMessage: string;
  progress: ProgressInfo | null;
  error: string | null;
  images: ImageEntry[];
  searchResults: SimilarityResult[];
  sourceImage: ImageEntry | null;
  folderPath: string;
  compareFolderPath: string;
  cacheStats: { cached_count: number; cache_size_bytes: number; is_valid: boolean } | null;
}

export function useInference() {
  const [state, setState] = useState<InferenceState>({
    status: 'idle',
    statusMessage: 'Ready',
    progress: null,
    error: null,
    images: [],
    searchResults: [],
    sourceImage: null,
    folderPath: '',
    compareFolderPath: '',
    cacheStats: null,
  });

  const setStatus = useCallback((status: AppStatus, message: string) => {
    setState(prev => ({ ...prev, status, statusMessage: message, error: null }));
  }, []);

  // Live progress from the backend while DINOv3 features are being extracted
  useEffect(() => {
    const unlisten = listen<ProgressInfo>('extraction-progress', (event) => {
      const p = event.payload;
      setState(prev => ({
        ...prev,
        progress: p,
        // Only override the message while a long-running operation is active
        statusMessage: prev.status !== 'idle' && prev.status !== 'error' ? p.message : prev.statusMessage,
      }));
    });
    return () => {
      unlisten.then(fn => fn());
    };
  }, []);

  const setError = useCallback((error: string) => {
    setState(prev => ({ ...prev, status: 'error', statusMessage: 'Error', error }));
  }, []);

  const scanFolder = useCallback(async (path: string, recursive: boolean) => {
    setStatus('scanning', 'Scanning folder...');
    try {
      const images = await api.scanFolder(path, recursive);
      setState(prev => ({
        ...prev,
        images,
        folderPath: path,
        status: 'idle',
        statusMessage: `Loaded ${images.length} images`,
      }));
      // Fetch cache stats
      try {
        const stats = await api.getCacheStats(path);
        setState(prev => ({ ...prev, cacheStats: stats }));
      } catch {
        // Cache stats are optional
      }
    } catch (e) {
      setError(`Failed to scan folder: ${e}`);
    }
  }, [setStatus, setError]);

  const extractFeatures = useCallback(async (folderPath: string) => {
    setStatus('extracting', 'Extracting features...');
    setState(prev => ({ ...prev, progress: null }));
    try {
      const progress = await api.extractFeatures(folderPath);
      setState(prev => ({
        ...prev,
        progress,
        status: 'idle',
        statusMessage: progress.message || 'Feature extraction complete',
      }));
      // Refresh cache stats after extraction populated the folder cache
      try {
        const stats = await api.getCacheStats(folderPath);
        setState(prev => ({ ...prev, cacheStats: stats }));
      } catch {
        // Cache stats are optional
      }
    } catch (e) {
      setError(`Failed to extract features: ${e}`);
    }
  }, [setStatus, setError]);

  const searchSimilar = useCallback(async (sourcePath: string, folderPath: string, threshold: number) => {
    setStatus('searching', 'Searching for similar images...');
    setState(prev => ({ ...prev, progress: null }));
    try {
      const results = await api.searchSimilar(sourcePath, folderPath, threshold);
      setState(prev => ({
        ...prev,
        searchResults: results,
        status: 'idle',
        statusMessage: `Found ${results.length} similar images`,
      }));
    } catch (e) {
      setError(`Failed to search: ${e}`);
    }
  }, [setStatus, setError]);

  const compareFolders = useCallback(async (source: string, target: string, threshold: number) => {
    setStatus('comparing', 'Comparing folders...');
    try {
      const results = await api.compareFolders(source, target, threshold);
      setState(prev => ({
        ...prev,
        searchResults: results,
        status: 'idle',
        statusMessage: `Found ${results.length} similar images`,
      }));
    } catch (e) {
      setError(`Failed to compare folders: ${e}`);
    }
  }, [setStatus, setError]);

  const setSourceImage = useCallback((image: ImageEntry | null) => {
    setState(prev => ({ ...prev, sourceImage: image, searchResults: [] }));
  }, []);

  const setCompareFolderPath = useCallback((path: string) => {
    setState(prev => ({ ...prev, compareFolderPath: path }));
  }, []);

  const clearResults = useCallback(() => {
    setState(prev => ({
      ...prev,
      searchResults: [],
      sourceImage: null,
      status: 'idle',
      statusMessage: 'Ready',
      error: null,
    }));
  }, []);

  const clearCache = useCallback(async (folderPath: string) => {
    try {
      await api.clearCache(folderPath);
      setState(prev => ({ ...prev, cacheStats: null }));
    } catch (e) {
      setError(`Failed to clear cache: ${e}`);
    }
  }, [setError]);

  return {
    state,
    scanFolder,
    extractFeatures,
    searchSimilar,
    compareFolders,
    setSourceImage,
    setCompareFolderPath,
    clearResults,
    clearCache,
  };
}
