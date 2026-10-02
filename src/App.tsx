/**
 * App - Main application component
 * Image Similarity Finder with Fluent Design
 */
import { useState, useCallback, useEffect, useMemo } from 'react';
import {
  FluentProvider,
  webDarkTheme,
  Toaster,
  useToastController,
  Toast,
  ToastTitle,
  ToastBody,
} from '@fluentui/react-components';
import { Toolbar } from './components/Toolbar';
import { ImageGrid } from './components/ImageGrid';
import { SimilarityPanel } from './components/SimilarityPanel';
import { StatusBar } from './components/StatusBar';
import { SettingsDialog, type BackendKind } from './components/SettingsDialog';
import { useInference } from './hooks/useInference';
import { api, type ImageEntry, type SimilarityResult } from './services/tauriApi';
import './styles/fluent.css';

function useThemeDetection() {
  const [isDark, setIsDark] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setIsDark(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return isDark;
}

function AppContent() {
  const isDark = useThemeDetection();
  const theme = isDark ? webDarkTheme : webDarkTheme; // Use dark theme for Fluent Design aesthetics
  const { dispatchToast } = useToastController();

  const {
    state,
    scanFolder,
    extractFeatures,
    searchSimilar,
    setSourceImage,
    setCompareFolderPath,
    clearResults,
    clearCache,
  } = useInference();

  const [recursive, setRecursive] = useState(true);
  const [similarity, setSimilarity] = useState(90);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const isLoading = state.status !== 'idle' && state.status !== 'error';
  // Overlays are panel-scoped: folder-level operations cover the left grid,
  // search-level operations cover only the results panel — a running search
  // must not freeze browsing in the grid.
  const isFolderLoading = state.status === 'scanning' || state.status === 'extracting';
  const isResultsLoading = state.status === 'searching' || state.status === 'comparing';

  // Handle folder selection
  const handleFolderSelect = useCallback((path: string) => {
    scanFolder(path, recursive);
  }, [scanFolder, recursive]);

  // Handle extract features
  const handleExtractFeatures = useCallback(() => {
    if (state.folderPath) {
      extractFeatures(state.folderPath);
    }
  }, [extractFeatures, state.folderPath]);

  // Handle compare
  const handleCompare = useCallback(() => {
    if (state.sourceImage && state.compareFolderPath) {
      if (state.sourceImage.path) {
        searchSimilar(state.sourceImage.path, state.folderPath, similarity / 100);
      }
    }
  }, [state.sourceImage, state.compareFolderPath, state.folderPath, searchSimilar, similarity]);

  // Set a result item as the new search source (same as the left grid).
  // Reuse the full ImageEntry from the loaded list when available.
  const handleSetSourceFromResult = useCallback((result: SimilarityResult) => {
    const existing = state.images.find(img => img.path === result.path);
    setSourceImage(existing ?? {
      path: result.path,
      file_name: result.file_name,
      file_size: 0,
      modified: 0,
      thumbnail: result.thumbnail,
    });
  }, [state.images, setSourceImage]);

  // Right-click -> set as source AND search immediately (no Compare click).
  // Searches the loaded folder, exactly like the Compare button.
  const handleDirectSearch = useCallback((image: ImageEntry) => {
    if (!state.folderPath || !image.path) return;
    setSourceImage(image);
    searchSimilar(image.path, state.folderPath, similarity / 100);
  }, [state.folderPath, setSourceImage, searchSimilar, similarity]);

  // Handle load model
  const handleLoadModel = useCallback(async (backend: BackendKind, path?: string) => {
    try {
      const msg = await api.loadModel(backend, path);
      dispatchToast(
        <Toast><ToastTitle>Model Loaded</ToastTitle><ToastBody>{msg}</ToastBody></Toast>,
        { intent: 'success' }
      );
    } catch (e) {
      dispatchToast(
        <Toast><ToastTitle>Error</ToastTitle><ToastBody>{String(e)}</ToastBody></Toast>,
        { intent: 'error' }
      );
    }
  }, [dispatchToast]);

  // Handle clear cache
  const handleClearCache = useCallback(async (folderPath: string) => {
    try {
      await clearCache(folderPath);
      dispatchToast(
        <Toast><ToastTitle>Cache Cleared</ToastTitle><ToastBody>Cache has been cleared.</ToastBody></Toast>,
        { intent: 'success' }
      );
    } catch (e) {
      dispatchToast(
        <Toast><ToastTitle>Error</ToastTitle><ToastBody>{String(e)}</ToastBody></Toast>,
        { intent: 'error' }
      );
    }
  }, [clearCache, dispatchToast]);

  // Show error toast on error status
  useEffect(() => {
    if (state.status === 'error' && state.error) {
      dispatchToast(
        <Toast><ToastTitle>Error</ToastTitle><ToastBody>{state.error}</ToastBody></Toast>,
        { intent: 'error' }
      );
    }
  }, [state.status, state.error, dispatchToast]);

  const loadingMessage = useMemo(() => {
    switch (state.status) {
      case 'scanning': return 'Scanning folder...';
      case 'extracting': return 'Extracting features...';
      case 'searching': return 'Searching for similar images...';
      case 'comparing': return 'Comparing folders...';
      default: return undefined;
    }
  }, [state.status]);

  return (
    <FluentProvider theme={theme} style={{ height: '100vh' }}>
      <div className="app-container" data-theme={isDark ? 'dark' : 'light'}>
        {/* Toolbar */}
        <Toolbar
          folderPath={state.folderPath}
          recursive={recursive}
          similarity={similarity}
          isLoading={isLoading}
          onFolderSelect={handleFolderSelect}
          onRecursiveChange={setRecursive}
          onSimilarityChange={setSimilarity}
          onExtractFeatures={handleExtractFeatures}
          onClear={clearResults}
          onOpenSettings={() => setSettingsOpen(true)}
        />

        {/* Main content */}
        <div className="app-main">
          {/* Left panel - Image Grid */}
          <div className="left-panel">
            <ImageGrid
              images={state.images}
              isLoading={isFolderLoading}
              loadingMessage={loadingMessage}
              sourceImage={state.sourceImage}
              onSetSource={setSourceImage}
              onSearchSimilar={handleDirectSearch}
            />
          </div>

          {/* Divider */}
          <div className="panel-divider" />

          {/* Right panel - Similarity */}
          <div className="right-panel">
            <SimilarityPanel
              sourceImage={state.sourceImage}
              results={state.searchResults}
              compareFolderPath={state.compareFolderPath}
              isLoading={isResultsLoading}
              loadingMessage={loadingMessage}
              onSelectCompareFolder={setCompareFolderPath}
              onCompare={handleCompare}
              similarity={similarity}
              onSetSource={handleSetSourceFromResult}
              onSearchSimilar={handleDirectSearch}
            />
          </div>
        </div>

        {/* Status Bar */}
        <StatusBar
          status={state.status}
          statusMessage={state.statusMessage}
          progress={state.progress}
          imageCount={state.images.length}
          resultCount={state.searchResults.length}
          cacheStats={state.cacheStats}
        />

        {/* Settings Dialog */}
        <SettingsDialog
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          onLoadModel={handleLoadModel}
          onClearCache={handleClearCache}
          folderPath={state.folderPath}
          isLoading={isLoading}
        />

        {/* Toaster */}
        <Toaster position="bottom-end" />
      </div>
    </FluentProvider>
  );
}

function App() {
  return <AppContent />;
}

export default App;
