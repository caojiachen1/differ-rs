/**
 * SimilarityPanel - Right panel showing compare folder, source image, and results
 */
import { useCallback, useState, useEffect } from 'react';
import {
  Button,
  Text,
  Spinner,
  Menu,
  MenuTrigger,
  MenuList,
  MenuItem,
  MenuPopover,
} from '@fluentui/react-components';
import {
  FolderOpenRegular,
  TargetRegular,
  ImageRegular,
  ArrowRightRegular,
} from '@fluentui/react-icons';
import { open } from '@tauri-apps/plugin-dialog';
import { open as shellOpen } from '@tauri-apps/plugin-shell';
import { api } from '../services/tauriApi';
import type { ImageEntry, SimilarityResult } from '../services/tauriApi';

interface SimilarityPanelProps {
  sourceImage: ImageEntry | null;
  results: SimilarityResult[];
  compareFolderPath: string;
  isLoading: boolean;
  loadingMessage?: string;
  onSelectCompareFolder: (path: string) => void;
  onCompare: () => void;
  similarity: number;
  onSetSource: (result: SimilarityResult) => void;
}

export function SimilarityPanel({
  sourceImage,
  results,
  compareFolderPath,
  isLoading,
  loadingMessage,
  onSelectCompareFolder,
  onCompare,
  similarity,
  onSetSource,
}: SimilarityPanelProps) {
  const [thumbnailCache, setThumbnailCache] = useState<Map<string, string>>(new Map());
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; result: SimilarityResult } | null>(null);

  // Same context-menu behavior as the left grid
  const handleContextMenu = useCallback((e: React.MouseEvent, result: SimilarityResult) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, result });
  }, []);

  const handleDoubleClick = useCallback(async (result: SimilarityResult) => {
    try {
      await shellOpen(result.path);
    } catch (e) {
      console.error('Failed to open image:', e);
    }
  }, []);

  const handleOpenLocation = useCallback(async (result: SimilarityResult) => {
    try {
      // Reveal (and highlight) the file in the system file manager
      await api.showInFolder(result.path);
    } catch (e) {
      console.error('Failed to open location:', e);
    }
  }, []);

  // Close context menu on click elsewhere
  useEffect(() => {
    if (contextMenu) {
      const handler = () => setContextMenu(null);
      window.addEventListener('click', handler);
      return () => window.removeEventListener('click', handler);
    }
  }, [contextMenu]);

  const handleSelectCompareFolder = useCallback(async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Select Compare Folder',
      });
      if (selected) {
        onSelectCompareFolder(selected);
      }
    } catch (e) {
      console.error('Failed to open folder dialog:', e);
    }
  }, [onSelectCompareFolder]);

  const getThumbnailSrc = useCallback((result: SimilarityResult): string => {
    const cached = thumbnailCache.get(result.path);
    if (cached) return cached;

    if (result.thumbnail) {
      const src = `data:image/jpeg;base64,${result.thumbnail}`;
      setThumbnailCache(prev => new Map(prev).set(result.path, src));
      return src;
    }

    return `asset://localhost/${encodeURIComponent(result.path)}`;
  }, [thumbnailCache]);

  const getSourceThumbSrc = useCallback((image: ImageEntry): string => {
    if (image.thumbnail) {
      return `data:image/jpeg;base64,${image.thumbnail}`;
    }
    return `asset://localhost/${encodeURIComponent(image.path)}`;
  }, []);

  const getSimilarityClass = (sim: number): string => {
    const pct = sim * 100;
    if (pct >= 90) return 'high';
    if (pct >= 80) return 'medium';
    return 'low';
  };

  const formatSimilarity = (sim: number): string => {
    return `${Math.round(sim * 100)}%`;
  };

  return (
    <div className="similarity-panel" style={{ position: 'relative' }}>
      {/* Compare Folder Card */}
      <div className="panel-card">
        <div className="panel-card-title">Compare Folder</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <Button
            icon={<FolderOpenRegular />}
            onClick={handleSelectCompareFolder}
            disabled={isLoading}
            size="small"
            style={{ flex: 1 }}
          >
            {compareFolderPath ? 'Change Folder' : 'Select Folder'}
          </Button>
          {compareFolderPath && (
            <Button
              icon={<ArrowRightRegular />}
              onClick={onCompare}
              disabled={isLoading || !sourceImage}
              appearance="primary"
              size="small"
            >
              Compare
            </Button>
          )}
        </div>
        {compareFolderPath && (
          <Text
            size={200}
            style={{
              display: 'block',
              marginTop: 8,
              color: 'var(--text-tertiary)',
              wordBreak: 'break-all',
            }}
          >
            {compareFolderPath}
          </Text>
        )}
      </div>

      {/* Source Image Card */}
      <div className="panel-card">
        <div className="panel-card-title">Source Image</div>
        {sourceImage ? (
          <div className="source-image-card">
            <img
              src={getSourceThumbSrc(sourceImage)}
              alt={sourceImage.file_name}
              className="source-image-thumb"
            />
            <div className="source-image-info">
              <div className="source-image-name" title={sourceImage.file_name}>
                {sourceImage.file_name}
              </div>
              <div className="source-image-path" title={sourceImage.path}>
                {sourceImage.path}
              </div>
              <Text size={200} style={{ color: 'var(--text-tertiary)', marginTop: 4 }}>
                Threshold: {similarity}%
              </Text>
            </div>
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '16px 0' }}>
            <TargetRegular style={{ fontSize: 32, opacity: 0.4 }} />
            <div className="empty-state-text" style={{ fontSize: 12 }}>
              Right-click an image to set as search source
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      <div className="panel-card" style={{ flex: 1, overflow: 'auto' }}>
        <div className="panel-card-title">
          Results {results.length > 0 && `(${results.length})`}
        </div>
        {results.length > 0 ? (
          <div className="results-grid">
            {results.map(result => (
              <div
                key={result.path}
                className="result-card"
                title={`${result.file_name} - ${formatSimilarity(result.similarity)}`}
                onContextMenu={(e) => handleContextMenu(e, result)}
                onDoubleClick={() => handleDoubleClick(result)}
                role="button"
                tabIndex={0}
                aria-label={result.file_name}
              >
                <img
                  src={getThumbnailSrc(result)}
                  alt={result.file_name}
                  loading="lazy"
                />
                <span className={`result-badge ${getSimilarityClass(result.similarity)}`}>
                  {formatSimilarity(result.similarity)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '24px 0' }}>
            <ImageRegular style={{ fontSize: 32, opacity: 0.4 }} />
            <div className="empty-state-text" style={{ fontSize: 12 }}>
              {sourceImage
                ? compareFolderPath
                  ? 'Click "Compare" to find similar images'
                  : 'Select a compare folder'
                : 'Set a source image to begin'}
            </div>
          </div>
        )}
      </div>

      {/* Loading overlay */}
      {isLoading && (
        <div className="loading-overlay">
          <Spinner size="large" />
          <div className="loading-overlay-text">{loadingMessage || 'Processing...'}</div>
        </div>
      )}

      {/* Context Menu (same style/options as the left grid) */}
      {contextMenu && (
        <Menu open={true} onOpenChange={() => setContextMenu(null)}>
          <MenuTrigger disableButtonEnhancement>
            <div
              style={{
                position: 'fixed',
                left: contextMenu.x,
                top: contextMenu.y,
                width: 1,
                height: 1,
              }}
            />
          </MenuTrigger>
          <MenuPopover style={{ position: 'fixed', left: contextMenu.x, top: contextMenu.y }}>
            <MenuList>
              <MenuItem
                icon={<TargetRegular />}
                onClick={() => {
                  onSetSource(contextMenu.result);
                  setContextMenu(null);
                }}
              >
                Set as Search Source
              </MenuItem>
              <MenuItem
                icon={<FolderOpenRegular />}
                onClick={() => {
                  handleOpenLocation(contextMenu.result);
                  setContextMenu(null);
                }}
              >
                Open File Location
              </MenuItem>
            </MenuList>
          </MenuPopover>
        </Menu>
      )}
    </div>
  );
}
