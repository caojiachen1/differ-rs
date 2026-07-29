/**
 * ImageGrid - Left panel with search and image thumbnail grid
 */
import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  Input,
  Spinner,
  Menu,
  MenuTrigger,
  MenuList,
  MenuItem,
  MenuPopover,
} from '@fluentui/react-components';
import {
  SearchRegular,
  ImageRegular,
  TargetRegular,
  FolderOpenRegular,
} from '@fluentui/react-icons';
import { api } from '../services/tauriApi';
import type { ImageEntry } from '../services/tauriApi';
import { ImageViewer } from './ImageViewer';

interface ImageGridProps {
  images: ImageEntry[];
  isLoading: boolean;
  loadingMessage?: string;
  sourceImage: ImageEntry | null;
  onSetSource: (image: ImageEntry) => void;
}

export function ImageGrid({
  images,
  isLoading,
  loadingMessage,
  sourceImage,
  onSetSource,
}: ImageGridProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; image: ImageEntry } | null>(null);
  const [thumbnailCache, setThumbnailCache] = useState<Map<string, string>>(new Map());
  const [preview, setPreview] = useState<ImageEntry | null>(null);

  // Filter images by search query
  const filteredImages = useMemo(() => {
    if (!searchQuery) return images;
    const q = searchQuery.toLowerCase();
    return images.filter(img => img.file_name.toLowerCase().includes(q));
  }, [images, searchQuery]);

  // Generate thumbnail URL from base64 or file path
  const getThumbnailSrc = useCallback((image: ImageEntry): string => {
    // Check cache first
    const cached = thumbnailCache.get(image.path);
    if (cached) return cached;

    // Use base64 thumbnail if available
    if (image.thumbnail) {
      const src = `data:image/jpeg;base64,${image.thumbnail}`;
      setThumbnailCache(prev => new Map(prev).set(image.path, src));
      return src;
    }

    // Fallback: use file path directly (works in Tauri with asset protocol)
    return `asset://localhost/${encodeURIComponent(image.path)}`;
  }, [thumbnailCache]);

  // Handle context menu
  const handleContextMenu = useCallback((e: React.MouseEvent, image: ImageEntry) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, image });
  }, []);

  const handleDoubleClick = useCallback((image: ImageEntry) => {
    setPreview(image);
  }, []);

  const handleOpenLocation = useCallback(async (image: ImageEntry) => {
    try {
      // Reveal (and highlight) the file in the system file manager
      await api.showInFolder(image.path);
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

  return (
    <div className="image-grid-container" style={{ position: 'relative' }}>
      {/* Search bar */}
      <div className="image-grid-search">
        <Input
          contentBefore={<SearchRegular />}
          placeholder="Search images..."
          value={searchQuery}
          onChange={(_, data) => setSearchQuery(data.value)}
          size="small"
          style={{ width: '100%' }}
          aria-label="Search images"
        />
      </div>

      {/* Grid */}
      <div className="image-grid-scroll">
        {filteredImages.length === 0 && !isLoading ? (
          <div className="empty-state">
            <ImageRegular className="empty-state-icon" />
            <div className="empty-state-text">
              {images.length === 0
                ? 'Select a folder to load images'
                : 'No images match your search'}
            </div>
          </div>
        ) : (
          <div className="image-grid">
            {filteredImages.map(image => (
              <div
                key={image.path}
                className={`image-card${sourceImage?.path === image.path ? ' selected' : ''}`}
                onContextMenu={(e) => handleContextMenu(e, image)}
                onDoubleClick={() => handleDoubleClick(image)}
                role="button"
                tabIndex={0}
                aria-label={image.file_name}
              >
                <img
                  src={getThumbnailSrc(image)}
                  alt={image.file_name}
                  loading="lazy"
                />
                <div className="image-card-overlay">
                  {image.file_name}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Loading overlay */}
      {isLoading && (
        <div className="loading-overlay">
          <Spinner size="large" />
          <div className="loading-overlay-text">{loadingMessage || 'Loading...'}</div>
        </div>
      )}

      {/* Image preview lightbox */}
      {preview && (
        <ImageViewer
          path={preview.path}
          fileName={preview.file_name}
          placeholderSrc={getThumbnailSrc(preview)}
          onClose={() => setPreview(null)}
        />
      )}

      {/* Context Menu */}
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
                  onSetSource(contextMenu.image);
                  setContextMenu(null);
                }}
              >
                Set as Search Source
              </MenuItem>
              <MenuItem
                icon={<FolderOpenRegular />}
                onClick={() => {
                  handleOpenLocation(contextMenu.image);
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
