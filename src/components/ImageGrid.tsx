/**
 * ImageGrid - Left panel with search and virtualized image thumbnail grid
 *
 * Two performance-critical designs for very large folders (50k+ images):
 *
 * 1. Windowed rendering: only the visible rows (plus a small overscan)
 *    exist in the DOM. Scroll position drives which slice of `images` is
 *    rendered; the DOM node count stays constant (~dozens) regardless of
 *    the folder size. This is what keeps drag, right-click and clicks
 *    responsive on 50k-image folders.
 *
 * 2. Lazy thumbnails: the scan is metadata-only; thumbnails are fetched
 *    from the backend in batches for exactly the rendered window (micro-
 *    batched and de-duplicated) and memoized here. No thumbnail is
 *    generated for images that are never scrolled into view.
 */
import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
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

// Grid geometry (must match the CSS card size in styles)
const CARD = 150;      // px, square card (width == height)
const GAP = 8;         // px, gap between cards
const OVERSCAN_ROWS = 3;

export function ImageGrid({
  images,
  isLoading,
  loadingMessage,
  sourceImage,
  onSetSource,
}: ImageGridProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; image: ImageEntry } | null>(null);
  const [preview, setPreview] = useState<ImageEntry | null>(null);
  const [thumbnails, setThumbnails] = useState<Map<string, string>>(new Map());

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportH, setViewportH] = useState(800);
  const [viewportW, setViewportW] = useState(600);

  // Filter images by search query
  const filteredImages = useMemo(() => {
    if (!searchQuery) return images;
    const q = searchQuery.toLowerCase();
    return images.filter(img => img.file_name.toLowerCase().includes(q));
  }, [images, searchQuery]);

  // --- Virtualization geometry ---
  const cols = Math.max(1, Math.floor((viewportW - GAP) / (CARD + GAP)));
  const total = filteredImages.length;
  const totalRows = Math.ceil(total / cols);
  const rowH = CARD + GAP;
  const firstRow = Math.max(0, Math.floor(scrollTop / rowH) - OVERSCAN_ROWS);
  const lastRow = Math.min(totalRows, Math.ceil((scrollTop + viewportH) / rowH) + OVERSCAN_ROWS);

  const visible: { image: ImageEntry; index: number }[] = useMemo(() => {
    const out: { image: ImageEntry; index: number }[] = [];
    for (let row = firstRow; row < lastRow; row++) {
      for (let col = 0; col < cols; col++) {
        const index = row * cols + col;
        if (index >= total) break;
        out.push({ image: filteredImages[index], index });
      }
    }
    return out;
  }, [filteredImages, firstRow, lastRow, cols, total]);

  // Track scroll + viewport size
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => setScrollTop(el.scrollTop);
    el.addEventListener('scroll', onScroll, { passive: true });
    const ro = new ResizeObserver(() => {
      setViewportH(el.clientHeight);
      setViewportW(el.clientWidth);
    });
    ro.observe(el);
    setViewportH(el.clientHeight);
    setViewportW(el.clientWidth);
    return () => {
      el.removeEventListener('scroll', onScroll);
      ro.disconnect();
    };
  }, []);

  // Reset scroll when the folder changes
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
    setThumbnails(new Map());
  }, [images]);

  // --- Lazy thumbnails: micro-batch requests for the visible window ---
  // `inflight` deliberately survives effect re-runs: paths queued during a
  // scroll burst stay queued until flushed, instead of being dropped by
  // the cleanup of the next effect run.
  const mounted = useRef(true);
  const inflight = useRef<Set<string>>(new Set());
  const pending = useRef<Map<string, number>>(new Map()); // path -> attempt count
  const batchTimer = useRef<number | null>(null);
  const flush = useRef<() => void>(() => {});

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (batchTimer.current !== null) clearTimeout(batchTimer.current);
    };
  }, []);

  flush.current = () => {
    batchTimer.current = null;
    const paths = Array.from(inflight.current);
    inflight.current.clear();
    if (paths.length === 0) return;
    api.getThumbnails(paths, 150)
      .then(results => {
        if (!mounted.current) return;
        setThumbnails(prev => {
          const next = new Map(prev);
          paths.forEach((p, i) => {
            const b64 = results[i];
            if (b64) next.set(p, `data:image/jpeg;base64,${b64}`);
            else {
              // Record the failure; after 3 attempts stop retrying
              const n = (pending.current.get(p) ?? 0) + 1;
              pending.current.set(p, n);
              if (n >= 3) next.set(p, ''); // marker: failed permanently
            }
          });
          return next;
        });
      })
      .catch(e => {
        console.error('Thumbnail batch failed:', e);
        for (const p of paths) {
          pending.current.set(p, (pending.current.get(p) ?? 0) + 1);
        }
      });
  };

  useEffect(() => {
    // Collect paths in the visible window that still lack a thumbnail
    for (const { image } of visible) {
      if (
        !thumbnails.get(image.path) &&
        !image.thumbnail &&
        (pending.current.get(image.path) ?? 0) < 3 &&
        !inflight.current.has(image.path)
      ) {
        inflight.current.add(image.path);
      }
    }
    if (inflight.current.size > 0 && batchTimer.current === null) {
      batchTimer.current = window.setTimeout(() => flush.current(), 50);
    }
  }, [visible, thumbnails]);

  // Force-refresh one thumbnail (double-click): bypasses the Rust-side
  // cache, resets the retry counter, and shows the fresh result.
  const reloadThumbnail = useCallback(async (image: ImageEntry) => {
    pending.current.delete(image.path);
    setThumbnails(prev => {
      const next = new Map(prev);
      next.delete(image.path);
      return next;
    });
    try {
      const [b64] = await api.getThumbnails([image.path], 150, true);
      if (b64) {
        setThumbnails(prev => new Map(prev).set(image.path, `data:image/jpeg;base64,${b64}`));
      }
    } catch (e) {
      console.error('Thumbnail reload failed:', e);
    }
  }, []);

  const getThumbnailSrc = useCallback(
    (image: ImageEntry): string | undefined => {
      const t = thumbnails.get(image.path);
      if (t !== undefined) return t || undefined; // '' = known failure
      if (image.thumbnail) return `data:image/jpeg;base64,${image.thumbnail}`;
      return undefined;
    },
    [thumbnails]
  );

  // Handle context menu
  const handleContextMenu = useCallback((e: React.MouseEvent, image: ImageEntry) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, image });
  }, []);

  const handleDoubleClick = useCallback((image: ImageEntry) => {
    // Re-generate this image's thumbnail (bypasses caches) and open the
    // full-size preview.
    void reloadThumbnail(image);
    setPreview(image);
  }, [reloadThumbnail]);

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

      {/* Virtualized grid */}
      <div className="image-grid-scroll" ref={scrollRef}>
        {total === 0 && !isLoading ? (
          <div className="empty-state">
            <ImageRegular className="empty-state-icon" />
            <div className="empty-state-text">
              {images.length === 0
                ? 'Select a folder to load images'
                : 'No images match your search'}
            </div>
          </div>
        ) : (
          <div
            style={{
              position: 'relative',
              height: totalRows * rowH,
              width: '100%',
            }}
          >
            {visible.map(({ image, index }) => {
              const row = Math.floor(index / cols);
              const col = index % cols;
              const src = getThumbnailSrc(image);
              return (
                <div
                  key={image.path}
                  className={`image-card${sourceImage?.path === image.path ? ' selected' : ''}`}
                  style={{
                    position: 'absolute',
                    left: col * (CARD + GAP),
                    top: row * rowH,
                    width: CARD,
                    height: CARD,
                  }}
                  onContextMenu={(e) => handleContextMenu(e, image)}
                  onDoubleClick={() => handleDoubleClick(image)}
                  role="button"
                  tabIndex={0}
                  aria-label={image.file_name}
                >
                  {src ? (
                    <img src={src} alt={image.file_name} />
                  ) : (
                    <div className="image-card-placeholder" />
                  )}
                  <div className="image-card-overlay">
                    {image.file_name}
                  </div>
                </div>
              );
            })}
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
