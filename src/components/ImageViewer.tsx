/**
 * ImageViewer - centered lightbox overlay for viewing an image at full size.
 * Click on the backdrop (blank area) or press Escape to close.
 */
import { useEffect, useState } from 'react';
import { Spinner } from '@fluentui/react-components';
import { api } from '../services/tauriApi';

interface ImageViewerProps {
  path: string;
  fileName: string;
  /** Fallback shown while the full image loads (e.g. thumbnail data URL) */
  placeholderSrc?: string;
  onClose: () => void;
}

function mimeFromPath(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  switch (ext) {
    case 'png': return 'image/png';
    case 'gif': return 'image/gif';
    case 'webp': return 'image/webp';
    case 'bmp': return 'image/bmp';
    default: return 'image/jpeg';
  }
}

export function ImageViewer({ path, fileName, placeholderSrc, onClose }: ImageViewerProps) {
  const [fullSrc, setFullSrc] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.readImage(path)
      .then(b64 => {
        if (!cancelled) setFullSrc(`data:${mimeFromPath(path)};base64,${b64}`);
      })
      .catch(e => console.error('Failed to load full image:', e));
    return () => { cancelled = true; };
  }, [path]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const src = fullSrc ?? placeholderSrc;

  return (
    <div className="image-viewer-overlay" onClick={onClose}>
      {src ? (
        <img
          src={src}
          alt={fileName}
          className="image-viewer-img"
          onClick={(e) => e.stopPropagation()}
          onDoubleClick={(e) => e.stopPropagation()}
        />
      ) : (
        <Spinner size="large" />
      )}
      <div className="image-viewer-caption" onClick={(e) => e.stopPropagation()}>
        {fileName}
      </div>
    </div>
  );
}
