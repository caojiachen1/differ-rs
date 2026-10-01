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

function mimeFromBase64(b64: string): string {
  // Sniff the magic bytes of the decoded payload — web-saved files often
  // carry a mismatched extension, so the path suffix cannot be trusted.
  try {
    const bytes = atob(b64.slice(0, 32));
    const sig = bytes.slice(0, 12);
    if (sig.startsWith('\xFF\xD8\xFF')) return 'image/jpeg';
    if (sig.startsWith('\x89PNG')) return 'image/png';
    if (sig.startsWith('GIF8')) return 'image/gif';
    if (sig.startsWith('BM')) return 'image/bmp';
    if (sig.startsWith('RIFF') && sig.includes('WEBP')) return 'image/webp';
    if (sig.includes('ftyp')) return 'image/avif';
  } catch {
    // fall through to the extension-based default
  }
  return 'image/jpeg';
}

export function ImageViewer({ path, fileName, placeholderSrc, onClose }: ImageViewerProps) {
  const [fullSrc, setFullSrc] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.readImage(path)
      .then(b64 => {
        if (!cancelled) setFullSrc(`data:${mimeFromBase64(b64)};base64,${b64}`);
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
