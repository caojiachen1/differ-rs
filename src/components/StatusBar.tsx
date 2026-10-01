/**
 * StatusBar - Bottom status bar with status info and progress
 */
import { Spinner, ProgressBar } from '@fluentui/react-components';
import {
  CheckmarkCircleRegular,
  ErrorCircleRegular,
  InfoRegular,
} from '@fluentui/react-icons';
import type { AppStatus } from '../hooks/useInference';
import type { ProgressInfo, CacheStats } from '../services/tauriApi';

interface StatusBarProps {
  status: AppStatus;
  statusMessage: string;
  progress: ProgressInfo | null;
  imageCount: number;
  resultCount: number;
  cacheStats: CacheStats | null;
}

export function StatusBar({
  status,
  statusMessage,
  progress,
  imageCount,
  resultCount,
  cacheStats,
}: StatusBarProps) {
  const isActive = status !== 'idle' && status !== 'error';

  const getStatusIcon = () => {
    if (status === 'error') {
      return <ErrorCircleRegular style={{ color: '#d13438' }} />;
    }
    if (isActive) {
      return <Spinner size="tiny" />;
    }
    return <CheckmarkCircleRegular style={{ color: '#0f7b0f' }} />;
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="status-bar">
      <div className="status-bar-left">
        <span className="status-indicator">
          {getStatusIcon()}
          <span>{statusMessage}</span>
        </span>
        {isActive && progress && progress.total > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginLeft: 8 }}>
            <ProgressBar
              value={progress.processed / progress.total}
              max={1}
              thickness="medium"
              style={{ width: 160 }}
            />
            <span style={{ color: 'var(--text-tertiary)' }}>
              {progress.processed} / {progress.total}
              {(progress.cache_hits > 0 || progress.cache_misses > 0) && (
                <> (cache {progress.cache_hits} | new {progress.cache_misses})</>
              )}
              {!!progress.images_per_second && progress.images_per_second > 0 && (
                <> · {progress.images_per_second >= 100
                  ? Math.round(progress.images_per_second)
                  : progress.images_per_second.toFixed(1)} img/s</>
              )}
            </span>
          </span>
        )}
        {!isActive && progress && progress.total > 0 && (
          <span style={{ marginLeft: 8, color: 'var(--text-tertiary)' }}>
            ({progress.processed} / {progress.total})
          </span>
        )}
      </div>

      <div className="status-bar-right">
        {imageCount > 0 && (
          <span>
            <InfoRegular style={{ marginRight: 4, verticalAlign: 'middle' }} />
            Images: {imageCount}
          </span>
        )}
        {resultCount > 0 && (
          <span>Results: {resultCount}</span>
        )}
        {cacheStats && (
          <span>
            Cache: {cacheStats.cached_count} items ({formatBytes(cacheStats.cache_size_bytes)})
          </span>
        )}
      </div>
    </div>
  );
}
