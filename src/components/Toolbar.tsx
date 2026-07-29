/**
 * Toolbar - Top toolbar with folder selection, similarity slider, and actions
 */
import { useCallback } from 'react';
import {
  Button,
  Checkbox,
  Slider,
  Tooltip,
  Divider,
} from '@fluentui/react-components';
import {
  FolderOpenRegular,
  DeleteRegular,
  BrainCircuitRegular,
  SettingsRegular,
} from '@fluentui/react-icons';
import { open } from '@tauri-apps/plugin-dialog';

interface ToolbarProps {
  folderPath: string;
  recursive: boolean;
  similarity: number;
  isLoading: boolean;
  onFolderSelect: (path: string) => void;
  onRecursiveChange: (recursive: boolean) => void;
  onSimilarityChange: (value: number) => void;
  onExtractFeatures: () => void;
  onClear: () => void;
  onOpenSettings: () => void;
}

export function Toolbar({
  folderPath,
  recursive,
  similarity,
  isLoading,
  onFolderSelect,
  onRecursiveChange,
  onSimilarityChange,
  onExtractFeatures,
  onClear,
  onOpenSettings,
}: ToolbarProps) {
  const handleSelectFolder = useCallback(async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Select Image Folder',
      });
      if (selected) {
        onFolderSelect(selected);
      }
    } catch (e) {
      console.error('Failed to open folder dialog:', e);
    }
  }, [onFolderSelect]);

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <Tooltip content="Select folder" relationship="label">
          <Button
            icon={<FolderOpenRegular />}
            onClick={handleSelectFolder}
            disabled={isLoading}
            appearance="primary"
          >
            Select Folder
          </Button>
        </Tooltip>

        <Checkbox
          label="Include subfolders"
          checked={recursive}
          onChange={(_, data) => onRecursiveChange(data.checked === true)}
          disabled={isLoading}
        />
      </div>

      {folderPath && (
        <>
          <Divider vertical style={{ height: 24 }} />
          <span className="toolbar-path" title={folderPath}>
            {folderPath}
          </span>
        </>
      )}

      <div style={{ flex: 1 }} />

      <div className="toolbar-group">
        <div className="toolbar-slider">
          <span className="toolbar-slider-label">Similarity:</span>
          <Slider
            min={50}
            max={100}
            step={1}
            value={similarity}
            onChange={(_, data) => onSimilarityChange(data.value)}
            disabled={isLoading}
            style={{ width: 120 }}
            aria-label="Similarity threshold"
          />
          <span className="toolbar-slider-value">{similarity}%</span>
        </div>
      </div>

      <Divider vertical style={{ height: 24 }} />

      <div className="toolbar-group">
        <Tooltip content="Extract features for all images" relationship="label">
          <Button
            icon={<BrainCircuitRegular />}
            onClick={onExtractFeatures}
            disabled={isLoading || !folderPath}
          >
            Extract
          </Button>
        </Tooltip>

        <Tooltip content="Clear results" relationship="label">
          <Button
            icon={<DeleteRegular />}
            onClick={onClear}
            disabled={isLoading}
          >
            Clear
          </Button>
        </Tooltip>

        <Tooltip content="Settings" relationship="label">
          <Button
            icon={<SettingsRegular />}
            onClick={onOpenSettings}
            appearance="subtle"
          />
        </Tooltip>
      </div>
    </div>
  );
}
