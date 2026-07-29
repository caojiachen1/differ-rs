/**
 * SettingsDialog - Settings dialog for model configuration and cache management
 */
import { useState, useCallback, useEffect } from 'react';
import {
  Dialog,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Input,
  Label,
  Divider,
  Text,
  Spinner,
  RadioGroup,
  Radio,
} from '@fluentui/react-components';
import {
  DismissRegular,
  BrainCircuitRegular,
  DeleteRegular,
  InfoRegular,
} from '@fluentui/react-icons';
import { api } from '../services/tauriApi';

export type BackendKind = 'ggml' | 'onnx';

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onLoadModel: (backend: BackendKind, path?: string) => void;
  onClearCache: (folderPath: string) => void;
  folderPath: string;
  isLoading: boolean;
}

export function SettingsDialog({
  open,
  onOpenChange,
  onLoadModel,
  onClearCache,
  folderPath,
  isLoading,
}: SettingsDialogProps) {
  const [modelPath, setModelPath] = useState('');
  const [backend, setBackend] = useState<BackendKind>('ggml');
  const [backendInfo, setBackendInfo] = useState('');

  // Show which backend/model is currently active whenever the dialog opens
  useEffect(() => {
    if (open) {
      api.getBackendInfo().then(setBackendInfo).catch(() => setBackendInfo(''));
    }
  }, [open, isLoading]);

  const handleLoadModel = useCallback(() => {
    onLoadModel(backend, modelPath || undefined);
  }, [backend, modelPath, onLoadModel]);

  const handleClearCache = useCallback(() => {
    if (folderPath) {
      onClearCache(folderPath);
    }
  }, [folderPath, onClearCache]);

  return (
    <Dialog open={open} onOpenChange={(_, data) => onOpenChange(data.open)}>
      <DialogSurface style={{ width: 480, maxWidth: '90vw' }}>
        <DialogBody>
          <DialogTitle action={<Button icon={<DismissRegular />} appearance="subtle" onClick={() => onOpenChange(false)} />}>
            Settings
          </DialogTitle>
          <DialogContent>
            {/* Model Section */}
            <div style={{ marginBottom: 20 }}>
              <Text weight="semibold" size={300} block style={{ marginBottom: 8 }}>
                <BrainCircuitRegular style={{ marginRight: 6, verticalAlign: 'text-bottom' }} />
                Model Configuration
              </Text>
              {backendInfo && (
                <Text size={200} block style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>
                  Current: {backendInfo}
                </Text>
              )}
              <Label size="small">Inference Backend</Label>
              <RadioGroup
                layout="horizontal"
                value={backend}
                onChange={(_, data) => setBackend(data.value as BackendKind)}
                style={{ marginBottom: 8 }}
              >
                <Radio value="ggml" label="GGML (CUDA, fastest)" />
                <Radio value="onnx" label="ONNX Runtime" />
              </RadioGroup>
              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
                <div style={{ flex: 1 }}>
                  <Label size="small" htmlFor="model-path">Model Path (optional, auto-detected)</Label>
                  <Input
                    id="model-path"
                    placeholder={backend === 'ggml' ? 'models/dinov3_vits16.bin' : 'models/model.onnx'}
                    value={modelPath}
                    onChange={(_, data) => setModelPath(data.value)}
                    size="small"
                    style={{ width: '100%', marginTop: 4 }}
                  />
                </div>
                <Button
                  size="small"
                  onClick={handleLoadModel}
                  disabled={isLoading}
                  appearance="primary"
                >
                  {isLoading ? <Spinner size="tiny" /> : 'Load'}
                </Button>
              </div>
            </div>

            <Divider style={{ margin: '16px 0' }} />

            {/* Cache Section */}
            <div style={{ marginBottom: 20 }}>
              <Text weight="semibold" size={300} block style={{ marginBottom: 8 }}>
                <DeleteRegular style={{ marginRight: 6, verticalAlign: 'text-bottom' }} />
                Cache Management
              </Text>
              <Text size={200} block style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>
                Clear cached features and thumbnails for the current folder.
              </Text>
              <Button
                size="small"
                onClick={handleClearCache}
                disabled={!folderPath || isLoading}
                icon={<DeleteRegular />}
              >
                Clear Cache
              </Button>
            </div>

            <Divider style={{ margin: '16px 0' }} />

            {/* About Section */}
            <div>
              <Text weight="semibold" size={300} block style={{ marginBottom: 8 }}>
                <InfoRegular style={{ marginRight: 6, verticalAlign: 'text-bottom' }} />
                About
              </Text>
              <Text size={200} block style={{ color: 'var(--text-secondary)' }}>
                Image Similarity Finder v0.1.0
              </Text>
              <Text size={200} block style={{ color: 'var(--text-secondary)' }}>
                Powered by DINOv3 (GGML CUDA / ONNX Runtime)
              </Text>
              <Text size={200} block style={{ color: 'var(--text-tertiary)', marginTop: 4 }}>
                Tauri 2.x + React + Fluent UI
              </Text>
            </div>
          </DialogContent>
          <DialogActions>
            <Button appearance="secondary" onClick={() => onOpenChange(false)}>
              Close
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  );
}
