import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Box,
  Typography,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Snackbar,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  Save,
  RestoreFromTrash,
  Info,
  Security,
  Speed,
  Storage,
  Psychology,
} from '@mui/icons-material';
import { SystemConfiguration } from '../../services/systemService';

interface SystemConfigurationProps {
  config: SystemConfiguration;
  onSaveConfig: (config: SystemConfiguration) => Promise<void>;
  onResetConfig: () => Promise<void>;
  loading?: boolean;
}

const SystemConfigurationComponent: React.FC<SystemConfigurationProps> = ({
  config,
  onSaveConfig,
  onResetConfig,
  loading = false,
}) => {
  const [localConfig, setLocalConfig] = useState<SystemConfiguration>(config);
  const [hasChanges, setHasChanges] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    setLocalConfig(config);
    setHasChanges(false);
  }, [config]);

  useEffect(() => {
    const configChanged = JSON.stringify(localConfig) !== JSON.stringify(config);
    setHasChanges(configChanged);
  }, [localConfig, config]);

  const handleSave = async () => {
    try {
      await onSaveConfig(localConfig);
      setSaveSuccess(true);
      setHasChanges(false);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save configuration');
    }
  };

  const handleReset = async () => {
    try {
      await onResetConfig();
      setSaveSuccess(true);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to reset configuration');
    }
  };

  const updateConfig = (section: keyof SystemConfiguration, key: string, value: any) => {
    setLocalConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
  };

  return (
    <>
      <Card sx={{ height: '100%' }}>
        <CardHeader
          title={
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="h6">System Configuration</Typography>
              {hasChanges && (
                <Chip
                  label="Unsaved Changes"
                  color="warning"
                  size="small"
                  variant="outlined"
                />
              )}
            </Box>
          }
          action={
            <Box display="flex" gap={1}>
              <Button
                variant="outlined"
                startIcon={<RestoreFromTrash />}
                onClick={handleReset}
                disabled={loading}
              >
                Reset to Defaults
              </Button>
              <Button
                variant="contained"
                startIcon={<Save />}
                onClick={handleSave}
                disabled={loading || !hasChanges}
              >
                Save Changes
              </Button>
            </Box>
          }
        />
        <CardContent sx={{ pt: 0 }}>
          <Box sx={{ maxHeight: 600, overflow: 'auto' }}>
            {/* RAG Pipeline Configuration */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Storage color="primary" />
                  <Typography variant="h6">RAG Pipeline</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box display="flex" flexDirection="column" gap={3}>
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Chunk Size
                      <Tooltip title="Size of text chunks for document processing">
                        <IconButton size="small">
                          <Info fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Typography>
                    <Slider
                      value={localConfig.ragPipeline.chunkSize}
                      onChange={(_, value) => updateConfig('ragPipeline', 'chunkSize', value)}
                      min={128}
                      max={4096}
                      step={128}
                      marks={[
                        { value: 512, label: '512' },
                        { value: 1024, label: '1024' },
                        { value: 2048, label: '2048' },
                      ]}
                      valueLabelDisplay="auto"
                    />
                  </Box>

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Overlap Size
                    </Typography>
                    <Slider
                      value={localConfig.ragPipeline.overlapSize}
                      onChange={(_, value) => updateConfig('ragPipeline', 'overlapSize', value)}
                      min={0}
                      max={512}
                      step={32}
                      marks={[
                        { value: 64, label: '64' },
                        { value: 128, label: '128' },
                        { value: 256, label: '256' },
                      ]}
                      valueLabelDisplay="auto"
                    />
                  </Box>

                  <FormControl fullWidth>
                    <InputLabel>Embedding Model</InputLabel>
                    <Select
                      value={localConfig.ragPipeline.embeddingModel}
                      label="Embedding Model"
                      onChange={(e) => updateConfig('ragPipeline', 'embeddingModel', e.target.value)}
                    >
                      <MenuItem value="all-mpnet-base-v2">all-mpnet-base-v2</MenuItem>
                      <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</MenuItem>
                      <MenuItem value="GraphCodeBERT">GraphCodeBERT</MenuItem>
                    </Select>
                  </FormControl>

                  <FormControlLabel
                    control={
                      <Switch
                        checked={localConfig.ragPipeline.rerankerEnabled}
                        onChange={(e) => updateConfig('ragPipeline', 'rerankerEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Reranker"
                  />

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Hybrid Search Weight (Vector vs Keyword)
                    </Typography>
                    <Slider
                      value={localConfig.ragPipeline.hybridSearchWeight}
                      onChange={(_, value) => updateConfig('ragPipeline', 'hybridSearchWeight', value)}
                      min={0}
                      max={1}
                      step={0.1}
                      marks={[
                        { value: 0, label: 'Keyword' },
                        { value: 0.5, label: 'Balanced' },
                        { value: 1, label: 'Vector' },
                      ]}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* LLM Settings */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Psychology color="primary" />
                  <Typography variant="h6">LLM Settings</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box display="flex" flexDirection="column" gap={3}>
                  <FormControl fullWidth>
                    <InputLabel>Model</InputLabel>
                    <Select
                      value={localConfig.llmSettings.model}
                      label="Model"
                      onChange={(e) => updateConfig('llmSettings', 'model', e.target.value)}
                    >
                      <MenuItem value="gpt-4">GPT-4</MenuItem>
                      <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo</MenuItem>
                      <MenuItem value="claude-3-opus">Claude 3 Opus</MenuItem>
                      <MenuItem value="claude-3-sonnet">Claude 3 Sonnet</MenuItem>
                    </Select>
                  </FormControl>

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Temperature: {localConfig.llmSettings.temperature}
                    </Typography>
                    <Slider
                      value={localConfig.llmSettings.temperature}
                      onChange={(_, value) => updateConfig('llmSettings', 'temperature', value)}
                      min={0}
                      max={2}
                      step={0.1}
                      marks={[
                        { value: 0, label: 'Deterministic' },
                        { value: 1, label: 'Balanced' },
                        { value: 2, label: 'Creative' },
                      ]}
                      valueLabelDisplay="auto"
                    />
                  </Box>

                  <TextField
                    label="Max Tokens"
                    type="number"
                    value={localConfig.llmSettings.maxTokens}
                    onChange={(e) => updateConfig('llmSettings', 'maxTokens', parseInt(e.target.value))}
                    fullWidth
                    inputProps={{ min: 1, max: 32000 }}
                  />

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Top P: {localConfig.llmSettings.topP}
                    </Typography>
                    <Slider
                      value={localConfig.llmSettings.topP}
                      onChange={(_, value) => updateConfig('llmSettings', 'topP', value)}
                      min={0}
                      max={1}
                      step={0.05}
                      valueLabelDisplay="auto"
                    />
                  </Box>

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Frequency Penalty: {localConfig.llmSettings.frequencyPenalty}
                    </Typography>
                    <Slider
                      value={localConfig.llmSettings.frequencyPenalty}
                      onChange={(_, value) => updateConfig('llmSettings', 'frequencyPenalty', value)}
                      min={-2}
                      max={2}
                      step={0.1}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Security Settings */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Security color="primary" />
                  <Typography variant="h6">Security</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box display="flex" flexDirection="column" gap={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localConfig.security.rateLimitEnabled}
                        onChange={(e) => updateConfig('security', 'rateLimitEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Rate Limiting"
                  />

                  <TextField
                    label="Max Requests Per Minute"
                    type="number"
                    value={localConfig.security.maxRequestsPerMinute}
                    onChange={(e) => updateConfig('security', 'maxRequestsPerMinute', parseInt(e.target.value))}
                    fullWidth
                    disabled={!localConfig.security.rateLimitEnabled}
                    inputProps={{ min: 1, max: 1000 }}
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={localConfig.security.inputSanitizationEnabled}
                        onChange={(e) => updateConfig('security', 'inputSanitizationEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Input Sanitization"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={localConfig.security.outputModerationEnabled}
                        onChange={(e) => updateConfig('security', 'outputModerationEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Output Moderation"
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Performance Settings */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Speed color="primary" />
                  <Typography variant="h6">Performance</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box display="flex" flexDirection="column" gap={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={localConfig.performance.cacheEnabled}
                        onChange={(e) => updateConfig('performance', 'cacheEnabled', e.target.checked)}
                      />
                    }
                    label="Enable Caching"
                  />

                  <TextField
                    label="Cache TTL (seconds)"
                    type="number"
                    value={localConfig.performance.cacheTtl}
                    onChange={(e) => updateConfig('performance', 'cacheTtl', parseInt(e.target.value))}
                    fullWidth
                    disabled={!localConfig.performance.cacheEnabled}
                    inputProps={{ min: 60, max: 86400 }}
                  />

                  <TextField
                    label="Max Concurrent Requests"
                    type="number"
                    value={localConfig.performance.maxConcurrentRequests}
                    onChange={(e) => updateConfig('performance', 'maxConcurrentRequests', parseInt(e.target.value))}
                    fullWidth
                    inputProps={{ min: 1, max: 100 }}
                  />

                  <TextField
                    label="Request Timeout (seconds)"
                    type="number"
                    value={localConfig.performance.requestTimeout}
                    onChange={(e) => updateConfig('performance', 'requestTimeout', parseInt(e.target.value))}
                    fullWidth
                    inputProps={{ min: 5, max: 300 }}
                  />
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>
        </CardContent>
      </Card>

      <Snackbar
        open={saveSuccess}
        autoHideDuration={3000}
        onClose={() => setSaveSuccess(false)}
      >
        <Alert severity="success" onClose={() => setSaveSuccess(false)}>
          Configuration saved successfully!
        </Alert>
      </Snackbar>

      <Snackbar
        open={Boolean(saveError)}
        autoHideDuration={5000}
        onClose={() => setSaveError(null)}
      >
        <Alert severity="error" onClose={() => setSaveError(null)}>
          {saveError}
        </Alert>
      </Snackbar>
    </>
  );
};

export default SystemConfigurationComponent;