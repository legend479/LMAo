import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import { toolsService, ToolSummary, ToolExecutionResult } from '../../services/toolsService';

const ToolsAdmin: React.FC = () => {
  const [tools, setTools] = useState<ToolSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [executingTool, setExecutingTool] = useState<string | null>(null);
  const [executionResult, setExecutionResult] = useState<ToolExecutionResult | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [resultDialogOpen, setResultDialogOpen] = useState(false);

  useEffect(() => {
    loadTools();
  }, []);

  const loadTools = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await toolsService.listTools();
      if (response.success && response.data) {
        setTools(response.data);
      } else {
        setError(response.error || 'Failed to load tools');
      }
    } catch (err) {
      setError('Failed to load tools');
    } finally {
      setLoading(false);
    }
  };

  const handleExecute = async (tool: ToolSummary) => {
    try {
      setExecutingTool(tool.name);
      setExecutionResult(null);
      setExecutionError(null);

      const response = await toolsService.executeTool(tool.name, {});
      if (response.success && response.data) {
        setExecutionResult(response.data);
      } else {
        setExecutionError(response.error || 'Tool execution failed');
      }
      setResultDialogOpen(true);
    } catch (err) {
      setExecutionError('Tool execution failed');
      setResultDialogOpen(true);
    } finally {
      setExecutingTool(null);
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Tools Management</Typography>
        <Button
          variant="outlined"
          size="small"
          onClick={loadTools}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {loading && (
        <Box display="flex" justifyContent="center" my={3}>
          <CircularProgress size={32} />
        </Box>
      )}

      {error && (
        <Typography color="error" variant="body2" mb={2}>
          {error}
        </Typography>
      )}

      {!loading && !error && tools.length === 0 && (
        <Typography variant="body2" color="text.secondary">
          No tools available.
        </Typography>
      )}

      {!loading && tools.length > 0 && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Category</TableCell>
                <TableCell>Tags</TableCell>
                <TableCell align="right">Usage</TableCell>
                <TableCell align="right">Success Rate</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tools.map((tool) => (
                <TableRow key={tool.name} hover>
                  <TableCell>{tool.name}</TableCell>
                  <TableCell>{tool.description}</TableCell>
                  <TableCell>{tool.category || '-'}</TableCell>
                  <TableCell>
                    {tool.tags && tool.tags.length > 0 ? (
                      <Box display="flex" flexWrap="wrap" gap={0.5}>
                        {tool.tags.map((tag) => (
                          <Chip key={tag} label={tag} size="small" />
                        ))}
                      </Box>
                    ) : (
                      <Typography variant="caption" color="text.secondary">
                        No tags
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell align="right">{tool.usage_count}</TableCell>
                  <TableCell align="right">
                    {(tool.success_rate * 100).toFixed(1)}%
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Execute tool">
                      <span>
                        <IconButton
                          size="small"
                          onClick={() => handleExecute(tool)}
                          disabled={!!executingTool}
                        >
                          {executingTool === tool.name ? (
                            <CircularProgress size={20} />
                          ) : (
                            <PlayArrow />
                          )}
                        </IconButton>
                      </span>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog
        open={resultDialogOpen}
        onClose={() => setResultDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Tool Execution Result</DialogTitle>
        <DialogContent dividers>
          {executionError && (
            <Typography color="error" gutterBottom>
              {executionError}
            </Typography>
          )}
          {executionResult && (
            <Box component="pre" sx={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>
              {JSON.stringify(executionResult, null, 2)}
            </Box>
          )}
          {!executionError && !executionResult && (
            <Typography variant="body2" color="text.secondary">
              No result available.
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResultDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ToolsAdmin;
