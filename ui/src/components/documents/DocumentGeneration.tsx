import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  CardActions,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Switch,
  FormControlLabel,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Cancel as CancelIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { documentService } from '../../services/documentService';
import {
  addGenerationRequest,
  updateGenerationRequest,
  removeGenerationRequest,
  addDocument,
  setError,
} from '../../store/slices/documentsSlice';
import { DocumentGenerationRequest, DocumentGenerationOptions } from '../../types';

interface DocumentGenerationProps {
  initialContent?: string;
  onGenerationComplete?: (documentId: string) => void;
}

const DocumentGeneration: React.FC<DocumentGenerationProps> = ({
  initialContent = '',
  onGenerationComplete,
}) => {
  const dispatch = useDispatch();
  const { generationRequests, error } = useSelector((state: RootState) => state.documents);
  
  const [content, setContent] = useState(initialContent);
  const [documentType, setDocumentType] = useState<'pdf' | 'docx' | 'ppt'>('pdf');
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [options, setOptions] = useState<DocumentGenerationOptions>({
    title: '',
    author: '',
    subject: '',
    keywords: [],
    includeTableOfContents: false,
    includePageNumbers: true,
    fontSize: 12,
    fontFamily: 'Arial',
    margins: {
      top: 1,
      bottom: 1,
      left: 1,
      right: 1,
    },
  });
  const [keywordInput, setKeywordInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // Poll for generation status updates
  useEffect(() => {
    const activeRequests = generationRequests.filter(
      req => req.status === 'pending' || req.status === 'processing'
    );

    if (activeRequests.length === 0) return;

    const pollInterval = setInterval(async () => {
      for (const request of activeRequests) {
        try {
          const result = await documentService.getGenerationStatus(request.id);
          if (result.success && result.data) {
            dispatch(updateGenerationRequest({
              id: request.id,
              updates: result.data,
            }));

            // If completed, add to documents
            if (result.data.status === 'completed') {
              // Fetch the completed document details
              // This would typically be returned with the status or fetched separately
              onGenerationComplete?.(request.id);
            }
          }
        } catch (error) {
          console.error('Failed to poll generation status:', error);
        }
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [generationRequests, dispatch, onGenerationComplete]);

  const handleGenerate = async () => {
    if (!content.trim()) {
      dispatch(setError('Content is required for document generation'));
      return;
    }

    setIsGenerating(true);
    dispatch(setError(null));

    try {
      const result = await documentService.generateDocument(content, documentType, options);
      
      if (result.success && result.data) {
        dispatch(addGenerationRequest(result.data));
      } else {
        dispatch(setError(result.error || 'Failed to start document generation'));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to start document generation'));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCancelGeneration = async (requestId: string) => {
    try {
      const result = await documentService.cancelGeneration(requestId);
      if (result.success) {
        dispatch(removeGenerationRequest(requestId));
      } else {
        dispatch(setError(result.error || 'Failed to cancel generation'));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to cancel generation'));
    }
  };

  const handleRefreshStatus = async (requestId: string) => {
    try {
      const result = await documentService.getGenerationStatus(requestId);
      if (result.success && result.data) {
        dispatch(updateGenerationRequest({
          id: requestId,
          updates: result.data,
        }));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to refresh status'));
    }
  };

  const handleAddKeyword = () => {
    if (keywordInput.trim() && !options.keywords?.includes(keywordInput.trim())) {
      setOptions(prev => ({
        ...prev,
        keywords: [...(prev.keywords || []), keywordInput.trim()],
      }));
      setKeywordInput('');
    }
  };

  const handleRemoveKeyword = (keyword: string) => {
    setOptions(prev => ({
      ...prev,
      keywords: prev.keywords?.filter(k => k !== keyword) || [],
    }));
  };

  const getStatusColor = (status: DocumentGenerationRequest['status']) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'processing':
        return 'primary';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const formatEstimatedTime = (date?: Date) => {
    if (!date) return 'Unknown';
    const now = new Date();
    const diff = date.getTime() - now.getTime();
    if (diff <= 0) return 'Any moment now';
    
    const minutes = Math.ceil(diff / (1000 * 60));
    if (minutes < 60) return `~${minutes} minutes`;
    
    const hours = Math.ceil(minutes / 60);
    return `~${hours} hours`;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Generate Document
      </Typography>

      {/* Content Input */}
      <TextField
        fullWidth
        multiline
        rows={8}
        label="Document Content"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Enter the content for your document..."
        sx={{ mb: 3 }}
      />

      {/* Document Type Selection */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Document Type</InputLabel>
            <Select
              value={documentType}
              label="Document Type"
              onChange={(e) => setDocumentType(e.target.value as 'pdf' | 'docx' | 'ppt')}
            >
              <MenuItem value="pdf">PDF</MenuItem>
              <MenuItem value="docx">Word Document (DOCX)</MenuItem>
              <MenuItem value="ppt">PowerPoint (PPT)</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            fullWidth
          >
            Advanced Options
          </Button>
        </Grid>
      </Grid>

      {/* Advanced Options */}
      {showAdvancedOptions && (
        <Accordion sx={{ mb: 3 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Document Options</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Title"
                  value={options.title}
                  onChange={(e) => setOptions(prev => ({ ...prev, title: e.target.value }))}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Author"
                  value={options.author}
                  onChange={(e) => setOptions(prev => ({ ...prev, author: e.target.value }))}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Subject"
                  value={options.subject}
                  onChange={(e) => setOptions(prev => ({ ...prev, subject: e.target.value }))}
                />
              </Grid>
              <Grid item xs={12}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Keywords
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                    {options.keywords?.map((keyword) => (
                      <Chip
                        key={keyword}
                        label={keyword}
                        onDelete={() => handleRemoveKeyword(keyword)}
                        size="small"
                      />
                    ))}
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                      size="small"
                      placeholder="Add keyword"
                      value={keywordInput}
                      onChange={(e) => setKeywordInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAddKeyword()}
                    />
                    <Button onClick={handleAddKeyword} disabled={!keywordInput.trim()}>
                      Add
                    </Button>
                  </Box>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={options.includeTableOfContents}
                      onChange={(e) => setOptions(prev => ({ 
                        ...prev, 
                        includeTableOfContents: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Table of Contents"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={options.includePageNumbers}
                      onChange={(e) => setOptions(prev => ({ 
                        ...prev, 
                        includePageNumbers: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Page Numbers"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Font Size"
                  value={options.fontSize}
                  onChange={(e) => setOptions(prev => ({ 
                    ...prev, 
                    fontSize: parseInt(e.target.value) || 12 
                  }))}
                  inputProps={{ min: 8, max: 72 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Font Family</InputLabel>
                  <Select
                    value={options.fontFamily}
                    label="Font Family"
                    onChange={(e) => setOptions(prev => ({ 
                      ...prev, 
                      fontFamily: e.target.value 
                    }))}
                  >
                    <MenuItem value="Arial">Arial</MenuItem>
                    <MenuItem value="Times New Roman">Times New Roman</MenuItem>
                    <MenuItem value="Calibri">Calibri</MenuItem>
                    <MenuItem value="Helvetica">Helvetica</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Generate Button */}
      <Button
        variant="contained"
        onClick={handleGenerate}
        disabled={isGenerating || !content.trim()}
        fullWidth
        sx={{ mb: 3 }}
      >
        {isGenerating ? 'Starting Generation...' : `Generate ${documentType.toUpperCase()}`}
      </Button>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Generation Requests */}
      {generationRequests.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Generation Progress ({generationRequests.length})
          </Typography>
          {generationRequests.map((request) => (
            <Card key={request.id} sx={{ mb: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="subtitle1">
                    {request.type.toUpperCase()} Document
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Chip
                      label={request.status}
                      color={getStatusColor(request.status) as any}
                      size="small"
                    />
                    <IconButton
                      size="small"
                      onClick={() => handleRefreshStatus(request.id)}
                      disabled={request.status === 'completed' || request.status === 'failed'}
                    >
                      <RefreshIcon />
                    </IconButton>
                  </Box>
                </Box>

                {(request.status === 'pending' || request.status === 'processing') && (
                  <Box sx={{ mb: 2 }}>
                    <LinearProgress
                      variant="determinate"
                      value={request.progress}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        {request.progress}% complete
                      </Typography>
                      {request.estimatedCompletion && (
                        <Typography variant="body2" color="text.secondary">
                          ETA: {formatEstimatedTime(request.estimatedCompletion)}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                )}

                {request.error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {request.error}
                  </Alert>
                )}

                <Typography variant="body2" color="text.secondary">
                  Started: {new Date(request.createdAt).toLocaleString()}
                </Typography>
              </CardContent>

              <CardActions>
                {request.status === 'completed' && (
                  <Button
                    startIcon={<DownloadIcon />}
                    onClick={() => {
                      // This would trigger download of the completed document
                      // The document ID would be available in the request or returned from the API
                    }}
                  >
                    Download
                  </Button>
                )}
                {(request.status === 'pending' || request.status === 'processing') && (
                  <Button
                    startIcon={<CancelIcon />}
                    onClick={() => handleCancelGeneration(request.id)}
                    color="error"
                  >
                    Cancel
                  </Button>
                )}
              </CardActions>
            </Card>
          ))}
        </Box>
      )}
    </Box>
  );
};

export default DocumentGeneration;