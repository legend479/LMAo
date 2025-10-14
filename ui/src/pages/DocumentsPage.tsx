import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  Grid,
  Card,
  CardContent,
  Fab,
  Badge,
  Alert,
} from '@mui/material';
import {
  Add as AddIcon,
  CloudUpload as UploadIcon,
  Description as DocumentIcon,
  History as HistoryIcon,
  Folder as FolderIcon,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { documentService } from '../services/documentService';
import {
  setDocuments,
  setAvailableTags,
  setLoading,
  setError,
} from '../store/slices/documentsSlice';
import { Document, UploadedFile, DocumentGenerationRequest } from '../types';

// Import our new components
import FileUpload from '../components/documents/FileUpload';
import DocumentGeneration from '../components/documents/DocumentGeneration';
import DocumentList from '../components/documents/DocumentList';
import DocumentHistory from '../components/documents/DocumentHistory';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`document-tabpanel-${index}`}
      aria-labelledby={`document-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
};

const DocumentsPage: React.FC = () => {
  const dispatch = useDispatch();
  const { 
    documents, 
    uploadedFiles, 
    generationRequests, 
    error 
  } = useSelector((state: RootState) => state.documents);

  const [activeTab, setActiveTab] = useState(0);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    dispatch(setLoading(true));
    
    try {
      // Load documents
      const documentsResult = await documentService.getDocuments();
      if (documentsResult.success && documentsResult.data) {
        dispatch(setDocuments(documentsResult.data));
      }

      // Load available tags
      const tagsResult = await documentService.getDocumentTags();
      if (tagsResult.success && tagsResult.data) {
        dispatch(setAvailableTags(tagsResult.data));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to load initial data'));
    } finally {
      dispatch(setLoading(false));
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleUploadComplete = (_files: UploadedFile[]) => {
    // Refresh documents list after successful upload
    loadInitialData();
  };

  const handleGenerationComplete = (_documentId: string) => {
    // Refresh documents list after successful generation
    loadInitialData();
  };

  const handleDocumentSelect = (document: Document) => {
    setSelectedDocumentId(document.id);
    setActiveTab(3); // Switch to history tab
  };

  const getActiveGenerationCount = () => {
    return generationRequests.filter(
      (req: DocumentGenerationRequest) => req.status === 'pending' || req.status === 'processing'
    ).length;
  };

  const getReadyDocumentsCount = () => {
    return documents.filter((doc: Document) => doc.status === 'ready').length;
  };

  const getRecentUploadsCount = () => {
    const oneDayAgo = new Date();
    oneDayAgo.setDate(oneDayAgo.getDate() - 1);
    return uploadedFiles.filter(
      (file: UploadedFile) => new Date(file.uploadedAt) > oneDayAgo && file.status === 'ready'
    ).length;
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Document Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload, generate, and manage your documents with advanced features
        </Typography>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="primary">
                    {documents.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Documents
                  </Typography>
                </Box>
                <DocumentIcon sx={{ fontSize: 40, color: 'primary.main', opacity: 0.7 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="success.main">
                    {getReadyDocumentsCount()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Ready to Download
                  </Typography>
                </Box>
                <Badge badgeContent={getActiveGenerationCount()} color="warning">
                  <DocumentIcon sx={{ fontSize: 40, color: 'success.main', opacity: 0.7 }} />
                </Badge>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="info.main">
                    {uploadedFiles.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Uploaded Files
                  </Typography>
                </Box>
                <Badge badgeContent={getRecentUploadsCount()} color="info">
                  <UploadIcon sx={{ fontSize: 40, color: 'info.main', opacity: 0.7 }} />
                </Badge>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="warning.main">
                    {getActiveGenerationCount()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Generating
                  </Typography>
                </Box>
                <HistoryIcon sx={{ fontSize: 40, color: 'warning.main', opacity: 0.7 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Main Content */}
      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="document management tabs">
            <Tab 
              label="All Documents" 
              icon={<DocumentIcon />} 
              iconPosition="start"
              id="document-tab-0"
              aria-controls="document-tabpanel-0"
            />
            <Tab 
              label="Upload Files" 
              icon={<UploadIcon />} 
              iconPosition="start"
              id="document-tab-1"
              aria-controls="document-tabpanel-1"
            />
            <Tab 
              label="Generate Documents" 
              icon={<AddIcon />} 
              iconPosition="start"
              id="document-tab-2"
              aria-controls="document-tabpanel-2"
            />
            <Tab 
              label="History & Organization" 
              icon={<FolderIcon />} 
              iconPosition="start"
              id="document-tab-3"
              aria-controls="document-tabpanel-3"
            />
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <TabPanel value={activeTab} index={0}>
          <DocumentList
            onDocumentSelect={handleDocumentSelect}
            selectable={true}
            showFilters={true}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <FileUpload
            onUploadComplete={handleUploadComplete}
            acceptedFileTypes={['.pdf', '.docx', '.pptx', '.txt', '.md']}
            maxFileSize={10 * 1024 * 1024} // 10MB
            maxFiles={10}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <DocumentGeneration
            onGenerationComplete={handleGenerationComplete}
          />
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          <DocumentHistory documentId={selectedDocumentId} />
        </TabPanel>
      </Paper>

      {/* Floating Action Buttons */}
      <Box sx={{ position: 'fixed', bottom: 24, right: 24, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Fab
          color="primary"
          aria-label="upload"
          onClick={() => setActiveTab(1)}
          sx={{ display: activeTab !== 1 ? 'flex' : 'none' }}
        >
          <UploadIcon />
        </Fab>
        <Fab
          color="secondary"
          aria-label="generate"
          onClick={() => setActiveTab(2)}
          sx={{ display: activeTab !== 2 ? 'flex' : 'none' }}
        >
          <AddIcon />
        </Fab>
        <Fab
          color="info"
          aria-label="history"
          onClick={() => setActiveTab(3)}
          sx={{ display: activeTab !== 3 ? 'flex' : 'none' }}
        >
          <HistoryIcon />
        </Fab>
      </Box>
    </Box>
  );
};

export default DocumentsPage;