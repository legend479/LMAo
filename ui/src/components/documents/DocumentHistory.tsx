import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Autocomplete,
  Alert,
} from '@mui/material';
import {
  History as HistoryIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Restore as RestoreIcon,
  Folder as FolderIcon,
  Label as TagIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';

import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
} from '@mui/lab';

import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { documentService } from '../../services/documentService';
import {
  setDocuments,
  updateDocument,
  setAvailableTags,
  addTag,
  setError,
} from '../../store/slices/documentsSlice';
import { Document } from '../../types';

interface DocumentVersion {
  id: string;
  documentId: string;
  version: number;
  name: string;
  size: number;
  createdAt: Date;
  createdBy: string;
  changes: string;
  downloadUrl: string;
}

interface DocumentFolder {
  id: string;
  name: string;
  description?: string;
  documentIds: string[];
  createdAt: Date;
  color?: string;
}

interface DocumentHistoryProps {
  documentId?: string | null;
}

const DocumentHistory: React.FC<DocumentHistoryProps> = ({ documentId }) => {
  const dispatch = useDispatch();
  const { documents, availableTags, error } = useSelector((state: RootState) => state.documents);
  
  const [activeTab, setActiveTab] = useState(0);
  const [versions, setVersions] = useState<DocumentVersion[]>([]);
  const [folders, setFolders] = useState<DocumentFolder[]>([]);
  const [showCreateFolder, setShowCreateFolder] = useState(false);
  const [showTagManager, setShowTagManager] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [newFolderDescription, setNewFolderDescription] = useState('');
  const [newFolderColor, setNewFolderColor] = useState('#1976d2');
  const [newTag, setNewTag] = useState('');
  const [editingTag, setEditingTag] = useState<string | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string>('');

  useEffect(() => {
    loadDocumentHistory();
    loadFolders();
    loadTags();
  }, [documentId]);

  const loadDocumentHistory = async () => {
    if (!documentId) return;

    try {
      // This would be an API call to get document versions
      // For now, we'll simulate some version data
      const mockVersions: DocumentVersion[] = [
        {
          id: '1',
          documentId,
          version: 3,
          name: 'Current Version',
          size: 1024000,
          createdAt: new Date(),
          createdBy: 'Current User',
          changes: 'Updated content and formatting',
          downloadUrl: `/api/documents/${documentId}/versions/3/download`,
        },
        {
          id: '2',
          documentId,
          version: 2,
          name: 'Previous Version',
          size: 980000,
          createdAt: new Date(Date.now() - 86400000), // 1 day ago
          createdBy: 'John Doe',
          changes: 'Added new sections and images',
          downloadUrl: `/api/documents/${documentId}/versions/2/download`,
        },
        {
          id: '3',
          documentId,
          version: 1,
          name: 'Initial Version',
          size: 512000,
          createdAt: new Date(Date.now() - 172800000), // 2 days ago
          createdBy: 'Jane Smith',
          changes: 'Initial document creation',
          downloadUrl: `/api/documents/${documentId}/versions/1/download`,
        },
      ];
      setVersions(mockVersions);
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to load document history'));
    }
  };

  const loadFolders = async () => {
    try {
      // This would be an API call to get folders
      // For now, we'll simulate some folder data
      const mockFolders: DocumentFolder[] = [
        {
          id: '1',
          name: 'Reports',
          description: 'Monthly and quarterly reports',
          documentIds: [],
          createdAt: new Date(),
          color: '#1976d2',
        },
        {
          id: '2',
          name: 'Presentations',
          description: 'Client presentations and proposals',
          documentIds: [],
          createdAt: new Date(),
          color: '#388e3c',
        },
        {
          id: '3',
          name: 'Templates',
          description: 'Document templates for reuse',
          documentIds: [],
          createdAt: new Date(),
          color: '#f57c00',
        },
      ];
      setFolders(mockFolders);
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to load folders'));
    }
  };

  const loadTags = async () => {
    try {
      const result = await documentService.getDocumentTags();
      if (result.success && result.data) {
        dispatch(setAvailableTags(result.data));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to load tags'));
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;

    try {
      // This would be an API call to create a folder
      const newFolder: DocumentFolder = {
        id: Date.now().toString(),
        name: newFolderName,
        description: newFolderDescription,
        documentIds: selectedDocuments,
        createdAt: new Date(),
        color: newFolderColor,
      };

      setFolders(prev => [...prev, newFolder]);
      setNewFolderName('');
      setNewFolderDescription('');
      setSelectedDocuments([]);
      setShowCreateFolder(false);
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to create folder'));
    }
  };

  const handleAddTag = async () => {
    if (!newTag.trim() || availableTags.includes(newTag.trim())) return;

    try {
      dispatch(addTag(newTag.trim()));
      setNewTag('');
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to add tag'));
    }
  };

  const handleAddDocumentsToFolder = async () => {
    if (!selectedFolder || selectedDocuments.length === 0) return;

    try {
      // This would be an API call to add documents to folder
      setFolders(prev => prev.map(folder => 
        folder.id === selectedFolder 
          ? { ...folder, documentIds: [...folder.documentIds, ...selectedDocuments] }
          : folder
      ));
      setSelectedDocuments([]);
      setSelectedFolder('');
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to add documents to folder'));
    }
  };

  const handleDownloadVersion = async (version: DocumentVersion) => {
    try {
      // This would trigger download of the specific version
      const link = document.createElement('a');
      link.href = version.downloadUrl;
      link.download = `${version.name}_v${version.version}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to download version'));
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const folderColors = [
    '#1976d2', '#388e3c', '#f57c00', '#d32f2f', '#7b1fa2', 
    '#303f9f', '#1976d2', '#0288d1', '#0097a7', '#00796b'
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Document Organization & History
      </Typography>

      <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Version History" />
        <Tab label="Folders" />
        <Tab label="Tags" />
      </Tabs>

      {/* Version History Tab */}
      {activeTab === 0 && (
        <Box>
          {documentId ? (
            versions.length > 0 ? (
              <Timeline>
                {versions.map((version, index) => (
                  <TimelineItem key={version.id}>
                    <TimelineSeparator>
                      <TimelineDot color={index === 0 ? 'primary' : 'grey'}>
                        <HistoryIcon />
                      </TimelineDot>
                      {index < versions.length - 1 && <TimelineConnector />}
                    </TimelineSeparator>
                    <TimelineContent>
                      <Card sx={{ mb: 2 }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                            <Typography variant="h6">
                              Version {version.version}
                              {index === 0 && <Chip label="Current" size="small" color="primary" sx={{ ml: 1 }} />}
                            </Typography>
                            <Box>
                              <IconButton onClick={() => handleDownloadVersion(version)}>
                                <DownloadIcon />
                              </IconButton>
                              {index > 0 && (
                                <IconButton>
                                  <RestoreIcon />
                                </IconButton>
                              )}
                            </Box>
                          </Box>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            {version.changes}
                          </Typography>
                          <Typography variant="caption" display="block">
                            {formatFileSize(version.size)} • Created by {version.createdBy} • {new Date(version.createdAt).toLocaleString()}
                          </Typography>
                        </CardContent>
                      </Card>
                    </TimelineContent>
                  </TimelineItem>
                ))}
              </Timeline>
            ) : (
              <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                No version history available
              </Typography>
            )
          ) : (
            <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              Select a document to view its version history
            </Typography>
          )}
        </Box>
      )}

      {/* Folders Tab */}
      {activeTab === 1 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6">Document Folders</Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setShowCreateFolder(true)}
            >
              Create Folder
            </Button>
          </Box>

          <Grid container spacing={2}>
            {folders.map((folder) => (
              <Grid item xs={12} sm={6} md={4} key={folder.id}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <FolderIcon sx={{ color: folder.color, mr: 1 }} />
                      <Typography variant="h6">{folder.name}</Typography>
                    </Box>
                    {folder.description && (
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {folder.description}
                      </Typography>
                    )}
                    <Typography variant="caption" color="text.secondary">
                      {folder.documentIds.length} documents
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Add Documents to Folder */}
          {documents.length > 0 && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Organize Documents
                </Typography>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} md={4}>
                    <Autocomplete
                      multiple
                      options={documents}
                      getOptionLabel={(doc) => doc.name}
                      value={documents.filter(doc => selectedDocuments.includes(doc.id))}
                      onChange={(_, newValue) => setSelectedDocuments(newValue.map(doc => doc.id))}
                      renderInput={(params) => (
                        <TextField {...params} label="Select Documents" />
                      )}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel>Target Folder</InputLabel>
                      <Select
                        value={selectedFolder}
                        label="Target Folder"
                        onChange={(e) => setSelectedFolder(e.target.value)}
                      >
                        {folders.map((folder) => (
                          <MenuItem key={folder.id} value={folder.id}>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <FolderIcon sx={{ color: folder.color, mr: 1 }} />
                              {folder.name}
                            </Box>
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Button
                      variant="contained"
                      onClick={handleAddDocumentsToFolder}
                      disabled={!selectedFolder || selectedDocuments.length === 0}
                      fullWidth
                    >
                      Add to Folder
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </Box>
      )}

      {/* Tags Tab */}
      {activeTab === 2 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6">Document Tags</Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setShowTagManager(true)}
            >
              Manage Tags
            </Button>
          </Box>

          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {availableTags.map((tag) => (
              <Chip
                key={tag}
                label={tag}
                icon={<TagIcon />}
                variant="outlined"
                onDelete={() => {
                  // Handle tag deletion
                }}
                onClick={() => setEditingTag(tag)}
              />
            ))}
          </Box>

          {availableTags.length === 0 && (
            <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              No tags available. Create tags to organize your documents.
            </Typography>
          )}
        </Box>
      )}

      {/* Create Folder Dialog */}
      <Dialog open={showCreateFolder} onClose={() => setShowCreateFolder(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Folder</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 1 }}>
            <TextField
              fullWidth
              label="Folder Name"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Description (optional)"
              value={newFolderDescription}
              onChange={(e) => setNewFolderDescription(e.target.value)}
              sx={{ mb: 2 }}
            />
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Folder Color
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {folderColors.map((color) => (
                  <Box
                    key={color}
                    sx={{
                      width: 32,
                      height: 32,
                      backgroundColor: color,
                      borderRadius: 1,
                      cursor: 'pointer',
                      border: newFolderColor === color ? '3px solid #000' : '1px solid #ccc',
                    }}
                    onClick={() => setNewFolderColor(color)}
                  />
                ))}
              </Box>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreateFolder(false)}>Cancel</Button>
          <Button onClick={handleCreateFolder} variant="contained" disabled={!newFolderName.trim()}>
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Tag Manager Dialog */}
      <Dialog open={showTagManager} onClose={() => setShowTagManager(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Manage Tags</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 1 }}>
            <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
              <TextField
                fullWidth
                label="New Tag"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
              />
              <Button onClick={handleAddTag} variant="contained" disabled={!newTag.trim()}>
                Add
              </Button>
            </Box>
            <Typography variant="subtitle2" gutterBottom>
              Existing Tags
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {availableTags.map((tag) => (
                <Chip
                  key={tag}
                  label={tag}
                  onDelete={() => {
                    // Handle tag deletion
                  }}
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowTagManager(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
};

export default DocumentHistory;