import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  IconButton,
  Grid,
  Chip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  Checkbox,
  FormControlLabel,
  Tooltip,
  Alert,
  Pagination,
  Skeleton,
} from '@mui/material';
import {
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Preview as PreviewIcon,
  MoreVert as MoreIcon,
  Edit as EditIcon,
  Share as ShareIcon,
  FilterList as FilterIcon,
  Search as SearchIcon,
  Sort as SortIcon,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { documentService } from '../../services/documentService';
import {
  setDocuments,
  updateDocument,
  removeDocument,
  setFilter,
  setSearchQuery,
  toggleDocumentSelection,
  setSelectedDocuments,
  clearSelection,
  setLoading,
  setError,
} from '../../store/slices/documentsSlice';
import { Document, DocumentFilter } from '../../types';

interface DocumentListProps {
  onDocumentSelect?: (document: Document) => void;
  selectable?: boolean;
  showFilters?: boolean;
}

const DocumentList: React.FC<DocumentListProps> = ({
  onDocumentSelect,
  selectable = false,
  showFilters = true,
}) => {
  const dispatch = useDispatch();
  const { 
    documents, 
    loading, 
    error, 
    filter, 
    searchQuery, 
    selectedDocuments 
  } = useSelector((state: RootState) => state.documents);

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [showFilterDialog, setShowFilterDialog] = useState(false);
  const [showPreviewDialog, setShowPreviewDialog] = useState(false);
  const [previewDocument, setPreviewDocument] = useState<Document | null>(null);
  const [editingDocument, setEditingDocument] = useState<Document | null>(null);
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState<'name' | 'createdAt' | 'size'>('createdAt');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  
  const itemsPerPage = 12;

  // Load documents on component mount and when filters change
  useEffect(() => {
    loadDocuments();
  }, [filter, searchQuery, sortBy, sortOrder, page]);

  const loadDocuments = async () => {
    dispatch(setLoading(true));
    dispatch(setError(null));

    try {
      const result = await documentService.getDocuments(filter);
      if (result.success && result.data) {
        // Apply client-side filtering and sorting
        let filteredDocs = result.data;

        // Search filter
        if (searchQuery) {
          filteredDocs = filteredDocs.filter(doc =>
            doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            doc.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            doc.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
          );
        }

        // Sort documents
        filteredDocs.sort((a, b) => {
          let aValue: any, bValue: any;
          
          switch (sortBy) {
            case 'name':
              aValue = a.name.toLowerCase();
              bValue = b.name.toLowerCase();
              break;
            case 'createdAt':
              aValue = new Date(a.createdAt).getTime();
              bValue = new Date(b.createdAt).getTime();
              break;
            case 'size':
              aValue = a.size;
              bValue = b.size;
              break;
            default:
              return 0;
          }

          if (sortOrder === 'asc') {
            return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
          } else {
            return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
          }
        });

        dispatch(setDocuments(filteredDocs));
      } else {
        dispatch(setError(result.error || 'Failed to load documents'));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to load documents'));
    } finally {
      dispatch(setLoading(false));
    }
  };

  const handleDownload = async (document: Document) => {
    try {
      await documentService.downloadDocument(document.id, document.name);
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to download document'));
    }
  };

  const handleDelete = async (document: Document) => {
    if (window.confirm(`Are you sure you want to delete "${document.name}"?`)) {
      try {
        const result = await documentService.deleteDocument(document.id);
        if (result.success) {
          dispatch(removeDocument(document.id));
        } else {
          dispatch(setError(result.error || 'Failed to delete document'));
        }
      } catch (error: any) {
        dispatch(setError(error.message || 'Failed to delete document'));
      }
    }
  };

  const handlePreview = async (document: Document) => {
    setPreviewDocument(document);
    setShowPreviewDialog(true);
  };

  const handleEdit = (document: Document) => {
    setEditingDocument({ ...document });
  };

  const handleSaveEdit = async () => {
    if (!editingDocument) return;

    try {
      const result = await documentService.updateDocument(editingDocument.id, {
        name: editingDocument.name,
        description: editingDocument.description,
        tags: editingDocument.tags,
      });

      if (result.success && result.data) {
        dispatch(updateDocument({
          id: editingDocument.id,
          updates: result.data,
        }));
        setEditingDocument(null);
      } else {
        dispatch(setError(result.error || 'Failed to update document'));
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Failed to update document'));
    }
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>, document: Document) => {
    setAnchorEl(event.currentTarget);
    setSelectedDoc(document);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedDoc(null);
  };

  const handleFilterChange = (newFilter: Partial<DocumentFilter>) => {
    dispatch(setFilter({ ...filter, ...newFilter }));
    setPage(1);
  };

  const handleSearchChange = (query: string) => {
    dispatch(setSearchQuery(query));
    setPage(1);
  };

  const handleSelectAll = () => {
    if (selectedDocuments.length === paginatedDocuments.length) {
      dispatch(clearSelection());
    } else {
      dispatch(setSelectedDocuments(paginatedDocuments.map(doc => doc.id)));
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status: Document['status']) => {
    switch (status) {
      case 'ready':
        return 'success';
      case 'error':
        return 'error';
      case 'generating':
        return 'warning';
      default:
        return 'default';
    }
  };

  // Pagination
  const startIndex = (page - 1) * itemsPerPage;
  const paginatedDocuments = documents.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(documents.length / itemsPerPage);

  return (
    <Box>
      {/* Header with search and filters */}
      {showFilters && (
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                <Button
                  startIcon={<FilterIcon />}
                  onClick={() => setShowFilterDialog(true)}
                  variant="outlined"
                >
                  Filters
                </Button>
                <Button
                  startIcon={<SortIcon />}
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  variant="outlined"
                >
                  {sortOrder === 'asc' ? 'A-Z' : 'Z-A'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>
      )}

      {/* Selection controls */}
      {selectable && documents.length > 0 && (
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={selectedDocuments.length === paginatedDocuments.length && paginatedDocuments.length > 0}
                indeterminate={selectedDocuments.length > 0 && selectedDocuments.length < paginatedDocuments.length}
                onChange={handleSelectAll}
              />
            }
            label={`Select All (${selectedDocuments.length} selected)`}
          />
          {selectedDocuments.length > 0 && (
            <Button
              color="error"
              onClick={() => {
                if (window.confirm(`Delete ${selectedDocuments.length} selected documents?`)) {
                  // Handle bulk delete
                  selectedDocuments.forEach(id => {
                    const doc = documents.find(d => d.id === id);
                    if (doc) handleDelete(doc);
                  });
                }
              }}
            >
              Delete Selected ({selectedDocuments.length})
            </Button>
          )}
        </Box>
      )}

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Documents grid */}
      {loading ? (
        <Grid container spacing={2}>
          {Array.from({ length: 6 }).map((_, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card>
                <CardContent>
                  <Skeleton variant="text" width="80%" height={24} />
                  <Skeleton variant="text" width="60%" height={20} />
                  <Skeleton variant="text" width="40%" height={16} />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : documents.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h6" color="text.secondary">
            No documents found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {searchQuery || filter.type !== 'all' || filter.status !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Upload or generate your first document to get started'}
          </Typography>
        </Box>
      ) : (
        <>
          <Grid container spacing={2}>
            {paginatedDocuments.map((document) => (
              <Grid item xs={12} sm={6} md={4} key={document.id}>
                <Card 
                  sx={{ 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column',
                    cursor: onDocumentSelect ? 'pointer' : 'default',
                    border: selectedDocuments.includes(document.id) ? 2 : 0,
                    borderColor: 'primary.main',
                  }}
                  onClick={() => {
                    if (selectable) {
                      dispatch(toggleDocumentSelection(document.id));
                    } else if (onDocumentSelect) {
                      onDocumentSelect(document);
                    }
                  }}
                >
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="h6" component="h3" noWrap sx={{ flexGrow: 1, mr: 1 }}>
                        {document.name}
                      </Typography>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleMenuClick(e, document);
                        }}
                      >
                        <MoreIcon />
                      </IconButton>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                      <Chip
                        label={document.type.toUpperCase()}
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        label={document.status}
                        size="small"
                        color={getStatusColor(document.status) as any}
                      />
                    </Box>

                    {document.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {document.description}
                      </Typography>
                    )}

                    <Typography variant="caption" color="text.secondary" display="block">
                      Size: {formatFileSize(document.size)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Created: {new Date(document.createdAt).toLocaleDateString()}
                    </Typography>

                    {document.tags && document.tags.length > 0 && (
                      <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {document.tags.slice(0, 3).map((tag) => (
                          <Chip key={tag} label={tag} size="small" variant="outlined" />
                        ))}
                        {document.tags.length > 3 && (
                          <Chip label={`+${document.tags.length - 3}`} size="small" variant="outlined" />
                        )}
                      </Box>
                    )}
                  </CardContent>

                  <CardActions>
                    <Tooltip title="Download">
                      <IconButton
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownload(document);
                        }}
                        disabled={document.status !== 'ready'}
                      >
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Preview">
                      <IconButton
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePreview(document);
                        }}
                        disabled={document.status !== 'ready'}
                      >
                        <PreviewIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Pagination */}
          {totalPages > 1 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={(_, newPage) => setPage(newPage)}
                color="primary"
              />
            </Box>
          )}
        </>
      )}

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { selectedDoc && handleDownload(selectedDoc); handleMenuClose(); }}>
          <DownloadIcon sx={{ mr: 1 }} /> Download
        </MenuItem>
        <MenuItem onClick={() => { selectedDoc && handlePreview(selectedDoc); handleMenuClose(); }}>
          <PreviewIcon sx={{ mr: 1 }} /> Preview
        </MenuItem>
        <MenuItem onClick={() => { selectedDoc && handleEdit(selectedDoc); handleMenuClose(); }}>
          <EditIcon sx={{ mr: 1 }} /> Edit Details
        </MenuItem>
        <MenuItem onClick={() => { selectedDoc && handleDelete(selectedDoc); handleMenuClose(); }}>
          <DeleteIcon sx={{ mr: 1 }} /> Delete
        </MenuItem>
      </Menu>

      {/* Filter Dialog */}
      <Dialog open={showFilterDialog} onClose={() => setShowFilterDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Filter Documents</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Document Type</InputLabel>
                <Select
                  value={filter.type || 'all'}
                  label="Document Type"
                  onChange={(e) => handleFilterChange({ type: e.target.value as any })}
                >
                  <MenuItem value="all">All Types</MenuItem>
                  <MenuItem value="pdf">PDF</MenuItem>
                  <MenuItem value="docx">Word Document</MenuItem>
                  <MenuItem value="ppt">PowerPoint</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={filter.status || 'all'}
                  label="Status"
                  onChange={(e) => handleFilterChange({ status: e.target.value as any })}
                >
                  <MenuItem value="all">All Status</MenuItem>
                  <MenuItem value="ready">Ready</MenuItem>
                  <MenuItem value="generating">Generating</MenuItem>
                  <MenuItem value="error">Error</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={sortBy}
                  label="Sort By"
                  onChange={(e) => setSortBy(e.target.value as any)}
                >
                  <MenuItem value="createdAt">Date Created</MenuItem>
                  <MenuItem value="name">Name</MenuItem>
                  <MenuItem value="size">File Size</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowFilterDialog(false)}>Cancel</Button>
          <Button onClick={() => setShowFilterDialog(false)} variant="contained">Apply</Button>
        </DialogActions>
      </Dialog>

      {/* Edit Document Dialog */}
      <Dialog open={Boolean(editingDocument)} onClose={() => setEditingDocument(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Document Details</DialogTitle>
        <DialogContent>
          {editingDocument && (
            <Box sx={{ mt: 1 }}>
              <TextField
                fullWidth
                label="Document Name"
                value={editingDocument.name}
                onChange={(e) => setEditingDocument({ ...editingDocument, name: e.target.value })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                value={editingDocument.description || ''}
                onChange={(e) => setEditingDocument({ ...editingDocument, description: e.target.value })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Tags (comma-separated)"
                value={editingDocument.tags?.join(', ') || ''}
                onChange={(e) => setEditingDocument({ 
                  ...editingDocument, 
                  tags: e.target.value.split(',').map(tag => tag.trim()).filter(Boolean)
                })}
                helperText="Enter tags separated by commas"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditingDocument(null)}>Cancel</Button>
          <Button onClick={handleSaveEdit} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog 
        open={showPreviewDialog} 
        onClose={() => setShowPreviewDialog(false)} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>
          Document Preview: {previewDocument?.name}
        </DialogTitle>
        <DialogContent>
          {previewDocument && (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body1" color="text.secondary">
                Document preview functionality will be implemented based on document type.
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                Type: {previewDocument.type.toUpperCase()} | 
                Size: {formatFileSize(previewDocument.size)} |
                Created: {new Date(previewDocument.createdAt).toLocaleDateString()}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPreviewDialog(false)}>Close</Button>
          {previewDocument && (
            <Button 
              onClick={() => handleDownload(previewDocument)} 
              variant="contained"
              startIcon={<DownloadIcon />}
            >
              Download
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DocumentList;