import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Alert,
  Chip,
  Paper,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  InsertDriveFile as FileIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { documentService } from '../../services/documentService';
import {
  addUploadedFile,
  updateUploadedFile,
  removeUploadedFile,
  setUploadProgress,
  clearUploadProgress,
  setError,
} from '../../store/slices/documentsSlice';
import { UploadedFile } from '../../types';

interface FileUploadProps {
  onUploadComplete?: (files: UploadedFile[]) => void;
  acceptedFileTypes?: string[];
  maxFileSize?: number;
  maxFiles?: number;
}

const DropzoneContainer = styled(Paper)(({ theme, isDragActive }: { theme: any; isDragActive: boolean }) => ({
  border: `2px dashed ${isDragActive ? theme.palette.primary.main : theme.palette.grey[300]}`,
  borderRadius: theme.spacing(2),
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'all 0.2s ease-in-out',
  backgroundColor: isDragActive ? theme.palette.action.hover : 'transparent',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: theme.palette.action.hover,
  },
}));

const FileUpload: React.FC<FileUploadProps> = ({
  onUploadComplete,
  acceptedFileTypes = ['.pdf', '.docx', '.pptx', '.txt', '.md'],
  maxFileSize = 10 * 1024 * 1024, // 10MB
  maxFiles = 10,
}) => {
  const dispatch = useDispatch();
  const { uploadedFiles, uploadProgress, error } = useSelector((state: RootState) => state.documents);
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setUploading(true);
    dispatch(setError(null));

    try {
      const uploadPromises = acceptedFiles.map(async (file) => {
        const fileId = `${file.name}-${Date.now()}`;
        
        // Create initial uploaded file entry
        const uploadedFile: UploadedFile = {
          id: fileId,
          name: file.name,
          type: file.type,
          size: file.size,
          uploadedAt: new Date(),
          status: 'uploading',
          progress: 0,
        };
        
        dispatch(addUploadedFile(uploadedFile));

        try {
          const result = await documentService.uploadFile(file, (progress) => {
            dispatch(setUploadProgress({ id: fileId, progress }));
            dispatch(updateUploadedFile({
              id: fileId,
              updates: { progress, status: 'uploading' }
            }));
          });

          if (result.success && result.data) {
            dispatch(updateUploadedFile({
              id: fileId,
              updates: {
                ...result.data,
                status: 'ready',
                progress: 100,
              }
            }));
            dispatch(clearUploadProgress(fileId));
            return result.data;
          } else {
            throw new Error(result.error || 'Upload failed');
          }
        } catch (error: any) {
          dispatch(updateUploadedFile({
            id: fileId,
            updates: {
              status: 'error',
              error: error.message,
            }
          }));
          dispatch(clearUploadProgress(fileId));
          throw error;
        }
      });

      const results = await Promise.allSettled(uploadPromises);
      const successfulUploads = results
        .filter((result): result is PromiseFulfilledResult<UploadedFile> => 
          result.status === 'fulfilled'
        )
        .map(result => result.value);

      const failedUploads = results.filter(result => result.status === 'rejected');

      if (failedUploads.length > 0) {
        dispatch(setError(`${failedUploads.length} files failed to upload`));
      }

      if (successfulUploads.length > 0 && onUploadComplete) {
        onUploadComplete(successfulUploads);
      }
    } catch (error: any) {
      dispatch(setError(error.message || 'Upload failed'));
    } finally {
      setUploading(false);
    }
  }, [dispatch, onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: acceptedFileTypes.reduce((acc, type) => {
      acc[type] = [];
      return acc;
    }, {} as Record<string, string[]>),
    maxSize: maxFileSize,
    maxFiles,
    disabled: uploading,
  });

  const handleRemoveFile = (fileId: string) => {
    dispatch(removeUploadedFile(fileId));
    dispatch(clearUploadProgress(fileId));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'ready':
        return <CheckIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'uploading':
      case 'processing':
        return <FileIcon color="primary" />;
      default:
        return <FileIcon />;
    }
  };

  const getStatusColor = (status: UploadedFile['status']) => {
    switch (status) {
      case 'ready':
        return 'success';
      case 'error':
        return 'error';
      case 'uploading':
      case 'processing':
        return 'primary';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <DropzoneContainer {...getRootProps()} isDragActive={isDragActive} elevation={0}>
        <input {...getInputProps()} />
        <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          or
        </Typography>
        <Button variant="contained" disabled={uploading}>
          Browse Files
        </Button>
        <Typography variant="caption" display="block" sx={{ mt: 2 }}>
          Supported formats: {acceptedFileTypes.join(', ')}
        </Typography>
        <Typography variant="caption" display="block">
          Max file size: {formatFileSize(maxFileSize)} | Max files: {maxFiles}
        </Typography>
      </DropzoneContainer>

      {/* File rejection errors */}
      {fileRejections.length > 0 && (
        <Alert severity="error" sx={{ mt: 2 }}>
          <Typography variant="subtitle2">Some files were rejected:</Typography>
          {fileRejections.map(({ file, errors }) => (
            <Typography key={file.name} variant="body2">
              {file.name}: {errors.map(e => e.message).join(', ')}
            </Typography>
          ))}
        </Alert>
      )}

      {/* General error */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Uploaded files list */}
      {uploadedFiles.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Uploaded Files ({uploadedFiles.length})
          </Typography>
          <List>
            {uploadedFiles.map((file) => (
              <ListItem key={file.id} divider>
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                  {getStatusIcon(file.status)}
                </Box>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body1">{file.name}</Typography>
                      <Chip
                        label={file.status}
                        size="small"
                        color={getStatusColor(file.status) as any}
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {formatFileSize(file.size)} • {file.type}
                      </Typography>
                      {file.status === 'uploading' && (
                        <Box sx={{ mt: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={file.progress}
                            sx={{ height: 4, borderRadius: 2 }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {file.progress}%
                          </Typography>
                        </Box>
                      )}
                      {file.error && (
                        <Typography variant="caption" color="error">
                          Error: {file.error}
                        </Typography>
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    onClick={() => handleRemoveFile(file.id)}
                    disabled={file.status === 'uploading'}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Box>
  );
};

export default FileUpload;