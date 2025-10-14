import axios from 'axios';
import { 
  Document, 
  UploadedFile, 
  DocumentGenerationRequest, 
  DocumentGenerationOptions,
  DocumentFilter,
  DocumentPreview,
  ApiResponse 
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const documentService = {
  // Document management
  async getDocuments(filter?: DocumentFilter): Promise<ApiResponse<Document[]>> {
    try {
      const response = await api.get('/documents', { params: filter });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch documents',
      };
    }
  },

  async getDocument(id: string): Promise<ApiResponse<Document>> {
    try {
      const response = await api.get(`/documents/${id}`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch document',
      };
    }
  },

  async deleteDocument(id: string): Promise<ApiResponse<void>> {
    try {
      const response = await api.delete(`/documents/${id}`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to delete document',
      };
    }
  },

  async updateDocument(id: string, updates: Partial<Document>): Promise<ApiResponse<Document>> {
    try {
      const response = await api.patch(`/documents/${id}`, updates);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to update document',
      };
    }
  },

  // File upload
  async uploadFile(
    file: File, 
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<UploadedFile>> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to upload file',
      };
    }
  },

  async uploadMultipleFiles(
    files: File[],
    onProgress?: (fileId: string, progress: number) => void
  ): Promise<ApiResponse<UploadedFile[]>> {
    try {
      const uploadPromises = files.map((file, index) => {
        const fileId = `${file.name}-${index}`;
        return this.uploadFile(file, (progress) => {
          onProgress?.(fileId, progress);
        });
      });

      const results = await Promise.all(uploadPromises);
      const successfulUploads = results
        .filter(result => result.success)
        .map(result => result.data!);

      const failedUploads = results.filter(result => !result.success);

      if (failedUploads.length > 0) {
        return {
          success: false,
          error: `${failedUploads.length} files failed to upload`,
          data: successfulUploads,
        };
      }

      return {
        success: true,
        data: successfulUploads,
      };
    } catch (error: any) {
      return {
        success: false,
        error: 'Failed to upload files',
      };
    }
  },

  // Document generation
  async generateDocument(
    content: string,
    type: 'pdf' | 'docx' | 'ppt',
    options?: DocumentGenerationOptions
  ): Promise<ApiResponse<DocumentGenerationRequest>> {
    try {
      const response = await api.post('/documents/generate', {
        content,
        type,
        options,
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to start document generation',
      };
    }
  },

  async getGenerationStatus(requestId: string): Promise<ApiResponse<DocumentGenerationRequest>> {
    try {
      const response = await api.get(`/documents/generate/${requestId}/status`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to get generation status',
      };
    }
  },

  async cancelGeneration(requestId: string): Promise<ApiResponse<void>> {
    try {
      const response = await api.post(`/documents/generate/${requestId}/cancel`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to cancel generation',
      };
    }
  },

  // Document download and preview
  async downloadDocument(id: string, filename?: string): Promise<void> {
    try {
      const response = await api.get(`/documents/${id}/download`, {
        responseType: 'blob',
      });

      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Get filename from response headers or use provided filename
      const contentDisposition = response.headers['content-disposition'];
      let downloadFilename = filename;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          downloadFilename = filenameMatch[1];
        }
      }
      
      link.download = downloadFilename || 'document';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to download document');
    }
  },

  async getDocumentPreview(id: string): Promise<ApiResponse<DocumentPreview>> {
    try {
      const response = await api.get(`/documents/${id}/preview`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to get document preview',
      };
    }
  },

  // Document search and filtering
  async searchDocuments(query: string, filter?: DocumentFilter): Promise<ApiResponse<Document[]>> {
    try {
      const response = await api.get('/documents/search', {
        params: { q: query, ...filter },
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to search documents',
      };
    }
  },

  // Document organization
  async getDocumentTags(): Promise<ApiResponse<string[]>> {
    try {
      const response = await api.get('/documents/tags');
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch document tags',
      };
    }
  },

  async addDocumentTag(documentId: string, tag: string): Promise<ApiResponse<Document>> {
    try {
      const response = await api.post(`/documents/${documentId}/tags`, { tag });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to add tag',
      };
    }
  },

  async removeDocumentTag(documentId: string, tag: string): Promise<ApiResponse<Document>> {
    try {
      const response = await api.delete(`/documents/${documentId}/tags/${tag}`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to remove tag',
      };
    }
  },
};