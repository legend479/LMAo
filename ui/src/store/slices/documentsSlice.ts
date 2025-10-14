import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Document, UploadedFile, DocumentGenerationRequest, DocumentFilter } from '../../types';

interface DocumentsState {
  documents: Document[];
  uploadedFiles: UploadedFile[];
  generationRequests: DocumentGenerationRequest[];
  loading: boolean;
  error: string | null;
  uploadProgress: { [key: string]: number };
  filter: DocumentFilter;
  searchQuery: string;
  selectedDocuments: string[];
  availableTags: string[];
}

const initialState: DocumentsState = {
  documents: [],
  uploadedFiles: [],
  generationRequests: [],
  loading: false,
  error: null,
  uploadProgress: {},
  filter: { type: 'all', status: 'all' },
  searchQuery: '',
  selectedDocuments: [],
  availableTags: [],
};

const documentsSlice = createSlice({
  name: 'documents',
  initialState,
  reducers: {
    // Document management
    setDocuments: (state, action: PayloadAction<Document[]>) => {
      state.documents = action.payload;
    },
    addDocument: (state, action: PayloadAction<Document>) => {
      state.documents.unshift(action.payload);
    },
    updateDocument: (state, action: PayloadAction<{ id: string; updates: Partial<Document> }>) => {
      const index = state.documents.findIndex(d => d.id === action.payload.id);
      if (index !== -1) {
        state.documents[index] = { ...state.documents[index], ...action.payload.updates };
      }
    },
    removeDocument: (state, action: PayloadAction<string>) => {
      state.documents = state.documents.filter(d => d.id !== action.payload);
      state.selectedDocuments = state.selectedDocuments.filter(id => id !== action.payload);
    },
    
    // File upload management
    setUploadedFiles: (state, action: PayloadAction<UploadedFile[]>) => {
      state.uploadedFiles = action.payload;
    },
    addUploadedFile: (state, action: PayloadAction<UploadedFile>) => {
      state.uploadedFiles.unshift(action.payload);
    },
    updateUploadedFile: (state, action: PayloadAction<{ id: string; updates: Partial<UploadedFile> }>) => {
      const index = state.uploadedFiles.findIndex(f => f.id === action.payload.id);
      if (index !== -1) {
        state.uploadedFiles[index] = { ...state.uploadedFiles[index], ...action.payload.updates };
      }
    },
    removeUploadedFile: (state, action: PayloadAction<string>) => {
      state.uploadedFiles = state.uploadedFiles.filter(f => f.id !== action.payload);
    },
    
    // Generation request management
    setGenerationRequests: (state, action: PayloadAction<DocumentGenerationRequest[]>) => {
      state.generationRequests = action.payload;
    },
    addGenerationRequest: (state, action: PayloadAction<DocumentGenerationRequest>) => {
      state.generationRequests.unshift(action.payload);
    },
    updateGenerationRequest: (state, action: PayloadAction<{ id: string; updates: Partial<DocumentGenerationRequest> }>) => {
      const index = state.generationRequests.findIndex(r => r.id === action.payload.id);
      if (index !== -1) {
        state.generationRequests[index] = { ...state.generationRequests[index], ...action.payload.updates };
      }
    },
    removeGenerationRequest: (state, action: PayloadAction<string>) => {
      state.generationRequests = state.generationRequests.filter(r => r.id !== action.payload);
    },
    
    // UI state management
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setUploadProgress: (state, action: PayloadAction<{ id: string; progress: number }>) => {
      state.uploadProgress[action.payload.id] = action.payload.progress;
    },
    clearUploadProgress: (state, action: PayloadAction<string>) => {
      delete state.uploadProgress[action.payload];
    },
    
    // Filter and search
    setFilter: (state, action: PayloadAction<DocumentFilter>) => {
      state.filter = action.payload;
    },
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload;
    },
    
    // Selection management
    setSelectedDocuments: (state, action: PayloadAction<string[]>) => {
      state.selectedDocuments = action.payload;
    },
    toggleDocumentSelection: (state, action: PayloadAction<string>) => {
      const id = action.payload;
      if (state.selectedDocuments.includes(id)) {
        state.selectedDocuments = state.selectedDocuments.filter(docId => docId !== id);
      } else {
        state.selectedDocuments.push(id);
      }
    },
    clearSelection: (state) => {
      state.selectedDocuments = [];
    },
    
    // Tags management
    setAvailableTags: (state, action: PayloadAction<string[]>) => {
      state.availableTags = action.payload;
    },
    addTag: (state, action: PayloadAction<string>) => {
      if (!state.availableTags.includes(action.payload)) {
        state.availableTags.push(action.payload);
      }
    },
  },
});

export const {
  setDocuments,
  addDocument,
  updateDocument,
  removeDocument,
  setUploadedFiles,
  addUploadedFile,
  updateUploadedFile,
  removeUploadedFile,
  setGenerationRequests,
  addGenerationRequest,
  updateGenerationRequest,
  removeGenerationRequest,
  setLoading,
  setError,
  setUploadProgress,
  clearUploadProgress,
  setFilter,
  setSearchQuery,
  setSelectedDocuments,
  toggleDocumentSelection,
  clearSelection,
  setAvailableTags,
  addTag,
} = documentsSlice.actions;

export default documentsSlice.reducer;