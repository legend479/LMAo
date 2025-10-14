// Core application types and interfaces

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'user' | 'admin';
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  language: string;
  outputComplexity: 'simple' | 'intermediate' | 'advanced';
  preferredFormats: string[];
  notifications: NotificationSettings;
}

export interface NotificationSettings {
  email: boolean;
  browser: boolean;
  taskCompletion: boolean;
  systemAlerts: boolean;
}

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  timestamp: Date;
  type: 'text' | 'code' | 'document' | 'error';
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  toolsUsed?: string[];
  executionTime?: number;
  confidence?: number;
  sources?: string[];
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  status: 'active' | 'archived';
}

export interface Document {
  id: string;
  name: string;
  type: 'pdf' | 'docx' | 'ppt';
  size: number;
  createdAt: Date;
  downloadUrl: string;
  status: 'generating' | 'ready' | 'error';
  description?: string;
  tags?: string[];
  generatedBy?: string;
  previewUrl?: string;
}

export interface UploadedFile {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: Date;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  progress: number;
  error?: string;
}

export interface DocumentGenerationRequest {
  id: string;
  type: 'pdf' | 'docx' | 'ppt';
  content: string;
  template?: string;
  options?: DocumentGenerationOptions;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  createdAt: Date;
  estimatedCompletion?: Date;
  error?: string;
}

export interface DocumentGenerationOptions {
  title?: string;
  author?: string;
  subject?: string;
  keywords?: string[];
  includeTableOfContents?: boolean;
  includePageNumbers?: boolean;
  fontSize?: number;
  fontFamily?: string;
  margins?: {
    top: number;
    bottom: number;
    left: number;
    right: number;
  };
}

export interface DocumentFilter {
  type?: 'pdf' | 'docx' | 'ppt' | 'all';
  status?: 'generating' | 'ready' | 'error' | 'all';
  dateRange?: {
    start: Date;
    end: Date;
  };
  tags?: string[];
  searchQuery?: string;
}

export interface DocumentPreview {
  id: string;
  documentId: string;
  previewUrl: string;
  thumbnailUrl?: string;
  pageCount?: number;
  generatedAt: Date;
}

export interface SystemMetrics {
  cpu: number;
  memory: number;
  activeUsers: number;
  requestsPerMinute: number;
  averageResponseTime: number;
  errorRate: number;
  diskUsage?: number;
  networkIO?: {
    bytesIn: number;
    bytesOut: number;
  };
  timestamp: Date;
}

export interface SystemAlert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  resolved: boolean;
  source: string;
  severity: number;
}

export interface PerformanceMetric {
  timestamp: Date;
  value: number;
  label: string;
}

export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime: number;
  uptime: number;
  lastCheck: Date;
  dependencies: string[];
}

export interface UserSession {
  id: string;
  userId: string;
  startTime: Date;
  lastActivity: Date;
  ipAddress: string;
  userAgent: string;
  active: boolean;
}

export interface ToolExecution {
  id: string;
  toolName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  progress: number;
  result?: any;
  error?: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface WebSocketMessage {
  type: 'message' | 'typing' | 'status' | 'metrics' | 'tool_update';
  payload: any;
  timestamp: Date;
}

export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon: string;
  roles?: string[];
  children?: NavigationItem[];
}

export interface AppState {
  user: User | null;
  conversations: Conversation[];
  currentConversation: string | null;
  documents: Document[];
  systemMetrics: SystemMetrics;
  toolExecutions: ToolExecution[];
  isConnected: boolean;
  loading: boolean;
  error: string | null;
}