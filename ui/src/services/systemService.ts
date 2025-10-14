import axios from 'axios';
import { SystemMetrics, User, ApiResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: ServiceStatus[];
  uptime: number;
  version: string;
  lastCheck: Date;
}

export interface ServiceStatus {
  name: string;
  status: 'online' | 'offline' | 'degraded';
  responseTime: number;
  lastCheck: Date;
  url?: string;
  error?: string;
}

export interface UsageAnalytics {
  totalUsers: number;
  activeUsers: number;
  totalConversations: number;
  totalDocuments: number;
  popularTools: ToolUsageStats[];
  userActivity: UserActivityStats[];
  systemLoad: SystemLoadStats[];
}

export interface ToolUsageStats {
  toolName: string;
  usageCount: number;
  successRate: number;
  averageExecutionTime: number;
}

export interface UserActivityStats {
  date: string;
  activeUsers: number;
  newUsers: number;
  totalSessions: number;
  averageSessionDuration: number;
}

export interface SystemLoadStats {
  timestamp: string;
  cpu: number;
  memory: number;
  requestsPerMinute: number;
  responseTime: number;
}

export interface SystemConfiguration {
  ragPipeline: {
    chunkSize: number;
    overlapSize: number;
    embeddingModel: string;
    rerankerEnabled: boolean;
    hybridSearchWeight: number;
  };
  llmSettings: {
    model: string;
    temperature: number;
    maxTokens: number;
    topP: number;
    frequencyPenalty: number;
  };
  security: {
    rateLimitEnabled: boolean;
    maxRequestsPerMinute: number;
    inputSanitizationEnabled: boolean;
    outputModerationEnabled: boolean;
  };
  performance: {
    cacheEnabled: boolean;
    cacheTtl: number;
    maxConcurrentRequests: number;
    requestTimeout: number;
  };
}

class SystemService {
  private axiosInstance;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: `${API_BASE_URL}/api/system`,
      timeout: 10000,
    });

    // Add auth token to requests
    this.axiosInstance.interceptors.request.use((config) => {
      const token = localStorage.getItem('authToken');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.axiosInstance.get<ApiResponse<SystemHealth>>('/health');
    return response.data.data!;
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await this.axiosInstance.get<ApiResponse<SystemMetrics>>('/metrics');
    return response.data.data!;
  }

  async getUsageAnalytics(timeRange: '1h' | '24h' | '7d' | '30d' = '24h'): Promise<UsageAnalytics> {
    const response = await this.axiosInstance.get<ApiResponse<UsageAnalytics>>(`/analytics?range=${timeRange}`);
    return response.data.data!;
  }

  async getSystemConfiguration(): Promise<SystemConfiguration> {
    const response = await this.axiosInstance.get<ApiResponse<SystemConfiguration>>('/config');
    return response.data.data!;
  }

  async updateSystemConfiguration(config: Partial<SystemConfiguration>): Promise<SystemConfiguration> {
    const response = await this.axiosInstance.put<ApiResponse<SystemConfiguration>>('/config', config);
    return response.data.data!;
  }

  async getAllUsers(page: number = 1, limit: number = 20): Promise<{ users: User[]; total: number; page: number; totalPages: number }> {
    const response = await this.axiosInstance.get<ApiResponse<{ users: User[]; total: number; page: number; totalPages: number }>>(`/users?page=${page}&limit=${limit}`);
    return response.data.data!;
  }

  async updateUser(userId: string, updates: Partial<User>): Promise<User> {
    const response = await this.axiosInstance.put<ApiResponse<User>>(`/users/${userId}`, updates);
    return response.data.data!;
  }

  async deleteUser(userId: string): Promise<void> {
    await this.axiosInstance.delete(`/users/${userId}`);
  }

  async getUserActivity(userId: string, timeRange: '1h' | '24h' | '7d' | '30d' = '24h'): Promise<UserActivityStats[]> {
    const response = await this.axiosInstance.get<ApiResponse<UserActivityStats[]>>(`/users/${userId}/activity?range=${timeRange}`);
    return response.data.data!;
  }

  async exportSystemLogs(timeRange: '1h' | '24h' | '7d' | '30d' = '24h'): Promise<Blob> {
    const response = await this.axiosInstance.get(`/logs/export?range=${timeRange}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async restartService(serviceName: string): Promise<void> {
    await this.axiosInstance.post(`/services/${serviceName}/restart`);
  }

  async getSystemAlerts(): Promise<Array<{
    id: string;
    type: 'info' | 'warning' | 'error';
    message: string;
    timestamp: Date;
    resolved: boolean;
  }>> {
    const response = await this.axiosInstance.get<ApiResponse<Array<{
      id: string;
      type: 'info' | 'warning' | 'error';
      message: string;
      timestamp: Date;
      resolved: boolean;
    }>>>('/alerts');
    return response.data.data!;
  }

  async resolveAlert(alertId: string): Promise<void> {
    await this.axiosInstance.put(`/alerts/${alertId}/resolve`);
  }
}

export const systemService = new SystemService();