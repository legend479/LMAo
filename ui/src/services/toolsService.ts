import axios from 'axios';
import { ApiResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface ToolSummary {
  name: string;
  description: string;
  parameters: Record<string, any>;
  required_params: string[];
  category?: string | null;
  tags: string[];
  usage_count: number;
  success_rate: number;
}

export interface ToolExecutionResult {
  tool_name: string;
  result: any;
  status: string;
  execution_time: number;
  metadata: Record<string, any>;
}

export const toolsService = {
  async listTools(): Promise<ApiResponse<ToolSummary[]>> {
    try {
      const response = await api.get<ToolSummary[]>('/v1/tools');
      return {
        success: true,
        data: response.data,
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch tools',
      };
    }
  },

  async executeTool(toolName: string, parameters: Record<string, any> = {}): Promise<ApiResponse<ToolExecutionResult>> {
    try {
      const response = await api.post<ToolExecutionResult>(`/v1/tools/${toolName}/execute`, {
        parameters,
      });
      return {
        success: true,
        data: response.data,
      };
    } catch (error: any) {
      const detail = error.response?.data;
      return {
        success: false,
        error:
          detail?.message ||
          detail?.error ||
          error.response?.data?.message ||
          'Tool execution failed',
        data: detail,
      };
    }
  },
};
