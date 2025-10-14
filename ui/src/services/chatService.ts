import axios from 'axios';
import { Conversation, Message, ApiResponse } from '../types';

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

export const chatService = {
  // Conversation management
  async getConversations(): Promise<ApiResponse<Conversation[]>> {
    try {
      const response = await api.get('/conversations');
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch conversations',
      };
    }
  },

  async createConversation(title?: string): Promise<ApiResponse<Conversation>> {
    try {
      const response = await api.post('/conversations', { title });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to create conversation',
      };
    }
  },

  async getConversation(id: string): Promise<ApiResponse<Conversation>> {
    try {
      const response = await api.get(`/conversations/${id}`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch conversation',
      };
    }
  },

  async updateConversation(id: string, updates: Partial<Conversation>): Promise<ApiResponse<Conversation>> {
    try {
      const response = await api.patch(`/conversations/${id}`, updates);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to update conversation',
      };
    }
  },

  async deleteConversation(id: string): Promise<ApiResponse<void>> {
    try {
      const response = await api.delete(`/conversations/${id}`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to delete conversation',
      };
    }
  },

  // Message management
  async getMessages(conversationId: string, page = 1, limit = 50): Promise<ApiResponse<Message[]>> {
    try {
      const response = await api.get(`/conversations/${conversationId}/messages`, {
        params: { page, limit },
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to fetch messages',
      };
    }
  },

  async sendMessage(conversationId: string, content: string): Promise<ApiResponse<Message>> {
    try {
      const response = await api.post(`/conversations/${conversationId}/messages`, {
        content,
        timestamp: new Date(),
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to send message',
      };
    }
  },

  // Search functionality
  async searchConversations(query: string): Promise<ApiResponse<Conversation[]>> {
    try {
      const response = await api.get('/conversations/search', {
        params: { q: query },
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to search conversations',
      };
    }
  },

  async searchMessages(conversationId: string, query: string): Promise<ApiResponse<Message[]>> {
    try {
      const response = await api.get(`/conversations/${conversationId}/messages/search`, {
        params: { q: query },
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.message || 'Failed to search messages',
      };
    }
  },
};