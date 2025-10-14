import { io, Socket } from 'socket.io-client';
import { store } from '../store/store';
import { setConnected, addMessage, setTyping } from '../store/slices/chatSlice';
import { updateMetrics, addToolExecution, updateToolExecution } from '../store/slices/systemSlice';
import { addNotification } from '../store/slices/uiSlice';
import { WebSocketMessage, Message, SystemMetrics, ToolExecution } from '../types';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(token: string): void {
    if (this.socket?.connected) {
      return;
    }

    this.socket = io(process.env.REACT_APP_WS_URL || 'ws://localhost:8000', {
      auth: {
        token,
      },
      transports: ['websocket'],
    });

    this.setupEventListeners();
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    store.dispatch(setConnected(false));
  }

  sendMessage(content: string, conversationId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('message', {
        content,
        conversationId,
        timestamp: new Date(),
      });
    }
  }

  joinConversation(conversationId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('join_conversation', { conversationId });
    }
  }

  leaveConversation(conversationId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('leave_conversation', { conversationId });
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      store.dispatch(setConnected(true));
      store.dispatch(addNotification({
        type: 'success',
        message: 'Connected to SE SME Agent',
        autoHide: true,
      }));
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      store.dispatch(setConnected(false));
      this.handleReconnect();
    });

    this.socket.on('message', (data: Message) => {
      store.dispatch(addMessage(data));
    });

    this.socket.on('typing', (data: { isTyping: boolean; userId: string }) => {
      store.dispatch(setTyping(data.isTyping));
    });

    this.socket.on('system_metrics', (data: SystemMetrics) => {
      store.dispatch(updateMetrics(data));
    });

    this.socket.on('tool_execution_start', (data: ToolExecution) => {
      store.dispatch(addToolExecution(data));
    });

    this.socket.on('tool_execution_update', (data: { id: string; updates: Partial<ToolExecution> }) => {
      store.dispatch(updateToolExecution(data));
    });

    this.socket.on('error', (error: any) => {
      console.error('WebSocket error:', error);
      store.dispatch(addNotification({
        type: 'error',
        message: `Connection error: ${error.message || 'Unknown error'}`,
        autoHide: true,
      }));
    });

    this.socket.on('connect_error', (error: any) => {
      console.error('WebSocket connection error:', error);
      store.dispatch(addNotification({
        type: 'error',
        message: 'Failed to connect to SE SME Agent',
        autoHide: true,
      }));
    });
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.socket?.connect();
      }, delay);
    } else {
      store.dispatch(addNotification({
        type: 'error',
        message: 'Unable to reconnect to SE SME Agent. Please refresh the page.',
        autoHide: false,
      }));
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

export const websocketService = new WebSocketService();