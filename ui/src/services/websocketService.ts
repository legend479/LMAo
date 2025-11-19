import { store } from '../store/store';
import { setConnected, addMessage, setTyping } from '../store/slices/chatSlice';
import { updateMetrics, addToolExecution, updateToolExecution } from '../store/slices/systemSlice';
import { addNotification } from '../store/slices/uiSlice';
import { WebSocketMessage, Message, SystemMetrics, ToolExecution } from '../types';

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private baseUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
  private sessionId: string | null = null;

  connect(token: string, sessionId: string | 'new' = 'new'): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) return;

    // Build websocket URL: e.g. ws://host/api/v1/chat/ws/{sessionId}?token=...
    const sid = sessionId || 'new';
    const url = `${this.baseUrl.replace(/\/+$/, '')}/api/v1/chat/ws/${encodeURIComponent(
      sid
    )}?token=${encodeURIComponent(token)}`;

    try {
      this.socket = new WebSocket(url);
      this.sessionId = sid;

      this.socket.onopen = () => {
        console.log('WebSocket connected');
        store.dispatch(setConnected(true));
        store.dispatch(
          addNotification({ type: 'success', message: 'Connected to SE SME Agent', autoHide: true })
        );
        this.reconnectAttempts = 0;
      };

      this.socket.onclose = (ev) => {
        console.log('WebSocket disconnected', ev);
        store.dispatch(setConnected(false));
        this.handleReconnect(token, this.sessionId || 'new');
      };

      this.socket.onerror = (err) => {
        console.error('WebSocket error', err);
        store.dispatch(
          addNotification({ type: 'error', message: 'WebSocket connection error', autoHide: true })
        );
      };

      this.socket.onmessage = (ev) => {
        try {
          const parsed: WebSocketMessage = JSON.parse(ev.data);
          this.handleIncoming(parsed);
        } catch (e) {
          console.warn('Received non-JSON WebSocket message', ev.data);
        }
      };

    } catch (e) {
      console.error('Failed to create WebSocket', e);
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    store.dispatch(setConnected(false));
  }

  sendMessage(content: string, conversationId?: string): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const payload = {
        type: 'message',
        data: {
          message: content,
          conversationId,
          timestamp: new Date().toISOString(),
        },
      };
      this.socket.send(JSON.stringify(payload));
    }
  }

  // To join a conversation, reconnect to the server with the session id in the path
  joinConversation(conversationId: string, token: string): void {
    this.disconnect();
    this.connect(token, conversationId);
  }

  leaveConversation(conversationId?: string): void {
    // Close the connection; server will cleanup session
    this.disconnect();
  }

  private handleIncoming(msg: WebSocketMessage): void {
    const { type, data } = msg;

    switch (type) {
      case 'message':
        store.dispatch(addMessage(data as Message));
        break;
      case 'typing':
        store.dispatch(setTyping((data as any).is_typing || (data as any).isTyping || false));
        break;
      case 'system':
        // handle system messages like welcome
        if ((data as any).metrics) {
          store.dispatch(updateMetrics((data as any).metrics as SystemMetrics));
        }
        break;
      case 'system_metrics':
        store.dispatch(updateMetrics(data as SystemMetrics));
        break;
      case 'tool_execution_start':
        store.dispatch(addToolExecution(data as ToolExecution));
        break;
      case 'tool_execution_update':
        store.dispatch(updateToolExecution(data as { id: string; updates: Partial<ToolExecution> }));
        break;
      case 'error':
        store.dispatch(
          addNotification({ type: 'error', message: (data as any).message || 'Server error', autoHide: false })
        );
        break;
      default:
        console.debug('Unhandled WebSocket message type', type, data);
    }
  }

  private handleReconnect(token: string, sessionId: string | 'new') {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect(token, sessionId);
      }, delay);
    } else {
      store.dispatch(
        addNotification({ type: 'error', message: 'Unable to reconnect to SE SME Agent. Please refresh the page.', autoHide: false })
      );
    }
  }

  isConnected(): boolean {
    return this.socket != null && this.socket.readyState === WebSocket.OPEN;
  }
}

export const websocketService = new WebSocketService();