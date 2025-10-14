import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Conversation, Message } from '../../types';

interface ChatState {
  conversations: Conversation[];
  currentConversationId: string | null;
  messages: Message[];
  isTyping: boolean;
  isConnected: boolean;
  loading: boolean;
  error: string | null;
}

const initialState: ChatState = {
  conversations: [],
  currentConversationId: null,
  messages: [],
  isTyping: false,
  isConnected: false,
  loading: false,
  error: null,
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    setConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    setConversations: (state, action: PayloadAction<Conversation[]>) => {
      state.conversations = action.payload;
    },
    addConversation: (state, action: PayloadAction<Conversation>) => {
      state.conversations.unshift(action.payload);
    },
    setCurrentConversation: (state, action: PayloadAction<string>) => {
      state.currentConversationId = action.payload;
      const conversation = state.conversations.find(c => c.id === action.payload);
      state.messages = conversation?.messages || [];
    },
    addMessage: (state, action: PayloadAction<Message>) => {
      state.messages.push(action.payload);
      // Update the conversation's messages
      const conversationIndex = state.conversations.findIndex(
        c => c.id === state.currentConversationId
      );
      if (conversationIndex !== -1) {
        state.conversations[conversationIndex].messages.push(action.payload);
        state.conversations[conversationIndex].updatedAt = new Date();
      }
    },
    updateMessage: (state, action: PayloadAction<{ id: string; updates: Partial<Message> }>) => {
      const messageIndex = state.messages.findIndex(m => m.id === action.payload.id);
      if (messageIndex !== -1) {
        state.messages[messageIndex] = { ...state.messages[messageIndex], ...action.payload.updates };
      }
    },
    setTyping: (state, action: PayloadAction<boolean>) => {
      state.isTyping = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearMessages: (state) => {
      state.messages = [];
    },
    deleteConversation: (state, action: PayloadAction<string>) => {
      state.conversations = state.conversations.filter(c => c.id !== action.payload);
      if (state.currentConversationId === action.payload) {
        state.currentConversationId = null;
        state.messages = [];
      }
    },
  },
});

export const {
  setConnected,
  setConversations,
  addConversation,
  setCurrentConversation,
  addMessage,
  updateMessage,
  setTyping,
  setLoading,
  setError,
  clearMessages,
  deleteConversation,
} = chatSlice.actions;

export default chatSlice.reducer;