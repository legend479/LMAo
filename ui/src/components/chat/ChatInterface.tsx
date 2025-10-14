import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Typography,
  Divider,
  Alert,
  Fab,
  Tooltip,
} from '@mui/material';
import {
  KeyboardArrowDown as ScrollDownIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import { addMessage, setLoading, setError } from '../../store/slices/chatSlice';
import { addNotification } from '../../store/slices/uiSlice';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';
import TypingIndicator from './TypingIndicator';
import MessageSearch from './MessageSearch';
import LoadingSpinner from '../common/LoadingSpinner';
import { chatService } from '../../services/chatService';
import { websocketService } from '../../services/websocketService';
import { Message } from '../../types';

interface ChatInterfaceProps {
  conversationId: string | null;
}

const ChatContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.default,
}));

const ChatHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  borderBottom: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.paper,
}));

const MessagesContainer = styled(Box)(({ theme }) => ({
  flex: 1,
  overflow: 'auto',
  padding: theme.spacing(1, 2),
  position: 'relative',
}));

const InputContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  borderTop: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.paper,
}));

const ScrollToBottomFab = styled(Fab)(({ theme }) => ({
  position: 'absolute',
  bottom: theme.spacing(2),
  right: theme.spacing(2),
  zIndex: 1,
}));

const EmptyState = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  textAlign: 'center',
  padding: theme.spacing(4),
}));

const ChatInterface: React.FC<ChatInterfaceProps> = ({ conversationId }) => {
  const dispatch = useDispatch();
  const { messages, isTyping, isConnected, loading, error } = useSelector(
    (state: RootState) => state.chat
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [searchOpen, setSearchOpen] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  // Handle scroll events to show/hide scroll button
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      
      setShowScrollButton(!isNearBottom);
      setAutoScroll(isNearBottom);
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // Load messages when conversation changes
  useEffect(() => {
    if (conversationId) {
      loadMessages(conversationId);
      websocketService.joinConversation(conversationId);
    }

    return () => {
      if (conversationId) {
        websocketService.leaveConversation(conversationId);
      }
    };
  }, [conversationId]);

  const loadMessages = async (convId: string) => {
    dispatch(setLoading(true));
    try {
      const response = await chatService.getMessages(convId);
      if (response.success && response.data) {
        // Messages are loaded through WebSocket or would be set here
        // For now, we'll rely on the Redux state
      } else {
        dispatch(setError(response.error || 'Failed to load messages'));
      }
    } catch (error) {
      dispatch(setError('Failed to load messages'));
    } finally {
      dispatch(setLoading(false));
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!conversationId) return;

    // Add user message immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
      type: 'text',
    };

    dispatch(addMessage(userMessage));

    try {
      // Send via WebSocket for real-time response
      websocketService.sendMessage(content, conversationId);
      
      // Also send via API for persistence
      const response = await chatService.sendMessage(conversationId, content);
      if (!response.success) {
        dispatch(addNotification({
          type: 'error',
          message: response.error || 'Failed to send message',
          autoHide: true,
        }));
      }
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        message: 'Failed to send message',
        autoHide: true,
      }));
    }
  };

  const handleCopyMessage = (content: string) => {
    dispatch(addNotification({
      type: 'success',
      message: 'Message copied to clipboard',
      autoHide: true,
    }));
  };

  const handleMessageFeedback = (messageId: string, feedback: 'positive' | 'negative') => {
    // Handle feedback logic here
    dispatch(addNotification({
      type: 'info',
      message: `Feedback recorded: ${feedback}`,
      autoHide: true,
    }));
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    setAutoScroll(true);
  };

  const handleRetryConnection = () => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      websocketService.connect(token);
    }
  };

  if (!conversationId) {
    return (
      <ChatContainer>
        <EmptyState>
          <Typography variant="h5" gutterBottom>
            Welcome to SE SME Agent
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Select a conversation or start a new one to begin chatting with your AI assistant.
          </Typography>
        </EmptyState>
      </ChatContainer>
    );
  }

  return (
    <ChatContainer>
      {/* Chat Header */}
      <ChatHeader>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">
            SE SME Agent Chat
          </Typography>
          <Box display="flex" alignItems="center" gap={1}>
            <Tooltip title="Search messages">
              <Fab
                size="small"
                onClick={() => setSearchOpen(true)}
              >
                <SearchIcon />
              </Fab>
            </Tooltip>
            {!isConnected && (
              <Tooltip title="Reconnect">
                <Fab
                  size="small"
                  color="primary"
                  onClick={handleRetryConnection}
                >
                  <RefreshIcon />
                </Fab>
              </Tooltip>
            )}
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: isConnected ? 'success.main' : 'error.main',
              }}
            />
            <Typography variant="caption" color="text.secondary">
              {isConnected ? 'Connected' : 'Disconnected'}
            </Typography>
          </Box>
        </Box>
      </ChatHeader>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" onClose={() => dispatch(setError(null))}>
          {error}
        </Alert>
      )}

      {/* Messages Container */}
      <MessagesContainer ref={messagesContainerRef}>
        {loading && messages.length === 0 ? (
          <LoadingSpinner message="Loading conversation..." />
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onCopy={handleCopyMessage}
                onFeedback={handleMessageFeedback}
              />
            ))}
            
            {/* Typing Indicator */}
            <TypingIndicator isVisible={isTyping} />
            
            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </>
        )}

        {/* Scroll to bottom button */}
        {showScrollButton && (
          <ScrollToBottomFab
            size="small"
            color="primary"
            onClick={scrollToBottom}
          >
            <ScrollDownIcon />
          </ScrollToBottomFab>
        )}
      </MessagesContainer>

      {/* Message Input */}
      <InputContainer>
        <MessageInput
          onSendMessage={handleSendMessage}
          disabled={!isConnected || loading}
          loading={loading}
          placeholder={
            isConnected 
              ? "Ask me anything about software engineering..." 
              : "Connecting to SE SME Agent..."
          }
        />
      </InputContainer>

      {/* Message Search Dialog */}
      {conversationId && (
        <MessageSearch
          open={searchOpen}
          onClose={() => setSearchOpen(false)}
          conversationId={conversationId}
          onMessageSelect={(message: Message) => {
            // Scroll to message or highlight it
            console.log('Selected message:', message);
          }}
        />
      )}
    </ChatContainer>
  );
};

export default ChatInterface;