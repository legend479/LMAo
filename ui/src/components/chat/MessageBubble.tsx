import React from 'react';
import { Box, Typography, Chip, Tooltip, IconButton } from '@mui/material';
import { styled } from '@mui/material/styles';
import { ContentCopy as CopyIcon, ThumbUp, ThumbDown } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
// @ts-ignore
import { Prism as SyntaxHighlighter, SyntaxHighlighterProps } from 'react-syntax-highlighter';
// @ts-ignore
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { Message } from '../../types';

interface MessageBubbleProps {
  message: Message;
  onCopy?: (content: string) => void;
  onFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void;
}

const MessageContainer = styled(Box)<{ sender: 'user' | 'agent' }>(({ theme, sender }) => ({
  display: 'flex',
  justifyContent: sender === 'user' ? 'flex-end' : 'flex-start',
  marginBottom: theme.spacing(2),
}));

const MessageBubbleStyled = styled(Box)<{ sender: 'user' | 'agent' }>(({ theme, sender }) => ({
  maxWidth: '70%',
  padding: theme.spacing(1.5, 2),
  borderRadius: theme.spacing(2),
  backgroundColor: sender === 'user' 
    ? theme.palette.primary.main 
    : theme.palette.grey[100],
  color: sender === 'user' 
    ? theme.palette.primary.contrastText 
    : theme.palette.text.primary,
  position: 'relative',
  wordBreak: 'break-word',
  
  '&:hover .message-actions': {
    opacity: 1,
  },
  
  ...(theme.palette.mode === 'dark' && sender === 'agent' && {
    backgroundColor: theme.palette.grey[800],
  }),
}));

const MessageActions = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: -8,
  right: 8,
  opacity: 0,
  transition: 'opacity 0.2s ease',
  display: 'flex',
  gap: theme.spacing(0.5),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.spacing(1),
  padding: theme.spacing(0.5),
  boxShadow: theme.shadows[2],
}));

const MetadataContainer = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(1),
  display: 'flex',
  flexWrap: 'wrap',
  gap: theme.spacing(0.5),
}));

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onCopy, onFeedback }) => {
  const theme = useSelector((state: RootState) => state.ui.theme);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    onCopy?.(message.content);
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const renderMarkdown = (content: string) => {
    return (
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            
            if (!inline && language) {
              
              return (
                <SyntaxHighlighter
                  style={theme === 'dark' ? (oneDark as any) : (oneLight as any)}
                  language={language}
                  PreTag="div"
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              );
            }
            
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          p: ({ children }) => (
            <Typography variant="body1" component="p" sx={{ mb: 1, '&:last-child': { mb: 0 } }}>
              {children}
            </Typography>
          ),
          h1: ({ children }) => (
            <Typography variant="h5" component="h1" sx={{ mb: 1, fontWeight: 600 }}>
              {children}
            </Typography>
          ),
          h2: ({ children }) => (
            <Typography variant="h6" component="h2" sx={{ mb: 1, fontWeight: 600 }}>
              {children}
            </Typography>
          ),
          h3: ({ children }) => (
            <Typography variant="subtitle1" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
              {children}
            </Typography>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <MessageContainer sender={message.sender}>
      <MessageBubbleStyled sender={message.sender}>
        <Box>
          {message.type === 'code' ? (
            <SyntaxHighlighter
              style={theme === 'dark' ? (oneDark as any) : (oneLight as any)}
              language="javascript"
              PreTag="div"
            >
              {message.content}
            </SyntaxHighlighter>
          ) : (
            renderMarkdown(message.content)
          )}
        </Box>

        {/* Message metadata */}
        {message.metadata && (
          <MetadataContainer>
            {message.metadata.toolsUsed && message.metadata.toolsUsed.length > 0 && (
              <Tooltip title="Tools used in this response">
                <Chip
                  label={`${message.metadata.toolsUsed.length} tools`}
                  size="small"
                  variant="outlined"
                />
              </Tooltip>
            )}
            {message.metadata.executionTime && (
              <Tooltip title="Response time">
                <Chip
                  label={`${message.metadata.executionTime}ms`}
                  size="small"
                  variant="outlined"
                />
              </Tooltip>
            )}
            {message.metadata.confidence && (
              <Tooltip title="Confidence level">
                <Chip
                  label={`${Math.round(message.metadata.confidence * 100)}%`}
                  size="small"
                  variant="outlined"
                  color={message.metadata.confidence > 0.8 ? 'success' : 'warning'}
                />
              </Tooltip>
            )}
          </MetadataContainer>
        )}

        {/* Timestamp */}
        <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.7 }}>
          {formatTimestamp(message.timestamp)}
        </Typography>

        {/* Message actions */}
        <MessageActions className="message-actions">
          <Tooltip title="Copy message">
            <IconButton size="small" onClick={handleCopy}>
              <CopyIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          {message.sender === 'agent' && onFeedback && (
            <>
              <Tooltip title="Helpful">
                <IconButton 
                  size="small" 
                  onClick={() => onFeedback(message.id, 'positive')}
                >
                  <ThumbUp fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Not helpful">
                <IconButton 
                  size="small" 
                  onClick={() => onFeedback(message.id, 'negative')}
                >
                  <ThumbDown fontSize="small" />
                </IconButton>
              </Tooltip>
            </>
          )}
        </MessageActions>
      </MessageBubbleStyled>
    </MessageContainer>
  );
};

export default MessageBubble;