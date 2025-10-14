import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Paper,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Send as SendIcon,
  AttachFile as AttachIcon,
  Mic as MicIcon,
  Stop as StopIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface MessageInputProps {
  onSendMessage: (content: string) => void;
  onFileUpload?: (file: File) => void;
  disabled?: boolean;
  loading?: boolean;
  placeholder?: string;
}

const InputContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  display: 'flex',
  alignItems: 'flex-end',
  gap: theme.spacing(1),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.spacing(2),
  border: `1px solid ${theme.palette.divider}`,
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    border: 'none',
    '& fieldset': {
      border: 'none',
    },
  },
  '& .MuiInputBase-input': {
    padding: theme.spacing(1, 0),
    maxHeight: '120px',
    overflowY: 'auto',
  },
}));

const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  onFileUpload,
  disabled = false,
  loading = false,
  placeholder = "Type your message...",
}) => {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const textFieldRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled && !loading) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onFileUpload) {
      onFileUpload(file);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleVoiceRecording = () => {
    if (isRecording) {
      // Stop recording logic would go here
      setIsRecording(false);
    } else {
      // Start recording logic would go here
      setIsRecording(true);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textFieldRef.current) {
      textFieldRef.current.style.height = 'auto';
      textFieldRef.current.style.height = `${textFieldRef.current.scrollHeight}px`;
    }
  }, [message]);

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <InputContainer elevation={1}>
        {/* File upload button */}
        <Tooltip title="Attach file">
          <IconButton
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            size="small"
          >
            <AttachIcon />
          </IconButton>
        </Tooltip>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          hidden
          onChange={handleFileSelect}
          accept=".txt,.md,.pdf,.doc,.docx"
        />

        {/* Message input */}
        <StyledTextField
          ref={textFieldRef}
          fullWidth
          multiline
          maxRows={6}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          disabled={disabled}
          variant="outlined"
        />

        {/* Voice recording button */}
        <Tooltip title={isRecording ? "Stop recording" : "Voice message"}>
          <IconButton
            onClick={handleVoiceRecording}
            disabled={disabled}
            size="small"
            color={isRecording ? "error" : "default"}
          >
            {isRecording ? <StopIcon /> : <MicIcon />}
          </IconButton>
        </Tooltip>

        {/* Send button */}
        <Tooltip title="Send message">
          <IconButton
            type="submit"
            disabled={!message.trim() || disabled || loading}
            color="primary"
            size="small"
          >
            {loading ? (
              <CircularProgress size={20} />
            ) : (
              <SendIcon />
            )}
          </IconButton>
        </Tooltip>
      </InputContainer>
    </Box>
  );
};

export default MessageInput;