import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  List,
  ListItem,
  CircularProgress,
  Alert,
  AppBar,
  Toolbar,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface SystemHealth {
  status: string;
  version?: string;
  services?: Record<string, any>;
  error?: string;
}

const SimpleChatApp: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setSystemHealth(response.data);
    } catch (err: any) {
      console.error('Health check failed:', err);
      setSystemHealth({ status: 'unhealthy', error: err.message });
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/chat/message`, {
        message: inputMessage,
        session_id: sessionId
      });

      const agentMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.response,
        timestamp: response.data.timestamp
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (err: any) {
      console.error('Error sending message:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to send message');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <SmartToyIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            SE SME Agent
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                backgroundColor: systemHealth?.status === 'healthy' ? 'green' : 'red',
                mr: 1
              }}
            />
            <Typography variant="body2">
              {systemHealth?.status || 'Unknown'}
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* System Status */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Status
                </Typography>
                {systemHealth ? (
                  <Box>
                    <Typography variant="body2">
                      Status: {systemHealth.status}
                    </Typography>
                    <Typography variant="body2">
                      Version: {systemHealth.version || 'Unknown'}
                    </Typography>
                    {systemHealth.services && (
                      <Typography variant="body2">
                        Services: {Object.keys(systemHealth.services).length} active
                      </Typography>
                    )}
                  </Box>
                ) : (
                  <CircularProgress size={20} />
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Chat Interface */}
          <Grid item xs={12}>
            <Paper sx={{ height: '60vh', display: 'flex', flexDirection: 'column' }}>
              {/* Messages */}
              <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                {messages.length === 0 ? (
                  <Box sx={{ textAlign: 'center', mt: 4 }}>
                    <SmartToyIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary">
                      Welcome to SE SME Agent
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Ask me anything about software engineering!
                    </Typography>
                  </Box>
                ) : (
                  <List>
                    {messages.map((message) => (
                      <ListItem
                        key={message.id}
                        sx={{
                          flexDirection: 'column',
                          alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                          mb: 1
                        }}
                      >
                        <Paper
                          sx={{
                            p: 2,
                            maxWidth: '70%',
                            backgroundColor: message.role === 'user' ? 'primary.main' : 'grey.800'
                          }}
                        >
                          <Typography variant="body1">
                            {message.content}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </Typography>
                        </Paper>
                      </ListItem>
                    ))}
                  </List>
                )}
                {loading && (
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                    <CircularProgress size={24} />
                    <Typography variant="body2" sx={{ ml: 2 }}>
                      Agent is thinking...
                    </Typography>
                  </Box>
                )}
              </Box>

              {/* Error Display */}
              {error && (
                <Box sx={{ p: 2 }}>
                  <Alert severity="error" onClose={() => setError(null)}>
                    {error}
                  </Alert>
                </Box>
              )}

              {/* Input */}
              <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    multiline
                    maxRows={4}
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask me about software engineering..."
                    disabled={loading}
                    variant="outlined"
                    size="small"
                  />
                  <Button
                    variant="contained"
                    onClick={sendMessage}
                    disabled={loading || !inputMessage.trim()}
                    endIcon={<SendIcon />}
                  >
                    Send
                  </Button>
                </Box>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default SimpleChatApp;