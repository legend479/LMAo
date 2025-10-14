import React, { useEffect, useState } from 'react';
import { Box, Grid, Dialog, DialogTitle, DialogContent, TextField, DialogActions } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { 
  setConversations, 
  addConversation, 
  setCurrentConversation,
  updateMessage 
} from '../store/slices/chatSlice';
import { addNotification } from '../store/slices/uiSlice';
import ConversationList from '../components/chat/ConversationList';
import ChatInterface from '../components/chat/ChatInterface';
import Button from '../components/common/Button';
import { chatService } from '../services/chatService';
import { websocketService } from '../services/websocketService';
import { Conversation } from '../types';

const ChatPage: React.FC = () => {
  const { conversationId } = useParams<{ conversationId: string }>();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { conversations, currentConversationId } = useSelector((state: RootState) => state.chat);
  const { token } = useSelector((state: RootState) => state.auth);
  const [editDialog, setEditDialog] = useState<{ open: boolean; conversation: Conversation | null }>({
    open: false,
    conversation: null,
  });
  const [newTitle, setNewTitle] = useState('');

  // Initialize WebSocket connection
  useEffect(() => {
    if (token) {
      websocketService.connect(token);
    }

    return () => {
      websocketService.disconnect();
    };
  }, [token]);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Set current conversation from URL
  useEffect(() => {
    if (conversationId && conversationId !== currentConversationId) {
      dispatch(setCurrentConversation(conversationId));
    }
  }, [conversationId, currentConversationId, dispatch]);

  const loadConversations = async () => {
    try {
      const response = await chatService.getConversations();
      if (response.success && response.data) {
        dispatch(setConversations(response.data));
      } else {
        dispatch(addNotification({
          type: 'error',
          message: response.error || 'Failed to load conversations',
          autoHide: true,
        }));
      }
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        message: 'Failed to load conversations',
        autoHide: true,
      }));
    }
  };

  const handleNewConversation = async () => {
    try {
      const response = await chatService.createConversation();
      if (response.success && response.data) {
        dispatch(addConversation(response.data));
        navigate(`/chat/${response.data.id}`);
        dispatch(addNotification({
          type: 'success',
          message: 'New conversation created',
          autoHide: true,
        }));
      } else {
        dispatch(addNotification({
          type: 'error',
          message: response.error || 'Failed to create conversation',
          autoHide: true,
        }));
      }
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        message: 'Failed to create conversation',
        autoHide: true,
      }));
    }
  };

  const handleEditConversation = (conversation: Conversation) => {
    setEditDialog({ open: true, conversation });
    setNewTitle(conversation.title);
  };

  const handleSaveEdit = async () => {
    if (!editDialog.conversation || !newTitle.trim()) return;

    try {
      const response = await chatService.updateConversation(
        editDialog.conversation.id,
        { title: newTitle.trim() }
      );
      
      if (response.success) {
        // Reload conversations to get updated data
        await loadConversations();
        dispatch(addNotification({
          type: 'success',
          message: 'Conversation renamed',
          autoHide: true,
        }));
      } else {
        dispatch(addNotification({
          type: 'error',
          message: response.error || 'Failed to rename conversation',
          autoHide: true,
        }));
      }
    } catch (error) {
      dispatch(addNotification({
        type: 'error',
        message: 'Failed to rename conversation',
        autoHide: true,
      }));
    }

    setEditDialog({ open: false, conversation: null });
    setNewTitle('');
  };

  const handleCancelEdit = () => {
    setEditDialog({ open: false, conversation: null });
    setNewTitle('');
  };

  return (
    <Box sx={{ height: 'calc(100vh - 64px)', display: 'flex' }}>
      <Grid container sx={{ height: '100%' }}>
        {/* Conversation List */}
        <Grid item xs={12} md={4} lg={3} sx={{ height: '100%' }}>
          <ConversationList
            onNewConversation={handleNewConversation}
            onEditConversation={handleEditConversation}
          />
        </Grid>

        {/* Chat Interface */}
        <Grid item xs={12} md={8} lg={9} sx={{ height: '100%' }}>
          <ChatInterface conversationId={currentConversationId} />
        </Grid>
      </Grid>

      {/* Edit Conversation Dialog */}
      <Dialog open={editDialog.open} onClose={handleCancelEdit} maxWidth="sm" fullWidth>
        <DialogTitle>Rename Conversation</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            label="Conversation Title"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            margin="normal"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSaveEdit();
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button variant="text" onClick={handleCancelEdit}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSaveEdit} disabled={!newTitle.trim()}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ChatPage;