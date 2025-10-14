import React, { useState } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  IconButton,
  Typography,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  Chip,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Search as SearchIcon,
  MoreVert as MoreIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Archive as ArchiveIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import { setCurrentConversation, deleteConversation } from '../../store/slices/chatSlice';
import { Conversation } from '../../types';
import Button from '../common/Button';

interface ConversationListProps {
  onNewConversation: () => void;
  onEditConversation?: (conversation: Conversation) => void;
}

const ConversationContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.paper,
  borderRight: `1px solid ${theme.palette.divider}`,
}));

const SearchContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

const ConversationListStyled = styled(List)(({ theme }) => ({
  flex: 1,
  overflow: 'auto',
  padding: 0,
}));

const ConversationItem = styled(ListItemButton)<{ selected?: boolean }>(({ theme, selected }) => ({
  borderBottom: `1px solid ${theme.palette.divider}`,
  padding: theme.spacing(1.5, 2),
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
  ...(selected && {
    backgroundColor: theme.palette.primary.light,
    '&:hover': {
      backgroundColor: theme.palette.primary.light,
    },
  }),
}));

const ConversationList: React.FC<ConversationListProps> = ({
  onNewConversation,
  onEditConversation,
}) => {
  const dispatch = useDispatch();
  const { conversations, currentConversationId } = useSelector((state: RootState) => state.chat);
  const [searchQuery, setSearchQuery] = useState('');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);

  const filteredConversations = conversations.filter(conversation =>
    conversation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conversation.messages.some(message =>
      message.content.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  const handleConversationSelect = (conversationId: string) => {
    dispatch(setCurrentConversation(conversationId));
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, conversation: Conversation) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
    setSelectedConversation(conversation);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedConversation(null);
  };

  const handleDeleteConversation = () => {
    if (selectedConversation) {
      dispatch(deleteConversation(selectedConversation.id));
    }
    handleMenuClose();
  };

  const handleEditConversation = () => {
    if (selectedConversation && onEditConversation) {
      onEditConversation(selectedConversation);
    }
    handleMenuClose();
  };

  const formatLastMessage = (conversation: Conversation) => {
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    if (!lastMessage) return 'No messages';
    
    const preview = lastMessage.content.length > 50 
      ? `${lastMessage.content.substring(0, 50)}...` 
      : lastMessage.content;
    
    return `${lastMessage.sender === 'user' ? 'You: ' : 'Agent: '}${preview}`;
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const messageDate = new Date(date);
    const diffInHours = (now.getTime() - messageDate.getTime()) / (1000 * 60 * 60);
    
    if (diffInHours < 24) {
      return messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffInHours < 168) { // 7 days
      return messageDate.toLocaleDateString([], { weekday: 'short' });
    } else {
      return messageDate.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  return (
    <ConversationContainer>
      {/* Header with new conversation button */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Button
          variant="primary"
          fullWidth
          startIcon={<AddIcon />}
          onClick={onNewConversation}
        >
          New Conversation
        </Button>
      </Box>

      {/* Search */}
      <SearchContainer>
        <TextField
          fullWidth
          size="small"
          placeholder="Search conversations..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
      </SearchContainer>

      {/* Conversation list */}
      <ConversationListStyled>
        {filteredConversations.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              {searchQuery ? 'No conversations found' : 'No conversations yet'}
            </Typography>
          </Box>
        ) : (
          filteredConversations.map((conversation) => (
            <ConversationItem
              key={conversation.id}
              selected={conversation.id === currentConversationId}
              onClick={() => handleConversationSelect(conversation.id)}
            >
              <ListItemIcon>
                <ChatIcon />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2" noWrap sx={{ flex: 1 }}>
                      {conversation.title}
                    </Typography>
                    {conversation.status === 'archived' && (
                      <Chip label="Archived" size="small" variant="outlined" />
                    )}
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary" noWrap>
                      {formatLastMessage(conversation)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTimestamp(conversation.updatedAt)}
                    </Typography>
                  </Box>
                }
              />
              <IconButton
                size="small"
                onClick={(e) => handleMenuOpen(e, conversation)}
              >
                <MoreIcon />
              </IconButton>
            </ConversationItem>
          ))
        )}
      </ConversationListStyled>

      {/* Context menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEditConversation}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Rename</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => console.log('Archive conversation')}>
          <ListItemIcon>
            <ArchiveIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Archive</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleDeleteConversation}>
          <ListItemIcon>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>
    </ConversationContainer>
  );
};

export default ConversationList;