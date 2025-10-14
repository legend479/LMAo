import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  List,
  ListItem,
  ListItemText,
  Typography,
  Box,
  Chip,
  InputAdornment,
  IconButton,
} from '@mui/material';
import {
  Search as SearchIcon,
  Close as CloseIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import { chatService } from '../../services/chatService';
import { Message } from '../../types';
import LoadingSpinner from '../common/LoadingSpinner';

interface MessageSearchProps {
  open: boolean;
  onClose: () => void;
  conversationId: string;
  onMessageSelect?: (message: Message) => void;
}

const SearchResult = styled(ListItem)(({ theme }) => ({
  borderBottom: `1px solid ${theme.palette.divider}`,
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

const HighlightedText = styled('span')(({ theme }) => ({
  backgroundColor: theme.palette.warning.light,
  padding: theme.spacing(0, 0.5),
  borderRadius: theme.spacing(0.5),
}));

const MessageSearch: React.FC<MessageSearchProps> = ({
  open,
  onClose,
  conversationId,
  onMessageSelect,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  // Load recent searches from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('recent_searches');
    if (saved) {
      try {
        setRecentSearches(JSON.parse(saved));
      } catch (error) {
        console.error('Failed to parse recent searches:', error);
      }
    }
  }, []);

  // Debounced search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    const timeoutId = setTimeout(() => {
      performSearch(searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, conversationId]);

  const performSearch = async (query: string) => {
    if (!query.trim() || !conversationId) return;

    setLoading(true);
    try {
      const response = await chatService.searchMessages(conversationId, query);
      if (response.success && response.data) {
        setSearchResults(response.data);
        
        // Save to recent searches
        const newRecentSearches = [
          query,
          ...recentSearches.filter(s => s !== query)
        ].slice(0, 5);
        
        setRecentSearches(newRecentSearches);
        localStorage.setItem('recent_searches', JSON.stringify(newRecentSearches));
      }
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMessageSelect = (message: Message) => {
    onMessageSelect?.(message);
    onClose();
  };

  const handleRecentSearchSelect = (query: string) => {
    setSearchQuery(query);
  };

  const highlightText = (text: string, query: string) => {
    if (!query.trim()) return text;

    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) =>
      regex.test(part) ? (
        <HighlightedText key={index}>{part}</HighlightedText>
      ) : (
        part
      )
    );
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">Search Messages</Typography>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {/* Search Input */}
        <TextField
          fullWidth
          placeholder="Search messages..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{ mb: 2 }}
        />

        {/* Recent Searches */}
        {!searchQuery && recentSearches.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Recent Searches
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {recentSearches.map((query, index) => (
                <Chip
                  key={index}
                  label={query}
                  size="small"
                  icon={<HistoryIcon />}
                  onClick={() => handleRecentSearchSelect(query)}
                  clickable
                />
              ))}
            </Box>
          </Box>
        )}

        {/* Loading */}
        {loading && (
          <Box display="flex" justifyContent="center" py={4}>
            <LoadingSpinner message="Searching messages..." />
          </Box>
        )}

        {/* Search Results */}
        {!loading && searchResults.length > 0 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} found
            </Typography>
            <List>
              {searchResults.map((message) => (
                <SearchResult
                  key={message.id}
                  onClick={() => handleMessageSelect(message)}
                >
                  <ListItemText
                    primary={
                      <Box>
                        <Typography variant="body2" component="div">
                          {highlightText(
                            message.content.length > 200
                              ? `${message.content.substring(0, 200)}...`
                              : message.content,
                            searchQuery
                          )}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box display="flex" alignItems="center" gap={1} mt={1}>
                        <Chip
                          label={message.sender === 'user' ? 'You' : 'Agent'}
                          size="small"
                          color={message.sender === 'user' ? 'primary' : 'secondary'}
                        />
                        <Typography variant="caption" color="text.secondary">
                          {formatTimestamp(message.timestamp)}
                        </Typography>
                      </Box>
                    }
                  />
                </SearchResult>
              ))}
            </List>
          </Box>
        )}

        {/* No Results */}
        {!loading && searchQuery && searchResults.length === 0 && (
          <Box textAlign="center" py={4}>
            <Typography variant="body2" color="text.secondary">
              No messages found for "{searchQuery}"
            </Typography>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default MessageSearch;