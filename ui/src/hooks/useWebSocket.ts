import { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { websocketService } from '../services/websocketService';
import { addNotification } from '../store/slices/uiSlice';

export const useWebSocket = () => {
  const dispatch = useDispatch();
  const { token, isAuthenticated } = useSelector((state: RootState) => state.auth);
  const { isConnected } = useSelector((state: RootState) => state.chat);

  useEffect(() => {
    if (isAuthenticated && token && !isConnected) {
      try {
        websocketService.connect(token);
      } catch (error) {
        dispatch(addNotification({
          type: 'error',
          message: 'Failed to connect to SE SME Agent',
          autoHide: true,
        }));
      }
    }

    return () => {
      if (isConnected) {
        websocketService.disconnect();
      }
    };
  }, [isAuthenticated, token, isConnected, dispatch]);

  return {
    isConnected,
    sendMessage: websocketService.sendMessage.bind(websocketService),
    joinConversation: websocketService.joinConversation.bind(websocketService),
    leaveConversation: websocketService.leaveConversation.bind(websocketService),
  };
};