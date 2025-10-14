import React from 'react';
import { Box, Typography } from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';

interface TypingIndicatorProps {
  isVisible: boolean;
  userName?: string;
}

const bounce = keyframes`
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-10px);
  }
`;

const TypingContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(1, 2),
  marginBottom: theme.spacing(1),
  opacity: 0,
  transform: 'translateY(10px)',
  transition: 'all 0.3s ease',
  '&.visible': {
    opacity: 1,
    transform: 'translateY(0)',
  },
}));

const TypingBubble = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.grey[200],
  borderRadius: theme.spacing(2),
  padding: theme.spacing(1, 1.5),
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  ...(theme.palette.mode === 'dark' && {
    backgroundColor: theme.palette.grey[700],
  }),
}));

const Dot = styled(Box)(({ theme }) => ({
  width: 6,
  height: 6,
  borderRadius: '50%',
  backgroundColor: theme.palette.text.secondary,
  animation: `${bounce} 1.4s infinite ease-in-out`,
  '&:nth-of-type(1)': {
    animationDelay: '-0.32s',
  },
  '&:nth-of-type(2)': {
    animationDelay: '-0.16s',
  },
}));

const TypingIndicator: React.FC<TypingIndicatorProps> = ({ 
  isVisible, 
  userName = 'SE SME Agent' 
}) => {
  if (!isVisible) return null;

  return (
    <TypingContainer className={isVisible ? 'visible' : ''}>
      <TypingBubble>
        <Dot />
        <Dot />
        <Dot />
      </TypingBubble>
      <Typography variant="caption" color="text.secondary">
        {userName} is typing...
      </Typography>
    </TypingContainer>
  );
};

export default TypingIndicator;