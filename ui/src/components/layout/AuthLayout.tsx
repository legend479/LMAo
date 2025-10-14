import React from 'react';
import { Box, Container, Paper, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

interface AuthLayoutProps {
  children: React.ReactNode;
}

const StyledContainer = styled(Container)(({ theme }) => ({
  minHeight: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: `linear-gradient(135deg, ${theme.palette.primary.light} 0%, ${theme.palette.primary.main} 100%)`,
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  borderRadius: theme.spacing(2),
  maxWidth: 400,
  width: '100%',
  boxShadow: theme.shadows[10],
}));

const AuthLayout: React.FC<AuthLayoutProps> = ({ children }) => {
  return (
    <StyledContainer maxWidth={false}>
      <StyledPaper elevation={10}>
        <Box textAlign="center" mb={3}>
          <Typography variant="h4" component="h1" gutterBottom>
            SE SME Agent
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Software Engineering Subject Matter Expert
          </Typography>
        </Box>
        {children}
      </StyledPaper>
    </StyledContainer>
  );
};

export default AuthLayout;