import React, { useState } from 'react';
import { Box, TextField, Typography, Alert } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { loginStart, loginSuccess, loginFailure } from '../store/slices/authSlice';
import Button from '../components/common/Button';

const LoginPage: React.FC = () => {
  const dispatch = useDispatch();
  const { loading, error } = useSelector((state: RootState) => state.auth);
  const [credentials, setCredentials] = useState({ username: '', password: '' });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    dispatch(loginStart());

    try {
      // Mock authentication - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      if (credentials.username === 'admin' && credentials.password === 'admin') {
        dispatch(loginSuccess({
          user: {
            id: '1',
            username: 'admin',
            email: 'admin@example.com',
            role: 'admin',
            preferences: {
              theme: 'light',
              language: 'en',
              outputComplexity: 'advanced',
              preferredFormats: ['pdf', 'docx'],
              notifications: {
                email: true,
                browser: true,
                taskCompletion: true,
                systemAlerts: true,
              },
            },
          },
          token: 'mock-jwt-token',
        }));
      } else if (credentials.username === 'user' && credentials.password === 'user') {
        dispatch(loginSuccess({
          user: {
            id: '2',
            username: 'user',
            email: 'user@example.com',
            role: 'user',
            preferences: {
              theme: 'light',
              language: 'en',
              outputComplexity: 'intermediate',
              preferredFormats: ['pdf'],
              notifications: {
                email: false,
                browser: true,
                taskCompletion: true,
                systemAlerts: false,
              },
            },
          },
          token: 'mock-jwt-token',
        }));
      } else {
        dispatch(loginFailure('Invalid credentials'));
      }
    } catch (error) {
      dispatch(loginFailure('Login failed'));
    }
  };

  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setCredentials(prev => ({ ...prev, [field]: e.target.value }));
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Typography variant="h5" gutterBottom textAlign="center">
        Sign In
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <TextField
        fullWidth
        label="Username"
        value={credentials.username}
        onChange={handleChange('username')}
        margin="normal"
        required
        helperText="Use 'admin' or 'user' for demo"
      />
      
      <TextField
        fullWidth
        label="Password"
        type="password"
        value={credentials.password}
        onChange={handleChange('password')}
        margin="normal"
        required
        helperText="Use 'admin' or 'user' for demo"
      />
      
      <Button
        type="submit"
        fullWidth
        variant="primary"
        loading={loading}
        sx={{ mt: 3, mb: 2 }}
      >
        Sign In
      </Button>
      
      <Typography variant="body2" color="text.secondary" textAlign="center">
        Demo credentials: admin/admin or user/user
      </Typography>
    </Box>
  );
};

export default LoginPage;