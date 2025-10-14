import React from 'react';
import { Button as MuiButton, ButtonProps as MuiButtonProps, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

interface ButtonProps extends Omit<MuiButtonProps, 'variant'> {
  variant?: 'primary' | 'secondary' | 'outlined' | 'text' | 'danger';
  loading?: boolean;
  fullWidth?: boolean;
}

const StyledButton = styled(MuiButton)<{ customvariant?: string }>(({ theme, customvariant }) => ({
  textTransform: 'none',
  fontWeight: 500,
  borderRadius: theme.spacing(1),
  padding: theme.spacing(1, 2),
  
  ...(customvariant === 'primary' && {
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
    '&:hover': {
      backgroundColor: theme.palette.primary.dark,
    },
  }),
  
  ...(customvariant === 'secondary' && {
    backgroundColor: theme.palette.secondary.main,
    color: theme.palette.secondary.contrastText,
    '&:hover': {
      backgroundColor: theme.palette.secondary.dark,
    },
  }),
  
  ...(customvariant === 'danger' && {
    backgroundColor: theme.palette.error.main,
    color: theme.palette.error.contrastText,
    '&:hover': {
      backgroundColor: theme.palette.error.dark,
    },
  }),
}));

const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  loading = false,
  disabled,
  ...props
}) => {
  const muiVariant = variant === 'outlined' ? 'outlined' : variant === 'text' ? 'text' : 'contained';
  
  return (
    <StyledButton
      variant={muiVariant}
      customvariant={variant}
      disabled={disabled || loading}
      startIcon={loading ? <CircularProgress size={16} /> : props.startIcon}
      {...props}
    >
      {children}
    </StyledButton>
  );
};

export default Button;