import React from 'react';
import { Card as MuiCard, CardContent, CardHeader, CardActions, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
  elevation?: number;
  className?: string;
}

const StyledCard = styled(MuiCard)(({ theme }) => ({
  borderRadius: theme.spacing(1.5),
  boxShadow: theme.shadows[1],
  transition: 'box-shadow 0.2s ease-in-out',
  '&:hover': {
    boxShadow: theme.shadows[3],
  },
}));

const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  children,
  actions,
  elevation = 1,
  className,
}) => {
  return (
    <StyledCard elevation={elevation} className={className}>
      {(title || subtitle) && (
        <CardHeader
          title={title && <Typography variant="h6">{title}</Typography>}
          subheader={subtitle && <Typography variant="body2" color="text.secondary">{subtitle}</Typography>}
        />
      )}
      <CardContent>{children}</CardContent>
      {actions && <CardActions>{actions}</CardActions>}
    </StyledCard>
  );
};

export default Card;