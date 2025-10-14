import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Info,
  Warning,
  Error,
  CheckCircle,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

interface MetricsCardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: 'up' | 'down' | 'flat';
  trendValue?: number;
  status?: 'healthy' | 'warning' | 'error' | 'info';
  progress?: number;
  maxValue?: number;
  description?: string;
  icon?: React.ReactNode;
  onClick?: () => void;
}

const StyledCard = styled(Card)<{ status?: string }>(({ theme, status }) => ({
  height: '100%',
  cursor: 'pointer',
  transition: 'all 0.2s ease-in-out',
  borderLeft: `4px solid ${
    status === 'error' ? theme.palette.error.main :
    status === 'warning' ? theme.palette.warning.main :
    status === 'info' ? theme.palette.info.main :
    theme.palette.success.main
  }`,
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[4],
  },
}));

const TrendIcon = styled(Box)<{ trend?: string }>(({ theme, trend }) => ({
  display: 'flex',
  alignItems: 'center',
  color: 
    trend === 'up' ? theme.palette.success.main :
    trend === 'down' ? theme.palette.error.main :
    theme.palette.text.secondary,
}));

const StatusIcon = ({ status }: { status?: string }) => {
  switch (status) {
    case 'error':
      return <Error color="error" />;
    case 'warning':
      return <Warning color="warning" />;
    case 'info':
      return <Info color="info" />;
    default:
      return <CheckCircle color="success" />;
  }
};

const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  unit,
  trend,
  trendValue,
  status = 'healthy',
  progress,
  maxValue,
  description,
  icon,
  onClick,
}) => {
  const formatValue = (val: number | string): string => {
    if (typeof val === 'string') return val;
    
    if (val >= 1000000) {
      return `${(val / 1000000).toFixed(1)}M`;
    } else if (val >= 1000) {
      return `${(val / 1000).toFixed(1)}K`;
    }
    return val.toFixed(1);
  };

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp fontSize="small" />;
      case 'down':
        return <TrendingDown fontSize="small" />;
      default:
        return <TrendingFlat fontSize="small" />;
    }
  };

  const progressValue = progress !== undefined ? progress : 
    (typeof value === 'number' && maxValue) ? (value / maxValue) * 100 : 0;

  return (
    <StyledCard status={status} onClick={onClick}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Box display="flex" alignItems="baseline" gap={0.5}>
              <Typography variant="h4" component="div" fontWeight="bold">
                {formatValue(value)}
              </Typography>
              {unit && (
                <Typography variant="body2" color="text.secondary">
                  {unit}
                </Typography>
              )}
            </Box>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            {icon}
            <StatusIcon status={status} />
            {description && (
              <Tooltip title={description}>
                <IconButton size="small">
                  <Info fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        {(trend || trendValue !== undefined) && (
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <TrendIcon trend={trend}>
              {getTrendIcon()}
            </TrendIcon>
            {trendValue !== undefined && (
              <Typography variant="body2" color="text.secondary">
                {trendValue > 0 ? '+' : ''}{trendValue.toFixed(1)}%
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary">
              vs last period
            </Typography>
          </Box>
        )}

        {(progress !== undefined || maxValue) && (
          <Box>
            <LinearProgress
              variant="determinate"
              value={Math.min(progressValue, 100)}
              color={
                progressValue > 90 ? 'error' :
                progressValue > 75 ? 'warning' :
                'primary'
              }
              sx={{ height: 6, borderRadius: 3 }}
            />
            <Box display="flex" justifyContent="space-between" mt={0.5}>
              <Typography variant="caption" color="text.secondary">
                {progressValue.toFixed(1)}%
              </Typography>
              {maxValue && (
                <Typography variant="caption" color="text.secondary">
                  Max: {formatValue(maxValue)}
                </Typography>
              )}
            </Box>
          </Box>
        )}

        {status !== 'healthy' && (
          <Box mt={1}>
            <Chip
              size="small"
              label={status.charAt(0).toUpperCase() + status.slice(1)}
              color={
                status === 'error' ? 'error' :
                status === 'warning' ? 'warning' :
                'info'
              }
              variant="outlined"
            />
          </Box>
        )}
      </CardContent>
    </StyledCard>
  );
};

export default MetricsCard;