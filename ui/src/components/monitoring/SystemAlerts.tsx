import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Typography,
  Box,
  Button,
  Collapse,
  Alert,
  AlertTitle,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  Error,
  Warning,
  Info,
  CheckCircle,
  ExpandMore,
  ExpandLess,
  Close,
  Refresh,
  FilterList,
  NotificationsActive,
} from '@mui/icons-material';
import { SystemAlert } from '../../types';

interface SystemAlertsProps {
  alerts: SystemAlert[];
  onResolveAlert?: (alertId: string) => void;
  onRefresh?: () => void;
  maxHeight?: number;
}

const SystemAlerts: React.FC<SystemAlertsProps> = ({
  alerts,
  onResolveAlert,
  onRefresh,
  maxHeight = 400,
}) => {
  const [expandedAlert, setExpandedAlert] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'unresolved' | 'critical'>('unresolved');

  const getAlertIcon = (type: SystemAlert['type']) => {
    switch (type) {
      case 'critical':
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

  const getAlertColor = (type: SystemAlert['type']) => {
    switch (type) {
      case 'critical':
        return 'error';
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'success';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    switch (filter) {
      case 'unresolved':
        return !alert.resolved;
      case 'critical':
        return alert.type === 'critical' && !alert.resolved;
      default:
        return true;
    }
  });

  const unresolvedCount = alerts.filter(a => !a.resolved).length;
  const criticalCount = alerts.filter(a => a.type === 'critical' && !a.resolved).length;

  const handleToggleExpand = (alertId: string) => {
    setExpandedAlert(expandedAlert === alertId ? null : alertId);
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={
          <Box display="flex" alignItems="center" gap={1}>
            <Badge badgeContent={unresolvedCount} color="error">
              <NotificationsActive />
            </Badge>
            <Typography variant="h6">System Alerts</Typography>
            {criticalCount > 0 && (
              <Chip
                label={`${criticalCount} Critical`}
                color="error"
                size="small"
                variant="outlined"
              />
            )}
          </Box>
        }
        action={
          <Box display="flex" alignItems="center" gap={1}>
            <Button
              size="small"
              startIcon={<FilterList />}
              onClick={() => {
                const filters = ['all', 'unresolved', 'critical'] as const;
                const currentIndex = filters.indexOf(filter);
                const nextIndex = (currentIndex + 1) % filters.length;
                setFilter(filters[nextIndex]);
              }}
            >
              {filter.charAt(0).toUpperCase() + filter.slice(1)}
            </Button>
            <IconButton onClick={onRefresh} size="small">
              <Refresh />
            </IconButton>
          </Box>
        }
      />
      <CardContent sx={{ pt: 0, pb: 1 }}>
        {filteredAlerts.length === 0 ? (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            py={4}
          >
            <CheckCircle color="success" sx={{ fontSize: 48, mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              No alerts to display
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {filter === 'unresolved' ? 'All alerts have been resolved' : 'System is running smoothly'}
            </Typography>
          </Box>
        ) : (
          <List
            sx={{
              maxHeight: maxHeight,
              overflow: 'auto',
              '& .MuiListItem-root': {
                borderRadius: 1,
                mb: 1,
                border: '1px solid',
                borderColor: 'divider',
              },
            }}
          >
            {filteredAlerts.map((alert) => (
              <React.Fragment key={alert.id}>
                <ListItem
                  sx={{
                    bgcolor: alert.resolved ? 'grey.50' : 'background.paper',
                    opacity: alert.resolved ? 0.7 : 1,
                  }}
                >
                  <ListItemIcon>
                    {getAlertIcon(alert.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography
                          variant="subtitle2"
                          sx={{
                            textDecoration: alert.resolved ? 'line-through' : 'none',
                          }}
                        >
                          {alert.title}
                        </Typography>
                        <Chip
                          label={alert.type}
                          color={getAlertColor(alert.type) as any}
                          size="small"
                          variant="outlined"
                        />
                        {alert.resolved && (
                          <Chip
                            label="Resolved"
                            color="success"
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {alert.source} • {formatTimestamp(alert.timestamp)}
                        </Typography>
                        {expandedAlert !== alert.id && (
                          <Typography
                            variant="body2"
                            sx={{
                              mt: 0.5,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {alert.message}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      {!alert.resolved && onResolveAlert && (
                        <Tooltip title="Mark as resolved">
                          <IconButton
                            size="small"
                            onClick={() => onResolveAlert(alert.id)}
                            color="success"
                          >
                            <CheckCircle />
                          </IconButton>
                        </Tooltip>
                      )}
                      <IconButton
                        size="small"
                        onClick={() => handleToggleExpand(alert.id)}
                      >
                        {expandedAlert === alert.id ? <ExpandLess /> : <ExpandMore />}
                      </IconButton>
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
                
                <Collapse in={expandedAlert === alert.id} timeout="auto" unmountOnExit>
                  <Box sx={{ px: 2, pb: 2 }}>
                    <Alert severity={getAlertColor(alert.type) as any} variant="outlined">
                      <AlertTitle>Alert Details</AlertTitle>
                      <Typography variant="body2" paragraph>
                        {alert.message}
                      </Typography>
                      <Box display="flex" gap={2} flexWrap="wrap">
                        <Typography variant="caption">
                          <strong>Source:</strong> {alert.source}
                        </Typography>
                        <Typography variant="caption">
                          <strong>Severity:</strong> {alert.severity}/10
                        </Typography>
                        <Typography variant="caption">
                          <strong>Time:</strong> {new Date(alert.timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                    </Alert>
                  </Box>
                </Collapse>
              </React.Fragment>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemAlerts;