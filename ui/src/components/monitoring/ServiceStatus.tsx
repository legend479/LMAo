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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Tooltip,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  RestartAlt,
  MoreVert,
  Timeline,
  Settings,
  Info,
} from '@mui/icons-material';
import { ServiceHealth } from '../../types';

interface ServiceStatusProps {
  services: ServiceHealth[];
  onRestartService?: (serviceName: string) => void;
  onViewLogs?: (serviceName: string) => void;
  onConfigureService?: (serviceName: string) => void;
}

const ServiceStatus: React.FC<ServiceStatusProps> = ({
  services,
  onRestartService,
  onViewLogs,
  onConfigureService,
}) => {
  const [selectedService, setSelectedService] = useState<ServiceHealth | null>(null);
  const [restartDialog, setRestartDialog] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<{ [key: string]: HTMLElement | null }>({});

  const getStatusIcon = (status: ServiceHealth['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'degraded':
        return <Warning color="warning" />;
      case 'unhealthy':
        return <Error color="error" />;
      default:
        return <Error color="disabled" />;
    }
  };

  const getStatusColor = (status: ServiceHealth['status']) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatUptime = (uptime: number) => {
    const days = Math.floor(uptime / (24 * 60 * 60));
    const hours = Math.floor((uptime % (24 * 60 * 60)) / (60 * 60));
    const minutes = Math.floor((uptime % (60 * 60)) / 60);

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const handleMenuClick = (serviceName: string, event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl({ ...anchorEl, [serviceName]: event.currentTarget });
  };

  const handleMenuClose = (serviceName: string) => {
    setAnchorEl({ ...anchorEl, [serviceName]: null });
  };

  const handleRestartConfirm = (serviceName: string) => {
    onRestartService?.(serviceName);
    setRestartDialog(null);
  };

  const healthyCount = services.filter(s => s.status === 'healthy').length;
  const degradedCount = services.filter(s => s.status === 'degraded').length;
  const unhealthyCount = services.filter(s => s.status === 'unhealthy').length;

  return (
    <>
      <Card sx={{ height: '100%' }}>
        <CardHeader
          title={
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="h6">Service Status</Typography>
              <Box display="flex" gap={1}>
                {healthyCount > 0 && (
                  <Chip
                    label={`${healthyCount} Healthy`}
                    color="success"
                    size="small"
                    variant="outlined"
                  />
                )}
                {degradedCount > 0 && (
                  <Chip
                    label={`${degradedCount} Degraded`}
                    color="warning"
                    size="small"
                    variant="outlined"
                  />
                )}
                {unhealthyCount > 0 && (
                  <Chip
                    label={`${unhealthyCount} Unhealthy`}
                    color="error"
                    size="small"
                    variant="outlined"
                  />
                )}
              </Box>
            </Box>
          }
        />
        <CardContent sx={{ pt: 0 }}>
          <List>
            {services.map((service) => (
              <ListItem
                key={service.name}
                sx={{
                  borderRadius: 1,
                  mb: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                  bgcolor: 'background.paper',
                }}
              >
                <ListItemIcon>
                  {getStatusIcon(service.status)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="subtitle2">
                        {service.name}
                      </Typography>
                      <Chip
                        label={service.status}
                        color={getStatusColor(service.status) as any}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Response Time: {service.responseTime}ms • Uptime: {formatUptime(service.uptime)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Last Check: {new Date(service.lastCheck).toLocaleTimeString()}
                      </Typography>
                      {service.dependencies.length > 0 && (
                        <Box mt={0.5}>
                          <Typography variant="caption" color="text.secondary">
                            Dependencies: {service.dependencies.join(', ')}
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <Tooltip title="Response Time">
                      <Box display="flex" alignItems="center" gap={0.5}>
                        <Typography variant="caption" color="text.secondary">
                          {service.responseTime}ms
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={Math.min((service.responseTime / 1000) * 100, 100)}
                          color={
                            service.responseTime > 500 ? 'error' :
                            service.responseTime > 200 ? 'warning' :
                            'success'
                          }
                          sx={{ width: 40, height: 4 }}
                        />
                      </Box>
                    </Tooltip>
                    
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuClick(service.name, e)}
                    >
                      <MoreVert />
                    </IconButton>
                    
                    <Menu
                      anchorEl={anchorEl[service.name]}
                      open={Boolean(anchorEl[service.name])}
                      onClose={() => handleMenuClose(service.name)}
                    >
                      <MenuItem onClick={() => {
                        setSelectedService(service);
                        handleMenuClose(service.name);
                      }}>
                        <Info sx={{ mr: 1 }} />
                        View Details
                      </MenuItem>
                      <MenuItem onClick={() => {
                        onViewLogs?.(service.name);
                        handleMenuClose(service.name);
                      }}>
                        <Timeline sx={{ mr: 1 }} />
                        View Logs
                      </MenuItem>
                      <MenuItem onClick={() => {
                        onConfigureService?.(service.name);
                        handleMenuClose(service.name);
                      }}>
                        <Settings sx={{ mr: 1 }} />
                        Configure
                      </MenuItem>
                      <MenuItem onClick={() => {
                        setRestartDialog(service.name);
                        handleMenuClose(service.name);
                      }}>
                        <RestartAlt sx={{ mr: 1 }} />
                        Restart Service
                      </MenuItem>
                    </Menu>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Service Details Dialog */}
      <Dialog
        open={Boolean(selectedService)}
        onClose={() => setSelectedService(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Service Details: {selectedService?.name}
        </DialogTitle>
        <DialogContent>
          {selectedService && (
            <Box>
              <Box display="flex" alignItems="center" gap={2} mb={2}>
                {getStatusIcon(selectedService.status)}
                <Typography variant="h6">
                  Status: {selectedService.status}
                </Typography>
              </Box>
              
              <Box display="grid" gridTemplateColumns="1fr 1fr" gap={2}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <Typography variant="body2">
                    Response Time: {selectedService.responseTime}ms
                  </Typography>
                  <Typography variant="body2">
                    Uptime: {formatUptime(selectedService.uptime)}
                  </Typography>
                  <Typography variant="body2">
                    Last Check: {new Date(selectedService.lastCheck).toLocaleString()}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Dependencies
                  </Typography>
                  {selectedService.dependencies.length > 0 ? (
                    selectedService.dependencies.map((dep) => (
                      <Chip
                        key={dep}
                        label={dep}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 0.5, mb: 0.5 }}
                      />
                    ))
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No dependencies
                    </Typography>
                  )}
                </Box>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedService(null)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Restart Confirmation Dialog */}
      <Dialog
        open={Boolean(restartDialog)}
        onClose={() => setRestartDialog(null)}
      >
        <DialogTitle>Restart Service</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to restart the "{restartDialog}" service?
            This may cause temporary service interruption.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRestartDialog(null)}>
            Cancel
          </Button>
          <Button
            onClick={() => restartDialog && handleRestartConfirm(restartDialog)}
            color="warning"
            variant="contained"
          >
            Restart
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ServiceStatus;