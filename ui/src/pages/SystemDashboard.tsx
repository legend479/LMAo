import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Alert,
  Snackbar,
  IconButton,
  Tooltip,
  Button,
} from '@mui/material';
import {
  Refresh,
  Settings,
  Download,
  Fullscreen,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { updateMetrics, addAlert } from '../store/slices/systemSlice';
import MetricsCard from '../components/monitoring/MetricsCard';
import RealTimeChart from '../components/monitoring/RealTimeChart';
import SystemAlerts from '../components/monitoring/SystemAlerts';
import ServiceStatus from '../components/monitoring/ServiceStatus';
import { systemService, SystemHealth } from '../services/systemService';
import { SystemMetrics, PerformanceMetric, SystemAlert, ServiceHealth } from '../types';

const SystemDashboard: React.FC = () => {
  const dispatch = useDispatch();
  const { metrics, isHealthy, alerts } = useSelector((state: RootState) => state.system);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceMetric[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchSystemData();
    
    // Set up auto-refresh
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(fetchSystemData, 30000); // Refresh every 30 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const fetchSystemData = async () => {
    try {
      setLoading(true);
      
      // Fetch system health
      const health = await systemService.getSystemHealth();
      setSystemHealth(health);
      
      // Fetch current metrics
      const currentMetrics = await systemService.getSystemMetrics();
      dispatch(updateMetrics(currentMetrics));
      
      // Update performance history
      setPerformanceHistory(prev => {
        const newHistory = [...prev, {
          timestamp: new Date(),
          value: currentMetrics.cpu,
          label: 'CPU Usage',
        }];
        // Keep only last 50 data points
        return newHistory.slice(-50);
      });
      
      // Mock services data (in real implementation, this would come from the API)
      const mockServices: ServiceHealth[] = [
        {
          name: 'API Server',
          status: health.status === 'healthy' ? 'healthy' : 'degraded',
          responseTime: Math.random() * 200 + 50,
          uptime: 86400,
          lastCheck: new Date(),
          dependencies: ['Database', 'Redis'],
        },
        {
          name: 'Agent Server',
          status: 'healthy',
          responseTime: Math.random() * 150 + 30,
          uptime: 86400,
          lastCheck: new Date(),
          dependencies: ['LLM Service', 'Vector DB'],
        },
        {
          name: 'RAG Pipeline',
          status: 'healthy',
          responseTime: Math.random() * 300 + 100,
          uptime: 86400,
          lastCheck: new Date(),
          dependencies: ['Elasticsearch', 'Embedding Service'],
        },
        {
          name: 'Web UI',
          status: 'healthy',
          responseTime: Math.random() * 100 + 20,
          uptime: 86400,
          lastCheck: new Date(),
          dependencies: ['API Server'],
        },
      ];
      setServices(mockServices);
      
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch system data';
      setError(errorMessage);
      dispatch(addAlert({
        type: 'error',
        message: errorMessage,
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchSystemData();
  };

  const handleResolveAlert = async (alertId: string) => {
    try {
      await systemService.resolveAlert(alertId);
      // In a real implementation, this would update the alerts from the server
    } catch (err) {
      setError('Failed to resolve alert');
    }
  };

  const handleRestartService = async (serviceName: string) => {
    try {
      await systemService.restartService(serviceName);
      dispatch(addAlert({
        type: 'info',
        message: `Service "${serviceName}" restart initiated`,
      }));
      // Refresh data after restart
      setTimeout(fetchSystemData, 2000);
    } catch (err) {
      setError(`Failed to restart service: ${serviceName}`);
    }
  };

  const getSystemStatus = () => {
    if (!systemHealth) return 'info';
    if (systemHealth.status === 'healthy' && isHealthy) return 'healthy';
    if (systemHealth.status === 'degraded' || metrics.cpu > 80) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">System Monitoring</Typography>
        <Box display="flex" alignItems="center" gap={1}>
          <Button
            variant={autoRefresh ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            Auto Refresh: {autoRefresh ? 'ON' : 'OFF'}
          </Button>
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>
          <Tooltip title="Export Report">
            <IconButton>
              <Download />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton>
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* System Overview Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="System Status"
            value={systemHealth?.status || 'Unknown'}
            status={getSystemStatus()}
            description={`Uptime: ${systemHealth ? Math.floor(systemHealth.uptime / 3600) : 0}h`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="CPU Usage"
            value={metrics.cpu}
            unit="%"
            progress={metrics.cpu}
            maxValue={100}
            status={metrics.cpu > 80 ? 'error' : metrics.cpu > 60 ? 'warning' : 'healthy'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Memory Usage"
            value={metrics.memory}
            unit="%"
            progress={metrics.memory}
            maxValue={100}
            status={metrics.memory > 85 ? 'error' : metrics.memory > 70 ? 'warning' : 'healthy'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Active Users"
            value={metrics.activeUsers}
            status="healthy"
            description="Currently online"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Requests/Min"
            value={metrics.requestsPerMinute}
            status="healthy"
            description="Current load"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Response Time"
            value={metrics.averageResponseTime}
            unit="ms"
            status={metrics.averageResponseTime > 1000 ? 'error' : metrics.averageResponseTime > 500 ? 'warning' : 'healthy'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Error Rate"
            value={(metrics.errorRate * 100).toFixed(2)}
            unit="%"
            status={metrics.errorRate > 0.05 ? 'error' : metrics.errorRate > 0.02 ? 'warning' : 'healthy'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Services"
            value={services.filter(s => s.status === 'healthy').length}
            unit={`/${services.length}`}
            progress={(services.filter(s => s.status === 'healthy').length / services.length) * 100}
            status={services.every(s => s.status === 'healthy') ? 'healthy' : 'warning'}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Real-time Performance Chart */}
        <Grid item xs={12} lg={8}>
          <RealTimeChart
            title="CPU Usage Over Time"
            data={performanceHistory}
            height={300}
            color="#1976d2"
            unit="%"
            onRefresh={handleRefresh}
          />
        </Grid>

        {/* System Alerts */}
        <Grid item xs={12} lg={4}>
          <SystemAlerts
            alerts={alerts.map(alert => ({
              id: alert.id,
              type: alert.type as SystemAlert['type'],
              title: alert.message,
              message: alert.message,
              timestamp: alert.timestamp,
              resolved: alert.dismissed,
              source: 'System Monitor',
              severity: alert.type === 'error' ? 8 : alert.type === 'warning' ? 5 : 3,
            }))}
            onResolveAlert={handleResolveAlert}
            onRefresh={handleRefresh}
          />
        </Grid>

        {/* Service Status */}
        <Grid item xs={12}>
          <ServiceStatus
            services={services}
            onRestartService={handleRestartService}
            onViewLogs={(serviceName) => console.log('View logs for:', serviceName)}
            onConfigureService={(serviceName) => console.log('Configure service:', serviceName)}
          />
        </Grid>
      </Grid>

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={5000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SystemDashboard;