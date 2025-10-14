import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  People,
  Chat,
  Description,
  Build,
  Refresh,
  Download,
} from '@mui/icons-material';
import { UsageAnalytics } from '../../services/systemService';
import MetricsCard from '../monitoring/MetricsCard';
import RealTimeChart from '../monitoring/RealTimeChart';

interface AnalyticsDashboardProps {
  analytics: UsageAnalytics;
  timeRange: '1h' | '24h' | '7d' | '30d';
  onTimeRangeChange: (range: '5m' | '15m' | '1h' | '6h' | '24h' | '7d' | '30d') => void;
  onRefresh: () => void;
  loading?: boolean;
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  analytics,
  timeRange,
  onTimeRangeChange,
  onRefresh,
  loading = false,
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'users' | 'sessions' | 'tools'>('users');

  const getChartData = () => {
    switch (selectedMetric) {
      case 'users':
        return analytics.userActivity.map(activity => ({
          timestamp: new Date(activity.date),
          value: activity.activeUsers,
          label: 'Active Users',
        }));
      case 'sessions':
        return analytics.userActivity.map(activity => ({
          timestamp: new Date(activity.date),
          value: activity.totalSessions,
          label: 'Total Sessions',
        }));
      case 'tools':
        return analytics.systemLoad.map(load => ({
          timestamp: new Date(load.timestamp),
          value: load.requestsPerMinute,
          label: 'Requests/Min',
        }));
      default:
        return [];
    }
  };

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  const getTimeRangeLabel = (range: string) => {
    switch (range) {
      case '1h': return 'Last Hour';
      case '24h': return 'Last 24 Hours';
      case '7d': return 'Last 7 Days';
      case '30d': return 'Last 30 Days';
      default: return range;
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Analytics Dashboard</Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <ToggleButtonGroup
            value={timeRange}
            exclusive
            onChange={(_, value) => value && onTimeRangeChange(value)}
            size="small"
          >
            <ToggleButton value="1h">1H</ToggleButton>
            <ToggleButton value="24h">24H</ToggleButton>
            <ToggleButton value="7d">7D</ToggleButton>
            <ToggleButton value="30d">30D</ToggleButton>
          </ToggleButtonGroup>
          <Tooltip title="Refresh Data">
            <IconButton onClick={onRefresh} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>
          <Tooltip title="Export Report">
            <IconButton>
              <Download />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Overview Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Total Users"
            value={analytics.totalUsers}
            icon={<People />}
            status="healthy"
            description={`${analytics.activeUsers} currently active`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Active Users"
            value={analytics.activeUsers}
            unit={`/${analytics.totalUsers}`}
            progress={(analytics.activeUsers / analytics.totalUsers) * 100}
            icon={<TrendingUp />}
            status="healthy"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Total Conversations"
            value={analytics.totalConversations}
            icon={<Chat />}
            status="healthy"
            description="All time conversations"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Documents Generated"
            value={analytics.totalDocuments}
            icon={<Description />}
            status="healthy"
            description="All time documents"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Activity Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Typography variant="h6">
                    User Activity - {getTimeRangeLabel(timeRange)}
                  </Typography>
                  <ToggleButtonGroup
                    value={selectedMetric}
                    exclusive
                    onChange={(_, value) => value && setSelectedMetric(value)}
                    size="small"
                  >
                    <ToggleButton value="users">Users</ToggleButton>
                    <ToggleButton value="sessions">Sessions</ToggleButton>
                    <ToggleButton value="tools">Tools</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
              }
            />
            <CardContent>
              <RealTimeChart
                title=""
                data={getChartData()}
                height={300}
                color="#1976d2"
                timeRange={timeRange}
                onTimeRangeChange={onTimeRangeChange}
                onRefresh={onRefresh}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Top Tools */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: '100%' }}>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" gap={1}>
                  <Build />
                  <Typography variant="h6">Popular Tools</Typography>
                </Box>
              }
            />
            <CardContent>
              <Box display="flex" flexDirection="column" gap={2}>
                {analytics.popularTools.map((tool, index) => (
                  <Box key={tool.toolName}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle2">
                        {tool.toolName}
                      </Typography>
                      <Chip
                        label={`${tool.usageCount} uses`}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80 }}>
                        Success Rate:
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={tool.successRate * 100}
                        color={tool.successRate > 0.9 ? 'success' : tool.successRate > 0.7 ? 'warning' : 'error'}
                        sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {(tool.successRate * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      Avg. execution: {tool.averageExecutionTime.toFixed(1)}ms
                    </Typography>
                    {index < analytics.popularTools.length - 1 && (
                      <Box sx={{ borderBottom: '1px solid', borderColor: 'divider', mt: 2 }} />
                    )}
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* User Activity Table */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title={
                <Typography variant="h6">Detailed User Activity</Typography>
              }
            />
            <CardContent>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Date</TableCell>
                      <TableCell align="right">Active Users</TableCell>
                      <TableCell align="right">New Users</TableCell>
                      <TableCell align="right">Total Sessions</TableCell>
                      <TableCell align="right">Avg. Session Duration</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {analytics.userActivity.slice(0, 10).map((activity) => (
                      <TableRow key={activity.date} hover>
                        <TableCell>
                          {new Date(activity.date).toLocaleDateString()}
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontWeight="medium">
                            {activity.activeUsers}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={activity.newUsers}
                            size="small"
                            color={activity.newUsers > 0 ? 'success' : 'default'}
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="right">
                          {activity.totalSessions}
                        </TableCell>
                        <TableCell align="right">
                          {formatDuration(activity.averageSessionDuration)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* System Load Chart */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title={<Typography variant="h6">System Performance</Typography>}
            />
            <CardContent>
              <RealTimeChart
                title=""
                data={analytics.systemLoad.map(load => ({
                  timestamp: new Date(load.timestamp),
                  value: load.cpu,
                  label: 'CPU Usage %',
                }))}
                height={200}
                color="#ff9800"
                unit="%"
                timeRange={timeRange}
                onTimeRangeChange={onTimeRangeChange}
                onRefresh={onRefresh}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalyticsDashboard;