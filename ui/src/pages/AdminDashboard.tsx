import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Tabs,
  Tab,
  Paper,
  Alert,
  Snackbar,
} from '@mui/material';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import UserManagement from '../components/admin/UserManagement';
import SystemConfigurationComponent from '../components/admin/SystemConfiguration';
import AnalyticsDashboard from '../components/analytics/AnalyticsDashboard';
import ToolsAdmin from '../components/admin/ToolsAdmin';
import { systemService, UsageAnalytics, SystemConfiguration } from '../services/systemService';
import { User } from '../types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`admin-tabpanel-${index}`}
      aria-labelledby={`admin-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

const AdminDashboard: React.FC = () => {
  const { user } = useSelector((state: RootState) => state.auth);
  const [tabValue, setTabValue] = useState(0);
  const [users, setUsers] = useState<User[]>([]);
  const [totalUsers, setTotalUsers] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [analytics, setAnalytics] = useState<UsageAnalytics | null>(null);
  const [systemConfig, setSystemConfig] = useState<SystemConfiguration | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (user?.role === 'admin') {
      fetchInitialData();
    }
  }, [user]);

  const fetchInitialData = async () => {
    try {
      setLoading(true);
      await Promise.all([
        fetchUsers(),
        fetchAnalytics(),
        fetchSystemConfig(),
      ]);
    } catch (err) {
      setError('Failed to load admin dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const fetchUsers = async (page: number = 0) => {
    try {
      const response = await systemService.getAllUsers(page + 1, 20);
      setUsers(response.users);
      setTotalUsers(response.total);
      setCurrentPage(response.page - 1);
      setTotalPages(response.totalPages);
    } catch (err) {
      setError('Failed to fetch users');
    }
  };

  const fetchAnalytics = async () => {
    try {
      const analyticsData = await systemService.getUsageAnalytics(timeRange);
      setAnalytics(analyticsData);
    } catch (err) {
      setError('Failed to fetch analytics data');
    }
  };

  const fetchSystemConfig = async () => {
    try {
      const config = await systemService.getSystemConfiguration();
      setSystemConfig(config);
    } catch (err) {
      setError('Failed to fetch system configuration');
    }
  };

  const handleUpdateUser = async (userId: string, updates: Partial<User>) => {
    try {
      const updatedUser = await systemService.updateUser(userId, updates);
      setUsers(prev => prev.map(u => u.id === userId ? updatedUser : u));
      setSuccess('User updated successfully');
    } catch (err) {
      setError('Failed to update user');
    }
  };

  const handleDeleteUser = async (userId: string) => {
    try {
      await systemService.deleteUser(userId);
      setUsers(prev => prev.filter(u => u.id !== userId));
      setTotalUsers(prev => prev - 1);
      setSuccess('User deleted successfully');
    } catch (err) {
      setError('Failed to delete user');
    }
  };

  const handleCreateUser = async (userData: Omit<User, 'id'>) => {
    try {
      // In a real implementation, this would call the API to create a user
      const newUser: User = {
        ...userData,
        id: Date.now().toString(), // Mock ID generation
      };
      setUsers(prev => [newUser, ...prev]);
      setTotalUsers(prev => prev + 1);
      setSuccess('User created successfully');
    } catch (err) {
      setError('Failed to create user');
    }
  };

  const handleSaveConfig = async (config: SystemConfiguration) => {
    try {
      const updatedConfig = await systemService.updateSystemConfiguration(config);
      setSystemConfig(updatedConfig);
      setSuccess('Configuration saved successfully');
    } catch (err) {
      setError('Failed to save configuration');
    }
  };

  const handleResetConfig = async () => {
    try {
      // In a real implementation, this would reset to default configuration
      await fetchSystemConfig();
      setSuccess('Configuration reset to defaults');
    } catch (err) {
      setError('Failed to reset configuration');
    }
  };

  const handleTimeRangeChange = (range: '5m' | '15m' | '1h' | '6h' | '24h' | '7d' | '30d') => {
    setTimeRange(range as '1h' | '24h' | '7d' | '30d');
    fetchAnalytics();
  };

  // Check if user is admin
  if (user?.role !== 'admin') {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Alert severity="error">
          Access denied. Admin privileges required.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Admin Dashboard
      </Typography>

      <Paper sx={{ width: '100%' }}>
        <Tabs
          value={tabValue}
          onChange={(_, newValue) => setTabValue(newValue)}
          aria-label="admin dashboard tabs"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="User Management" />
          <Tab label="System Configuration" />
          <Tab label="Analytics" />
          <Tab label="Tools" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <UserManagement
            users={users}
            totalUsers={totalUsers}
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={fetchUsers}
            onUpdateUser={handleUpdateUser}
            onDeleteUser={handleDeleteUser}
            onCreateUser={handleCreateUser}
            loading={loading}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {systemConfig && (
            <SystemConfigurationComponent
              config={systemConfig}
              onSaveConfig={handleSaveConfig}
              onResetConfig={handleResetConfig}
              loading={loading}
            />
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {analytics && (
            <AnalyticsDashboard
              analytics={analytics}
              timeRange={timeRange}
              onTimeRangeChange={handleTimeRangeChange}
              onRefresh={fetchAnalytics}
              loading={loading}
            />
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <ToolsAdmin />
        </TabPanel>
      </Paper>

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={5000}
        onClose={() => setError(null)}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={Boolean(success)}
        autoHideDuration={3000}
        onClose={() => setSuccess(null)}
      >
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default AdminDashboard;