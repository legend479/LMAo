import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Description as DocumentIcon,
  Dashboard as DashboardIcon,
  Settings as SettingsIcon,
  AdminPanelSettings as AdminIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import { logout } from '../../store/slices/authSlice';
import { NavigationItem } from '../../types';

interface SidebarProps {
  open: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ open }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useDispatch();
  const user = useSelector((state: RootState) => state.auth.user);

  const navigationItems: NavigationItem[] = [
    {
      id: 'chat',
      label: 'Chat',
      path: '/chat',
      icon: 'chat',
    },
    {
      id: 'documents',
      label: 'Documents',
      path: '/documents',
      icon: 'document',
    },
    {
      id: 'system',
      label: 'System Monitor',
      path: '/system',
      icon: 'dashboard',
    },
    {
      id: 'settings',
      label: 'Settings',
      path: '/settings',
      icon: 'settings',
    },
  ];

  const adminItems: NavigationItem[] = [
    {
      id: 'admin',
      label: 'Admin Dashboard',
      path: '/admin',
      icon: 'admin',
      roles: ['admin'],
    },
  ];

  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'chat':
        return <ChatIcon />;
      case 'document':
        return <DocumentIcon />;
      case 'dashboard':
        return <DashboardIcon />;
      case 'settings':
        return <SettingsIcon />;
      case 'admin':
        return <AdminIcon />;
      default:
        return <ChatIcon />;
    }
  };

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
  };

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: 240,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
          mt: 8,
        },
      }}
    >
      <Box sx={{ overflow: 'auto', height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Navigation Items */}
        <List>
          {navigationItems.map((item) => (
            <ListItem key={item.id} disablePadding>
              <ListItemButton
                selected={isActive(item.path)}
                onClick={() => handleNavigation(item.path)}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: 'primary.light',
                    '&:hover': {
                      backgroundColor: 'primary.light',
                    },
                  },
                }}
              >
                <ListItemIcon>{getIcon(item.icon)}</ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        {/* Admin Section */}
        {user?.role === 'admin' && (
          <>
            <Divider />
            <Box sx={{ px: 2, py: 1 }}>
              <Typography variant="overline" color="text.secondary">
                Administration
              </Typography>
            </Box>
            <List>
              {adminItems.map((item) => (
                <ListItem key={item.id} disablePadding>
                  <ListItemButton
                    selected={isActive(item.path)}
                    onClick={() => handleNavigation(item.path)}
                    sx={{
                      '&.Mui-selected': {
                        backgroundColor: 'primary.light',
                        '&:hover': {
                          backgroundColor: 'primary.light',
                        },
                      },
                    }}
                  >
                    <ListItemIcon>{getIcon(item.icon)}</ListItemIcon>
                    <ListItemText primary={item.label} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </>
        )}

        {/* Logout */}
        <Box sx={{ mt: 'auto' }}>
          <Divider />
          <List>
            <ListItem disablePadding>
              <ListItemButton onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon />
                </ListItemIcon>
                <ListItemText primary="Logout" />
              </ListItemButton>
            </ListItem>
          </List>
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;