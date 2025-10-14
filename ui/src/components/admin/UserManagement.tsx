import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  IconButton,
  Button,
  Chip,
  Typography,
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Avatar,
  Menu,
  Tooltip,
  Alert,
} from '@mui/material';
import {
  MoreVert,
  Edit,
  Delete,
  Block,
  CheckCircle,
  PersonAdd,
  Search,
  FilterList,
  Download,
} from '@mui/icons-material';
import { User } from '../../types';

interface UserManagementProps {
  users: User[];
  totalUsers: number;
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  onUpdateUser?: (userId: string, updates: Partial<User>) => void;
  onDeleteUser?: (userId: string) => void;
  onCreateUser?: (user: Omit<User, 'id'>) => void;
  loading?: boolean;
}

interface UserFormData {
  username: string;
  email: string;
  role: 'user' | 'admin';
  preferences: User['preferences'];
}

const UserManagement: React.FC<UserManagementProps> = ({
  users,
  totalUsers,
  currentPage,
  totalPages,
  onPageChange,
  onUpdateUser,
  onDeleteUser,
  onCreateUser,
  loading = false,
}) => {
  const [anchorEl, setAnchorEl] = useState<{ [key: string]: HTMLElement | null }>({});
  const [editDialog, setEditDialog] = useState<User | null>(null);
  const [deleteDialog, setDeleteDialog] = useState<User | null>(null);
  const [createDialog, setCreateDialog] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState<'all' | 'user' | 'admin'>('all');
  const [formData, setFormData] = useState<UserFormData>({
    username: '',
    email: '',
    role: 'user',
    preferences: {
      theme: 'light',
      language: 'en',
      outputComplexity: 'intermediate',
      preferredFormats: ['pdf'],
      notifications: {
        email: true,
        browser: true,
        taskCompletion: true,
        systemAlerts: false,
      },
    },
  });

  const handleMenuClick = (userId: string, event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl({ ...anchorEl, [userId]: event.currentTarget });
  };

  const handleMenuClose = (userId: string) => {
    setAnchorEl({ ...anchorEl, [userId]: null });
  };

  const handleEditUser = (user: User) => {
    setEditDialog(user);
    setFormData({
      username: user.username,
      email: user.email,
      role: user.role,
      preferences: user.preferences,
    });
  };

  const handleSaveUser = () => {
    if (editDialog) {
      onUpdateUser?.(editDialog.id, {
        username: formData.username,
        email: formData.email,
        role: formData.role,
        preferences: formData.preferences,
      });
      setEditDialog(null);
    }
  };

  const handleCreateUser = () => {
    onCreateUser?.({
      username: formData.username,
      email: formData.email,
      role: formData.role,
      preferences: formData.preferences,
    });
    setCreateDialog(false);
    setFormData({
      username: '',
      email: '',
      role: 'user',
      preferences: {
        theme: 'light',
        language: 'en',
        outputComplexity: 'intermediate',
        preferredFormats: ['pdf'],
        notifications: {
          email: true,
          browser: true,
          taskCompletion: true,
          systemAlerts: false,
        },
      },
    });
  };

  const handleDeleteUser = () => {
    if (deleteDialog) {
      onDeleteUser?.(deleteDialog.id);
      setDeleteDialog(null);
    }
  };

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = roleFilter === 'all' || user.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  const getUserInitials = (username: string) => {
    return username.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  };

  const formatLastActivity = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <>
      <Card sx={{ height: '100%' }}>
        <CardHeader
          title={
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Typography variant="h6">User Management</Typography>
              <Box display="flex" alignItems="center" gap={1}>
                <TextField
                  size="small"
                  placeholder="Search users..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
                  }}
                  sx={{ width: 200 }}
                />
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Role</InputLabel>
                  <Select
                    value={roleFilter}
                    label="Role"
                    onChange={(e) => setRoleFilter(e.target.value as any)}
                  >
                    <MenuItem value="all">All Roles</MenuItem>
                    <MenuItem value="user">Users</MenuItem>
                    <MenuItem value="admin">Admins</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  variant="contained"
                  startIcon={<PersonAdd />}
                  onClick={() => setCreateDialog(true)}
                >
                  Add User
                </Button>
              </Box>
            </Box>
          }
        />
        <CardContent sx={{ pt: 0 }}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>User</TableCell>
                  <TableCell>Role</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Preferences</TableCell>
                  <TableCell>Last Activity</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredUsers.map((user) => (
                  <TableRow key={user.id} hover>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={2}>
                        <Avatar sx={{ width: 32, height: 32 }}>
                          {getUserInitials(user.username)}
                        </Avatar>
                        <Typography variant="subtitle2">
                          {user.username}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={user.role}
                        color={user.role === 'admin' ? 'primary' : 'default'}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {user.email}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={0.5} flexWrap="wrap">
                        <Chip
                          label={user.preferences.theme}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          label={user.preferences.outputComplexity}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {formatLastActivity(new Date())}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <IconButton
                        size="small"
                        onClick={(e) => handleMenuClick(user.id, e)}
                      >
                        <MoreVert />
                      </IconButton>
                      <Menu
                        anchorEl={anchorEl[user.id]}
                        open={Boolean(anchorEl[user.id])}
                        onClose={() => handleMenuClose(user.id)}
                      >
                        <MenuItem onClick={() => {
                          handleEditUser(user);
                          handleMenuClose(user.id);
                        }}>
                          <Edit sx={{ mr: 1 }} />
                          Edit User
                        </MenuItem>
                        <MenuItem onClick={() => {
                          setDeleteDialog(user);
                          handleMenuClose(user.id);
                        }}>
                          <Delete sx={{ mr: 1 }} />
                          Delete User
                        </MenuItem>
                      </Menu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <TablePagination
            component="div"
            count={totalUsers}
            page={currentPage}
            onPageChange={(_, page) => onPageChange(page)}
            rowsPerPage={20}
            rowsPerPageOptions={[20]}
          />
        </CardContent>
      </Card>

      {/* Edit User Dialog */}
      <Dialog
        open={Boolean(editDialog)}
        onClose={() => setEditDialog(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit User</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={2} pt={1}>
            <TextField
              label="Username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              fullWidth
            />
            <TextField
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              fullWidth
            />
            <FormControl fullWidth>
              <InputLabel>Role</InputLabel>
              <Select
                value={formData.role}
                label="Role"
                onChange={(e) => setFormData({ ...formData, role: e.target.value as any })}
              >
                <MenuItem value="user">User</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>
            
            <Typography variant="subtitle2" sx={{ mt: 2 }}>
              Preferences
            </Typography>
            
            <FormControl fullWidth>
              <InputLabel>Theme</InputLabel>
              <Select
                value={formData.preferences.theme}
                label="Theme"
                onChange={(e) => setFormData({
                  ...formData,
                  preferences: { ...formData.preferences, theme: e.target.value as any }
                })}
              >
                <MenuItem value="light">Light</MenuItem>
                <MenuItem value="dark">Dark</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth>
              <InputLabel>Output Complexity</InputLabel>
              <Select
                value={formData.preferences.outputComplexity}
                label="Output Complexity"
                onChange={(e) => setFormData({
                  ...formData,
                  preferences: { ...formData.preferences, outputComplexity: e.target.value as any }
                })}
              >
                <MenuItem value="simple">Simple</MenuItem>
                <MenuItem value="intermediate">Intermediate</MenuItem>
                <MenuItem value="advanced">Advanced</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(null)}>
            Cancel
          </Button>
          <Button onClick={handleSaveUser} variant="contained">
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create User Dialog */}
      <Dialog
        open={createDialog}
        onClose={() => setCreateDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New User</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={2} pt={1}>
            <TextField
              label="Username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              fullWidth
              required
            />
            <FormControl fullWidth>
              <InputLabel>Role</InputLabel>
              <Select
                value={formData.role}
                label="Role"
                onChange={(e) => setFormData({ ...formData, role: e.target.value as any })}
              >
                <MenuItem value="user">User</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialog(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreateUser} variant="contained">
            Create User
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={Boolean(deleteDialog)}
        onClose={() => setDeleteDialog(null)}
      >
        <DialogTitle>Delete User</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            This action cannot be undone.
          </Alert>
          <Typography>
            Are you sure you want to delete user "{deleteDialog?.username}"?
            All associated data will be permanently removed.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(null)}>
            Cancel
          </Button>
          <Button onClick={handleDeleteUser} color="error" variant="contained">
            Delete User
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default UserManagement;