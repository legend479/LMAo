import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Button,
  Divider,
  Alert,
  Snackbar,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  Save,
  RestoreFromTrash,
  Palette,
  Language,
  Notifications,
  Security,
  Download,
  Visibility,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { updateUserPreferences } from '../store/slices/authSlice';
import { UserPreferences } from '../types';

const SettingsPage: React.FC = () => {
  const dispatch = useDispatch();
  const { user } = useSelector((state: RootState) => state.auth);
  const [preferences, setPreferences] = useState<UserPreferences>(
    user?.preferences || {
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
    }
  );
  const [hasChanges, setHasChanges] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (user?.preferences) {
      setPreferences(user.preferences);
    }
  }, [user]);

  useEffect(() => {
    const preferencesChanged = JSON.stringify(preferences) !== JSON.stringify(user?.preferences);
    setHasChanges(preferencesChanged);
  }, [preferences, user]);

  const handleSave = async () => {
    try {
      dispatch(updateUserPreferences(preferences));
      setSaveSuccess(true);
      setHasChanges(false);
    } catch (error) {
      setSaveError('Failed to save preferences');
    }
  };

  const handleReset = () => {
    if (user?.preferences) {
      setPreferences(user.preferences);
      setHasChanges(false);
    }
  };

  const updatePreference = <K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const updateNotificationPreference = <K extends keyof UserPreferences['notifications']>(
    key: K,
    value: UserPreferences['notifications'][K]
  ) => {
    setPreferences(prev => ({
      ...prev,
      notifications: {
        ...prev.notifications,
        [key]: value,
      },
    }));
  };

  const togglePreferredFormat = (format: string) => {
    setPreferences(prev => ({
      ...prev,
      preferredFormats: prev.preferredFormats.includes(format)
        ? prev.preferredFormats.filter(f => f !== format)
        : [...prev.preferredFormats, format],
    }));
  };

  const availableFormats = ['pdf', 'docx', 'ppt', 'txt', 'md'];
  const availableLanguages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
  ];

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Settings</Typography>
        <Box display="flex" gap={1}>
          <Button
            variant="outlined"
            startIcon={<RestoreFromTrash />}
            onClick={handleReset}
            disabled={!hasChanges}
          >
            Reset Changes
          </Button>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSave}
            disabled={!hasChanges}
          >
            Save Preferences
          </Button>
        </Box>
      </Box>

      {hasChanges && (
        <Alert severity="info" sx={{ mb: 3 }}>
          You have unsaved changes. Don't forget to save your preferences.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Appearance Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" gap={1}>
                  <Palette color="primary" />
                  <Typography variant="h6">Appearance</Typography>
                </Box>
              }
            />
            <CardContent>
              <Box display="flex" flexDirection="column" gap={3}>
                <FormControl fullWidth>
                  <InputLabel>Theme</InputLabel>
                  <Select
                    value={preferences.theme}
                    label="Theme"
                    onChange={(e) => updatePreference('theme', e.target.value as any)}
                  >
                    <MenuItem value="light">Light</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>Language</InputLabel>
                  <Select
                    value={preferences.language}
                    label="Language"
                    onChange={(e) => updatePreference('language', e.target.value)}
                  >
                    {availableLanguages.map((lang) => (
                      <MenuItem key={lang.code} value={lang.code}>
                        {lang.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>Output Complexity</InputLabel>
                  <Select
                    value={preferences.outputComplexity}
                    label="Output Complexity"
                    onChange={(e) => updatePreference('outputComplexity', e.target.value as any)}
                  >
                    <MenuItem value="simple">Simple</MenuItem>
                    <MenuItem value="intermediate">Intermediate</MenuItem>
                    <MenuItem value="advanced">Advanced</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Document Preferences */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" gap={1}>
                  <Download color="primary" />
                  <Typography variant="h6">Document Preferences</Typography>
                </Box>
              }
            />
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Preferred Document Formats
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Select your preferred formats for document generation
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} mt={2}>
                {availableFormats.map((format) => (
                  <Chip
                    key={format}
                    label={format.toUpperCase()}
                    color={preferences.preferredFormats.includes(format) ? 'primary' : 'default'}
                    variant={preferences.preferredFormats.includes(format) ? 'filled' : 'outlined'}
                    onClick={() => togglePreferredFormat(format)}
                    clickable
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" gap={1}>
                  <Notifications color="primary" />
                  <Typography variant="h6">Notification Preferences</Typography>
                </Box>
              }
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preferences.notifications.email}
                        onChange={(e) => updateNotificationPreference('email', e.target.checked)}
                      />
                    }
                    label="Email Notifications"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Receive notifications via email
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preferences.notifications.browser}
                        onChange={(e) => updateNotificationPreference('browser', e.target.checked)}
                      />
                    }
                    label="Browser Notifications"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Show browser push notifications
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preferences.notifications.taskCompletion}
                        onChange={(e) => updateNotificationPreference('taskCompletion', e.target.checked)}
                      />
                    }
                    label="Task Completion"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Notify when tasks are completed
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preferences.notifications.systemAlerts}
                        onChange={(e) => updateNotificationPreference('systemAlerts', e.target.checked)}
                      />
                    }
                    label="System Alerts"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Receive system status alerts
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Account Information */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" gap={1}>
                  <Security color="primary" />
                  <Typography variant="h6">Account Information</Typography>
                </Box>
              }
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Username"
                    value={user?.username || ''}
                    fullWidth
                    disabled
                    helperText="Username cannot be changed"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Email"
                    value={user?.email || ''}
                    fullWidth
                    disabled
                    helperText="Contact admin to change email"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Role"
                    value={user?.role || ''}
                    fullWidth
                    disabled
                    helperText="Role is assigned by administrators"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Account Status
                    </Typography>
                    <Chip
                      label="Active"
                      color="success"
                      variant="outlined"
                      icon={<Visibility />}
                    />
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Advanced Settings */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Advanced Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Alert severity="info" sx={{ mb: 2 }}>
                These settings are for advanced users. Changing them may affect system performance.
              </Alert>
              
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Enable Analytics"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Allow collection of usage analytics to improve the service
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Auto-save Conversations"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Automatically save conversation history
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={<Switch />}
                    label="Debug Mode"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Show additional debugging information
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Keyboard Shortcuts"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Enable keyboard shortcuts for faster navigation
                  </Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>

      <Snackbar
        open={saveSuccess}
        autoHideDuration={3000}
        onClose={() => setSaveSuccess(false)}
      >
        <Alert severity="success" onClose={() => setSaveSuccess(false)}>
          Preferences saved successfully!
        </Alert>
      </Snackbar>

      <Snackbar
        open={Boolean(saveError)}
        autoHideDuration={5000}
        onClose={() => setSaveError(null)}
      >
        <Alert severity="error" onClose={() => setSaveError(null)}>
          {saveError}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SettingsPage;