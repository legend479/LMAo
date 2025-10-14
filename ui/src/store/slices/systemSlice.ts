import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { SystemMetrics, ToolExecution } from '../../types';

interface SystemState {
  metrics: SystemMetrics;
  toolExecutions: ToolExecution[];
  isHealthy: boolean;
  lastUpdate: Date | null;
  alerts: Alert[];
}

interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error';
  message: string;
  timestamp: Date;
  dismissed: boolean;
}

const initialState: SystemState = {
  metrics: {
    cpu: 0,
    memory: 0,
    activeUsers: 0,
    requestsPerMinute: 0,
    averageResponseTime: 0,
    errorRate: 0,
    timestamp: new Date(),
  },
  toolExecutions: [],
  isHealthy: true,
  lastUpdate: null,
  alerts: [],
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    updateMetrics: (state, action: PayloadAction<SystemMetrics>) => {
      state.metrics = action.payload;
      state.lastUpdate = new Date();
      state.isHealthy = action.payload.errorRate < 0.05 && action.payload.cpu < 90;
    },
    addToolExecution: (state, action: PayloadAction<ToolExecution>) => {
      state.toolExecutions.unshift(action.payload);
      // Keep only last 50 executions
      if (state.toolExecutions.length > 50) {
        state.toolExecutions = state.toolExecutions.slice(0, 50);
      }
    },
    updateToolExecution: (state, action: PayloadAction<{ id: string; updates: Partial<ToolExecution> }>) => {
      const index = state.toolExecutions.findIndex(t => t.id === action.payload.id);
      if (index !== -1) {
        state.toolExecutions[index] = { ...state.toolExecutions[index], ...action.payload.updates };
      }
    },
    addAlert: (state, action: PayloadAction<Omit<Alert, 'id' | 'timestamp' | 'dismissed'>>) => {
      const alert: Alert = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date(),
        dismissed: false,
      };
      state.alerts.unshift(alert);
      // Keep only last 20 alerts
      if (state.alerts.length > 20) {
        state.alerts = state.alerts.slice(0, 20);
      }
    },
    dismissAlert: (state, action: PayloadAction<string>) => {
      const index = state.alerts.findIndex(a => a.id === action.payload);
      if (index !== -1) {
        state.alerts[index].dismissed = true;
      }
    },
    clearDismissedAlerts: (state) => {
      state.alerts = state.alerts.filter(a => !a.dismissed);
    },
  },
});

export const {
  updateMetrics,
  addToolExecution,
  updateToolExecution,
  addAlert,
  dismissAlert,
  clearDismissedAlerts,
} = systemSlice.actions;

export default systemSlice.reducer;