import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
  Menu,
  MenuItem,
  Tooltip,
} from '@mui/material';
import {
  MoreVert,
  Fullscreen,
  Download,
  Refresh,
} from '@mui/icons-material';
import { PerformanceMetric } from '../../types';

interface RealTimeChartProps {
  title: string;
  data: PerformanceMetric[];
  height?: number;
  color?: string;
  unit?: string;
  timeRange?: '5m' | '15m' | '1h' | '6h' | '24h' | '7d' | '30d';
  onTimeRangeChange?: (range: '5m' | '15m' | '1h' | '6h' | '24h' | '7d' | '30d') => void;
  onRefresh?: () => void;
  onExport?: () => void;
  onFullscreen?: () => void;
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({
  title,
  data,
  height = 300,
  color = '#1976d2',
  unit = '',
  timeRange = '15m',
  onTimeRangeChange,
  onRefresh,
  onExport,
  onFullscreen,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    drawChart();
  }, [data, height]);

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const chartHeight = rect.height;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const innerHeight = chartHeight - padding * 2;

    // Clear canvas
    ctx.clearRect(0, 0, width, chartHeight);

    // Find min/max values
    const values = data.map(d => d.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = maxValue - minValue || 1;

    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (innerHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Vertical grid lines
    const timePoints = Math.min(data.length, 10);
    for (let i = 0; i <= timePoints; i++) {
      const x = padding + (chartWidth / timePoints) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, chartHeight - padding);
      ctx.stroke();
    }

    // Draw area under curve
    if (data.length > 1) {
      const gradient = ctx.createLinearGradient(0, padding, 0, chartHeight - padding);
      gradient.addColorStop(0, `${color}20`);
      gradient.addColorStop(1, `${color}05`);

      ctx.fillStyle = gradient;
      ctx.beginPath();

      data.forEach((point, index) => {
        const x = padding + (chartWidth / (data.length - 1)) * index;
        const y = chartHeight - padding - ((point.value - minValue) / valueRange) * innerHeight;

        if (index === 0) {
          ctx.moveTo(x, chartHeight - padding);
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.lineTo(width - padding, chartHeight - padding);
      ctx.closePath();
      ctx.fill();
    }

    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((point, index) => {
      const x = padding + (chartWidth / (data.length - 1)) * index;
      const y = chartHeight - padding - ((point.value - minValue) / valueRange) * innerHeight;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw data points
    ctx.fillStyle = color;
    data.forEach((point, index) => {
      const x = padding + (chartWidth / (data.length - 1)) * index;
      const y = chartHeight - padding - ((point.value - minValue) / valueRange) * innerHeight;

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw Y-axis labels
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 5; i++) {
      const value = maxValue - (valueRange / 5) * i;
      const y = padding + (innerHeight / 5) * i + 4;
      ctx.fillText(`${value.toFixed(1)}${unit}`, padding - 10, y);
    }

    // Draw X-axis labels (time)
    ctx.textAlign = 'center';
    const labelCount = Math.min(data.length, 6);
    for (let i = 0; i < labelCount; i++) {
      const dataIndex = Math.floor((data.length - 1) * (i / (labelCount - 1)));
      const point = data[dataIndex];
      if (point) {
        const x = padding + (chartWidth / (labelCount - 1)) * i;
        const time = new Date(point.timestamp).toLocaleTimeString('en-US', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
        });
        ctx.fillText(time, x, chartHeight - 10);
      }
    }
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleRefresh = () => {
    setIsAnimating(true);
    onRefresh?.();
    setTimeout(() => setIsAnimating(false), 1000);
    handleMenuClose();
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={title}
        action={
          <Box display="flex" alignItems="center" gap={1}>
            <ToggleButtonGroup
              value={timeRange}
              exclusive
              onChange={(_, value) => value && onTimeRangeChange?.(value)}
              size="small"
            >
              <ToggleButton value="5m">5m</ToggleButton>
              <ToggleButton value="15m">15m</ToggleButton>
              <ToggleButton value="1h">1h</ToggleButton>
              <ToggleButton value="6h">6h</ToggleButton>
              <ToggleButton value="24h">24h</ToggleButton>
              <ToggleButton value="7d">7d</ToggleButton>
              <ToggleButton value="30d">30d</ToggleButton>
            </ToggleButtonGroup>

            <IconButton onClick={handleMenuClick}>
              <MoreVert />
            </IconButton>

            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
            >
              <MenuItem onClick={handleRefresh}>
                <Refresh sx={{ mr: 1, animation: isAnimating ? 'spin 1s linear infinite' : 'none' }} />
                Refresh
              </MenuItem>
              <MenuItem onClick={() => { onExport?.(); handleMenuClose(); }}>
                <Download sx={{ mr: 1 }} />
                Export Data
              </MenuItem>
              <MenuItem onClick={() => { onFullscreen?.(); handleMenuClose(); }}>
                <Fullscreen sx={{ mr: 1 }} />
                Fullscreen
              </MenuItem>
            </Menu>
          </Box>
        }
      />
      <CardContent sx={{ pt: 0 }}>
        <Box
          component="canvas"
          ref={canvasRef}
          sx={{
            width: '100%',
            height: height,
            display: 'block',
          }}
        />
        {data.length === 0 && (
          <Box
            display="flex"
            alignItems="center"
            justifyContent="center"
            height={height}
          >
            <Typography variant="body2" color="text.secondary">
              No data available
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RealTimeChart;