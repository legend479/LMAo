FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY ui/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY ui/ ./

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001 && \
    chown -R nextjs:nodejs /app
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start the application
CMD ["npm", "start"]