# LMA-o: API Reference Guide

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket API](#websocket-api)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

## Overview

The LMA-o API provides programmatic access to all system features including chat, document management, tool execution, and administration.

**Base URL**: `http://localhost:8000`  
**API Version**: v1  
**Protocol**: REST + WebSocket  
**Authentication**: JWT Bearer Token

### API Conventions

- All timestamps are in ISO 8601 format (UTC)
- All request/response bodies use JSON
- HTTP status codes follow REST conventions
- Pagination uses `offset` and `limit` parameters
- Filtering uses query parameters

## Authentication

### Login

Authenticate and receive access token.

**Endpoint**: `POST /auth/login`

**Request**:
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "uuid",
    "username": "user@example.com",
    "role": "user"
  }
}
```

### Refresh Token

Obtain new access token using refresh token.

**Endpoint**: `POST /auth/refresh`

**Request**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Logout

Invalidate current session.

**Endpoint**: `POST /auth/logout`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "message": "Successfully logged out"
}
```

## REST API Endpoints

### Health Check

Check system health and status.

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "api_server": "healthy",
    "agent_server": "healthy",
    "rag_pipeline": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "elasticsearch": "healthy"
  },
  "timestamp": "2024-11-19T10:30:00Z"
}
```

### Chat Endpoints

#### Send Message

Send a chat message and receive response.

**Endpoint**: `POST /api/v1/chat/message`

**Headers**:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request**:
```json
{
  "message": "Explain microservices architecture",
  "session_id": "optional-session-id",
  "context": {
    "previous_topic": "monolithic architecture"
  },
  "options": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "stream": false
  }
}
```

**Response** (200 OK):
```json
{
  "response": "Microservices architecture is...",
  "session_id": "uuid",
  "timestamp": "2024-11-19T10:30:00Z",
  "metadata": {
    "model_used": "gpt-4",
    "tokens_used": 450,
    "execution_time": 2.3,
    "tools_used": ["knowledge_retrieval"],
    "confidence": 0.95
  }
}
```

#### Get Conversation History

Retrieve conversation history for a session.

**Endpoint**: `GET /api/v1/chat/history/{session_id}`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Query Parameters**:
- `limit` (optional): Number of messages (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response** (200 OK):
```json
{
  "session_id": "uuid",
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "Hello",
      "timestamp": "2024-11-19T10:30:00Z"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "Hello! How can I help you?",
      "timestamp": "2024-11-19T10:30:01Z",
      "metadata": {
        "model": "gpt-4",
        "tokens": 12
      }
    }
  ],
  "total": 2,
  "has_more": false
}
```

#### Delete Conversation

Delete a conversation and its history.

**Endpoint**: `DELETE /api/v1/chat/history/{session_id}`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "message": "Conversation deleted successfully",
  "session_id": "uuid"
}
```

### Document Endpoints

#### Upload Document

Upload and ingest a document into the knowledge base.

**Endpoint**: `POST /api/v1/documents/upload`

**Headers**:
```
Authorization: Bearer <access_token>
Content-Type: multipart/form-data
```

**Request**:
```
file: <binary file>
metadata: {
  "title": "Document Title",
  "category": "technical",
  "tags": ["python", "tutorial"],
  "language": "en"
}
```

**Response** (200 OK):
```json
{
  "document_id": "uuid",
  "filename": "document.pdf",
  "status": "success",
  "chunks_processed": 42,
  "processing_time": 3.5,
  "metadata": {
    "file_size": 1048576,
    "file_type": "pdf",
    "pages": 10
  }
}
```

#### Search Documents

Search the knowledge base.

**Endpoint**: `POST /api/v1/documents/search`

**Headers**:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request**:
```json
{
  "query": "How to implement authentication in FastAPI?",
  "filters": {
    "category": "technical",
    "tags": ["fastapi", "authentication"],
    "language": "en"
  },
  "max_results": 10,
  "search_type": "hybrid",
  "options": {
    "enable_reranking": true,
    "enable_query_reformulation": true
  }
}
```

**Response** (200 OK):
```json
{
  "query": "How to implement authentication in FastAPI?",
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "content": "To implement authentication in FastAPI...",
      "score": 0.95,
      "metadata": {
        "filename": "fastapi_guide.pdf",
        "page_number": 5,
        "section": "Authentication"
      },
      "highlights": [
        "implement <em>authentication</em> in <em>FastAPI</em>"
      ]
    }
  ],
  "total_results": 42,
  "processing_time": 0.15,
  "metadata": {
    "search_type": "hybrid",
    "reranking_applied": true,
    "query_reformulated": false
  }
}
```

#### List Documents

List all documents in the knowledge base.

**Endpoint**: `GET /api/v1/documents`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Query Parameters**:
- `limit` (optional): Number of documents (default: 50)
- `offset` (optional): Pagination offset (default: 0)
- `category` (optional): Filter by category
- `tags` (optional): Filter by tags (comma-separated)

**Response** (200 OK):
```json
{
  "documents": [
    {
      "id": "uuid",
      "filename": "document.pdf",
      "file_type": "pdf",
      "file_size": 1048576,
      "uploaded_at": "2024-11-19T10:30:00Z",
      "chunks_count": 42,
      "metadata": {
        "title": "Document Title",
        "category": "technical",
        "tags": ["python", "tutorial"]
      }
    }
  ],
  "total": 100,
  "has_more": true
}
```

#### Delete Document

Delete a document from the knowledge base.

**Endpoint**: `DELETE /api/v1/documents/{document_id}`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "message": "Document deleted successfully",
  "document_id": "uuid"
}
```

### Tool Endpoints

#### List Available Tools

Get list of all available tools.

**Endpoint**: `GET /api/v1/tools`

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "tools": [
    {
      "name": "knowledge_retrieval",
      "description": "Retrieve information from knowledge base",
      "category": "information",
      "capabilities": ["search", "retrieval"],
      "parameters": {
        "query": {
          "type": "string",
          "required": true,
          "description": "Search query"
        },
        "max_results": {
          "type": "integer",
          "required": false,
          "default": 10,
          "description": "Maximum number of results"
        }
      },
      "performance": {
        "average_execution_time": 0.5,
        "success_rate": 0.98
      }
    }
  ],
  "total": 10
}
```

#### Execute Tool

Execute a specific tool.

**Endpoint**: `POST /api/v1/tools/{tool_name}/execute`

**Headers**:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request**:
```json
{
  "parameters": {
    "query": "Python best practices",
    "max_results": 5
  },
  "session_id": "optional-session-id",
  "options": {
    "timeout": 30,
    "priority": "normal"
  }
}
```

**Response** (200 OK):
```json
{
  "tool_name": "knowledge_retrieval",
  "result": {
    "data": [
      {
        "content": "Python best practices include...",
        "score": 0.95
      }
    ]
  },
  "status": "success",
  "execution_time": 0.5,
  "metadata": {
    "execution_id": "uuid",
    "timestamp": "2024-11-19T10:30:00Z"
  }
}
```

### Admin Endpoints

#### System Statistics

Get system statistics and metrics.

**Endpoint**: `GET /admin/stats`

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "system": {
    "uptime": 86400,
    "version": "1.0.0",
    "environment": "production"
  },
  "usage": {
    "total_requests": 10000,
    "total_users": 100,
    "active_sessions": 25,
    "documents_indexed": 500,
    "total_tokens_used": 1000000
  },
  "performance": {
    "average_response_time": 1.2,
    "requests_per_minute": 50,
    "error_rate": 0.01
  },
  "resources": {
    "cpu_usage": 45.5,
    "memory_usage": 60.2,
    "disk_usage": 30.0
  }
}
```

#### User Management

List all users (admin only).

**Endpoint**: `GET /admin/users`

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Query Parameters**:
- `limit` (optional): Number of users (default: 50)
- `offset` (optional): Pagination offset (default: 0)
- `role` (optional): Filter by role

**Response** (200 OK):
```json
{
  "users": [
    {
      "id": "uuid",
      "username": "user@example.com",
      "role": "user",
      "created_at": "2024-11-19T10:30:00Z",
      "last_login": "2024-11-19T10:30:00Z",
      "is_active": true
    }
  ],
  "total": 100,
  "has_more": true
}
```

## WebSocket API

### Chat WebSocket

Real-time chat communication.

**Endpoint**: `ws://localhost:8000/api/v1/chat/ws/{session_id}`

**Headers**:
```
Authorization: Bearer <access_token>
```

#### Client → Server Messages

**Send Message**:
```json
{
  "type": "message",
  "content": "Hello, how are you?",
  "metadata": {
    "timestamp": "2024-11-19T10:30:00Z"
  }
}
```

**Typing Indicator**:
```json
{
  "type": "typing",
  "is_typing": true
}
```

**Ping**:
```json
{
  "type": "ping"
}
```

#### Server → Client Messages

**Response Message**:
```json
{
  "type": "message",
  "content": "I'm doing well, thank you!",
  "metadata": {
    "timestamp": "2024-11-19T10:30:01Z",
    "model": "gpt-4",
    "tokens": 12
  }
}
```

**Streaming Chunk**:
```json
{
  "type": "stream",
  "chunk": "I'm ",
  "is_final": false
}
```

**Status Update**:
```json
{
  "type": "status",
  "status": "processing",
  "message": "Searching knowledge base...",
  "progress": 50
}
```

**Error**:
```json
{
  "type": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again later.",
    "retry_after": 60
  }
}
```

**Pong**:
```json
{
  "type": "pong"
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional context"
    },
    "request_id": "uuid",
    "timestamp": "2024-11-19T10:30:00Z"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Codes

**Authentication Errors**:
- `AUTH_INVALID_CREDENTIALS`: Invalid username or password
- `AUTH_TOKEN_EXPIRED`: Access token has expired
- `AUTH_TOKEN_INVALID`: Invalid or malformed token
- `AUTH_INSUFFICIENT_PERMISSIONS`: User lacks required permissions

**Validation Errors**:
- `VALIDATION_FAILED`: Request validation failed
- `VALIDATION_MISSING_FIELD`: Required field missing
- `VALIDATION_INVALID_TYPE`: Invalid field type
- `VALIDATION_INVALID_VALUE`: Invalid field value

**Resource Errors**:
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `RESOURCE_ALREADY_EXISTS`: Resource already exists
- `RESOURCE_CONFLICT`: Resource conflict

**Rate Limiting Errors**:
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `QUOTA_EXCEEDED`: Usage quota exceeded

**Service Errors**:
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable
- `SERVICE_TIMEOUT`: Service request timeout
- `INTERNAL_ERROR`: Internal server error

## Rate Limiting

### Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1700395200
```

### Rate Limits

**Default Limits** (per user):
- Chat messages: 60 requests/minute
- Document uploads: 10 requests/minute
- Document searches: 100 requests/minute
- Tool executions: 30 requests/minute

**Global Limits**:
- API requests: 1000 requests/minute
- WebSocket connections: 100 concurrent connections

### Handling Rate Limits

When rate limit is exceeded (429 status):

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 60,
      "retry_after": 30
    }
  }
}
```

**Best Practices**:
1. Implement exponential backoff
2. Cache responses when possible
3. Use WebSocket for real-time features
4. Batch requests when applicable
5. Monitor rate limit headers

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Authors**: Raveesh Vyas, Prakhar Singhal
