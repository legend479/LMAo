# Configuration Management Best Practices

## Environment Variables
Use environment variables for configuration that changes between deployments.

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
API_KEY=your-secret-key
DEBUG=true
```

## Configuration Files
Use structured configuration files for complex settings.

```yaml
# config.yaml
database:
  host: localhost
  port: 5432
  name: myapp
  
redis:
  host: localhost
  port: 6379
  
logging:
  level: INFO
  format: json
```

## Security Considerations
- Never commit secrets to version control
- Use secret management systems
- Rotate credentials regularly
- Implement least privilege access
