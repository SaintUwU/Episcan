# EpiScan Deployment Guide

## Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher (for production)
- Git

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd EpiScan
   python setup.py
   ```

2. **Configure Environment**
   - Copy `env.example` to `.env`
   - Update API keys in `.env` file

3. **Start the Application**
   ```bash
   # Start the web application
   python app.py
   
   # Start the data collection scheduler (in another terminal)
   python run_scheduler.py
   ```

4. **Access the Dashboard**
   - Open http://localhost:5000 in your browser

## Environment Configuration

### Required API Keys

1. **Twitter API**
   - Get API keys from https://developer.twitter.com/
   - Add to `.env`:
     ```
     TWITTER_API_KEY=your_api_key
     TWITTER_API_SECRET=your_api_secret
     TWITTER_ACCESS_TOKEN=your_access_token
     TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
     TWITTER_BEARER_TOKEN=your_bearer_token
     ```

2. **News API**
   - Get API key from https://newsapi.org/
   - Add to `.env`:
     ```
     NEWS_API_KEY=your_news_api_key
     ```

3. **Database**
   - For production, use PostgreSQL:
     ```
     DATABASE_URL=postgresql://username:password@localhost/episcan
     ```

## Production Deployment

### Using Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 5000
   
   CMD ["python", "app.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t episcan .
   docker run -p 5000:5000 episcan
   ```

### Using Heroku

1. **Create Procfile**
   ```
   web: python app.py
   worker: python run_scheduler.py
   ```

2. **Deploy**
   ```bash
   heroku create episcan-app
   heroku addons:create heroku-postgresql:hobby-dev
   git push heroku main
   ```

### Using AWS/GCP/Azure

1. **Set up PostgreSQL database**
2. **Configure environment variables**
3. **Deploy using your preferred method**

## Data Collection Schedule

The system automatically collects data on the following schedule:

- **Twitter**: Every 2 hours
- **Google Trends**: Daily at 6 AM
- **News**: Every 4 hours  
- **WHO Data**: Daily at 8 AM
- **Full Collection**: Daily at midnight

## Monitoring

### Health Check
- Endpoint: `GET /api/health`
- Returns system status and timestamp

### Statistics
- Endpoint: `GET /api/stats`
- Returns data collection statistics

### Logs
- Application logs are written to console
- For production, configure proper logging

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check DATABASE_URL in .env
   - Ensure PostgreSQL is running
   - Verify database exists

2. **API Rate Limits**
   - Twitter API has rate limits
   - News API has daily limits
   - Google Trends may be blocked

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

### Debug Mode

Enable debug mode by setting in `.env`:
```
FLASK_DEBUG=True
FLASK_ENV=development
```

## Security Considerations

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Database Security**
   - Use strong passwords
   - Enable SSL connections
   - Restrict network access

3. **Application Security**
   - Use HTTPS in production
   - Implement proper authentication
   - Regular security updates

## Performance Optimization

1. **Database Indexing**
   - Add indexes on frequently queried columns
   - Monitor query performance

2. **Caching**
   - Implement Redis for caching
   - Cache API responses

3. **Scaling**
   - Use load balancers
   - Implement horizontal scaling
   - Monitor resource usage

## Backup and Recovery

1. **Database Backups**
   ```bash
   pg_dump episcan > backup.sql
   ```

2. **Data Backup**
   - Regular exports of collected data
   - Store backups securely

3. **Recovery Procedures**
   - Document recovery steps
   - Test backup restoration

## Maintenance

1. **Regular Updates**
   - Update dependencies
   - Apply security patches
   - Monitor for new features

2. **Data Cleanup**
   - Archive old data
   - Clean up temporary files
   - Monitor disk usage

3. **Performance Monitoring**
   - Monitor response times
   - Track error rates
   - Analyze usage patterns

