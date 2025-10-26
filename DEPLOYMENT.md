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


## Data Collection Schedule

The system automatically collects data on the following schedule:

- **Twitter**: Every 2 hours
- **Google Trends**: Daily at 6 AM
- **News**: Every 4 hours  
- **WHO Data**: Daily at 8 AM
- **Full Collection**: Daily at midnight



## Troubleshooting

### Common Issues

2. **API Rate Limits**
   - Twitter API has rate limits
   - News API has daily limits
   - Google Trends may be blocked

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility


## Security Considerations

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly


