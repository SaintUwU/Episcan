# EpiScan API Keys Setup Guide

This guide explains how to obtain and configure all the required API keys for the EpiScan data collection system.

## 🔑 Required API Keys

The EpiScan system requires API keys for the following services:

1. **Twitter/X API** - For social media data collection
2. **News API** - For news article collection
3. **Google Trends** - For search trend data (optional, has free tier)

## 📋 API Keys Summary

| Service | Required | Free Tier | Cost | Purpose |
|---------|----------|-----------|------|---------|
| Twitter API | ✅ Yes | Limited | $100+/month | Social media posts |
| News API | ✅ Yes | 1000 requests/day | $449/month | News articles |
| Google Trends | ❌ No | Free | Free | Search trends |

## 🐦 Twitter/X API Setup

### 1. Create Twitter Developer Account
1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Sign in with your Twitter account
3. Apply for developer access
4. Complete the application form

### 2. Create a Twitter App
1. Go to [developer.twitter.com/en/portal/dashboard](https://developer.twitter.com/en/portal/dashboard)
2. Click "Create App"
3. Fill in app details:
   - App name: `EpiScan`
   - Description: `Disease outbreak prediction system for Kenya`
   - Website: Your website URL
   - Use case: Research/Educational

### 3. Generate API Keys
1. Go to your app's "Keys and Tokens" tab
2. Generate the following:
   - **API Key** (Consumer Key)
   - **API Secret Key** (Consumer Secret)
   - **Access Token**
   - **Access Token Secret**
   - **Bearer Token** (for v2 API)

### 4. Set Environment Variables
```bash
export TWITTER_API_KEY="HOadu5scH1tnxzWhBy3L49zJ8"
export TWITTER_API_SECRET="qfn7o05bn2qdgKnVF8vOGc0xHZpRTq68zLazFw5HfkGF7M1MED"
export TWITTER_ACCESS_TOKEN="1246048252316864512-bBfG2FBGLlVDRyLzD8PWAck2K5HrmT"
export TWITTER_ACCESS_TOKEN_SECRET="PtGenUsmJiQ47I0oBhumBXYHvSKDEW7coe1GYuiAyq9kX"
export TWITTER_BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAAAy84QEAAAAATSa3fnK8klFr9S8GpcbUerhQMKc%3DuUafLmxZpLkOTVcRumYbtIlfk9j9ieo9TTFbDkVBRmu0hk8vXi"
```

## 📰 News API Setup

### 1. Create News API Account
1. Go to [newsapi.org](https://newsapi.org)
2. Sign up for a free account
3. Verify your email address

### 2. Get API Key
1. Go to your [dashboard](https://newsapi.org/account)
2. Copy your API key

### 3. Set Environment Variable
```bash
export NEWS_API_KEY="e5fe9c7de25b4f8a8e96ca663ddac97c"
```

### 4. Upgrade Plan (Optional)
- **Free**: 1,000 requests/day
- **Business**: $449/month for 250,000 requests/day
- **Enterprise**: Custom pricing

## 📊 Google Trends Setup

### 1. No API Key Required
Google Trends doesn't require an API key for basic usage. The `pytrends` library uses the public Google Trends interface.

### 2. Optional: Proxies for Rate Limiting
If you encounter rate limiting, you can use proxies:

```bash
export GOOGLE_TRENDS_PROXY="your_proxy_url_here"
```

## 🔧 Environment Configuration

### 1. Create `.env` File
Create a `.env` file in your project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://episcan_user:episcan_pass@localhost/episcan_db

# Twitter API Configuration
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# News API Configuration
NEWS_API_KEY=your_news_api_key

# Google Trends (Optional)
GOOGLE_TRENDS_PROXY=your_proxy_url

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

### 2. Load Environment Variables
Make sure your `.env` file is loaded:

```python
from dotenv import load_dotenv
load_dotenv()
```

## 🧪 Testing API Keys

### 1. Test Twitter API
```python
import os
import tweepy

# Test Twitter API
api_key = os.getenv('TWITTER_API_KEY')
api_secret = os.getenv('TWITTER_API_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

if all([api_key, api_secret, access_token, access_token_secret]):
    print("✅ Twitter API keys found")
else:
    print("❌ Twitter API keys missing")
```

### 2. Test News API
```python
import os
import requests

# Test News API
news_api_key = os.getenv('NEWS_API_KEY')
if news_api_key:
    url = f"https://newsapi.org/v2/top-headlines?country=ke&apiKey={news_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        print("✅ News API key working")
    else:
        print("❌ News API key invalid")
else:
    print("❌ News API key missing")
```

### 3. Test Data Collection
```bash
# Test data collection with your API keys
python app.py collect-data
```

## 💰 Cost Estimation

### Monthly Costs (Approximate)
- **Twitter API**: $100-500/month (depending on usage)
- **News API**: $0 (free tier) to $449/month
- **Google Trends**: $0 (free)
- **Total**: $100-949/month

### Free Tier Limitations
- **Twitter**: Limited to 500,000 tweets/month
- **News API**: 1,000 requests/day
- **Google Trends**: No limits (but rate limiting applies)

## 🚀 Production Setup

### 1. Environment Variables in Production
```bash
# Set in your production environment
export TWITTER_API_KEY="prod_twitter_key"
export TWITTER_API_SECRET="prod_twitter_secret"
export TWITTER_ACCESS_TOKEN="prod_access_token"
export TWITTER_ACCESS_TOKEN_SECRET="prod_access_secret"
export TWITTER_BEARER_TOKEN="prod_bearer_token"
export NEWS_API_KEY="prod_news_key"
```

### 2. Docker Environment
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV TWITTER_API_KEY=""
ENV TWITTER_API_SECRET=""
ENV TWITTER_ACCESS_TOKEN=""
ENV TWITTER_ACCESS_TOKEN_SECRET=""
ENV TWITTER_BEARER_TOKEN=""
ENV NEWS_API_KEY=""

# Rest of your Dockerfile...
```

### 3. Kubernetes Secrets
```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: episcan-api-keys
type: Opaque
data:
  twitter-api-key: <base64-encoded-key>
  twitter-api-secret: <base64-encoded-secret>
  news-api-key: <base64-encoded-key>
```

## 🔒 Security Best Practices

### 1. Never Commit API Keys
```bash
# Add to .gitignore
.env
*.env
config/secrets.json
```

### 2. Use Environment Variables
```python
# Good
api_key = os.getenv('TWITTER_API_KEY')

# Bad
api_key = "your_actual_key_here"
```

### 3. Rotate Keys Regularly
- Change API keys every 90 days
- Monitor API usage for anomalies
- Use different keys for development/production

### 4. Monitor Usage
```python
# Add usage monitoring
def log_api_usage(service, endpoint, response):
    logger.info(f"{service} API call to {endpoint}: {response.status_code}")
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Twitter API Rate Limiting
```
Error: 429 Too Many Requests
```
**Solution**: Implement exponential backoff:
```python
import time
import random

def exponential_backoff(attempt):
    delay = (2 ** attempt) + random.uniform(0, 1)
    time.sleep(delay)
```

#### 2. News API Quota Exceeded
```
Error: 429 Too Many Requests
```
**Solution**: 
- Upgrade to paid plan
- Implement request caching
- Reduce collection frequency

#### 3. Google Trends Rate Limiting
```
Error: 429 Too Many Requests
```
**Solution**:
- Use proxies
- Add delays between requests
- Implement retry logic

#### 4. Invalid API Keys
```
Error: 401 Unauthorized
```
**Solution**:
- Check key format
- Verify key is active
- Check environment variables

## 📞 Support

### API Support
- **Twitter**: [developer.twitter.com/support](https://developer.twitter.com/support)
- **News API**: [newsapi.org/contact](https://newsapi.org/contact)
- **Google Trends**: No official support (community forums)

### EpiScan Support
- Check logs for specific error messages
- Verify API key configuration
- Test individual collectors
- Check rate limits and quotas

## 🎯 Quick Start Checklist

- [ ] Create Twitter Developer Account
- [ ] Generate Twitter API Keys
- [ ] Create News API Account
- [ ] Get News API Key
- [ ] Set up `.env` file
- [ ] Test API keys
- [ ] Run data collection
- [ ] Monitor usage and costs

---

**EpiScan API Setup** - Configure your data collection APIs for disease surveillance


