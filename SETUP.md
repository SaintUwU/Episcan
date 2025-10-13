# EpiScan Setup Guide

## Quick Start

### 1. Environment Setup

Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Database Configuration

Initialize the database:
```bash
flask db upgrade
```

Create admin account:
```bash
python create_admin.py
```

### 3. API Configuration (Optional)

Some data sources require API keys. Create a `.env` file:

```
# Twitter API (optional)
TWITTER_BEARER_TOKEN=your_token_here

# News API (optional)
NEWS_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost/episcan
# Or for SQLite: DATABASE_URL=sqlite:///instance/episcan.db

# Flask
FLASK_SECRET_KEY=your_secret_key_here
```

See `API_KEYS_SETUP.md` for detailed API configuration.

### 4. Data Collection

Initial data collection:
```bash
# Collect from public sources
python collect_data.py --days 30

# Or generate test data
python generate_test_data.py --tweets 500 --news 200
```

### 5. Train Models

```bash
python ml_models/train_models.py
```

### 6. Run Application

```bash
python app.py
```

Access at: http://localhost:5000

## Data Management

### Importing Historical Data

Prepare CSV files with required columns (see `csv_templates/` for examples).

Import:
```bash
python import_data.py --twitter historical_tweets.csv --news historical_articles.csv
```

### Automated Collection

Run the scheduler for continuous data collection:
```bash
python run_scheduler.py
```

This runs in the background and collects data at scheduled intervals:
- Twitter: Every 2 hours
- News: Every 4 hours  
- Google Trends: Daily at 6 AM
- WHO Data: Daily at 8 AM

## Model Training

Train models with your collected data:

```bash
# Basic training
python ml_models/train_models.py

# Specify training period
python ml_models/train_models.py --days 90

# With hyperparameter tuning (slower but better results)
python ml_models/train_models.py --hyperparameter-tuning
```

## Troubleshooting

### Database Connection Issues
- Check DATABASE_URL in .env
- Ensure PostgreSQL is running
- Or use SQLite for development

### API Rate Limits
- Reduce collection frequency
- Use smaller time ranges (--days 7)
- Focus on public data sources (WHO, News RSS)

### Insufficient Training Data
- Collect more data over time
- Generate test data: `python generate_test_data.py --tweets 1000 --news 500`
- Import historical datasets if available

### Model Training Fails
- Ensure you have at least 500 records
- Check data quality: `python check_data.py`
- Review logs for specific errors

## Production Deployment

For production deployment, see `DEPLOYMENT.md`.

Key considerations:
- Use PostgreSQL instead of SQLite
- Set up proper environment variables
- Configure reverse proxy (nginx)
- Use gunicorn or uwsgi for serving
- Set up SSL certificates
- Configure automated backups
- Monitor logs and performance

## Development

Run in development mode:
```bash
export FLASK_ENV=development
python app.py
```

Run tests:
```bash
python run_tests.py
```

## Support

Check logs for errors:
- Application logs: Check console output
- Data collection logs: Check `logs/` directory
- Database issues: Check PostgreSQL logs

For API configuration help, see `API_KEYS_SETUP.md`.

