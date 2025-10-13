# EpiScan - Disease Outbreak Prediction System

A machine learning system for predicting disease outbreaks in Kenya using social media data, news articles, and health authority reports.

## Features

- Multi-source data collection (Twitter, News, WHO, Google Trends)
- Machine learning models for outbreak prediction
- Real-time data processing and analysis
- Web dashboard for visualization
- Automated data collection scheduler

## Setup

### Prerequisites
- Python 3.9+
- PostgreSQL (or SQLite for development)
- Virtual environment

### Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp env.example .env
   # Edit .env with your API keys and database credentials
   ```

5. Initialize database:
   ```bash
   flask db upgrade
   ```

6. Create admin user:
   ```bash
   python create_admin.py
   ```

## Usage

### Data Collection

Collect data from all sources:
```bash
python collect_data.py --days 30 --twitter-max 500
```

Import data from CSV:
```bash
python import_data.py --twitter data/tweets.csv --news data/articles.csv
```

Generate test data:
```bash
python generate_test_data.py --tweets 500 --news 200
```

### Model Training

Train ML models:
```bash
python ml_models/train_models.py --days 90
```

### Running the Application

Start the web server:
```bash
python app.py
```

Start automated data collection:
```bash
python run_scheduler.py
```

## Project Structure

```
episcan/
├── app/                    # Flask application
│   ├── models/            # Database models
│   ├── routes/            # API routes
│   └── templates/         # HTML templates
├── data_collection/       # Data collection modules
├── ml_models/            # Machine learning models
├── migrations/           # Database migrations
└── tests/               # Unit tests
```

## API Documentation

See `API_KEYS_SETUP.md` for configuring external APIs.

## Testing

Run tests:
```bash
python run_tests.py
```

## Deployment

See `DEPLOYMENT.md` for production deployment instructions.

## License

MIT License

## Contributors

- EpiScan Development Team
