"""
Data preprocessing pipeline for EpiScan ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from sqlalchemy import create_engine, text
from app import create_app, db
from app.models.data_models import TwitterData, GoogleTrendsData, NewsData, WHOData
from app.models.prediction_models import OutbreakPrediction, Alert, ModelPerformance
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data preprocessing and preparation for ML training"""
    
    def __init__(self):
        self.app = create_app()
        self.kenyan_counties = [
            'nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika',
            'malindi', 'kitale', 'garissa', 'kakamega', 'kisii', 'nyeri',
            'meru', 'machakos', 'kiambu', 'kilifi', 'kwale', 'tana river',
            'lamu', 'taita taveta', 'turkana', 'west pokot', 'samburu',
            'trans nzoia', 'uasin gishu', 'elgeyo marakwet', 'nandi',
            'baringo', 'laikipia', 'narok', 'kajiado', 'kericho',
            'bomet', 'vihiga', 'bungoma', 'busia', 'siaya',
            'homa bay', 'migori', 'nyamira', 'kirinyaga', 'muranga',
            'makueni', 'kitui', 'embu', 'tharaka nithi', 'isiolo',
            'marsabit', 'mandera', 'wajir'
        ]
    
    def collect_training_data(self, days_back: int = 90) -> pd.DataFrame:
        """Collect and combine data from all sources for training"""
        logger.info(f"Collecting training data from last {days_back} days...")
        
        with self.app.app_context():
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_data = []
            
            # Collect Twitter data
            twitter_data = self._collect_twitter_data(start_date, end_date)
            all_data.extend(twitter_data)
            
            # Collect News data
            news_data = self._collect_news_data(start_date, end_date)
            all_data.extend(news_data)
            
            # Collect Google Trends data
            trends_data = self._collect_trends_data(start_date, end_date)
            all_data.extend(trends_data)
            
            # Collect WHO data
            who_data = self._collect_who_data(start_date, end_date)
            all_data.extend(who_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            if df.empty:
                logger.warning("No data collected for training")
                return df
            
            # Clean and standardize data
            df = self._clean_data(df)
            
            logger.info(f"Collected {len(df)} records for training")
            return df
    
    def _collect_twitter_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect Twitter data"""
        try:
            tweets = TwitterData.query.filter(
                TwitterData.created_at >= start_date,
                TwitterData.created_at <= end_date
            ).all()
            
            data = []
            for tweet in tweets:
                data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'county': tweet.county,
                    'created_at': tweet.created_at,
                    'sentiment_score': tweet.sentiment_score,
                    'sentiment_label': tweet.sentiment_label,
                    'keywords': tweet.keywords,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'source_type': 'twitter',
                    'username': tweet.username,
                    'location': tweet.location
                })
            
            logger.info(f"Collected {len(data)} Twitter records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {str(e)}")
            return []
    
    def _collect_news_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect News data"""
        try:
            news = NewsData.query.filter(
                NewsData.published_at >= start_date,
                NewsData.published_at <= end_date
            ).all()
            
            data = []
            for article in news:
                data.append({
                    'id': article.id,
                    'text': article.content,
                    'title': article.title,
                    'county': article.county,
                    'created_at': article.published_at,
                    'sentiment_score': article.sentiment_score,
                    'sentiment_label': article.sentiment_label,
                    'keywords': article.keywords,
                    'source': article.source,
                    'url': article.url,
                    'health_relevance_score': article.health_relevance_score,
                    'source_type': 'news',
                    'retweet_count': 0,
                    'favorite_count': 0
                })
            
            logger.info(f"Collected {len(data)} News records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting News data: {str(e)}")
            return []
    
    def _collect_trends_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect Google Trends data"""
        try:
            trends = GoogleTrendsData.query.filter(
                GoogleTrendsData.date >= start_date.date(),
                GoogleTrendsData.date <= end_date.date()
            ).all()
            
            data = []
            for trend in trends:
                data.append({
                    'id': trend.id,
                    'text': trend.keyword,  # Use keyword as text
                    'county': None,  # Trends data doesn't have county
                    'created_at': trend.date,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'keywords': [trend.keyword],
                    'interest_score': trend.interest_score,
                    'keyword': trend.keyword,
                    'geo_location': trend.geo_location,
                    'source_type': 'trends',
                    'retweet_count': 0,
                    'favorite_count': 0
                })
            
            logger.info(f"Collected {len(data)} Trends records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting Trends data: {str(e)}")
            return []
    
    def _collect_who_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect WHO data"""
        try:
            who_data = WHOData.query.filter(
                WHOData.published_at >= start_date,
                WHOData.published_at <= end_date
            ).all()
            
            data = []
            for item in who_data:
                data.append({
                    'id': item.id,
                    'text': item.content,
                    'title': item.title,
                    'county': None,  # WHO data doesn't have county
                    'created_at': item.published_at,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'keywords': [item.disease_type] if item.disease_type else [],
                    'disease_type': item.disease_type,
                    'outbreak_status': item.outbreak_status,
                    'case_count': item.case_count,
                    'source': item.source,
                    'url': item.url,
                    'source_type': 'who',
                    'retweet_count': 0,
                    'favorite_count': 0
                })
            
            logger.info(f"Collected {len(data)} WHO records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting WHO data: {str(e)}")
            return []
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the collected data"""
        logger.info("Cleaning and standardizing data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id', 'source_type'])
        
        # Handle missing values
        df['text'] = df['text'].fillna('')
        df['county'] = df['county'].fillna('')
        df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
        df['retweet_count'] = df['retweet_count'].fillna(0)
        df['favorite_count'] = df['favorite_count'].fillna(0)
        
        # Add interest_score if it doesn't exist
        if 'interest_score' not in df.columns:
            df['interest_score'] = 0
        else:
            df['interest_score'] = df['interest_score'].fillna(0)
        
        # Standardize county names
        df['county'] = df['county'].str.lower().str.strip()
        df['county'] = df['county'].apply(self._standardize_county_name)
        
        # Convert dates
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['date'] = df['created_at'].dt.date
        
        # Remove records with invalid dates
        df = df.dropna(subset=['created_at'])
        
        # Clean text data
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 10]  # Remove very short texts
        
        # Handle keywords
        df['keywords'] = df['keywords'].apply(self._process_keywords)
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df)} records")
        return df
    
    def _standardize_county_name(self, county: str) -> str:
        """Standardize county name"""
        if not county or pd.isna(county):
            return ''
        
        county_lower = county.lower().strip()
        
        # Direct matches
        if county_lower in self.kenyan_counties:
            return county_lower
        
        # Fuzzy matching for common variations
        county_mappings = {
            'nairobi': ['nairobi', 'nrb'],
            'mombasa': ['mombasa', 'mombasa county'],
            'kisumu': ['kisumu', 'kisumu county'],
            'nakuru': ['nakuru', 'nakuru county'],
            'eldoret': ['eldoret', 'uasin gishu'],
            'thika': ['thika', 'kiambu']
        }
        
        for standard_name, variations in county_mappings.items():
            if any(var in county_lower for var in variations):
                return standard_name
        
        return county_lower
    
    def _process_keywords(self, keywords) -> List[str]:
        """Process keywords field"""
        # Handle None or NaN
        if keywords is None:
            return []
        
        try:
            if pd.isna(keywords):
                return []
        except (ValueError, TypeError):
            # Handle arrays/lists that can't be checked with pd.isna
            pass
        
        if isinstance(keywords, list):
            return [str(k).lower().strip() for k in keywords if k is not None and str(k).strip()]
        
        if isinstance(keywords, str):
            # Try to parse as JSON or comma-separated
            try:
                import json
                return json.loads(keywords)
            except:
                return [k.strip() for k in keywords.split(',') if k.strip()]
        
        return []
    
    def create_training_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for training (synthetic for now)"""
        logger.info("Creating training targets...")
        
        # This is a simplified approach - in practice, you'd need actual outbreak data
        # For now, we'll create synthetic targets based on health indicators
        
        # Calculate health relevance score
        health_scores = []
        for _, row in df.iterrows():
            score = 0
            
            # Text-based indicators
            text = str(row.get('text', '')).lower()
            
            # Disease keywords
            disease_keywords = ['flu', 'malaria', 'cholera', 'covid', 'fever', 'outbreak', 'epidemic']
            score += sum(1 for keyword in disease_keywords if keyword in text)
            
            # Sentiment indicators
            if row.get('sentiment_score', 0) < -0.1:  # Negative sentiment
                score += 1
            
            # Social engagement (for social media)
            if row.get('source_type') == 'twitter':
                engagement = row.get('retweet_count', 0) + row.get('favorite_count', 0)
                if engagement > 10:
                    score += 1
            
            # Interest score (for trends)
            if row.get('source_type') == 'trends':
                interest = row.get('interest_score', 0)
                if interest > 50:
                    score += 1
            
            # WHO outbreak status
            if row.get('source_type') == 'who':
                outbreak_status = row.get('outbreak_status', '')
                if 'confirmed' in outbreak_status.lower():
                    score += 2
                elif 'suspected' in outbreak_status.lower():
                    score += 1
            
            health_scores.append(score)
        
        # Normalize scores to 0-1
        max_score = max(health_scores) if health_scores and max(health_scores) > 0 else 1
        health_scores_normalized = [score / max_score for score in health_scores]
        
        # Create binary target (outbreak occurred or not)
        # Use median threshold for balanced classes
        if health_scores_normalized:
            threshold = np.median(health_scores_normalized)
        else:
            threshold = 0.3
        
        df['outbreak_occurred'] = (np.array(health_scores_normalized) > threshold).astype(int)
        df['health_relevance_score_computed'] = health_scores_normalized
        
        # Add temporal features
        df['is_recent'] = (datetime.now() - df['created_at']).dt.days <= 7
        df['is_weekend'] = df['created_at'].dt.weekday >= 5
        
        logger.info(f"Created targets. Outbreak rate: {df['outbreak_occurred'].mean():.3f}")
        return df
    
    def prepare_ml_dataset(self, days_back: int = 90) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare complete ML dataset"""
        logger.info("Preparing ML dataset...")
        
        # Collect data
        df = self.collect_training_data(days_back)
        
        if df.empty:
            logger.warning("No data available for training")
            return df, []
        
        # Create targets
        df = self.create_training_targets(df)
        
        # Add additional features
        df = self._add_derived_features(df)
        
        # Get feature names (exclude string columns and non-numeric features)
        exclude_columns = ['id', 'text', 'title', 'county', 'created_at', 'date', 
                          'keywords', 'source', 'url', 'username', 'location',
                          'disease_type', 'outbreak_status', 'case_count',
                          'geo_location', 'keyword', 'source_type', 'sentiment_label']
        
        # Only include numeric columns
        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns and df[col].dtype in ['int64', 'float64', 'bool']:
                feature_columns.append(col)
        
        # Handle any remaining non-numeric columns by converting them
        for col in df.columns:
            if col not in exclude_columns and col not in feature_columns:
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        feature_columns.append(col)
                except:
                    # Skip columns that can't be converted
                    continue
        
        logger.info(f"ML dataset prepared: {len(df)} records, {len(feature_columns)} features")
        return df, feature_columns
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset"""
        # Text length
        df['text_length'] = df['text'].str.len()
        
        # Word count
        df['word_count'] = df['text'].str.split().str.len()
        
        # Has county information
        df['has_county'] = (df['county'] != '').astype(int)
        
        # Is from social media
        df['is_social_media'] = (df['source_type'] == 'twitter').astype(int)
        
        # Is from news
        df['is_news'] = (df['source_type'] == 'news').astype(int)
        
        # Is from official source
        df['is_official'] = df['source_type'].isin(['who', 'news']).astype(int)
        
        # Engagement score
        df['engagement_score'] = df['retweet_count'] + df['favorite_count']
        
        # High engagement
        df['high_engagement'] = (df['engagement_score'] > df['engagement_score'].quantile(0.8)).astype(int)
        
        return df
    
    def save_training_data(self, df: pd.DataFrame, filename: str = None):
        """Save training data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"
        
        filepath = os.path.join("ml_models", filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Training data saved to {filepath}")
        
        return filepath
    
    def load_training_data(self, filepath: str) -> pd.DataFrame:
        """Load training data from file"""
        df = pd.read_csv(filepath)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        logger.info(f"Training data loaded from {filepath}: {len(df)} records")
        return df


