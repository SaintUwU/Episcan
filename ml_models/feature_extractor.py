"""
Feature extraction module for EpiScan ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from collected data for ML models"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Health-related keywords for disease detection
        self.health_keywords = {
            'flu': ['flu', 'influenza', 'fever', 'cough', 'cold', 'sore throat'],
            'malaria': ['malaria', 'mosquito', 'fever', 'chills', 'headache'],
            'cholera': ['cholera', 'diarrhea', 'vomiting', 'dehydration', 'water'],
            'covid': ['covid', 'coronavirus', 'pandemic', 'quarantine', 'mask'],
            'dengue': ['dengue', 'fever', 'rash', 'headache', 'muscle pain'],
            'typhoid': ['typhoid', 'fever', 'headache', 'abdominal pain', 'diarrhea']
        }
        
        # Kenyan counties for geolocation
        self.kenyan_counties = [
            'nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika',
            'malindi', 'kitale', 'garissa', 'kakamega', 'kisii', 'nyeri',
            'meru', 'machakos', 'kiambu', 'kilifi', 'kwale', 'tana river',
            'lamu', 'taita taveta', 'turkana', 'west pokot', 'samburu',
            'trans nzoia', 'uasin gishu', 'elgeyo marakwet', 'nandi',
            'baringo', 'laikipia', 'nakuru', 'narok', 'kajiado', 'kericho',
            'bomet', 'kakamega', 'vihiga', 'bungoma', 'busia', 'siaya',
            'kisumu', 'homa bay', 'migori', 'kisii', 'nyamira', 'nyeri',
            'kirinyaga', 'muranga', 'kiambu', 'nairobi', 'machakos',
            'makueni', 'kitui', 'embu', 'meru', 'tharaka nithi', 'isiolo',
            'marsabit', 'mandera', 'wajir', 'garissa', 'tana river',
            'lamu', 'kilifi', 'mombasa', 'kwale', 'taita taveta'
        ]
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract features from text data"""
        if not text or pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 0.0,
                'sentiment_compound': 0.0,
                'health_keyword_count': 0,
                'disease_mentions': 0
            }
        
        # Basic text features
        text_length = len(text)
        word_count = len(text.split())
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Health keyword detection
        text_lower = text.lower()
        health_keyword_count = sum(1 for keyword_list in self.health_keywords.values() 
                                 for keyword in keyword_list if keyword in text_lower)
        
        # Disease mentions
        disease_mentions = 0
        for disease, keywords in self.health_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                disease_mentions += 1
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'sentiment_positive': sentiment_scores['pos'],
            'sentiment_negative': sentiment_scores['neg'],
            'sentiment_neutral': sentiment_scores['neu'],
            'sentiment_compound': sentiment_scores['compound'],
            'health_keyword_count': health_keyword_count,
            'disease_mentions': disease_mentions
        }
    
    def extract_temporal_features(self, date: datetime, reference_date: datetime = None) -> Dict[str, float]:
        """Extract temporal features from dates"""
        if reference_date is None:
            reference_date = datetime.now()
        
        time_diff = (reference_date - date).days
        
        return {
            'days_since_creation': time_diff,
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'hour_of_day': date.hour,
            'day_of_week': date.weekday(),
            'month': date.month,
            'is_recent': 1 if time_diff <= 7 else 0
        }
    
    def extract_geographical_features(self, county: str) -> Dict[str, float]:
        """Extract geographical features"""
        if not county or pd.isna(county):
            return {
                'county_encoded': 0,
                'is_urban': 0,
                'population_density': 0.0
            }
        
        county_lower = county.lower()
        
        # Urban counties (major cities)
        urban_counties = ['nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika']
        is_urban = 1 if county_lower in urban_counties else 0
        
        # Population density (simplified)
        population_density_map = {
            'nairobi': 1.0, 'mombasa': 0.8, 'kisumu': 0.6, 'nakuru': 0.5,
            'eldoret': 0.4, 'thika': 0.4, 'malindi': 0.3, 'kitale': 0.3
        }
        population_density = population_density_map.get(county_lower, 0.2)
        
        # County encoding
        county_encoded = self.kenyan_counties.index(county_lower) if county_lower in self.kenyan_counties else 0
        
        return {
            'county_encoded': county_encoded,
            'is_urban': is_urban,
            'population_density': population_density
        }
    
    def extract_social_media_features(self, retweet_count: int, favorite_count: int) -> Dict[str, float]:
        """Extract social media engagement features"""
        total_engagement = retweet_count + favorite_count
        
        return {
            'retweet_count': retweet_count,
            'favorite_count': favorite_count,
            'total_engagement': total_engagement,
            'engagement_score': np.log1p(total_engagement),  # Log scale for better distribution
            'has_engagement': 1 if total_engagement > 0 else 0
        }
    
    def extract_trend_features(self, interest_score: int, keyword: str) -> Dict[str, float]:
        """Extract Google Trends features"""
        # Normalize interest score (0-100 scale)
        normalized_interest = interest_score / 100.0
        
        # Check if keyword is health-related
        is_health_keyword = 1 if any(keyword.lower() in keywords for keywords in self.health_keywords.values()) else 0
        
        return {
            'interest_score': interest_score,
            'normalized_interest': normalized_interest,
            'is_health_keyword': is_health_keyword,
            'high_interest': 1 if interest_score > 50 else 0
        }
    
    def create_aggregated_features(self, data: pd.DataFrame, group_by: str, 
                                 time_window_days: int = 7) -> pd.DataFrame:
        """Create aggregated features over time windows"""
        end_date = data['date'].max()
        start_date = end_date - timedelta(days=time_window_days)
        
        window_data = data[data['date'] >= start_date]
        
        aggregated = window_data.groupby(group_by).agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'health_keyword_count': ['sum', 'mean'],
            'disease_mentions': ['sum', 'mean'],
            'total_engagement': ['sum', 'mean'],
            'interest_score': ['mean', 'max'],
            'text_length': ['mean', 'std']
        }).fillna(0)
        
        # Flatten column names
        aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
        
        return aggregated.reset_index()
    
    def extract_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the dataset"""
        logger.info("Starting feature extraction...")
        
        features_list = []
        
        for idx, row in data.iterrows():
            features = {}
            
            # Text features
            text_features = self.extract_text_features(row.get('text', ''))
            features.update(text_features)
            
            # Temporal features
            date = row.get('created_at', row.get('published_at', row.get('date')))
            if pd.notna(date):
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                temporal_features = self.extract_temporal_features(date)
                features.update(temporal_features)
            
            # Geographical features
            county = row.get('county', '')
            geo_features = self.extract_geographical_features(county)
            features.update(geo_features)
            
            # Social media features
            social_features = self.extract_social_media_features(
                row.get('retweet_count', 0),
                row.get('favorite_count', 0)
            )
            features.update(social_features)
            
            # Trend features
            trend_features = self.extract_trend_features(
                row.get('interest_score', 0),
                row.get('keyword', '')
            )
            features.update(trend_features)
            
            # Source type encoding
            source_type = row.get('source_type', 'unknown')
            features['source_twitter'] = 1 if source_type == 'twitter' else 0
            features['source_news'] = 1 if source_type == 'news' else 0
            features['source_trends'] = 1 if source_type == 'trends' else 0
            features['source_who'] = 1 if source_type == 'who' else 0
            
            # Add original data for reference
            features['original_id'] = row.get('id', idx)
            features['county'] = county
            features['date'] = date
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
        
        logger.info(f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features")
        
        return features_df
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            target_column: str = 'outbreak_occurred') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for model training"""
        
        # Select feature columns (exclude metadata)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['original_id', 'county', 'date', target_column]]
        
        X = features_df[feature_columns].values
        
        # Create target variable if it doesn't exist
        if target_column not in features_df.columns:
            # This would need to be created based on actual outbreak data
            # For now, create a synthetic target based on health keyword count and sentiment
            y = ((features_df['health_keyword_count'] > 2) & 
                 (features_df['sentiment_negative'] > 0.3)).astype(int)
        else:
            y = features_df[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def get_feature_names(self, features_df: pd.DataFrame) -> List[str]:
        """Get list of feature names for model interpretation"""
        feature_columns = [col for col in features_df.columns 
                          if col not in ['original_id', 'county', 'date', 'outbreak_occurred']]
        return feature_columns


