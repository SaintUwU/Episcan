"""
Tests for data collection modules
"""
import pytest
import os
from unittest.mock import Mock, patch
from data_collection.twitter_collector import TwitterCollector
from data_collection.google_trends import GoogleTrendsCollector
from data_collection.news_collector import NewsCollector
from data_collection.who_data import WHODataCollector

class TestTwitterCollector:
    """Test Twitter data collector"""
    
    def test_init(self):
        """Test TwitterCollector initialization"""
        with patch.dict(os.environ, {
            'TWITTER_API_KEY': 'test_key',
            'TWITTER_API_SECRET': 'test_secret',
            'TWITTER_ACCESS_TOKEN': 'test_token',
            'TWITTER_ACCESS_TOKEN_SECRET': 'test_token_secret'
        }):
            collector = TwitterCollector()
            assert collector.api_key == 'test_key'
            assert collector.api_secret == 'test_secret'
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        with patch.dict(os.environ, {
            'TWITTER_API_KEY': 'test_key',
            'TWITTER_API_SECRET': 'test_secret',
            'TWITTER_ACCESS_TOKEN': 'test_token',
            'TWITTER_ACCESS_TOKEN_SECRET': 'test_token_secret'
        }):
            collector = TwitterCollector()
            text = "I have flu symptoms and need to see a doctor"
            keywords = collector._extract_keywords(text)
            assert 'flu' in keywords
            assert 'doctor' in keywords
    
    def test_extract_county(self):
        """Test county extraction"""
        with patch.dict(os.environ, {
            'TWITTER_API_KEY': 'test_key',
            'TWITTER_API_SECRET': 'test_secret',
            'TWITTER_ACCESS_TOKEN': 'test_token',
            'TWITTER_ACCESS_TOKEN_SECRET': 'test_token_secret'
        }):
            collector = TwitterCollector()
            text = "Outbreak reported in Nairobi county"
            county = collector._extract_county(text)
            assert county == 'Nairobi'

class TestGoogleTrendsCollector:
    """Test Google Trends data collector"""
    
    def test_init(self):
        """Test GoogleTrendsCollector initialization"""
        collector = GoogleTrendsCollector()
        assert len(collector.health_keywords) > 0
        assert len(collector.kenya_health_terms) > 0

class TestNewsCollector:
    """Test News data collector"""
    
    def test_init(self):
        """Test NewsCollector initialization"""
        collector = NewsCollector()
        assert len(collector.news_sources) > 0
        assert len(collector.health_keywords) > 0
    
    def test_is_health_related(self):
        """Test health relevance detection"""
        collector = NewsCollector()
        
        # Health-related article
        health_article = {
            'title': 'Malaria outbreak in Kenya',
            'content': 'Health officials report increased cases'
        }
        assert collector._is_health_related(health_article) == True
        
        # Non-health article
        non_health_article = {
            'title': 'Football match results',
            'content': 'Team wins championship'
        }
        assert collector._is_health_related(non_health_article) == False

class TestWHODataCollector:
    """Test WHO data collector"""
    
    def test_init(self):
        """Test WHODataCollector initialization"""
        collector = WHODataCollector()
        assert len(collector.data_sources) > 0
        assert len(collector.disease_types) > 0
    
    def test_extract_disease_type(self):
        """Test disease type extraction"""
        collector = WHODataCollector()
        text = "Malaria outbreak confirmed in several counties"
        disease_type = collector._extract_disease_type(text)
        assert disease_type == 'malaria'
    
    def test_extract_outbreak_status(self):
        """Test outbreak status extraction"""
        collector = WHODataCollector()
        
        # Confirmed outbreak
        text = "Outbreak confirmed in the region"
        status = collector._extract_outbreak_status(text)
        assert status == 'confirmed'
        
        # Suspected outbreak
        text = "Suspected cases under investigation"
        status = collector._extract_outbreak_status(text)
        assert status == 'suspected'

if __name__ == "__main__":
    pytest.main([__file__])

