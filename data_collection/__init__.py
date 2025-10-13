"""
Data collection modules for EpiScan
"""
from .twitter_collector import TwitterCollector
from .google_trends import GoogleTrendsCollector
from .news_collector import NewsCollector
from .who_data import WHODataCollector

__all__ = [
    'TwitterCollector',
    'GoogleTrendsCollector', 
    'NewsCollector',
    'WHODataCollector'
]

