#!/usr/bin/env python3
"""
Data collection script for EpiScan
Collects health data from multiple sources for ML training
"""
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from data_collection.twitter_collector import TwitterCollector
from data_collection.google_trends import GoogleTrendsCollector
from data_collection.news_collector import NewsCollector
from data_collection.who_data import WHODataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_all_data(days_back=30, twitter_max_results=500):
    """Collect data from all configured sources"""
    
    app = create_app()
    results = {}
    
    with app.app_context():
        collectors = {
            'twitter': TwitterCollector(),
            'google_trends': GoogleTrendsCollector(),
            'news': NewsCollector(),
            'who': WHODataCollector()
        }
        
        # Collect WHO data
        try:
            who_results = collectors['who'].collect_and_save(days_back=days_back)
            results['who'] = who_results
            logger.info(f"WHO: {who_results['saved']} records")
        except Exception as e:
            logger.error(f"WHO collection failed: {str(e)}")
            results['who'] = {'error': str(e), 'saved': 0}
        
        # Collect Google Trends
        try:
            if days_back <= 7:
                timeframe = f'now {days_back}d'
            elif days_back <= 30:
                timeframe = 'today 1-m'
            elif days_back <= 90:
                timeframe = 'today 3-m'
            else:
                timeframe = 'today 12-m'
            
            trends_results = collectors['google_trends'].collect_all_health_trends(
                timeframe=timeframe
            )
            results['trends'] = trends_results
            logger.info(f"Trends: {trends_results.get('total_saved', 0)} records")
        except Exception as e:
            logger.error(f"Trends collection failed: {str(e)}")
            results['trends'] = {'error': str(e), 'total_saved': 0}
        
        # Collect news
        try:
            news_results = collectors['news'].collect_and_save(days_back=days_back)
            results['news'] = news_results
            logger.info(f"News: {news_results['saved']} records")
        except Exception as e:
            logger.error(f"News collection failed: {str(e)}")
            results['news'] = {'error': str(e), 'saved': 0}
        
        # Collect Twitter
        try:
            twitter_results = collectors['twitter'].collect_and_save(
                days_back=days_back,
                max_results=twitter_max_results
            )
            results['twitter'] = twitter_results
            logger.info(f"Twitter: {twitter_results['saved']} records")
        except Exception as e:
            logger.error(f"Twitter collection failed: {str(e)}")
            results['twitter'] = {'error': str(e), 'saved': 0}
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect health data from multiple sources')
    parser.add_argument('--days', type=int, default=30, help='Days to look back')
    parser.add_argument('--twitter-max', type=int, default=500, help='Max tweets per query')
    
    args = parser.parse_args()
    
    logger.info(f"Starting data collection (days={args.days})")
    results = collect_all_data(days_back=args.days, twitter_max_results=args.twitter_max)
    
    total = sum([
        results.get('who', {}).get('saved', 0),
        results.get('trends', {}).get('total_saved', 0),
        results.get('news', {}).get('saved', 0),
        results.get('twitter', {}).get('saved', 0)
    ])
    
    logger.info(f"Collection complete. Total records: {total}")


