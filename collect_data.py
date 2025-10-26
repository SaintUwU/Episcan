#!/usr/bin/env python3
"""
Enhanced data collection for EpiScan
Focuses on real-world data sources: WHO GHO, KHIS, News, Twitter, Google Trends
"""
import sys
import os
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from data_collection.news_collector import NewsCollector
from data_collection.who_gho_collector import WHOGHOCollector
from data_collection.twitter_collector import TwitterCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhanced_data_collection():
    """Enhanced data collection from WHO, News, and Twitter sources"""
    print("=" * 60)
    print("EPISCAN COMPREHENSIVE DATA COLLECTION")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    app = create_app()
    results = {}
    
    with app.app_context():
        print("Collecting data from multiple sources...")
        print()
        
        # 1. News data collection
        print("1. Collecting News data...")
        try:
            news_collector = NewsCollector()
            news_results = news_collector.collect_and_save(days_back=30)
            results['news'] = news_results
            print(f"    News API: {news_results.get('saved', 0)} records")
        except Exception as e:
            print(f"    News API: ERROR - {str(e)}")
            results['news'] = {'error': str(e), 'saved': 0}
        
        # 2. WHO Global Health Observatory data
        print("2. Collecting WHO GHO data...")
        try:
            who_gho_collector = WHOGHOCollector()
            who_gho_results = who_gho_collector.collect_and_save(years_back=3)
            results['who_gho'] = who_gho_results
            print(f"    WHO GHO: {who_gho_results.get('saved', 0)} records")
        except Exception as e:
            print(f"    WHO GHO: ERROR - {str(e)}")
            results['who_gho'] = {'error': str(e), 'saved': 0}
        
        # 3. Twitter data collection
        print("3. Collecting Twitter data...")
        try:
            twitter_collector = TwitterCollector()
            twitter_results = twitter_collector.collect_and_save(days_back=7)
            results['twitter'] = twitter_results
            print(f"    Twitter: {twitter_results.get('saved', 0)} records")
        except Exception as e:
            print(f"    Twitter: ERROR - {str(e)}")
            results['twitter'] = {'error': str(e), 'saved': 0}
        
        # Calculate totals
        total_saved = sum([r.get('saved', 0) for r in results.values()])
        
        print(f"\n" + "=" * 60)
        print("COMPREHENSIVE DATA COLLECTION RESULTS")
        print("=" * 60)
        print(f"News API: {results['news'].get('saved', 0)} records")
        print(f"WHO GHO: {results['who_gho'].get('saved', 0)} records")
        print(f"Twitter: {results['twitter'].get('saved', 0)} records")
        print(f"TOTAL: {total_saved} records")
        
        if total_saved > 0:
            print(f"\n[SUCCESS] Successfully collected {total_saved} new records!")
            print("Your training dataset has been expanded with real-world data.")
        else:
            print(f"\n[WARNING] No new records collected.")
            print("Consider checking your API keys and network connectivity.")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        return total_saved

# Synthetic data generation functions removed - now using real-world data sources

def main():
    """Main execution function"""
    print("Comprehensive data collection for EpiScan...")
    print("This will collect data from WHO GHO, News APIs, and Twitter.")
    print()
    
    total_collected = enhanced_data_collection()
    
    if total_collected > 0:
        print(f"\n[SUCCESS] Comprehensive data collection completed!")
        print(f"Collected {total_collected} new records from multiple sources.")
        print("\nNext steps:")
        print("1. Run: python train_models.py to retrain with new data")
        print("2. Check the dashboard for updated visualizations")
        print("3. Monitor data quality and collection success rates")
    else:
        print(f"\n[ERROR] No data collected.")
        print("Consider checking your API keys and network connectivity.")
        print("Some APIs may require authentication or have rate limits.")
    
    return 0

if __name__ == "__main__":
    exit(main())
