#!/usr/bin/env python3
"""
CSV data import utility
Import historical health data from CSV files
"""
import sys
import os
import pandas as pd
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.data_models import TwitterData, NewsData, GoogleTrendsData, WHOData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataImporter:
    """Handle CSV imports for various data sources"""
    
    def __init__(self):
        self.app = create_app()
    
    def import_twitter(self, csv_path):
        """Import Twitter data from CSV"""
        logger.info(f"Importing Twitter data from {csv_path}")
        
        with self.app.app_context():
            try:
                df = pd.read_csv(csv_path)
                
                required = ['tweet_id', 'text', 'username', 'created_at']
                if not all(col in df.columns for col in required):
                    return {'success': False, 'error': f'Missing required columns', 'imported': 0}
                
                imported = 0
                skipped = 0
                
                for _, row in df.iterrows():
                    try:
                        if TwitterData.query.filter_by(tweet_id=str(row['tweet_id'])).first():
                            skipped += 1
                            continue
                        
                        twitter_data = TwitterData(
                            tweet_id=str(row['tweet_id']),
                            text=str(row['text']),
                            user_id=str(row.get('user_id', row['username'])),
                            username=str(row['username']),
                            created_at=pd.to_datetime(row['created_at']),
                            location=str(row.get('location', '')) if pd.notna(row.get('location')) else None,
                            county=str(row.get('county', '')) if pd.notna(row.get('county')) else None,
                            sentiment_score=float(row.get('sentiment_score', 0)) if pd.notna(row.get('sentiment_score')) else None,
                            sentiment_label=str(row.get('sentiment_label', 'neutral')),
                            keywords=row.get('keywords', []) if isinstance(row.get('keywords'), list) else [],
                            retweet_count=int(row.get('retweet_count', 0)) if pd.notna(row.get('retweet_count')) else 0,
                            favorite_count=int(row.get('favorite_count', 0)) if pd.notna(row.get('favorite_count')) else 0
                        )
                        
                        db.session.add(twitter_data)
                        imported += 1
                        
                        if imported % 100 == 0:
                            db.session.commit()
                    
                    except Exception as e:
                        logger.warning(f"Error importing row: {str(e)}")
                        skipped += 1
                        continue
                
                db.session.commit()
                return {'success': True, 'imported': imported, 'skipped': skipped}
            
            except Exception as e:
                db.session.rollback()
                logger.error(f"Import failed: {str(e)}")
                return {'success': False, 'error': str(e), 'imported': 0}
    
    def import_news(self, csv_path):
        """Import news articles from CSV"""
        logger.info(f"Importing news data from {csv_path}")
        
        with self.app.app_context():
            try:
                df = pd.read_csv(csv_path)
                
                required = ['title', 'content', 'source', 'url', 'published_at']
                if not all(col in df.columns for col in required):
                    return {'success': False, 'error': 'Missing required columns', 'imported': 0}
                
                imported = 0
                skipped = 0
                
                for _, row in df.iterrows():
                    try:
                        if NewsData.query.filter_by(url=str(row['url'])).first():
                            skipped += 1
                            continue
                        
                        news_data = NewsData(
                            title=str(row['title']),
                            content=str(row['content']),
                            source=str(row['source']),
                            url=str(row['url']),
                            published_at=pd.to_datetime(row['published_at']),
                            county=str(row.get('county', '')) if pd.notna(row.get('county')) else None,
                            sentiment_score=float(row.get('sentiment_score', 0)) if pd.notna(row.get('sentiment_score')) else None,
                            sentiment_label=str(row.get('sentiment_label', 'neutral')),
                            keywords=row.get('keywords', []),
                            health_relevance_score=float(row.get('health_relevance_score', 0)) if pd.notna(row.get('health_relevance_score')) else None
                        )
                        
                        db.session.add(news_data)
                        imported += 1
                        
                        if imported % 100 == 0:
                            db.session.commit()
                    
                    except Exception as e:
                        logger.warning(f"Error importing row: {str(e)}")
                        skipped += 1
                
                db.session.commit()
                return {'success': True, 'imported': imported, 'skipped': skipped}
            
            except Exception as e:
                db.session.rollback()
                return {'success': False, 'error': str(e), 'imported': 0}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Import data from CSV files')
    parser.add_argument('--twitter', help='Twitter CSV file path')
    parser.add_argument('--news', help='News CSV file path')
    
    args = parser.parse_args()
    
    if not any([args.twitter, args.news]):
        parser.print_help()
        print("\nError: Specify at least one CSV file")
        sys.exit(1)
    
    importer = DataImporter()
    
    if args.twitter:
        result = importer.import_twitter(args.twitter)
        print(f"Twitter: {result['imported']} imported, {result.get('skipped', 0)} skipped")
    
    if args.news:
        result = importer.import_news(args.news)
        print(f"News: {result['imported']} imported, {result.get('skipped', 0)} skipped")


