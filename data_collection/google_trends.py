"""
Google Trends data collector for health-related keywords in Kenya
"""
import pytrends
from pytrends.request import TrendReq
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from app import db
from app.models.data_models import GoogleTrendsData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleTrendsCollector:
    """Collects Google Trends data for health-related keywords in Kenya"""
    
    def __init__(self):
        """Initialize Google Trends client"""
        self.pytrends = TrendReq(hl='en-US', tz=360)  # UTC+3 for Kenya
        
        # Health-related keywords for Kenya
        self.health_keywords = [
            'flu', 'malaria', 'cholera', 'covid', 'covid-19', 'outbreak', 'epidemic',
            'fever', 'cough', 'sick', 'illness', 'disease', 'health', 'hospital',
            'clinic', 'doctor', 'nurse', 'medication', 'vaccine', 'vaccination',
            'dengue', 'typhoid', 'diarrhea', 'pneumonia', 'tuberculosis', 'tb'
        ]
        
        # Kenya-specific health terms
        self.kenya_health_terms = [
            'malaria Kenya', 'cholera Kenya', 'flu Kenya', 'covid Kenya',
            'health Kenya', 'hospital Kenya', 'clinic Kenya', 'disease Kenya'
        ]
    
    def get_trends_data(self, keywords: List[str], timeframe: str = 'today 3-m') -> Dict:
        """
        Get Google Trends data for specified keywords
        
        Args:
            keywords: List of keywords to search
            timeframe: Time period for trends (e.g., 'today 3-m', 'today 1-y')
            
        Returns:
            Dictionary with trends data
        """
        try:
            # Build payload
            self.pytrends.build_payload(
                keywords,
                cat=0,  # All categories
                timeframe=timeframe,
                geo='KE',  # Kenya
                gprop=''  # Web search
            )
            
            # Get interest over time
            interest_data = self.pytrends.interest_over_time()
            
            if interest_data.empty:
                logger.warning(f"No trends data found for keywords: {keywords}")
                return {}
            
            # Process the data
            processed_data = []
            for keyword in keywords:
                if keyword in interest_data.columns:
                    for date, value in interest_data[keyword].items():
                        processed_data.append({
                            'keyword': keyword,
                            'date': date.date(),
                            'interest_score': int(value) if not pd.isna(value) else 0,
                            'geo_location': 'KE'
                        })
            
            logger.info(f"Collected trends data for {len(processed_data)} keyword-date combinations")
            return {'data': processed_data, 'timeframe': timeframe}
            
        except Exception as e:
            logger.error(f"Error getting trends data: {str(e)}")
            return {}
    
    def get_related_queries(self, keywords: List[str]) -> Dict:
        """
        Get related queries for health keywords
        
        Args:
            keywords: List of keywords to get related queries for
            
        Returns:
            Dictionary with related queries
        """
        try:
            related_queries = {}
            
            for keyword in keywords:
                try:
                    self.pytrends.build_payload([keyword], geo='KE')
                    related = self.pytrends.related_queries()
                    
                    if keyword in related and related[keyword]['top'] is not None:
                        related_queries[keyword] = related[keyword]['top']['query'].tolist()[:10]
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error getting related queries for {keyword}: {str(e)}")
                    continue
            
            return related_queries
            
        except Exception as e:
            logger.error(f"Error getting related queries: {str(e)}")
            return {}
    
    def get_geographic_data(self, keywords: List[str]) -> Dict:
        """
        Get geographic interest data for keywords
        
        Args:
            keywords: List of keywords to search
            
        Returns:
            Dictionary with geographic data
        """
        try:
            # Build payload
            self.pytrends.build_payload(keywords, geo='KE')
            
            # Get interest by region (Kenyan counties)
            geo_data = self.pytrends.interest_by_region(resolution='COUNTRY')
            
            if geo_data.empty:
                logger.warning("No geographic data found")
                return {}
            
            # Process geographic data
            processed_geo = {}
            for keyword in keywords:
                if keyword in geo_data.columns:
                    processed_geo[keyword] = geo_data[keyword].to_dict()
            
            return processed_geo
            
        except Exception as e:
            logger.error(f"Error getting geographic data: {str(e)}")
            return {}
    
    def save_trends_to_db(self, trends_data: List[Dict]) -> int:
        """
        Save trends data to database
        
        Args:
            trends_data: List of processed trends data
            
        Returns:
            Number of records saved
        """
        saved_count = 0
        
        try:
            for data_point in trends_data:
                # Check if data point already exists
                existing = GoogleTrendsData.query.filter_by(
                    keyword=data_point['keyword'],
                    date=data_point['date']
                ).first()
                
                if not existing:
                    # Create new GoogleTrendsData record
                    trends_record = GoogleTrendsData(
                        keyword=data_point['keyword'],
                        date=data_point['date'],
                        interest_score=data_point['interest_score'],
                        geo_location=data_point['geo_location']
                    )
                    
                    db.session.add(trends_record)
                    saved_count += 1
            
            db.session.commit()
            logger.info(f"Saved {saved_count} new trends records to database")
            
        except Exception as e:
            logger.error(f"Error saving trends data to database: {str(e)}")
            db.session.rollback()
        
        return saved_count
    
    def collect_all_health_trends(self, timeframe: str = 'today 3-m') -> Dict:
        """
        Collect trends data for all health keywords
        
        Args:
            timeframe: Time period for trends
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info("Starting Google Trends data collection...")
        
        # Collect data for individual health keywords
        individual_trends = self.get_trends_data(self.health_keywords, timeframe)
        
        # Collect data for Kenya-specific terms
        kenya_trends = self.get_trends_data(self.kenya_health_terms, timeframe)
        
        # Combine all data
        all_trends_data = []
        if individual_trends.get('data'):
            all_trends_data.extend(individual_trends['data'])
        if kenya_trends.get('data'):
            all_trends_data.extend(kenya_trends['data'])
        
        # Save to database
        saved_count = self.save_trends_to_db(all_trends_data)
        
        # Get related queries and geographic data
        related_queries = self.get_related_queries(self.health_keywords[:5])  # Limit to avoid rate limits
        geo_data = self.get_geographic_data(self.health_keywords[:5])
        
        return {
            'collected': len(all_trends_data),
            'saved': saved_count,
            'duplicates': len(all_trends_data) - saved_count,
            'related_queries': related_queries,
            'geographic_data': geo_data,
            'collection_date': datetime.utcnow().isoformat(),
            'timeframe': timeframe
        }
    
    def get_trending_health_topics(self, days: int = 7) -> List[Dict]:
        """
        Get currently trending health topics in Kenya
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of trending topics with scores
        """
        try:
            # Get recent trends data
            timeframe = f'today {days}d'
            trends_data = self.get_trends_data(self.health_keywords, timeframe)
            
            if not trends_data.get('data'):
                return []
            
            # Calculate average interest scores
            keyword_scores = {}
            for data_point in trends_data['data']:
                keyword = data_point['keyword']
                score = data_point['interest_score']
                
                if keyword not in keyword_scores:
                    keyword_scores[keyword] = []
                keyword_scores[keyword].append(score)
            
            # Calculate averages and sort by trending score
            trending_topics = []
            for keyword, scores in keyword_scores.items():
                avg_score = sum(scores) / len(scores)
                trending_topics.append({
                    'keyword': keyword,
                    'average_score': avg_score,
                    'data_points': len(scores),
                    'max_score': max(scores)
                })
            
            # Sort by average score (descending)
            trending_topics.sort(key=lambda x: x['average_score'], reverse=True)
            
            return trending_topics[:10]  # Top 10 trending topics
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []

if __name__ == "__main__":
    # Test the collector
    collector = GoogleTrendsCollector()
    results = collector.collect_all_health_trends(timeframe='today 7d')
    print(f"Collection results: {results}")
    
    # Get trending topics
    trending = collector.get_trending_health_topics(days=7)
    print(f"Trending health topics: {trending}")

