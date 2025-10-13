"""
News data collector for Kenyan health-related articles
"""
import requests
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from bs4 import BeautifulSoup
from app import db
from app.models.data_models import NewsData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    """Collects health-related news from Kenyan sources"""
    
    def __init__(self):
        """Initialize news collector"""
        self.news_api_key = os.getenv('NEWS_API_KEY')
        
        # Kenyan news sources
        self.news_sources = {
            'the-star': {
                'name': 'The Star',
                'url': 'https://www.the-star.co.ke',
                'rss': 'https://www.the-star.co.ke/rss',
                'api_source': 'the-star'
            },
            'standard-media': {
                'name': 'Standard Media',
                'url': 'https://www.standardmedia.co.ke',
                'rss': 'https://www.standardmedia.co.ke/rss',
                'api_source': 'standard-media'
            },
            'citizen-tv': {
                'name': 'Citizen TV',
                'url': 'https://citizentv.co.ke',
                'rss': 'https://citizentv.co.ke/rss',
                'api_source': 'kenya-citizen-tv'
            },
            'nation': {
                'name': 'Daily Nation',
                'url': 'https://www.nation.co.ke',
                'rss': 'https://www.nation.co.ke/rss',
                'api_source': 'nation'
            }
        }
        
        # Health-related keywords
        self.health_keywords = [
            'health', 'disease', 'outbreak', 'epidemic', 'flu', 'malaria', 'cholera',
            'covid', 'covid-19', 'fever', 'cough', 'sick', 'illness', 'hospital',
            'clinic', 'doctor', 'nurse', 'medication', 'vaccine', 'vaccination',
            'dengue', 'typhoid', 'diarrhea', 'pneumonia', 'tuberculosis', 'tb',
            'moh', 'ministry of health', 'who', 'world health organization'
        ]
        
        # Kenyan counties for geolocation
        self.kenyan_counties = [
            'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi',
            'Kitale', 'Garissa', 'Kakamega', 'Meru', 'Nyeri', 'Machakos', 'Kisii',
            'Kericho', 'Bungoma', 'Busia', 'Vihiga', 'Siaya', 'Homa Bay', 'Migori',
            'Nyamira', 'Trans Nzoia', 'Uasin Gishu', 'Elgeyo Marakwet', 'Nandi',
            'Baringo', 'Laikipia', 'Narok', 'Kajiado', 'Bomet', 'West Pokot',
            'Samburu', 'Turkana', 'Wajir', 'Mandera', 'Marsabit', 'Isiolo',
            'Tharaka Nithi', 'Embu', 'Kitui', 'Makueni', 'Taita Taveta', 'Kwale',
            'Kilifi', 'Tana River', 'Lamu'
        ]
    
    def collect_from_news_api(self, days_back: int = 7) -> List[Dict]:
        """
        Collect news using News API
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            List of news articles
        """
        if not self.news_api_key:
            logger.warning("News API key not found, skipping API collection")
            return []
        
        try:
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Collect from each Kenyan source
            all_articles = []
            for source_id, source_info in self.news_sources.items():
                try:
                    articles = self._fetch_from_news_api(source_info['api_source'], from_date)
                    all_articles.extend(articles)
                    
                    # Rate limiting
                    import time
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error collecting from {source_info['name']}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(all_articles)} articles from News API")
            return all_articles
            
        except Exception as e:
            logger.error(f"Error collecting from News API: {str(e)}")
            return []
    
    def _fetch_from_news_api(self, source: str, from_date: str) -> List[Dict]:
        """Fetch articles from specific source using News API"""
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'sources': source,
                'from': from_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            # Filter for health-related articles
            health_articles = []
            for article in articles:
                if self._is_health_related(article):
                    processed_article = self._process_news_article(article, source)
                    if processed_article:
                        health_articles.append(processed_article)
            
            return health_articles
            
        except Exception as e:
            logger.error(f"Error fetching from News API source {source}: {str(e)}")
            return []
    
    def collect_from_rss(self, days_back: int = 7) -> List[Dict]:
        """
        Collect news from RSS feeds
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            List of news articles
        """
        try:
            all_articles = []
            
            for source_id, source_info in self.news_sources.items():
                try:
                    articles = self._fetch_from_rss(source_info['rss'], source_info['name'])
                    all_articles.extend(articles)
                    
                except Exception as e:
                    logger.warning(f"Error collecting from RSS {source_info['name']}: {str(e)}")
                    continue
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_articles = [
                article for article in all_articles
                if article.get('published_at') and article['published_at'] >= cutoff_date
            ]
            
            logger.info(f"Collected {len(recent_articles)} articles from RSS feeds")
            return recent_articles
            
        except Exception as e:
            logger.error(f"Error collecting from RSS feeds: {str(e)}")
            return []
    
    def _fetch_from_rss(self, rss_url: str, source_name: str) -> List[Dict]:
        """Fetch articles from RSS feed"""
        try:
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            articles = []
            for item in items:
                try:
                    article = self._parse_rss_item(item, source_name)
                    if article and self._is_health_related(article):
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing RSS item: {str(e)}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS from {rss_url}: {str(e)}")
            return []
    
    def _parse_rss_item(self, item, source_name: str) -> Optional[Dict]:
        """Parse individual RSS item"""
        try:
            title = item.find('title')
            description = item.find('description')
            link = item.find('link')
            pub_date = item.find('pubDate')
            
            if not all([title, link]):
                return None
            
            # Parse publication date
            published_at = None
            if pub_date:
                try:
                    from dateutil import parser
                    published_at = parser.parse(pub_date.text)
                except:
                    published_at = datetime.now()
            
            return {
                'title': title.text.strip() if title else '',
                'content': description.text.strip() if description else '',
                'url': link.text.strip() if link else '',
                'source': source_name,
                'published_at': published_at or datetime.now()
            }
            
        except Exception as e:
            logger.warning(f"Error parsing RSS item: {str(e)}")
            return None
    
    def _is_health_related(self, article: Dict) -> bool:
        """Check if article is health-related"""
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        # Check for health keywords
        for keyword in self.health_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _process_news_article(self, article: Dict, source: str) -> Optional[Dict]:
        """Process news article and extract relevant information"""
        try:
            # Parse publication date
            published_at = None
            if article.get('publishedAt'):
                try:
                    from dateutil import parser
                    published_at = parser.parse(article['publishedAt'])
                except:
                    published_at = datetime.now()
            
            # Extract county information
            county = self._extract_county(article.get('title', '') + ' ' + article.get('content', ''))
            
            # Extract keywords
            keywords = self._extract_keywords(article.get('title', '') + ' ' + article.get('content', ''))
            
            return {
                'title': article.get('title', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'source': source,
                'published_at': published_at or datetime.now(),
                'county': county,
                'keywords': keywords
            }
            
        except Exception as e:
            logger.warning(f"Error processing article: {str(e)}")
            return None
    
    def _extract_county(self, text: str) -> Optional[str]:
        """Extract Kenyan county from text"""
        text_lower = text.lower()
        
        for county in self.kenyan_counties:
            if county.lower() in text_lower:
                return county
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract health-related keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.health_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def save_news_to_db(self, articles: List[Dict]) -> int:
        """
        Save news articles to database
        
        Args:
            articles: List of processed news articles
            
        Returns:
            Number of articles saved
        """
        saved_count = 0
        
        try:
            for article in articles:
                # Check if article already exists
                existing_article = NewsData.query.filter_by(url=article['url']).first()
                
                if not existing_article:
                    # Create new NewsData record
                    news_record = NewsData(
                        title=article['title'],
                        content=article['content'],
                        source=article['source'],
                        url=article['url'],
                        published_at=article['published_at'],
                        county=article.get('county'),
                        keywords=article.get('keywords', [])
                    )
                    
                    db.session.add(news_record)
                    saved_count += 1
            
            db.session.commit()
            logger.info(f"Saved {saved_count} new news articles to database")
            
        except Exception as e:
            logger.error(f"Error saving news articles to database: {str(e)}")
            db.session.rollback()
        
        return saved_count
    
    def collect_and_save(self, days_back: int = 7) -> Dict:
        """
        Main method to collect and save news articles
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info("Starting news data collection...")
        
        # Collect from News API
        api_articles = self.collect_from_news_api(days_back)
        
        # Collect from RSS feeds
        rss_articles = self.collect_from_rss(days_back)
        
        # Combine all articles
        all_articles = api_articles + rss_articles
        
        # Remove duplicates based on URL
        unique_articles = []
        seen_urls = set()
        for article in all_articles:
            if article['url'] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article['url'])
        
        # Save to database
        saved_count = self.save_news_to_db(unique_articles)
        
        return {
            'collected_api': len(api_articles),
            'collected_rss': len(rss_articles),
            'total_collected': len(all_articles),
            'unique_collected': len(unique_articles),
            'saved': saved_count,
            'duplicates': len(unique_articles) - saved_count,
            'collection_date': datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Test the collector
    collector = NewsCollector()
    results = collector.collect_and_save(days_back=7)
    print(f"Collection results: {results}")

