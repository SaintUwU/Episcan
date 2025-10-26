"""
News API data collector for EpiScan
Collects health-related news from various sources
"""
import requests
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from app import db
from app.models.data_models import NewsData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    """Collects health-related news from various sources"""
    
    def __init__(self):
        """Initialize News collector"""
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"
        
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set. News collection will be limited.")
        
        # Health-related keywords for Kenya
        self.health_keywords = [
            'malaria', 'tuberculosis', 'tb', 'hiv', 'aids', 'cholera', 'influenza', 'flu',
            'dengue', 'yellow fever', 'measles', 'mumps', 'rubella', 'polio', 'covid',
            'outbreak', 'epidemic', 'disease', 'health', 'medical', 'hospital', 'clinic',
            'vaccination', 'immunization', 'public health', 'ministry of health'
        ]
        
        # Kenya-specific health terms
        self.kenya_health_terms = [
            'kenya health', 'kenya ministry of health', 'kenya malaria', 'kenya tb',
            'kenya cholera', 'kenya outbreak', 'kenya epidemic', 'kenya vaccination',
            'nairobi health', 'mombasa health', 'kisumu health', 'nakuru health'
        ]
        
        # News sources to prioritize
        self.news_sources = [
            'bbc-news', 'cnn', 'reuters', 'associated-press', 'al-jazeera-english',
            'the-guardian-uk', 'independent', 'daily-mail', 'metro', 'mirror'
        ]
        
        logger.info("News Collector initialized successfully")
    
    def collect_and_save(self, days_back: int = 7) -> Dict:
        """Collect and save news data"""
        logger.info("Starting news data collection...")
        
        try:
            # Collect news data
            news_data = self._collect_news_data(days_back)
            
            if not news_data:
                logger.warning("No news data collected")
                return {'saved': 0, 'total_collected': 0, 'errors': ['No data collected']}
            
            # Save to database
            saved_count = self._save_news_data(news_data)
            
            logger.info(f"News collection completed: {saved_count} records saved")
            return {
                'saved': saved_count,
                'total_collected': len(news_data),
                'errors': []
            }
            
        except Exception as e:
            logger.error(f"Error in news collection: {str(e)}")
            return {'saved': 0, 'total_collected': 0, 'errors': [str(e)]}
    
    def _collect_news_data(self, days_back: int) -> List[Dict]:
        """Collect news data from various sources"""
        all_news = []
        
        # Collect from News API
        if self.news_api_key:
            news_api_data = self._collect_from_news_api(days_back)
            all_news.extend(news_api_data)
        
        # Collect from RSS feeds as fallback
        rss_data = self._collect_from_rss_feeds(days_back)
        all_news.extend(rss_data)
        
        # Remove duplicates based on URL
        unique_news = []
        seen_urls = set()
        
        for article in all_news:
            if article.get('url') not in seen_urls:
                unique_news.append(article)
                seen_urls.add(article.get('url'))
        
        logger.info(f"Collected {len(unique_news)} unique news articles")
        return unique_news
    
    def _collect_from_news_api(self, days_back: int) -> List[Dict]:
        """Collect data from News API"""
        news_data = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for health-related news
            for keyword in self.health_keywords[:5]:  # Limit to avoid rate limits
                try:
                    params = {
                        'q': f"{keyword} AND kenya",
                        'from': start_date.strftime('%Y-%m-%d'),
                        'to': end_date.strftime('%Y-%m-%d'),
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 20,
                        'apiKey': self.news_api_key
                    }
                    
                    response = requests.get(self.base_url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        if self._is_health_related(article):
                            processed_article = self._process_news_article(article)
                            if processed_article:
                                news_data.append(processed_article)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error collecting news for keyword '{keyword}': {str(e)}")
                    continue
            
            logger.info(f"Collected {len(news_data)} articles from News API")
            
        except Exception as e:
            logger.error(f"Error collecting from News API: {str(e)}")
        
        return news_data
    
    def _collect_from_rss_feeds(self, days_back: int) -> List[Dict]:
        """Collect data from RSS feeds as fallback"""
        news_data = []
        
        # RSS feeds for health news
        rss_feeds = [
            'https://www.who.int/rss-feeds/news-english.xml',
            'https://www.cdc.gov/rss/news.xml',
            'https://feeds.bbci.co.uk/news/health/rss.xml',
            'https://rss.cnn.com/rss/edition.rss'
        ]
        
        try:
            import feedparser
            
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Limit entries
                        if self._is_health_related_rss(entry):
                            processed_article = self._process_rss_article(entry)
                            if processed_article:
                                news_data.append(processed_article)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error parsing RSS feed {feed_url}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(news_data)} articles from RSS feeds")
            
        except ImportError:
            logger.warning("feedparser not installed. RSS collection skipped.")
        except Exception as e:
            logger.error(f"Error collecting from RSS feeds: {str(e)}")
        
        return news_data
    
    def _is_health_related(self, article: Dict) -> bool:
        """Check if article is health-related"""
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        
        text = f"{title} {description} {content}"
        
        # Check for health keywords
        for keyword in self.health_keywords:
            if keyword in text:
                return True
        
        # Check for Kenya-specific terms
        for term in self.kenya_health_terms:
            if term in text:
                return True
        
        return False
    
    def _is_health_related_rss(self, entry) -> bool:
        """Check if RSS entry is health-related"""
        title = getattr(entry, 'title', '').lower()
        summary = getattr(entry, 'summary', '').lower()
        
        text = f"{title} {summary}"
        
        for keyword in self.health_keywords:
            if keyword in text:
                return True
        
        return False
    
    def _process_news_article(self, article: Dict) -> Optional[Dict]:
        """Process news article for database storage"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')
            published_at = article.get('publishedAt', '')
            source = article.get('source', {}).get('name', 'Unknown')
            
            # Parse published date
            try:
                if published_at:
                    published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    published_date = datetime.now()
            except:
                published_date = datetime.now()
            
            # Extract disease type
            disease_type = self._extract_disease_type(title + ' ' + description)
            
            # Extract outbreak status
            outbreak_status = self._extract_outbreak_status(title + ' ' + description)
            
            # Extract affected counties
            affected_counties = self._extract_counties(title + ' ' + description)
            
            return {
                'title': title,
                'content': description,
                'url': url,
                'source': source,
                'published_at': published_date,
                'county': self._extract_primary_county(title + ' ' + description),
                'sentiment_score': 0.0,  # Default neutral sentiment
                'sentiment_label': 'neutral',
                'keywords': self._extract_keywords(title + ' ' + description),
                'health_relevance_score': self._calculate_relevance_score(title + ' ' + description)
            }
            
        except Exception as e:
            logger.warning(f"Error processing news article: {str(e)}")
            return None
    
    def _process_rss_article(self, entry) -> Optional[Dict]:
        """Process RSS article for database storage"""
        try:
            title = getattr(entry, 'title', '')
            summary = getattr(entry, 'summary', '')
            link = getattr(entry, 'link', '')
            published = getattr(entry, 'published', '')
            
            # Parse published date
            try:
                if published:
                    published_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                else:
                    published_date = datetime.now()
            except:
                published_date = datetime.now()
            
            # Extract disease type
            disease_type = self._extract_disease_type(title + ' ' + summary)
            
            # Extract outbreak status
            outbreak_status = self._extract_outbreak_status(title + ' ' + summary)
            
            # Extract affected counties
            affected_counties = self._extract_counties(title + ' ' + summary)
            
            return {
                'title': title,
                'content': summary,
                'url': link,
                'source': 'RSS Feed',
                'published_at': published_date,
                'county': self._extract_primary_county(title + ' ' + summary),
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'keywords': self._extract_keywords(title + ' ' + summary),
                'health_relevance_score': self._calculate_relevance_score(title + ' ' + summary)
            }
            
        except Exception as e:
            logger.warning(f"Error processing RSS article: {str(e)}")
            return None
    
    def _extract_disease_type(self, text: str) -> Optional[str]:
        """Extract disease type from text"""
        text_lower = text.lower()
        
        disease_mapping = {
            'malaria': ['malaria', 'mosquito', 'plasmodium'],
            'tuberculosis': ['tuberculosis', 'tb', 'tuberculous'],
            'hiv': ['hiv', 'aids', 'human immunodeficiency'],
            'cholera': ['cholera', 'vibrio'],
            'influenza': ['influenza', 'flu', 'h1n1', 'h3n2'],
            'covid': ['covid', 'coronavirus', 'sars-cov-2'],
            'dengue': ['dengue', 'dengue fever'],
            'measles': ['measles', 'rubeola'],
            'yellow fever': ['yellow fever', 'yellowfever']
        }
        
        for disease, keywords in disease_mapping.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return disease
        
        return None
    
    def _extract_outbreak_status(self, text: str) -> str:
        """Extract outbreak status from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['outbreak', 'epidemic', 'confirmed cases', 'spreading']):
            return 'confirmed'
        elif any(word in text_lower for word in ['suspected', 'investigating', 'monitoring', 'alert']):
            return 'suspected'
        else:
            return 'none'
    
    def _extract_counties(self, text: str) -> List[str]:
        """Extract affected counties from text"""
        text_lower = text.lower()
        
        kenyan_counties = [
            'nairobi', 'mombasa', 'kisumu', 'nakuru', 'eldoret', 'thika', 'malindi',
            'kitale', 'garissa', 'kakamega', 'meru', 'nyeri', 'machakos', 'kisii',
            'kericho', 'bungoma', 'busia', 'vihiga', 'siaya', 'homa bay', 'migori',
            'nyamira', 'trans nzoia', 'uasin gishu', 'elgeyo marakwet', 'nandi',
            'baringo', 'laikipia', 'narok', 'kajiado', 'bomet', 'west pokot',
            'samburu', 'turkana', 'wajir', 'mandera', 'marsabit', 'isiolo',
            'tharaka nithi', 'embu', 'kitui', 'makueni', 'taita taveta', 'kwale',
            'kilifi', 'tana river', 'lamu'
        ]
        
        affected_counties = []
        for county in kenyan_counties:
            if county in text_lower:
                affected_counties.append(county.title())
        
        return affected_counties if affected_counties else ['Kenya']
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score for the article"""
        text_lower = text.lower()
        score = 0.0
        
        # Base score for health keywords
        for keyword in self.health_keywords:
            if keyword in text_lower:
                score += 0.1
        
        # Bonus for Kenya-specific terms
        for term in self.kenya_health_terms:
            if term in text_lower:
                score += 0.2
        
        # Bonus for outbreak-related terms
        outbreak_terms = ['outbreak', 'epidemic', 'confirmed', 'cases', 'deaths']
        for term in outbreak_terms:
            if term in text_lower:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_primary_county(self, text: str) -> Optional[str]:
        """Extract the primary county mentioned in text"""
        counties = self._extract_counties(text)
        return counties[0] if counties else None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract health-related keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.health_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _save_news_data(self, news_data: List[Dict]) -> int:
        """Save news data to database"""
        saved_count = 0
        
        try:
            for article in news_data:
                try:
                    # Check if article already exists
                    existing = NewsData.query.filter_by(url=article['url']).first()
                    if existing:
                        continue
                    
                    # Create new record
                    news_record = NewsData(
                        title=article['title'],
                        content=article['content'],
                        url=article['url'],
                        source=article['source'],
                        published_at=article['published_at'],
                        county=article['county'],
                        sentiment_score=article['sentiment_score'],
                        sentiment_label=article['sentiment_label'],
                        keywords=article['keywords'],
                        health_relevance_score=article['health_relevance_score']
                    )
                    
                    db.session.add(news_record)
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error saving news article: {str(e)}")
                    continue
            
            db.session.commit()
            logger.info(f"Successfully saved {saved_count} news articles")
            
        except Exception as e:
            logger.error(f"Error saving news data to database: {str(e)}")
            db.session.rollback()
            saved_count = 0
        
        return saved_count
