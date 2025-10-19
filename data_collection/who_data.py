"""
WHO and Ministry of Health data collector
"""
import requests
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import feedparser
from bs4 import BeautifulSoup
from app import db
from app.models.data_models import WHOData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WHODataCollector:
    """Collects data from WHO and Ministry of Health sources"""
    
    def __init__(self):
        """Initialize WHO data collector"""
        # WHO and MOH data sources
        self.data_sources = {
            'who_africa': {
                'name': 'WHO Africa',
                'base_url': 'https://www.afro.who.int',
                'rss_url': 'https://www.afro.who.int/rss',
                'api_url': 'https://www.afro.who.int/api/news'
            },
            'who_global': {
                'name': 'WHO Global',
                'base_url': 'https://www.who.int',
                'rss_url': 'https://www.who.int/rss-feeds/news-english.xml',
                'api_url': 'https://www.who.int/api/news'
            },
            'moh_kenya': {
                'name': 'Ministry of Health Kenya',
                'base_url': 'https://www.health.go.ke',
                'rss_url': 'https://www.health.go.ke/rss',
                'api_url': None
            }
        }
        
        # Disease types to monitor
        self.disease_types = [
            'malaria', 'cholera', 'flu', 'influenza', 'covid', 'covid-19',
            'dengue', 'typhoid', 'diarrhea', 'pneumonia', 'tuberculosis',
            'measles', 'meningitis', 'yellow fever', 'ebola', 'lassa fever'
        ]
        
        # Outbreak status keywords
        self.outbreak_keywords = {
            'confirmed': ['confirmed', 'outbreak', 'epidemic', 'pandemic', 'alert'],
            'suspected': ['suspected', 'investigation', 'monitoring', 'watch'],
            'none': ['normal', 'stable', 'controlled', 'resolved']
        }
        
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
    
    def collect_from_who_africa(self, days_back: int = 30) -> List[Dict]:
        """
        Collect data from WHO Africa
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            List of WHO Africa reports
        """
        try:
            # Try RSS feed first
            rss_data = self._fetch_from_rss(
                self.data_sources['who_africa']['rss_url'],
                'WHO Africa'
            )
            
            # Filter by date and health relevance
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_data = [
                item for item in rss_data
                if item.get('published_at') and item['published_at'] >= cutoff_date
                and self._is_health_relevant(item)
            ]
            
            logger.info(f"Collected {len(recent_data)} items from WHO Africa")
            return recent_data
            
        except Exception as e:
            logger.error(f"Error collecting from WHO Africa: {str(e)}")
            return []
    
    def collect_from_who_global(self, days_back: int = 30) -> List[Dict]:
        """
        Collect data from WHO Global
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            List of WHO Global reports
        """
        try:
            # Try RSS feed first
            rss_data = self._fetch_from_rss(
                self.data_sources['who_global']['rss_url'],
                'WHO Global'
            )
            
            # Filter by date and Kenya relevance
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_data = [
                item for item in rss_data
                if item.get('published_at') and item['published_at'] >= cutoff_date
                and (self._is_health_relevant(item) or self._is_kenya_relevant(item))
            ]
            
            logger.info(f"Collected {len(recent_data)} items from WHO Global")
            return recent_data
            
        except Exception as e:
            logger.error(f"Error collecting from WHO Global: {str(e)}")
            return []
    
    def collect_from_moh_kenya(self, days_back: int = 30) -> List[Dict]:
        """
        Collect data from Ministry of Health Kenya
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            List of MOH Kenya reports
        """
        try:
            # Try RSS feed first
            rss_data = self._fetch_from_rss(
                self.data_sources['moh_kenya']['rss_url'],
                'MOH Kenya'
            )
            
            # Filter by date and health relevance
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_data = [
                item for item in rss_data
                if item.get('published_at') and item['published_at'] >= cutoff_date
                and self._is_health_relevant(item)
            ]
            
            logger.info(f"Collected {len(recent_data)} items from MOH Kenya")
            return recent_data
            
        except Exception as e:
            logger.error(f"Error collecting from MOH Kenya: {str(e)}")
            return []
    
    import feedparser

def _fetch_from_rss(self, rss_url: str, source_name: str) -> List[Dict]:
    """Fetch data from RSS feed using feedparser"""
    try:
        feed = feedparser.parse(rss_url)
        data_items = []
        for entry in feed.entries:
            published_at = None
            if hasattr(entry, 'published_parsed'):
                from datetime import datetime
                published_at = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed'):
                published_at = datetime(*entry.updated_parsed[:6])

            data_items.append({
                'title': getattr(entry, 'title', ''),
                'content': getattr(entry, 'summary', ''),
                'url': getattr(entry, 'link', ''),
                'source': source_name,
                'published_at': published_at or datetime.now()
            })
        return data_items
    except Exception as e:
        logger.error(f"Feedparser error for {rss_url}: {str(e)}")
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
            
            # Extract content from description or fetch full content
            content = description.text.strip() if description else ''
            
            return {
                'title': title.text.strip() if title else '',
                'content': content,
                'url': link.text.strip() if link else '',
                'source': source_name,
                'published_at': published_at or datetime.now()
            }
            
        except Exception as e:
            logger.warning(f"Error parsing RSS item: {str(e)}")
            return None
    
    def _is_health_relevant(self, item: Dict) -> bool:
        """Check if item is health-relevant"""
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        # Check for disease types
        for disease in self.disease_types:
            if disease.lower() in text:
                return True
        
        # Check for health keywords
        health_keywords = ['health', 'disease', 'outbreak', 'epidemic', 'pandemic', 'alert', 'warning']
        for keyword in health_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _is_kenya_relevant(self, item: Dict) -> bool:
        """Check if item is Kenya-relevant"""
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        
        kenya_keywords = ['kenya', 'kenyan', 'nairobi', 'mombasa', 'kisumu']
        for keyword in kenya_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _extract_disease_type(self, text: str) -> Optional[str]:
        """Extract disease type from text"""
        text_lower = text.lower()
        
        for disease in self.disease_types:
            if disease.lower() in text_lower:
                return disease
        
        return None
    
    def _extract_outbreak_status(self, text: str) -> str:
        """Extract outbreak status from text"""
        text_lower = text.lower()
        
        for status, keywords in self.outbreak_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return status
        
        return 'none'
    
    def _extract_affected_counties(self, text: str) -> List[str]:
        """Extract affected Kenyan counties from text"""
        text_lower = text.lower()
        affected_counties = []
        
        for county in self.kenyan_counties:
            if county.lower() in text_lower:
                affected_counties.append(county)
        
        return affected_counties
    
    def _extract_case_count(self, text: str) -> Optional[int]:
        """Extract case count from text"""
        import re
        
        # Look for patterns like "X cases", "X confirmed", "X infected"
        patterns = [
            r'(\d+)\s+cases?',
            r'(\d+)\s+confirmed',
            r'(\d+)\s+infected',
            r'(\d+)\s+patients?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def process_who_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw WHO data and extract structured information"""
        processed_data = []
        
        for item in raw_data:
            try:
                text = f"{item.get('title', '')} {item.get('content', '')}"
                
                processed_item = {
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'published_at': item.get('published_at'),
                    'disease_type': self._extract_disease_type(text),
                    'outbreak_status': self._extract_outbreak_status(text),
                    'affected_counties': self._extract_affected_counties(text),
                    'case_count': self._extract_case_count(text)
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing WHO data item: {str(e)}")
                continue
        
        return processed_data
    
    def save_who_data_to_db(self, who_data: List[Dict]) -> int:
        """
        Save WHO data to database
        
        Args:
            who_data: List of processed WHO data
            
        Returns:
            Number of records saved
        """
        saved_count = 0
        
        try:
            for data_item in who_data:
                # Check if data already exists
                existing = WHOData.query.filter_by(url=data_item['url']).first()
                
                if not existing:
                    # Create new WHOData record
                    who_record = WHOData(
                        title=data_item['title'],
                        content=data_item['content'],
                        source=data_item['source'],
                        url=data_item['url'],
                        published_at=data_item['published_at'],
                        disease_type=data_item.get('disease_type'),
                        outbreak_status=data_item.get('outbreak_status'),
                        affected_counties=data_item.get('affected_counties', []),
                        case_count=data_item.get('case_count')
                    )
                    
                    db.session.add(who_record)
                    saved_count += 1
            
            db.session.commit()
            logger.info(f"Saved {saved_count} new WHO data records to database")
            
        except Exception as e:
            logger.error(f"Error saving WHO data to database: {str(e)}")
            db.session.rollback()
        
        return saved_count
    
    def collect_and_save(self, days_back: int = 30) -> Dict:
        """
        Main method to collect and save WHO data
        
        Args:
            days_back: Number of days to search back
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info("Starting WHO data collection...")
        
        # Collect from all sources
        who_africa_data = self.collect_from_who_africa(days_back)
        who_global_data = self.collect_from_who_global(days_back)
        moh_kenya_data = self.collect_from_moh_kenya(days_back)
        
        # Combine all data
        all_raw_data = who_africa_data + who_global_data + moh_kenya_data
        
        # Process the data
        processed_data = self.process_who_data(all_raw_data)
        
        # Save to database
        saved_count = self.save_who_data_to_db(processed_data)
        
        return {
            'collected_who_africa': len(who_africa_data),
            'collected_who_global': len(who_global_data),
            'collected_moh_kenya': len(moh_kenya_data),
            'total_collected': len(all_raw_data),
            'processed': len(processed_data),
            'saved': saved_count,
            'duplicates': len(processed_data) - saved_count,
            'collection_date': datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Test the collector
    collector = WHODataCollector()
    results = collector.collect_and_save(days_back=30)
    print(f"Collection results: {results}")

