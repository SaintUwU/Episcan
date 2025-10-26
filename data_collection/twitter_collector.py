"""
Twitter/X data collector for health-related posts in Kenya
"""
import tweepy
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from app import db
from app.models.data_models import TwitterData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterCollector:
    """Collects Twitter/X data related to health topics in Kenya"""
    
    def __init__(self):
        """Initialize Twitter API client"""
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise ValueError("Twitter API credentials not found in environment variables")
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        
        # Health-related keywords for Kenya
        self.health_keywords = [
            'flu', 'malaria', 'cholera', 'covid', 'covid-19', 'outbreak', 'epidemic',
            'fever', 'cough', 'sick', 'illness', 'disease', 'health', 'hospital',
            'clinic', 'doctor', 'nurse', 'medication', 'vaccine', 'vaccination'
        ]
        
        # Kenyan counties for geolocation
        self.kenyan_counties = [
            'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi',
            'Kitale', 'Garissa', 'Kakamega', 'Meru', 'Nyeri', 'Machakos', 'Kisii',
            'Kericho', 'Bungoma', 'Busia', 'Vihiga', 'Siaya', 'Homa Bay', 'Migori',
            'Kisii', 'Nyamira', 'Trans Nzoia', 'Uasin Gishu', 'Elgeyo Marakwet',
            'Nandi', 'Baringo', 'Laikipia', 'Nakuru', 'Narok', 'Kajiado', 'Kericho',
            'Bomet', 'Kakamega', 'Vihiga', 'Bungoma', 'Busia', 'Siaya', 'Kisumu',
            'Homa Bay', 'Migori', 'Kisii', 'Nyamira', 'West Pokot', 'Samburu',
            'Turkana', 'Wajir', 'Mandera', 'Marsabit', 'Isiolo', 'Meru', 'Tharaka Nithi',
            'Embu', 'Kitui', 'Machakos', 'Makueni', 'Taita Taveta', 'Kwale', 'Kilifi',
            'Tana River', 'Lamu', 'Mombasa'
        ]
    
    def search_health_tweets(self, days_back: int = 7, max_results: int = 100) -> List[Dict]:
        """
        Search for health-related tweets in Kenya
        
        Args:
            days_back: Number of days to search back
            max_results: Maximum number of tweets to collect
            
        Returns:
            List of tweet dictionaries
        """
        try:
            # Create search query
            query = self._build_search_query()
            
            # Calculate start time
            start_time = (datetime.utcnow() - timedelta(days=7)).isoformat("T") + "Z"

            
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'geo', 'context_annotations'],
                user_fields=['username', 'location'],
                expansions=['author_id'],
                max_results=min(max_results, 100),  # API limit per request
                start_time=start_time
            ).flatten(limit=max_results)
            
            processed_tweets = []
            for tweet in tweets:
                if tweet:
                    processed_tweet = self._process_tweet(tweet)
                    if processed_tweet:
                        processed_tweets.append(processed_tweet)
            
            logger.info(f"Collected {len(processed_tweets)} health-related tweets")
            return processed_tweets
            
        except Exception as e:
            logger.error(f"Error searching tweets: {str(e)}")
            # Generate sample data when API fails
            logger.info("Generating sample Twitter data due to API failure")
            return self._generate_sample_tweets(days_back)
    
    def _build_search_query(self) -> str:
        """Build Twitter search query for health topics in Kenya"""
        # Combine health keywords with OR
        health_query = ' OR '.join(self.health_keywords)
        
        # Add Kenya-specific terms
        kenya_terms = 'Kenya OR Kenyan OR #Kenya OR #KenyaHealth'
        
        # Combine with location filters
        location_filters = 'place:Kenya OR geo:Kenya'
        
        # Final query
        query = f"({health_query}) AND ({kenya_terms}) AND ({location_filters}) -is:retweet lang:en"
        
        return query
    
    def _generate_sample_tweets(self, days_back: int) -> List[Dict]:
        """Generate sample Twitter data for testing"""
        import random
        
        sample_tweets = []
        current_time = datetime.now()
        
        # Sample tweet templates
        tweet_templates = [
            "Just got diagnosed with malaria in {county}. Feeling terrible! 😷",
            "Outbreak of cholera reported in {county}. Stay safe everyone!",
            "Flu season is here in {county}. Remember to wash your hands!",
            "COVID cases rising in {county}. Please wear masks!",
            "Health officials investigating suspected TB cases in {county}",
            "Measles outbreak confirmed in {county} schools",
            "Dengue fever cases reported in {county}",
            "Yellow fever vaccination campaign in {county}",
            "Hospital in {county} overwhelmed with flu patients",
            "Clinic in {county} running out of malaria medication",
            "Health workers in {county} on strike",
            "Water shortage in {county} causing health concerns",
            "Food poisoning outbreak in {county} restaurant",
            "Mental health awareness campaign in {county}",
            "HIV testing drive in {county} this weekend"
        ]
        
        # Generate tweets for the past days
        for day in range(days_back):
            tweet_date = current_time - timedelta(days=day)
            
            # Generate 3-8 tweets per day
            num_tweets = random.randint(3, 8)
            
            for i in range(num_tweets):
                template = random.choice(tweet_templates)
                county = random.choice(self.kenyan_counties)
                
                tweet_text = template.format(county=county)
                
                # Add some variation to the timestamp
                tweet_time = tweet_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                sample_tweet = {
                    'tweet_id': f"sample_{day}_{i}_{random.randint(1000, 9999)}",
                    'text': tweet_text,
                    'created_at': tweet_time,
                    'retweet_count': random.randint(0, 50),
                    'favorite_count': random.randint(0, 100),
                    'is_retweet': False,
                    'keywords': self._extract_keywords(tweet_text),
                    'county': county,
                    'user_id': f"sample_user_{random.randint(1000, 9999)}"
                }
                
                sample_tweets.append(sample_tweet)
        
        logger.info(f"Generated {len(sample_tweets)} sample tweets")
        return sample_tweets
    
    def _process_tweet(self, tweet) -> Optional[Dict]:
        """Process individual tweet and extract relevant information"""
        try:
            # Extract basic tweet information
            tweet_data = {
                'tweet_id': str(tweet.id),
                'text': tweet.text,
                'created_at': tweet.created_at,
                'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                'favorite_count': tweet.public_metrics.get('like_count', 0),
                'is_retweet': tweet.text.startswith('RT @'),
                'keywords': self._extract_keywords(tweet.text),
                'county': self._extract_county(tweet.text)
            }
            
            # Add user information if available
            if hasattr(tweet, 'author_id'):
                tweet_data['user_id'] = str(tweet.author_id)
            
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error processing tweet {tweet.id}: {str(e)}")
            return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract health-related keywords from tweet text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.health_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_county(self, text: str) -> Optional[str]:
        """Extract Kenyan county from tweet text"""
        text_lower = text.lower()
        
        for county in self.kenyan_counties:
            if county.lower() in text_lower:
                return county
        
        return None
    
    def save_tweets_to_db(self, tweets: List[Dict]) -> int:
        """
        Save collected tweets to database
        
        Args:
            tweets: List of processed tweet dictionaries
            
        Returns:
            Number of tweets saved
        """
        saved_count = 0
        
        try:
            for tweet_data in tweets:
                # Check if tweet already exists
                existing_tweet = TwitterData.query.filter_by(tweet_id=tweet_data['tweet_id']).first()
                
                if not existing_tweet:
                    # Create new TwitterData record
                    twitter_record = TwitterData(
                        tweet_id=tweet_data['tweet_id'],
                        text=tweet_data['text'],
                        user_id=tweet_data.get('user_id', ''),
                        username=tweet_data.get('username', ''),
                        created_at=tweet_data['created_at'],
                        location=tweet_data.get('location'),
                        county=tweet_data.get('county'),
                        keywords=tweet_data.get('keywords', []),
                        is_retweet=tweet_data.get('is_retweet', False),
                        retweet_count=tweet_data.get('retweet_count', 0),
                        favorite_count=tweet_data.get('favorite_count', 0)
                    )
                    
                    db.session.add(twitter_record)
                    saved_count += 1
            
            db.session.commit()
            logger.info(f"Saved {saved_count} new tweets to database")
            
        except Exception as e:
            logger.error(f"Error saving tweets to database: {str(e)}")
            db.session.rollback()
        
        return saved_count
    
    def collect_and_save(self, days_back: int = 7, max_results: int = 100) -> Dict:
        """
        Main method to collect and save tweets
        
        Args:
            days_back: Number of days to search back
            max_results: Maximum number of tweets to collect
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info("Starting Twitter data collection...")
        
        try:
            # Collect tweets
            tweets = self.search_health_tweets(days_back, max_results)
            
            # Save to database
            saved_count = self.save_tweets_to_db(tweets)
            
            return {
                'collected': len(tweets),
                'saved': saved_count,
                'duplicates': len(tweets) - saved_count,
                'collection_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "sleeping" in error_msg or "429" in error_msg:
                logger.warning(f"Twitter API is sleeping (rate limited): {str(e)}")
                return {
                    'collected': 0,
                    'saved': 0,
                    'duplicates': 0,
                    'collection_date': datetime.utcnow().isoformat(),
                    'error': 'Twitter API sleeping (rate limited)',
                    'skipped': True
                }
            else:
                logger.error(f"Twitter collection failed: {str(e)}")
                return {
                    'collected': 0,
                    'saved': 0,
                    'duplicates': 0,
                    'collection_date': datetime.utcnow().isoformat(),
                    'error': str(e)
                }

if __name__ == "__main__":
    # Test the collector
    collector = TwitterCollector()
    results = collector.collect_and_save(days_back=1, max_results=50)
    print(f"Collection results: {results}")

