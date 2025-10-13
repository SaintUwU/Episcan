#!/usr/bin/env python3
"""
Generate synthetic test data for development
"""
import sys
import os
import random
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models.data_models import TwitterData, NewsData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataGenerator:
    
    def __init__(self):
        self.app = create_app()
        self.diseases = ['malaria', 'cholera', 'flu', 'covid', 'dengue', 'typhoid']
        self.symptoms = ['fever', 'cough', 'headache', 'vomiting', 'fatigue', 'dizziness']
        self.counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika',
                        'Malindi', 'Kitale', 'Garissa', 'Kakamega', 'Meru', 'Nyeri']
    
    def generate_tweets(self, count=100):
        """Generate synthetic tweets"""
        logger.info(f"Generating {count} synthetic tweets")
        
        templates = [
            "Experiencing {symptom} and {symptom}. Could this be {disease}?",
            "Cases of {disease} reported in {county}. Stay safe!",
            "Health alert: {disease} cases increasing in {county} area",
            "{county} residents reporting {symptom}. What's happening?",
            "Feeling terrible - {symptom}, {symptom}. Hope it's not {disease}"
        ]
        
        with self.app.app_context():
            generated = 0
            
            for i in range(count):
                template = random.choice(templates)
                text = template.format(
                    disease=random.choice(self.diseases),
                    symptom=random.choice(self.symptoms),
                    county=random.choice(self.counties)
                )
                
                created_at = datetime.now() - timedelta(
                    days=random.randint(1, 90),
                    hours=random.randint(0, 23)
                )
                
                tweet_id = f"test_{int(datetime.now().timestamp())}_{i}"
                
                if TwitterData.query.filter_by(tweet_id=tweet_id).first():
                    continue
                
                county = random.choice(self.counties)
                
                twitter_data = TwitterData(
                    tweet_id=tweet_id,
                    text=text,
                    user_id=f"user_{random.randint(1000, 9999)}",
                    username=f"user{random.randint(1000, 9999)}",
                    created_at=created_at,
                    location=f"{county}, Kenya",
                    county=county.lower(),
                    sentiment_score=random.uniform(-0.3, -0.1),
                    sentiment_label='negative',
                    keywords=[],
                    retweet_count=random.randint(0, 50),
                    favorite_count=random.randint(0, 100)
                )
                
                db.session.add(twitter_data)
                generated += 1
                
                if generated % 50 == 0:
                    db.session.commit()
            
            db.session.commit()
            return generated
    
    def generate_news(self, count=50):
        """Generate synthetic news articles"""
        logger.info(f"Generating {count} synthetic news articles")
        
        headlines = [
            "{disease} cases surge in {county} County",
            "Health officials investigate {disease} outbreak in {county}",
            "{county} County reports increase in {disease} infections",
            "Ministry issues {disease} alert for {county}"
        ]
        
        sources = ['Daily Nation', 'The Standard', 'Capital FM', 'KBC News']
        
        with self.app.app_context():
            generated = 0
            
            for i in range(count):
                disease = random.choice(self.diseases)
                county = random.choice(self.counties)
                
                title = random.choice(headlines).format(disease=disease.title(), county=county)
                content = f"Health officials in {county} County have confirmed cases of {disease}. " \
                         f"Residents experiencing {random.choice(self.symptoms)} should seek medical care."
                
                published_at = datetime.now() - timedelta(days=random.randint(1, 90))
                url = f"https://test-news.ke/article/{int(published_at.timestamp())}_{i}"
                
                if NewsData.query.filter_by(url=url).first():
                    continue
                
                news_data = NewsData(
                    title=title,
                    content=content,
                    source=random.choice(sources),
                    url=url,
                    published_at=published_at,
                    county=county.lower(),
                    sentiment_score=random.uniform(-0.4, -0.2),
                    sentiment_label='negative',
                    keywords=[disease, county.lower()],
                    health_relevance_score=random.uniform(0.7, 0.95)
                )
                
                db.session.add(news_data)
                generated += 1
                
                if generated % 25 == 0:
                    db.session.commit()
            
            db.session.commit()
            return generated

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic test data')
    parser.add_argument('--tweets', type=int, default=0, help='Number of tweets to generate')
    parser.add_argument('--news', type=int, default=0, help='Number of articles to generate')
    
    args = parser.parse_args()
    
    if args.tweets == 0 and args.news == 0:
        parser.print_help()
        sys.exit(1)
    
    generator = TestDataGenerator()
    
    if args.tweets > 0:
        count = generator.generate_tweets(args.tweets)
        print(f"Generated {count} tweets")
    
    if args.news > 0:
        count = generator.generate_news(args.news)
        print(f"Generated {count} news articles")


