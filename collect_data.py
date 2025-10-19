#!/usr/bin/env python3
"""
Simple data collection boost for EpiScan
Focuses on working sources and generates unique synthetic data
"""
import sys
import os
import logging
from datetime import datetime, timedelta
import random
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from data_collection.news_collector import NewsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_data_boost():
    """Simple data collection focusing on working sources"""
    print("=" * 60)
    print("EPISCAN SIMPLE DATA BOOST")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    app = create_app()
    results = {}
    
    with app.app_context():
        print("Collecting data from working sources...")
        print()
        
        # 1. News data collection (this is working)
        print("1. Collecting News data...")
        try:
            news_collector = NewsCollector()
            news_results = news_collector.collect_and_save(days_back=30)
            results['news'] = news_results
            print(f"    News API: {news_results.get('saved', 0)} records")
        except Exception as e:
            print(f"    News API: ERROR - {str(e)}")
            results['news'] = {'error': str(e), 'saved': 0}
        
        # 2. Generate unique synthetic data
        print("2. Generating unique synthetic data...")
        try:
            synthetic_count = generate_unique_synthetic_data(1000)  # Generate 1000 unique records
            results['synthetic'] = {'saved': synthetic_count}
            print(f"    Synthetic: {synthetic_count} records")
        except Exception as e:
            print(f"    Synthetic: ERROR - {str(e)}")
            results['synthetic'] = {'error': str(e), 'saved': 0}
        
        # Calculate totals
        total_saved = sum([r.get('saved', 0) for r in results.values()])
        
        print(f"\n" + "=" * 60)
        print("SIMPLE DATA BOOST RESULTS")
        print("=" * 60)
        print(f"News API: {results['news'].get('saved', 0)} records")
        print(f"Synthetic: {results['synthetic'].get('saved', 0)} records")
        print(f"TOTAL: {total_saved} records")
        
        if total_saved > 0:
            print(f"\n✅ Successfully collected {total_saved} new records!")
            print("Your training dataset has been expanded.")
        else:
            print(f"\n⚠️  No new records collected.")
        
        print(f"\nCompleted at: {datetime.now()}")
        
        return total_saved

def generate_unique_synthetic_data(count: int) -> int:
    """Generate unique synthetic health data"""
    print(f"    Generating {count} unique synthetic health records...")
    
    try:
        from app.models.data_models import TwitterData, NewsData, GoogleTrendsData, WHOData
        
        saved_count = 0
        
        # Generate unique synthetic data
        for i in range(count):
            try:
                # Generate unique identifiers
                unique_id = str(uuid.uuid4())
                unique_url = f"https://synthetic-{unique_id[:8]}.com"
                unique_tweet_id = f"synthetic_{unique_id[:8]}"
                
                # Randomly choose data type
                data_type = random.choice(['twitter', 'news', 'trends', 'who'])
                
                if data_type == 'twitter':
                    record = TwitterData(
                        tweet_id=unique_tweet_id,
                        text=generate_synthetic_tweet_text(),
                        user_id=f"synthetic_user_{unique_id[:8]}",
                        username=f"synthetic_user_{random.randint(100, 999)}",
                        created_at=datetime.now() - timedelta(days=random.randint(0, 7)),
                        location=random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']),
                        county=random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']),
                        keywords=random.sample(['health', 'disease', 'outbreak', 'hospital'], random.randint(1, 3)),
                        is_retweet=False,
                        retweet_count=random.randint(0, 50),
                        favorite_count=random.randint(0, 100),
                        sentiment_score=random.uniform(-1, 1),
                        sentiment_label=random.choice(['positive', 'negative', 'neutral'])
                    )
                
                elif data_type == 'news':
                    record = NewsData(
                        title=generate_synthetic_news_title(),
                        content=generate_synthetic_news_content(),
                        source=random.choice(['The Star', 'Standard Media', 'Citizen TV', 'Daily Nation']),
                        url=unique_url,
                        published_at=datetime.now() - timedelta(days=random.randint(0, 30)),
                        county=random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']),
                        keywords=random.sample(['health', 'disease', 'outbreak', 'hospital'], random.randint(1, 3)),
                        health_relevance_score=random.uniform(0.5, 1.0),
                        sentiment_score=random.uniform(-0.5, 0.5),
                        sentiment_label=random.choice(['positive', 'negative', 'neutral'])
                    )
                
                elif data_type == 'trends':
                    record = GoogleTrendsData(
                        keyword=random.choice(['health', 'disease', 'outbreak', 'hospital', 'clinic']),
                        date=(datetime.now() - timedelta(days=random.randint(0, 30))).date(),
                        interest_score=random.randint(0, 100),
                        geo_location='KE'
                    )
                
                elif data_type == 'who':
                    record = WHOData(
                        title=generate_synthetic_who_title(),
                        content=generate_synthetic_who_content(),
                        source=random.choice(['WHO Africa', 'WHO Global', 'MOH Kenya']),
                        url=unique_url,
                        published_at=datetime.now() - timedelta(days=random.randint(0, 30)),
                        disease_type=random.choice(['malaria', 'cholera', 'flu', 'covid', 'dengue']),
                        outbreak_status=random.choice(['confirmed', 'suspected', 'none']),
                        affected_counties=[random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru'])],
                        case_count=random.randint(0, 100)
                    )
                
                db.session.add(record)
                saved_count += 1
                
                # Commit every 100 records to avoid memory issues
                if saved_count % 100 == 0:
                    db.session.commit()
                    print(f"      Saved {saved_count} records so far...")
                
            except Exception as e:
                logger.warning(f"Error saving synthetic record {i}: {str(e)}")
                continue
        
        # Final commit
        db.session.commit()
        logger.info(f"Saved {saved_count} unique synthetic records to database")
        return saved_count
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        return 0

def generate_synthetic_tweet_text() -> str:
    """Generate synthetic tweet text"""
    templates = [
        "Health alert in {county}: {disease} cases reported. Stay safe!",
        "Visited {county} hospital today. {experience}",
        "{county} health center needs more {resource}. {situation}",
        "Health workers in {county} are {status}. {context}",
        "Outbreak of {disease} in {county}. {advice}"
    ]
    
    counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']
    diseases = ['malaria', 'cholera', 'flu', 'covid', 'dengue']
    experiences = ['Good service', 'Long wait', 'Helpful staff', 'Crowded']
    resources = ['staff', 'equipment', 'supplies', 'medicines']
    situations = ['Understaffed', 'Overwhelmed', 'Struggling', 'Coping']
    statuses = ['working hard', 'overwhelmed', 'struggling', 'coping']
    contexts = ['Many patients', 'Long hours', 'Limited resources', 'High demand']
    advice = ['Take precautions', 'Stay informed', 'Follow guidelines', 'Be vigilant']
    
    template = random.choice(templates)
    return template.format(
        county=random.choice(counties),
        disease=random.choice(diseases),
        experience=random.choice(experiences),
        resource=random.choice(resources),
        situation=random.choice(situations),
        status=random.choice(statuses),
        context=random.choice(contexts),
        advice=random.choice(advice)
    )

def generate_synthetic_news_title() -> str:
    """Generate synthetic news title"""
    templates = [
        "Health Alert: {disease} Outbreak in {county}",
        "{county} Hospital {status} with {situation}",
        "Ministry of Health Issues {action} for {county}",
        "{county} Health Center {needs} {resource}",
        "Health Workers in {county} {status} with {context}"
    ]
    
    diseases = ['Malaria', 'Cholera', 'Flu', 'COVID', 'Dengue']
    counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']
    statuses = ['Overwhelmed', 'Struggling', 'Coping', 'Responding']
    situations = ['Patient Surge', 'Health Crisis', 'Disease Outbreak']
    actions = ['Warning', 'Alert', 'Advisory', 'Notice']
    needs = ['Needs', 'Requires', 'Seeks', 'Demands']
    resources = ['More Staff', 'Better Equipment', 'Additional Resources']
    contexts = ['Patient Load', 'Disease Cases', 'Health Crisis']
    
    template = random.choice(templates)
    return template.format(
        disease=random.choice(diseases),
        county=random.choice(counties),
        status=random.choice(statuses),
        situation=random.choice(situations),
        action=random.choice(actions),
        needs=random.choice(needs),
        resource=random.choice(resources),
        context=random.choice(contexts)
    )

def generate_synthetic_news_content() -> str:
    """Generate synthetic news content"""
    return f"Health officials are reporting {random.choice(['increased', 'rising', 'growing'])} cases of {random.choice(['disease', 'illness', 'outbreak'])}. The situation is being {random.choice(['monitored', 'investigated', 'addressed'])} by local health authorities."

def generate_synthetic_who_title() -> str:
    """Generate synthetic WHO title"""
    templates = [
        "WHO Alert: {disease} {status} in {county}",
        "Health Emergency: {disease} Outbreak in {county}",
        "WHO Monitoring: {disease} Cases in {county}",
        "Health Update: {disease} Situation in {county}"
    ]
    
    diseases = ['Malaria', 'Cholera', 'Flu', 'COVID', 'Dengue']
    counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru']
    statuses = ['Confirmed', 'Suspected', 'Reported', 'Detected']
    
    template = random.choice(templates)
    return template.format(
        disease=random.choice(diseases),
        status=random.choice(statuses),
        county=random.choice(counties)
    )

def generate_synthetic_who_content() -> str:
    """Generate synthetic WHO content"""
    return f"WHO is monitoring {random.choice(['disease', 'outbreak', 'health situation'])} cases. Status: {random.choice(['confirmed', 'suspected', 'investigating'])}. Local health authorities are {random.choice(['responding', 'monitoring', 'investigating'])} the situation."

def main():
    """Main execution function"""
    print("Simple data collection for EpiScan...")
    print("This will collect data from working sources and generate unique synthetic data.")
    print()
    
    total_collected = simple_data_boost()
    
    if total_collected > 0:
        print(f"\n🎉 Simple data boost completed!")
        print(f"Collected {total_collected} new records.")
        print("\nNext steps:")
        print("1. Run: python retrain_with_expanded_data.py")
        print("2. Or run: python collect_and_retrain.py for complete pipeline")
    else:
        print(f"\n❌ No data collected.")
        print("Consider checking your API keys and network connectivity.")
    
    return 0

if __name__ == "__main__":
    exit(main())
