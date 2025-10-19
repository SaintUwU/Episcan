#!/usr/bin/env python3
"""
Test script to verify API keys are working correctly
"""
import os
import sys
import requests
import tweepy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_twitter_api():
    """Test Twitter API credentials"""
    print(" Testing Twitter API...")
    
    try:
        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not all([api_key, api_secret, access_token, access_token_secret, bearer_token]):
            print(" Twitter API keys missing in environment variables")
            return False
        
        # Test API connection
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True
        )
        
        # Test with a simple search
        tweets = client.search_recent_tweets(
            query="health Kenya",
            max_results=10
        )
        
        if tweets.data:
            print(f" Twitter API working - Found {len(tweets.data)} tweets")
            return True
        else:
            print(" Twitter API working but no tweets found")
            return True
            
    except tweepy.TooManyRequests as e:
        print(f" Twitter API sleeping (rate limited): {str(e)}")
        print("  Skipping Twitter API test - API is rate limited")
        return "skipped"
    except tweepy.Unauthorized as e:
        print(f" Twitter API unauthorized: {str(e)}")
        return False
    except tweepy.Forbidden as e:
        print(f" Twitter API forbidden: {str(e)}")
        return False
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "sleeping" in error_msg or "429" in error_msg:
            print(f" Twitter API sleeping (rate limited): {str(e)}")
            print("  Skipping Twitter API test - API is rate limited")
            return "skipped"
        else:
            print(f" Twitter API error: {str(e)}")
            return False

def test_news_api():
    """Test News API credentials"""
    print("Testing News API...")
    
    try:
        news_api_key = os.getenv('NEWS_API_KEY')
        
        if not news_api_key:
            print("News API key missing in environment variables")
            return False
        
        # Test API with Kenya health news
        url = f"https://newsapi.org/v2/everything?q=health+Kenya&apiKey={news_api_key}&pageSize=5"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"News API working - Found {len(articles)} articles")
            return True
        elif response.status_code == 401:
            print(" News API key invalid")
            return False
        elif response.status_code == 429:
            print(" News API quota exceeded")
            return False
        else:
            print(f"News API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"News API error: {str(e)}")
        return False

def test_google_trends():
    """Test Google Trends (no API key required)"""
    print("Testing Google Trends...")
    
    try:
        from pytrends.request import TrendReq
        
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Test with a simple keyword
        pytrends.build_payload(['health Kenya'], timeframe='today 1-m')
        interest_over_time = pytrends.interest_over_time()
        
        if not interest_over_time.empty:
            print("Google Trends working - Data retrieved successfully")
            return True
        else:
            print(" Google Trends working but no data found")
            return True
            
    except Exception as e:
        print(f"Google Trends error: {str(e)}")
        return False

def test_database_connection():
    """Test database connection"""
    print("Testing database connection...")
    
    try:
        from app import create_app, db
        
        app = create_app()
        with app.app_context():
            # Test database connection
            result = db.session.execute("SELECT 1").scalar()
            if result == 1:
                print(" Database connection working")
                return True
            else:
                print(" Database connection failed")
                return False
                
    except Exception as e:
        print(f"Database error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("EpiScan API Keys Test")
    print("=" * 40)
    
    results = {
        'twitter': test_twitter_api(),
        'news': test_news_api(),
        'google_trends': test_google_trends(),
        'database': test_database_connection()
    }
    
    print("\n Test Results Summary:")
    print("-" * 30)
    
    skipped_services = []
    failed_services = []
    passed_services = []
    
    for service, success in results.items():
        if success == "skipped":
            status = " SKIPPED"
            skipped_services.append(service)
        elif success:
            status = " PASS"
            passed_services.append(service)
        else:
            status = " FAIL"
            failed_services.append(service)
        print(f"{service.upper()}: {status}")
    
    # Overall result
    all_passed = all(result for result in results.values() if result != "skipped")
    has_skipped = any(result == "skipped" for result in results.values())
    
    print("\n" + "=" * 40)
    if all_passed and not failed_services:
        if has_skipped:
            print("✅ Core services working! Some services were skipped due to rate limits.")
            print("You can now run: python quick_data_boost.py")
        else:
            print("🎉 All tests passed! Your API keys are working correctly.")
            print("You can now run: python quick_data_boost.py")
    elif failed_services:
        print("⚠️  Some tests failed. Please check your API key configuration.")
        print("See AUTHENTICATION_SETUP.md for detailed setup instructions.")
        if skipped_services:
            print(f"Skipped services (rate limited): {', '.join(skipped_services)}")
    else:
        print("✅ Tests completed with some services skipped due to rate limits.")
        print("You can still run data collection with available services.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


