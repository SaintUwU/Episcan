CSV Import Templates

These files show the expected format for importing historical data.

Required columns are marked with *, optional columns can be included for better data quality.

Twitter Data (twitter_template.csv):
* tweet_id, text, username, created_at
  Optional: location, county, sentiment_score, retweet_count, favorite_count

News Data (news_template.csv):
* title, content, source, url, published_at
  Optional: county, sentiment_score, health_relevance_score

WHO Data (who_template.csv):
* title, content, source, url, published_at
  Optional: disease_type, outbreak_status, case_count, affected_counties

Google Trends (trends_template.csv):
* keyword, date, interest_score
  Optional: geo_location (default: KE)

Usage:
  python import_data.py --twitter your_data.csv --news your_articles.csv

Date format: YYYY-MM-DD HH:MM:SS or YYYY-MM-DD



