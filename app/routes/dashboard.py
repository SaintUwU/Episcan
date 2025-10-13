"""
Dashboard routes for EpiScan frontend
"""
from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from app import db
from app.models.data_models import TwitterData, GoogleTrendsData, NewsData, WHOData
from app.models.prediction_models import OutbreakPrediction, Alert, TrendAnalysis
from datetime import datetime, timedelta
from sqlalchemy import func, desc

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@dashboard_bp.route('/api/dashboard-data')
@login_required
def dashboard_data():
    """Get dashboard data for visualization"""
    try:
        # Get recent data counts
        recent_tweets = TwitterData.query.filter(
            TwitterData.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        recent_news = NewsData.query.filter(
            NewsData.published_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        recent_trends = GoogleTrendsData.query.filter(
            GoogleTrendsData.date >= datetime.utcnow().date() - timedelta(days=7)
        ).count()
        
        # Get active alerts
        active_alerts = Alert.query.filter(Alert.is_active == True).all()
        
        # Get recent predictions
        recent_predictions = OutbreakPrediction.query.filter(
            OutbreakPrediction.prediction_date >= datetime.utcnow().date() - timedelta(days=7)
        ).order_by(desc(OutbreakPrediction.outbreak_probability)).limit(10).all()
        
        # Get county-wise data
        county_data = db.session.query(
            TwitterData.county,
            func.count(TwitterData.id).label('tweet_count')
        ).filter(
            TwitterData.county.isnot(None),
            TwitterData.created_at >= datetime.utcnow() - timedelta(days=7)
        ).group_by(TwitterData.county).all()
        
        # Get disease trends
        disease_trends = db.session.query(
            GoogleTrendsData.keyword,
            func.avg(GoogleTrendsData.interest_score).label('avg_interest')
        ).filter(
            GoogleTrendsData.date >= datetime.utcnow().date() - timedelta(days=7)
        ).group_by(GoogleTrendsData.keyword).order_by(desc('avg_interest')).limit(10).all()
        
        # Prepare dashboard data
        dashboard_data = {
            'summary': {
                'recent_tweets': recent_tweets,
                'recent_news': recent_news,
                'recent_trends': recent_trends,
                'active_alerts': len(active_alerts)
            },
            'alerts': [
                {
                    'id': alert.id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'title': alert.title,
                    'county': alert.county,
                    'created_at': alert.created_at.isoformat()
                }
                for alert in active_alerts
            ],
            'predictions': [
                {
                    'county': pred.county,
                    'disease_type': pred.disease_type,
                    'probability': pred.outbreak_probability,
                    'risk_level': pred.risk_level,
                    'date': pred.prediction_date.isoformat()
                }
                for pred in recent_predictions
            ],
            'county_data': [
                {
                    'county': county,
                    'tweet_count': count
                }
                for county, count in county_data
            ],
            'disease_trends': [
                {
                    'keyword': keyword,
                    'avg_interest': float(avg_interest)
                }
                for keyword, avg_interest in disease_trends
            ]
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/trends/<keyword>')
@login_required
def get_trend_data(keyword):
    """Get trend data for specific keyword"""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        trends = GoogleTrendsData.query.filter(
            GoogleTrendsData.keyword == keyword,
            GoogleTrendsData.date >= datetime.utcnow().date() - timedelta(days=days_back)
        ).order_by(GoogleTrendsData.date).all()
        
        data = [
            {
                'date': trend.date.isoformat(),
                'interest_score': trend.interest_score
            }
            for trend in trends
        ]
        
        return jsonify({
            'keyword': keyword,
            'data': data,
            'days_back': days_back
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard_bp.route('/api/county-data/<county>')
@login_required
def get_county_data(county):
    """Get data for specific county"""
    try:
        days_back = request.args.get('days', 7, type=int)
        
        # Get tweets for county
        tweets = TwitterData.query.filter(
            TwitterData.county == county,
            TwitterData.created_at >= datetime.utcnow() - timedelta(days=days_back)
        ).order_by(desc(TwitterData.created_at)).limit(50).all()
        
        # Get news for county
        news = NewsData.query.filter(
            NewsData.county == county,
            NewsData.published_at >= datetime.utcnow() - timedelta(days=days_back)
        ).order_by(desc(NewsData.published_at)).limit(20).all()
        
        # Get predictions for county
        predictions = OutbreakPrediction.query.filter(
            OutbreakPrediction.county == county,
            OutbreakPrediction.prediction_date >= datetime.utcnow().date() - timedelta(days=days_back)
        ).order_by(desc(OutbreakPrediction.prediction_date)).all()
        
        county_data = {
            'county': county,
            'tweets': [
                {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'sentiment_score': tweet.sentiment_score,
                    'keywords': tweet.keywords
                }
                for tweet in tweets
            ],
            'news': [
                {
                    'id': article.id,
                    'title': article.title,
                    'content': article.content[:200] + '...' if len(article.content) > 200 else article.content,
                    'source': article.source,
                    'published_at': article.published_at.isoformat(),
                    'sentiment_score': article.sentiment_score
                }
                for article in news
            ],
            'predictions': [
                {
                    'disease_type': pred.disease_type,
                    'probability': pred.outbreak_probability,
                    'risk_level': pred.risk_level,
                    'date': pred.prediction_date.isoformat()
                }
                for pred in predictions
            ]
        }
        
        return jsonify(county_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

