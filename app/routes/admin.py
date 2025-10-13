"""
Admin dashboard routes for EpiScan
"""
from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from app import db
from app.models.data_models import User, TwitterData, GoogleTrendsData, NewsData, WHOData
from app.models.prediction_models import OutbreakPrediction, Alert, TrendAnalysis
from datetime import datetime, timedelta
from sqlalchemy import func, desc
from functools import wraps

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            from flask import flash, redirect, url_for
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    """Admin dashboard page"""
    # Get user statistics
    total_users = User.query.count()
    admin_users = User.query.filter_by(role='admin').count()
    regular_users = User.query.filter_by(role='user').count()
    
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
    active_alerts = Alert.query.filter(Alert.is_active == True).count()
    
    # Get recent predictions
    recent_predictions = OutbreakPrediction.query.filter(
        OutbreakPrediction.prediction_date >= datetime.utcnow().date() - timedelta(days=7)
    ).order_by(desc(OutbreakPrediction.outbreak_probability)).limit(5).all()
    
    # Get all users for management
    all_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    dashboard_data = {
        'user_stats': {
            'total_users': total_users,
            'admin_users': admin_users,
            'regular_users': regular_users
        },
        'data_stats': {
            'recent_tweets': recent_tweets,
            'recent_news': recent_news,
            'recent_trends': recent_trends,
            'active_alerts': active_alerts
        },
        'recent_predictions': recent_predictions,
        'recent_users': all_users
    }
    
    return render_template('admin_dashboard.html', data=dashboard_data)

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """User management page"""
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

@admin_bp.route('/api/stats')
@login_required
@admin_required
def get_stats():
    """Get detailed statistics for admin dashboard"""
    try:
        # Calculate comprehensive statistics
        stats = {
            'users': {
                'total': User.query.count(),
                'admins': User.query.filter_by(role='admin').count(),
                'regular': User.query.filter_by(role='user').count(),
                'recent': User.query.filter(
                    User.created_at >= datetime.utcnow() - timedelta(days=30)
                ).count()
            },
            'data': {
                'twitter': {
                    'total': TwitterData.query.count(),
                    'recent': TwitterData.query.filter(
                        TwitterData.created_at >= datetime.utcnow() - timedelta(days=7)
                    ).count()
                },
                'news': {
                    'total': NewsData.query.count(),
                    'recent': NewsData.query.filter(
                        NewsData.published_at >= datetime.utcnow() - timedelta(days=7)
                    ).count()
                },
                'trends': {
                    'total': GoogleTrendsData.query.count(),
                    'recent': GoogleTrendsData.query.filter(
                        GoogleTrendsData.date >= datetime.utcnow().date() - timedelta(days=7)
                    ).count()
                },
                'who': {
                    'total': WHOData.query.count(),
                    'recent': WHOData.query.filter(
                        WHOData.published_at >= datetime.utcnow() - timedelta(days=30)
                    ).count()
                }
            },
            'predictions': {
                'total': OutbreakPrediction.query.count(),
                'recent': OutbreakPrediction.query.filter(
                    OutbreakPrediction.prediction_date >= datetime.utcnow().date() - timedelta(days=7)
                ).count(),
                'high_risk': OutbreakPrediction.query.filter(
                    OutbreakPrediction.risk_level.in_(['high', 'critical'])
                ).count()
            },
            'alerts': {
                'total': Alert.query.count(),
                'active': Alert.query.filter(Alert.is_active == True).count(),
                'critical': Alert.query.filter(
                    Alert.is_active == True,
                    Alert.severity == 'critical'
                ).count()
            }
        }
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


