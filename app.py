"""
EpiScan Main Application
"""
import os
from flask import Flask, redirect, url_for
from app import create_app, db
from flask_migrate import upgrade

# Create Flask app
app = create_app()

@app.route('/')
def index():
    """Redirect root to login page"""
    return redirect(url_for('auth.login'))

@app.shell_context_processor
def make_shell_context():
    """Make database available in shell context"""
    return {'db': db}

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized!")

@app.cli.command()
def create_admin():
    """Create the first admin user"""
    from app.models.data_models import User
    
    # Check if any users exist
    if User.query.first():
        print("Users already exist in the database. Skipping admin creation.")
        return
    
    # Create admin user
    admin = User(
        full_name="System Administrator",
        email="ericdeish@gmail.com",
        role="admin"
    )
    admin.set_password("admin123")  # Change this password in production!
    
    try:
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user created successfully!")
        print(f"   Email: {admin.email}")
        print(f"   Password: admin123")
        print("   ⚠️  Please change the password after first login!")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating admin user: {str(e)}")

@app.cli.command()
def seed_users():
    """Create admin and medical practitioner users"""
    from app.models.data_models import User
    
    # Check if any users exist
    if User.query.first():
        print("Users already exist in the database. Skipping user creation.")
        return
    
    # Create admin user
    admin = User(
        full_name="System Administrator",
        email="admin@episcan.ke",
        role="admin"
    )
    admin.set_password("admin123")
    
    # Create medical practitioner user
    practitioner = User(
        full_name="Dr. Jane Mwangi",
        email="jane.mwangi@episcan.ke",
        role="user"
    )
    practitioner.set_password("user123")
    
    try:
        db.session.add(admin)
        db.session.add(practitioner)
        db.session.commit()
        
        print("✅ Users created successfully!")
        print("\n📋 Login Credentials:")
        print("=" * 50)
        print("🔑 ADMIN USER:")
        print(f"   Email: {admin.email}")
        print(f"   Password: admin123")
        print(f"   Role: {admin.role.upper()}")
        print(f"   Dashboard: /admin/dashboard")
        print("\n👩‍⚕️ MEDICAL PRACTITIONER:")
        print(f"   Email: {practitioner.email}")
        print(f"   Password: user123")
        print(f"   Role: {practitioner.role.upper()}")
        print(f"   Dashboard: /user/dashboard")
        print("\n⚠️  IMPORTANT: Change these passwords after first login!")
        print("=" * 50)
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating users: {str(e)}")

@app.cli.command()
def collect_data():
    """Collect data from all sources"""
    from data_collection.twitter_collector import TwitterCollector
    from data_collection.google_trends import GoogleTrendsCollector
    from data_collection.who_data import WHODataCollector
    from data_collection.who_gho_collector import WHOGHOCollector
    
    print("Starting data collection...")
    
    # Collect Twitter data
    try:
        twitter_collector = TwitterCollector()
        twitter_results = twitter_collector.collect_and_save(days_back=7, max_results=100)
        print(f"Twitter collection: {twitter_results}")
    except Exception as e:
        print(f"Twitter collection failed: {str(e)}")
    
    # Collect Google Trends data
    try:
        trends_collector = GoogleTrendsCollector()
        trends_results = trends_collector.collect_all_health_trends(timeframe='today 7d')
        print(f"Google Trends collection: {trends_results}")
    except Exception as e:
        print(f"Google Trends collection failed: {str(e)}")
    
    # Collect WHO GHO data
    try:
        who_gho_collector = WHOGHOCollector()
        who_gho_results = who_gho_collector.collect_and_save(years_back=1)
        print(f"WHO GHO collection: {who_gho_results}")
    except Exception as e:
        print(f"WHO GHO collection failed: {str(e)}")
    
    # Collect WHO data
    try:
        who_collector = WHODataCollector()
        who_results = who_collector.collect_and_save(days_back=30)
        print(f"WHO collection: {who_results}")
    except Exception as e:
        print(f"WHO collection failed: {str(e)}")
    
    print("Data collection completed!")

@app.cli.command()
def train_models():
    """Train ML models for outbreak prediction"""
    import subprocess
    import sys
    
    print("Starting ML model training...")
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'ml_models/train_models.py',
            '--days', '90',
            '--hyperparameter-tuning'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
        else:
            print("❌ Model training failed!")
            
    except Exception as e:
        print(f"❌ Error running model training: {str(e)}")

@app.cli.command()
def evaluate_models():
    """Evaluate existing ML models"""
    import subprocess
    import sys
    
    print("Evaluating ML models...")
    
    try:
        # Run the evaluation script
        result = subprocess.run([
            sys.executable, 'ml_models/train_models.py',
            '--evaluate-only',
            '--days', '30'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("✅ Model evaluation completed successfully!")
        else:
            print("❌ Model evaluation failed!")
            
    except Exception as e:
        print(f"❌ Error running model evaluation: {str(e)}")

@app.cli.command()
def make_predictions():
    """Make outbreak predictions using trained models"""
    from ml_models.predictor import OutbreakPredictor
    from ml_models.data_processor import DataProcessor
    from datetime import datetime
    
    print("Making outbreak predictions...")
    
    try:
        # Initialize predictor
        predictor = OutbreakPredictor()
        predictor.load_models()
        
        # Get recent data
        data_processor = DataProcessor()
        df, _ = data_processor.prepare_ml_dataset(days_back=7)
        
        if df.empty:
            print("❌ No recent data available for predictions")
            return
        
        # Make predictions
        predictions = predictor.batch_predict(df)
        
        # Display results
        print(f"\n📊 PREDICTION RESULTS ({len(predictions)} predictions)")
        print("=" * 60)
        
        high_risk = predictions[predictions['outbreak_probability'] > 0.7]
        medium_risk = predictions[(predictions['outbreak_probability'] > 0.4) & 
                                (predictions['outbreak_probability'] <= 0.7)]
        
        print(f"🔴 High Risk Predictions: {len(high_risk)}")
        print(f"🟡 Medium Risk Predictions: {len(medium_risk)}")
        print(f"🟢 Low Risk Predictions: {len(predictions) - len(high_risk) - len(medium_risk)}")
        
        if len(high_risk) > 0:
            print(f"\n🚨 HIGH RISK ALERTS:")
            for _, pred in high_risk.iterrows():
                print(f"  - County: {pred['county']}, Probability: {pred['outbreak_probability']:.3f}")
        
        print(f"\n✅ Predictions completed successfully!")
        
    except Exception as e:
        print(f"❌ Error making predictions: {str(e)}")

@app.cli.command()
def test_api_keys():
    """Test all API keys and connections"""
    import subprocess
    import sys
    
    print("Testing API keys and connections...")
    
    try:
        # Run the API test script
        result = subprocess.run([
            sys.executable, 'test_api_keys.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("✅ All API keys are working correctly!")
        else:
            print("❌ Some API keys failed. Check the output above.")
            
    except Exception as e:
        print(f"❌ Error testing API keys: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

