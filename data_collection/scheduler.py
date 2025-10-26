"""
Data collection scheduler for EpiScan
"""
import schedule
import time
import logging
from datetime import datetime
from app import create_app, db
from data_collection.twitter_collector import TwitterCollector
from data_collection.google_trends import GoogleTrendsCollector
from data_collection.who_data import WHODataCollector
from data_collection.who_gho_collector import WHOGHOCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectionScheduler:
    """Schedules and manages data collection tasks"""
    
    def __init__(self):
        """Initialize the scheduler"""
        self.app = create_app()
        self.collectors = {
            'twitter': TwitterCollector(),
            'google_trends': GoogleTrendsCollector(),
            'who': WHODataCollector(),
            'who_gho': WHOGHOCollector()
        }
    
    def collect_twitter_data(self):
        """Collect Twitter data"""
        try:
            logger.info("Starting scheduled Twitter data collection...")
            with self.app.app_context():
                results = self.collectors['twitter'].collect_and_save(days_back=1, max_results=200)
                logger.info(f"Twitter collection completed: {results}")
        except Exception as e:
            logger.error(f"Twitter collection failed: {str(e)}")
    
    def collect_google_trends_data(self):
        """Collect Google Trends data"""
        try:
            logger.info("Starting scheduled Google Trends data collection...")
            with self.app.app_context():
                results = self.collectors['google_trends'].collect_all_health_trends(timeframe='today 7d')
                logger.info(f"Google Trends collection completed: {results}")
        except Exception as e:
            logger.error(f"Google Trends collection failed: {str(e)}")
    
    def collect_who_data(self):
        """Collect WHO data"""
        try:
            logger.info("Starting scheduled WHO data collection...")
            with self.app.app_context():
                results = self.collectors['who'].collect_and_save(days_back=7)
                logger.info(f"WHO collection completed: {results}")
        except Exception as e:
            logger.error(f"WHO collection failed: {str(e)}")
    
    def collect_who_gho_data(self):
        """Collect WHO GHO data"""
        try:
            logger.info("Starting scheduled WHO GHO data collection...")
            with self.app.app_context():
                results = self.collectors['who_gho'].collect_and_save(years_back=1)
                logger.info(f"WHO GHO collection completed: {results}")
        except Exception as e:
            logger.error(f"WHO GHO collection failed: {str(e)}")
    
    def collect_all_data(self):
        """Collect data from all sources"""
        logger.info("Starting scheduled data collection from all sources...")
        
        # Collect from all sources
        self.collect_twitter_data()
        self.collect_google_trends_data()
        self.collect_who_data()
        self.collect_who_gho_data()
        
        logger.info("Scheduled data collection completed")
    
    def setup_schedule(self):
        """Set up the data collection schedule"""
        # Twitter data collection every 2 hours
        schedule.every(2).hours.do(self.collect_twitter_data)
        
        # Google Trends data collection daily at 6 AM
        schedule.every().day.at("06:00").do(self.collect_google_trends_data)
        
        # WHO data collection daily at 8 AM
        schedule.every().day.at("08:00").do(self.collect_who_data)
        
        # WHO GHO data collection weekly on Sundays at 10 AM
        schedule.every().sunday.at("10:00").do(self.collect_who_gho_data)
        
        # Full data collection daily at midnight
        schedule.every().day.at("00:00").do(self.collect_all_data)
        
        logger.info("Data collection schedule configured")
    
    def run_scheduler(self):
        """Run the scheduler"""
        logger.info("Starting EpiScan data collection scheduler...")
        
        # Set up the schedule
        self.setup_schedule()
        
        # Run the scheduler
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    scheduler = DataCollectionScheduler()
    scheduler.run_scheduler()

