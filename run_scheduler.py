"""
Run the EpiScan data collection scheduler
"""
from data_collection.scheduler import DataCollectionScheduler

if __name__ == "__main__":
    scheduler = DataCollectionScheduler()
    scheduler.run_scheduler()

