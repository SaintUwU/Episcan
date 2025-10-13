#!/usr/bin/env python3
"""Database status check utility"""
from app import create_app, db
from app.models.data_models import TwitterData, NewsData, GoogleTrendsData, WHOData

def check_database_status():
    app = create_app()
    with app.app_context():
        tw = TwitterData.query.count()
        news = NewsData.query.count()
        trends = GoogleTrendsData.query.count()
        who = WHOData.query.count()
        total = tw + news + trends + who
        
        print('\nDatabase Status:')
        print(f'  Twitter:       {tw:>6} records')
        print(f'  News:          {news:>6} records')
        print(f'  Google Trends: {trends:>6} records')
        print(f'  WHO Data:      {who:>6} records')
        print(f'  {"-"*30}')
        print(f'  Total:         {total:>6} records')
        
        print(f'\nStatus: ', end='')
        if total >= 1000:
            print('Good - sufficient data for training')
        elif total >= 500:
            print('Adequate - can begin training')
        elif total >= 100:
            print('Limited - collect more data recommended')
        else:
            print('Insufficient - more data needed')
        print()

if __name__ == '__main__':
    check_database_status()

