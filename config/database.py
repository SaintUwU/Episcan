"""
Database configuration for EpiScan
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Database configuration
DATABASE_CONFIG = {
    'development': {
        'url': os.getenv('DATABASE_URL', 'sqlite:///episcan_dev.db'),
        'echo': True,
        'pool_pre_ping': True
    },
    'production': {
        'url': os.getenv
        ('DATABASE_URL',
         'postgresql+psycopg2://episcan_user:root@localhost:5432/episcan_db'
         ),
        'echo': False,
        'pool_pre_ping': True,
        'pool_recycle': 300
    },
    'testing': {
        'url': 'sqlite:///:memory:',
        'echo': False,
        'poolclass': StaticPool,
        'connect_args': {'check_same_thread': False}
    }
}

def get_database_url(environment='development'):
    """Get database URL for specified environment"""
    return DATABASE_CONFIG[environment]['url']

def create_database_engine(environment='development'):
    """Create database engine for specified environment"""
    config = DATABASE_CONFIG[environment]
    return create_engine(
        config['url'],
        echo=config.get('echo', False),
        pool_pre_ping=config.get('pool_pre_ping', True),
        pool_recycle=config.get('pool_recycle', 300),
        poolclass=config.get('poolclass')
    )

def get_session_factory(environment='development'):
    """Get session factory for specified environment"""
    engine = create_database_engine(environment)
    return sessionmaker(bind=engine)

