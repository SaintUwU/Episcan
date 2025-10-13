from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# SQLite database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./episcan.db"

# For SQLite, check_same_thread must be False when using the same connection across threads (as with FastAPI)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()




