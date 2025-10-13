from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from .database import Base


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    source = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    published_at = Column(DateTime, default=datetime.utcnow, nullable=False)




