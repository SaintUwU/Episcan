from datetime import datetime
from typing import Generator, List

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from .database import SessionLocal
from .init_db import init_db
from .models import Article


app = FastAPI(title="EpiScan API")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def root():
    return {"message": "EpiScan API is running 🚀"}


class ArticleCreate(BaseModel):
    title: str
    source: str
    url: HttpUrl


class ArticleRead(BaseModel):
    id: int
    title: str
    source: str
    url: HttpUrl
    published_at: datetime

    class Config:
        from_attributes = True


@app.post("/articles/", response_model=ArticleRead, status_code=status.HTTP_201_CREATED)
def create_article(payload: ArticleCreate, db: Session = Depends(get_db)):
    existing = db.query(Article).filter(Article.url == str(payload.url)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Article with this URL already exists")

    article = Article(
        title=payload.title,
        source=payload.source,
        url=str(payload.url),
    )
    db.add(article)
    db.commit()
    db.refresh(article)
    return article


@app.get("/articles/", response_model=List[ArticleRead])
def list_articles(db: Session = Depends(get_db)):
    articles = db.query(Article).order_by(Article.published_at.desc()).all()
    return articles




