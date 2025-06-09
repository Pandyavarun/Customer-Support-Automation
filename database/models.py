from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SupportTicket(Base):
    __tablename__ = 'support_tickets'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String, unique=True, index=True)
    author_id = Column(String, index=True)
    created_at = Column(DateTime)
    text = Column(Text)
    clean_text = Column(Text)
    is_inbound = Column(Boolean)
    response_category = Column(String)
    sentiment_score = Column(Float)
    urgency_score = Column(Float)
    frustration_score = Column(Float)
    response_time_minutes = Column(Float)
    thread_id = Column(String, index=True)
    is_resolved = Column(Boolean, default=False)
    resolution_time = Column(DateTime)