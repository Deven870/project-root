"""
SQLAlchemy ORM Models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Price(Base):
    """Price data model - 1min/daily OHLC"""
    __tablename__ = "prices"
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Signal(Base):
    """Trading signals"""
    __tablename__ = "signals"
    __table_args__ = (
        Index('idx_symbol_created', 'symbol', 'created_at'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, default=0.0)
    technical_score = Column(Float)
    sentiment_score = Column(Float)
    recommendation = Column(Text)
    analysis_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class Portfolio(Base):
    """Portfolio allocation"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=0)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Trade(Base):
    """Trade execution history"""
    __tablename__ = "trades"
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'created_at'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

class AnalysisCache(Base):
    """Analysis results cache"""
    __tablename__ = "analysis_cache"
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    analysis_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ttl = Column(Integer, default=180)  # Time to live in seconds
