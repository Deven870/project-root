"""
Pydantic schemas for signal endpoints
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SignalBase(BaseModel):
    symbol: str
    signal_type: str
    confidence: float
    technical_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    recommendation: Optional[str] = None

class SignalCreate(SignalBase):
    pass

class SignalResponse(SignalBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class PriceUpdate(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    timestamp: datetime

class PriceResponse(BaseModel):
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    timestamp: datetime
