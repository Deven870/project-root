"""
WebSocket connection manager for broadcasting
"""
from fastapi import WebSocket
from typing import Set
import logging
import json

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions = {}  # Track subscriptions per connection
    
    async def connect(self, websocket: WebSocket):
        """Accept and track a connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[id(websocket)] = set()
        logger.info(f"✅ Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a connection"""
        self.active_connections.discard(websocket)
        self.subscriptions.pop(id(websocket), None)
        logger.info(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, data: dict):
        """Broadcast to all connected clients"""
        if not self.active_connections:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_to_subscription(self, symbol: str, data: dict):
        """Broadcast to clients subscribed to a symbol"""
        message = json.dumps(data)
        disconnected = set()
        
        for connection in self.active_connections:
            conn_id = id(connection)
            if symbol in self.subscriptions.get(conn_id, set()):
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Send error: {e}")
                    disconnected.add(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    def subscribe(self, websocket: WebSocket, symbol: str):
        """Subscribe connection to symbol updates"""
        conn_id = id(websocket)
        if conn_id not in self.subscriptions:
            self.subscriptions[conn_id] = set()
        self.subscriptions[conn_id].add(symbol)
        logger.info(f"📌 Subscribed to {symbol}")
    
    def unsubscribe(self, websocket: WebSocket, symbol: str):
        """Unsubscribe from symbol"""
        conn_id = id(websocket)
        if conn_id in self.subscriptions:
            self.subscriptions[conn_id].discard(symbol)
            logger.info(f"📌 Unsubscribed from {symbol}")

# Global manager
manager = ConnectionManager()
