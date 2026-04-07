import { useEffect, useState } from 'react';

export function useWebSocket(url) {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connect = () => {
      try {
        ws = new WebSocket(url);

        ws.onopen = () => {
          console.log('✅ WebSocket connected:', url);
          setConnected(true);
          reconnectAttempts = 0;
        };

        ws.onmessage = (event) => {
          const message = JSON.parse(event.data);
          setData(message);
        };

        ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          setConnected(false);
        };

        ws.onclose = () => {
          console.log('Connection closed');
          setConnected(false);
          
          // Auto-reconnect
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            setTimeout(connect, 1000 * reconnectAttempts);
          }
        };
      } catch (error) {
        console.error('WebSocket connection error:', error);
        setConnected(false);
      }
    };

    connect();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [url]);

  return { data, connected };
}
