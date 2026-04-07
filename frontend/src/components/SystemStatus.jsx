import { useEffect, useState } from 'react';
import { fetchHealth } from '../services/api';

export function SystemStatus() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await fetchHealth();
        setHealth(data);
      } catch (error) {
        console.error('Health check failed:', error);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (!health) {
    return <div className="system-status p-4">Status unavailable</div>;
  }

  return (
    <div className="system-status p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <h3 className="text-lg font-bold mb-2">System Status</h3>
      
      <div className="space-y-2">
        <p className="flex justify-between">
          <span>Status:</span>
          <span className={health.status === 'healthy' ? 'text-green-600 font-bold' : 'text-red-600'}>
            {health.status === 'healthy' ? '🟢 Healthy' : '🔴 Error'}
          </span>
        </p>
        
        <p className="flex justify-between">
          <span>Database:</span>
          <span className="text-green-600">{health.database}</span>
        </p>
        
        <p className="flex justify-between">
          <span>Cache:</span>
          <span className={health.cache.includes('Connected') ? 'text-green-600' : 'text-yellow-600'}>
            {health.cache}
          </span>
        </p>
        
        <p className="flex justify-between">
          <span>Workers:</span>
          <span className="text-green-600">{health.workers}</span>
        </p>
        
        <p className="flex justify-between">
          <span>Active Connections:</span>
          <span className="text-blue-600 font-bold">{health.websockets}</span>
        </p>
      </div>
    </div>
  );
}
