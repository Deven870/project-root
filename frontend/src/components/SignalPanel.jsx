import { useEffect, useState } from 'react';
import { fetchSignal } from '../services/api';

export function SignalPanel({ symbol }) {
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadSignal = async () => {
      setLoading(true);
      try {
        const data = await fetchSignal(symbol);
        setSignal(data);
      } catch (error) {
        console.error('Error loading signal:', error);
      }
      setLoading(false);
    };

    loadSignal();
    
    // Refresh every 5 minutes
    const interval = setInterval(loadSignal, 300000);
    return () => clearInterval(interval);
  }, [symbol]);

  if (loading) {
    return <div className="signal-panel p-4">Loading...</div>;
  }

  if (!signal) {
    return <div className="signal-panel p-4">No signal available</div>;
  }

  const signalColor = {
    'BUY': 'bg-green-100 text-green-800',
    'SELL': 'bg-red-100 text-red-800',
    'HOLD': 'bg-yellow-100 text-yellow-800',
  };

  return (
    <div className="signal-panel p-4 border border-gray-300 rounded-lg">
      <h3 className="text-lg font-bold mb-2">{symbol}</h3>
      <div className={`px-3 py-2 rounded ${signalColor[signal.signal] || 'bg-gray-100'}`}>
        <p className="font-bold text-lg">{signal.signal}</p>
        <p className="text-sm">Confidence: {signal.confidence?.toFixed(1)}%</p>
      </div>
      <p className="mt-2 text-sm text-gray-600">{signal.recommendation}</p>
    </div>
  );
}
