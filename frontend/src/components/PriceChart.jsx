import { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

export function PriceChart({ symbol }) {
  const { data: price_update } = useWebSocket(`ws://localhost:8000/ws/prices/${symbol}`);
  const [price, setPrice] = useState(null);

  useEffect(() => {
    if (price_update?.data) {
      setPrice(price_update.data);
    }
  }, [price_update]);

  if (!price) {
    return (
      <div className="price-chart p-4 border border-gray-300 rounded-lg">
        <h3 className="text-lg font-bold">{symbol}</h3>
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }

  const changeColor = price.change_percent >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div className="price-chart p-4 border border-gray-300 rounded-lg">
      <h3 className="text-lg font-bold">{symbol}</h3>
      <p className="text-2xl font-bold">₹{price.price.toFixed(2)}</p>
      <p className={`text-lg ${changeColor}`}>
        {price.change_percent > 0 ? '▲' : '▼'} {Math.abs(price.change_percent).toFixed(2)}%
      </p>
      <p className="text-sm text-gray-500">{new Date(price.timestamp).toLocaleTimeString()}</p>
    </div>
  );
}
