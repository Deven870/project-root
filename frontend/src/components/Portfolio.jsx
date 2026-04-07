import { useEffect, useState } from 'react';
import { fetchPortfolio } from '../services/api';

export function PortfolioView() {
  const [portfolio, setPortfolio] = useState(null);

  useEffect(() => {
    const loadPortfolio = async () => {
      try {
        const data = await fetchPortfolio();
        setPortfolio(data);
      } catch (error) {
        console.error('Error loading portfolio:', error);
      }
    };

    loadPortfolio();
    const interval = setInterval(loadPortfolio, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (!portfolio) {
    return <div className="portfolio-view p-4">Loading...</div>;
  }

  return (
    <div className="portfolio-view p-4 border border-gray-300 rounded-lg">
      <h2 className="text-2xl font-bold mb-4">Portfolio Summary</h2>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 bg-gray-100 rounded">
          <p className="text-gray-600">Capital</p>
          <p className="text-2xl font-bold">₹{portfolio.capital?.toLocaleString()}</p>
        </div>
        
        <div className="p-3 bg-gray-100 rounded">
          <p className="text-gray-600">Current Value</p>
          <p className="text-2xl font-bold">₹{portfolio.current_value?.toLocaleString()}</p>
        </div>
        
        <div className={`p-3 rounded ${portfolio.pnl >= 0 ? 'bg-green-100' : 'bg-red-100'}`}>
          <p className="text-gray-600">P&L</p>
          <p className="text-2xl font-bold">₹{portfolio.pnl?.toLocaleString()}</p>
        </div>
        
        <div className={`p-3 rounded ${portfolio.pnl_percent >= 0 ? 'bg-green-100' : 'bg-red-100'}`}>
          <p className="text-gray-600">Return %</p>
          <p className="text-2xl font-bold">{portfolio.pnl_percent?.toFixed(2)}%</p>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="p-3 bg-blue-100 rounded">
          <p className="text-gray-600">Accuracy</p>
          <p className="text-2xl font-bold">{portfolio.accuracy}</p>
        </div>
        
        <div className="p-3 bg-purple-100 rounded">
          <p className="text-gray-600">Total Trades</p>
          <p className="text-2xl font-bold">{portfolio.trades}</p>
        </div>
      </div>
    </div>
  );
}
