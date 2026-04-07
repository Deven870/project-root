import { useState } from 'react';
import './App.css';
import { PortfolioView } from './components/Portfolio';
import { PriceChart } from './components/PriceChart';
import { SignalPanel } from './components/SignalPanel';
import { SystemStatus } from './components/SystemStatus';

function App() {
  const [topStocks] = useState([
    'RELIANCE.NS',
    'TCS.NS',
    'INFY.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS'
  ]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🚀 DigiTrader v5.0</h1>
          <p>Real-time Trading Platform for NSE Stocks</p>
        </div>
      </header>

      <main className="app-main">
        {/* Status Bar */}
        <section className="status-section">
          <SystemStatus />
        </section>

        {/* Portfolio Summary */}
        <section className="portfolio-section">
          <PortfolioView />
        </section>

        {/* Real-time Prices & Signals */}
        <section className="dashboard-section">
          <h2>Live Trading Dashboard</h2>
          
          <div className="dashboard-grid">
            <div className="prices-column">
              <h3>Real-Time Prices</h3>
              <div className="prices-grid">
                {topStocks.map(symbol => (
                  <PriceChart key={symbol} symbol={symbol} />
                ))}
              </div>
            </div>

            <div className="signals-column">
              <h3>Trading Signals</h3>
              <div className="signals-grid">
                {topStocks.map(symbol => (
                  <SignalPanel key={symbol} symbol={symbol} />
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>🚀 DigiTrader v5.0 | 80+ NSE Stocks | Real-time Analysis | 72.5% Accuracy</p>
      </footer>
    </div>
  );
}

export default App;
