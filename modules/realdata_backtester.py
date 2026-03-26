# modules/realdata_backtester.py
"""
Real Data Backtester
Tests ML models on actual NSE historical data with realistic position sizing and risk management.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from modules.feature_engineering import build_features
from modules.predictive_ml import predict_intraday, predict_long_term
from modules.risk_management import RiskManager


class RealdataBacktester:
    """
    Backtest trading strategy on real NSE data.
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        max_risk_per_trade: float = 0.02,
    ):
        """
        Initialize backtester.
        
        Args:
            tickers: List of NSE tickers (e.g., "RELIANCE.NS")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital in Rs
            max_risk_per_trade: Risk per trade as % of capital
        """
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        self.risk_manager = RiskManager(
            account_size=initial_capital,
            max_risk_per_trade=max_risk_per_trade
        )
        
        self.data = {}
        self.signals = {}
        self.trades = []
        self.equity_curve = [initial_capital]
    
    
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch real historical data for all tickers.
        """
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                
                if df.empty:
                    print(f"  ⚠️ {ticker}: No data fetched")
                    continue
                
                # Ensure required OHLCV columns
                df["Ticker"] = ticker
                df = df.rename(columns=str.lower)
                
                # Calculate daily returns
                df["returns"] = df["close"].pct_change()
                
                self.data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} rows")
            
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        
        return self.data
    
    
    def generate_signals(self, lookback: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals using ML models.
        """
        print("\nGenerating trading signals...")
        
        for ticker in self.data.keys():
            try:
                df = self.data[ticker].copy()
                
                # Build features
                df_features = build_features(df, ticker=ticker)
                
                if df_features is None or df_features.empty:
                    print(f"  ⚠️ {ticker}: Failed to build features")
                    continue
                
                # Generate signals
                signals = []
                for idx in range(lookback, len(df_features)):
                    row = df_features.iloc[idx]
                    
                    try:
                        # Get intraday and long-term predictions
                        trend_intra, conf_intra = predict_intraday(df_features.iloc[:idx+1])
                        trend_long, conf_long = predict_long_term(df_features.iloc[:idx+1])
                        
                        # Combine signals
                        final_trend = trend_intra if conf_intra > conf_long else trend_long
                        final_confidence = max(conf_intra, conf_long)
                        
                        signals.append({
                            "date": df.index[idx],
                            "close": row.get("close", 0),
                            "trend": final_trend,
                            "confidence": final_confidence,
                            "trend_intra": trend_intra,
                            "conf_intra": conf_intra,
                            "trend_long": trend_long,
                            "conf_long": conf_long,
                        })
                    except:
                        signals.append({
                            "date": df.index[idx],
                            "close": row.get("close", 0),
                            "trend": "Neutral",
                            "confidence": 0.5,
                        })
                
                if signals:
                    self.signals[ticker] = pd.DataFrame(signals)
                    print(f"  ✓ {ticker}: {len(signals)} signals generated")
            
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        
        return self.signals
    
    
    def run_backtest(
        self,
        use_stop_loss: bool = True,
        use_position_sizing: bool = True,
        position_hold_days: int = 5
    ) -> pd.DataFrame:
        """
        Run trading strategy backtest.
        
        Args:
            use_stop_loss: Use stop-loss orders
            use_position_sizing: Use risk-based position sizing
            position_hold_days: Max days to hold a position
            
        Returns:
            DataFrame of trades executed
        """
        print(f"\nRunning backtest ({self.start_date} to {self.end_date})...")
        print(f"Initial capital: Rs {self.initial_capital:,.0f}")
        
        active_trades = []  # Trades currently open
        
        for ticker in self.signals.keys():
            signals_df = self.signals[ticker]
            data_df = self.data[ticker]
            
            for sig_idx, signal in signals_df.iterrows():
                try:
                    entry_price = signal["close"]
                    entry_trend = signal["trend"]
                    confidence = signal["confidence"]
                    entry_date = signal["date"]
                    
                    # Skip if neutral or low confidence
                    if entry_trend == "Neutral" or confidence < 0.50:
                        continue
                    
                    # Calculate position size
                    if use_position_sizing:
                        # Estimate stop loss (3% for simplicity)
                        sl = entry_price * (0.97 if entry_trend == "Bullish" else 1.03)
                        pos_size = self.risk_manager.calculate_position_size(
                            entry_price, sl, confidence
                        )
                    else:
                        pos_size = 10  # Fixed 10 units
                    
                    # Check if we can take the trade
                    can_trade, reason = self.risk_manager.can_take_trade(
                        pos_size, entry_price, confidence
                    )
                    
                    if not can_trade:
                        continue
                    
                    # Calculate stop loss and take profit
                    if use_stop_loss:
                        sl = self.risk_manager.calculate_stop_loss(entry_price, entry_trend)
                        tp = entry_price + (entry_price - sl) * 2 if entry_trend == "Bullish" \
                             else entry_price - (sl - entry_price) * 2
                    else:
                        sl = None
                        tp = None
                    
                    # Open trade
                    trade = {
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "position_size": pos_size,
                        "trend": entry_trend,
                        "confidence": confidence,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "hold_days": 0,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "pnl": 0,
                        "pnl_percent": 0,
                    }
                    
                    active_trades.append(trade)
                
                except Exception as e:
                    pass
            
            # Close expired positions or profitable ones
            closed_trades = []
            for i, trade in enumerate(active_trades):
                try:
                    # Find exit price from data
                    if trade["exit_date"] is not None:
                        continue
                    
                    # Look for exit conditions
                    future_data = data_df.loc[entry_date:]
                    
                    for future_idx, (date, row) in enumerate(future_data.iterrows()):
                        if future_idx == 0:
                            continue  # Skip entry day
                        
                        current_price = row.get("Close", trade["entry_price"])
                        days_held = (date - trade["entry_date"]).days
                        
                        exit_reason = None
                        should_exit = False
                        
                        # Check stop loss
                        if trade["stop_loss"]:
                            if trade["trend"] == "Bullish" and current_price <= trade["stop_loss"]:
                                should_exit = True
                                exit_reason = "SL_HIT"
                            elif trade["trend"] == "Bearish" and current_price >= trade["stop_loss"]:
                                should_exit = True
                                exit_reason = "SL_HIT"
                        
                        # Check take profit
                        if trade["take_profit"]:
                            if trade["trend"] == "Bullish" and current_price >= trade["take_profit"]:
                                should_exit = True
                                exit_reason = "TP_HIT"
                            elif trade["trend"] == "Bearish" and current_price <= trade["take_profit"]:
                                should_exit = True
                                exit_reason = "TP_HIT"
                        
                        # Check max hold time
                        if days_held >= position_hold_days:
                            should_exit = True
                            exit_reason = "TIMEOUT"
                        
                        if should_exit:
                            trade["exit_date"] = date
                            trade["exit_price"] = current_price
                            trade["exit_reason"] = exit_reason
                            trade["hold_days"] = days_held
                            
                            # Calculate P&L
                            if trade["trend"] == "Bullish":
                                pnl = (current_price - trade["entry_price"]) * trade["position_size"]
                            else:
                                pnl = (trade["entry_price"] - current_price) * trade["position_size"]
                            
                            trade["pnl"] = pnl
                            trade["pnl_percent"] = (pnl / (trade["entry_price"] * trade["position_size"])) * 100
                            
                            closed_trades.append(i)
                            break
                
                except:
                    pass
            
            # Remove closed trades
            for idx in sorted(closed_trades, reverse=True):
                self.trades.append(active_trades.pop(idx))
        
        self.trades_df = pd.DataFrame(self.trades)
        return self.trades_df
    
    
    def get_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        """
        if self.trades_df.empty:
            return {"error": "No trades executed"}
        
        trades = self.trades_df
        
        # Base metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades["pnl"] > 0])
        losing_trades = len(trades[trades["pnl"] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades["pnl"].sum()
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0
        max_win = trades["pnl"].max()
        max_loss = trades["pnl"].min()
        
        # Return metrics
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Risk metrics
        returns = trades["pnl"].values
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Comparison against buy-and-hold
        avg_stock_return = self.trades_df[self.trades_df["pnl_percent"] > 0]["pnl_percent"].mean()
        
        report = {
            "backtest_period": f"{self.start_date} to {self.end_date}",
            "initial_capital": self.initial_capital,
            "final_capital": self.initial_capital + total_pnl,
            "total_pnl": round(total_pnl, 2),
            "total_return_percent": round(total_return_pct, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_percent": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_win": round(max_win, 2),
            "max_loss": round(max_loss, 2),
            "profit_factor": round(abs(avg_win * winning_trades / (avg_loss * losing_trades)), 2) if losing_trades > 0 else 0,
            "sharpe_ratio": round(sharpe, 2),
            "avg_hold_days": round(trades["hold_days"].mean(), 1),
            "timestamp": datetime.now().isoformat(),
        }
        
        return report
    
    
    def plot_equity_curve(self):
        """
        Generate equity curve plot.
        """
        if self.trades_df.empty:
            return None
        
        try:
            import plotly.graph_objects as go
            
            trades = self.trades_df.sort_values("exit_date")
            cumulative_pnl = trades["pnl"].cumsum()
            equity_curve = self.initial_capital + cumulative_pnl
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trades["exit_date"],
                y=equity_curve,
                mode="lines",
                name="Strategy Equity",
                line=dict(color="blue", width=2)
            ))
            
            fig.update_layout(
                title="Backtest Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity (Rs)",
                hovermode="x unified",
                template="plotly_dark"
            )
            
            return fig
        
        except:
            return None
