"""
Trading Bot Advanced Performance Analytics

Provides comprehensive performance analysis, metrics calculation, and optimization recommendations.
Analyzes trading patterns, win/loss sequences, drawdown periods, and statistical significance.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance classification levels"""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    POOR = "Poor"
    CRITICAL = "Critical"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    total_pnl: float
    total_return_pct: float
    
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    largest_win: float
    largest_loss: float
    
    max_drawdown: float
    max_drawdown_duration: int
    
    sharpe_ratio: float
    sortino_ratio: float
    
    average_trade_duration: float
    consecutive_wins: int
    consecutive_losses: int
    
    expectancy: float
    risk_reward_ratio: float


class PerformanceAnalyzer:
    """Advanced performance analytics engine"""
    
    def __init__(self, trades: List[Dict[str, Any]]):
        """
        Initialize analyzer with trades list
        
        Args:
            trades: List of trade dictionaries with keys:
                - entry_price, exit_price, quantity, entry_time, exit_time, pnl, status
        """
        self.trades = trades
        self.df = self._prepare_dataframe()
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Prepare trades as DataFrame for analysis"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamps
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        
        # Calculate trade duration if exits exist
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        
        return df
    
    def calculate_metrics(self, initial_capital: float = 300000) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if len(self.df) == 0:
            return self._empty_metrics()
        
        # Basic counts
        total_trades = len(self.df)
        winning_trades = len(self.df[self.df['pnl'] > 0])
        losing_trades = len(self.df[self.df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = self.df['pnl'].sum()
        total_return_pct = (total_pnl / initial_capital) * 100
        
        avg_win = self.df[self.df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.df[self.df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = self.df[self.df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.df[self.df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Win/loss extremes
        largest_win = self.df['pnl'].max()
        largest_loss = self.df['pnl'].min()
        
        # Drawdown analysis
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown()
        
        # Ratio metrics
        sharpe_ratio = self._calculate_sharpe_ratio(initial_capital)
        sortino_ratio = self._calculate_sortino_ratio(initial_capital)
        
        # Trade duration
        avg_trade_duration = self.df['duration'].mean() if 'duration' in self.df.columns else 0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._get_max_consecutive()
        
        # Expectancy (average P&L per trade)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk/Reward ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            average_trade_duration=avg_trade_duration,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            expectancy=expectancy,
            risk_reward_ratio=risk_reward
        )
    
    def _calculate_max_drawdown(self) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if len(self.df) < 2:
            return 0, 0
        
        # Calculate cumulative returns
        cumulative = (1 + self.df['pnl'] / 300000).cumprod()
        
        # Find running max
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Find duration
        dd_duration = 0
        if max_dd < 0:
            dd_periods = (drawdown == max_dd)
            dd_duration = len(drawdown[dd_periods])
        
        return abs(max_dd) * 100, dd_duration
    
    def _calculate_sharpe_ratio(self, initial_capital: float) -> float:
        """Calculate Sharpe ratio (risk-adjusted returns)"""
        if len(self.df) < 2:
            return 0
        
        returns = self.df['pnl'] / initial_capital
        excess_returns = returns - 0.0  # Assuming 0% risk-free rate
        
        if excess_returns.std() == 0:
            return 0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, initial_capital: float) -> float:
        """Calculate Sortino ratio (downside risk-adjusted returns)"""
        if len(self.df) < 2:
            return 0
        
        returns = self.df['pnl'] / initial_capital
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        excess_returns = returns.mean()
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0 if excess_returns == 0 else float('inf')
        
        return excess_returns / downside_std * np.sqrt(252)
    
    def _get_max_consecutive(self) -> Tuple[int, int]:
        """Get maximum consecutive wins and losses"""
        pnl_signs = (self.df['pnl'] > 0).astype(int)
        
        max_wins = 0
        max_losses = 0
        
        current_streak = 1
        for i in range(1, len(pnl_signs)):
            if pnl_signs.iloc[i] == pnl_signs.iloc[i-1]:
                current_streak += 1
            else:
                if pnl_signs.iloc[i-1] == 1:
                    max_wins = max(max_wins, current_streak)
                else:
                    max_losses = max(max_losses, current_streak)
                current_streak = 1
        
        return max_wins, max_losses
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics"""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_return_pct=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            largest_win=0,
            largest_loss=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            average_trade_duration=0,
            consecutive_wins=0,
            consecutive_losses=0,
            expectancy=0,
            risk_reward_ratio=0
        )
    
    def get_stock_analysis(self) -> Dict[str, Any]:
        """Analyze performance by stock"""
        if len(self.df) == 0:
            return {}
        
        by_stock = self.df.groupby('stock').agg({
            'pnl': ['sum', 'count', 'mean'],
        }).round(2)
        
        results = {}
        for stock in self.df['stock'].unique():
            stock_trades = self.df[self.df['stock'] == stock]
            wins = len(stock_trades[stock_trades['pnl'] > 0])
            
            results[stock] = {
                'total_trades': len(stock_trades),
                'total_pnl': stock_trades['pnl'].sum(),
                'win_rate': wins / len(stock_trades) * 100 if len(stock_trades) > 0 else 0,
                'avg_pnl': stock_trades['pnl'].mean()
            }
        
        return results
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Generate trading insights and recommendations"""
        insights = []
        metrics = self.calculate_metrics()
        
        # Win rate insights
        if metrics.win_rate < 0.4:
            insights.append({
                "level": "Critical",
                "message": "Win rate below 40% - Review signal quality",
                "recommendation": "Increase confidence threshold or adjust signal filters"
            })
        elif metrics.win_rate < 0.5:
            insights.append({
                "level": "Warning",
                "message": "Win rate below 50% - Profitability at risk",
                "recommendation": "Analyze losing trades for patterns"
            })
        elif metrics.win_rate > 0.7:
            insights.append({
                "level": "Positive",
                "message": "Strong win rate above 70%",
                "recommendation": "System performing well"
            })
        
        # Profit factor
        if metrics.profit_factor < 1.0:
            insights.append({
                "level": "Critical",
                "message": "Profit factor below 1.0 - Gross losses exceed gains",
                "recommendation": "Improve position sizing or exit strategy"
            })
        elif metrics.profit_factor > 2.0:
            insights.append({
                "level": "Positive",
                "message": f"Excellent profit factor: {metrics.profit_factor:.2f}",
                "recommendation": "Consider increasing position size"
            })
        
        # Risk/Reward
        if metrics.risk_reward_ratio < 1.0:
            insights.append({
                "level": "Warning",
                "message": "Risk exceeds reward per trade",
                "recommendation": "Adjust stop loss or take profit levels"
            })
        
        # Drawdown
        if metrics.max_drawdown > 0.15:  # 15%
            insights.append({
                "level": "Warning",
                "message": f"Significant drawdown: {metrics.max_drawdown:.1f}%",
                "recommendation": "Reduce position size or increase diversification"
            })
        
        return insights
    
    def format_metrics_for_display(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """Format metrics for display"""
        return {
            "Total Trades": str(metrics.total_trades),
            "Winning": f"{metrics.winning_trades} ({metrics.win_rate*100:.1f}%)",
            "Losing": f"{metrics.losing_trades} ({(1-metrics.win_rate)*100:.1f}%)",
            "Total P&L": f"₹{metrics.total_pnl:,.0f}",
            "Return %": f"{metrics.total_return_pct:+.2f}%",
            "Avg Win": f"₹{metrics.avg_win:,.0f}",
            "Avg Loss": f"₹{metrics.avg_loss:,.0f}",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Expectancy": f"₹{metrics.expectancy:,.0f}",
            "Risk/Reward": f"1:{metrics.risk_reward_ratio:.2f}",
            "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
            "Max Drawdown": f"{metrics.max_drawdown:.2f}%",
            "Consecutive Wins": str(metrics.consecutive_wins),
            "Consecutive Losses": str(metrics.consecutive_losses),
            "Avg Trade Duration": f"{metrics.average_trade_duration:.0f} min"
        }


# Export functions
def analyze_trading_performance(trades: List[Dict], initial_capital: float = 300000) -> Dict[str, Any]:
    """
    Comprehensive performance analysis
    
    Args:
        trades: List of trade records
        initial_capital: Starting capital
    
    Returns:
        Complete performance analysis
    """
    analyzer = PerformanceAnalyzer(trades)
    metrics = analyzer.calculate_metrics(initial_capital)
    
    return {
        "metrics": metrics,
        "stock_analysis": analyzer.get_stock_analysis(),
        "insights": analyzer.get_insights(),
        "formatted_display": analyzer.format_metrics_for_display(metrics)
    }


if __name__ == "__main__":
    # Example usage
    sample_trades = [
        {"stock": "RELIANCE", "pnl": 500, "entry_time": datetime.now(), "exit_time": datetime.now()},
        {"stock": "TCS", "pnl": -200, "entry_time": datetime.now(), "exit_time": datetime.now()},
        {"stock": "INFY", "pnl": 750, "entry_time": datetime.now(), "exit_time": datetime.now()},
    ]
    
    result = analyze_trading_performance(sample_trades)
    print("Performance Analysis:")
    print(json.dumps(result["formatted_display"], indent=2))
