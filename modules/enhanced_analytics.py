"""
Enhanced Analytics Module - Advanced Graphs, Visualizations & Performance Metrics
==================================================================================
Provides:
- Equity curves and drawdown analysis
- Performance metrics dashboard
- Risk-Return scatter plots
- Correlation heatmaps
- Distribution analysis
- Performance attribution
- Optimization metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class EquityCurveAnalyzer:
    """Analyze and visualize equity curves with drawdown analysis."""
    
    @staticmethod
    def calculate_equity_curve(daily_returns: np.ndarray, initial_capital: float = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate equity curve from daily returns.
        
        Args:
            daily_returns: Array of daily returns (decimal, e.g., 0.02 for 2%)
            initial_capital: Starting capital
            
        Returns:
            Tuple of (equity_values, cumulative_returns)
        """
        cumulative_returns = np.cumprod(1 + daily_returns) - 1
        equity_values = initial_capital * (1 + cumulative_returns)
        return equity_values, cumulative_returns
    
    @staticmethod
    def calculate_drawdown(equity_values: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate drawdown series and maximum drawdown.
        
        Args:
            equity_values: Array of portfolio values
            
        Returns:
            Tuple of (drawdown_series, max_drawdown)
        """
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max
        max_drawdown = np.min(drawdown)
        return drawdown, max_drawdown
    
    @staticmethod
    def calculate_metrics(daily_returns: np.ndarray, rf_rate: float = 0.06 / 252) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            daily_returns: Array of daily returns
            rf_rate: Risk-free rate (annual ~6%, converted to daily)
            
        Returns:
            Dictionary with performance metrics
        """
        annual_return = (1 + np.mean(daily_returns)) ** 252 - 1
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - rf_rate) / max(annual_volatility, 1e-6)
        
        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = np.max(np.sum(np.diff(np.sign(daily_returns)) != 0)) if len(daily_returns) > 1 else 0
        
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "monthly_return": (1 + np.mean(daily_returns)) ** 21 - 1,  # 21 trading days
            "best_day": np.max(daily_returns),
            "worst_day": np.min(daily_returns),
            "profit_factor": abs(np.sum(daily_returns[daily_returns > 0]) / np.sum(daily_returns[daily_returns < 0])) if np.sum(daily_returns[daily_returns < 0]) != 0 else 0,
        }
    
    @staticmethod
    def plot_equity_curve(daily_returns: np.ndarray, title: str = "Equity Curve", initial_capital: float = 100000) -> go.Figure:
        """Create equity curve plot with drawdown."""
        equity, cum_returns = EquityCurveAnalyzer.calculate_equity_curve(daily_returns, initial_capital)
        drawdown, max_dd = EquityCurveAnalyzer.calculate_drawdown(equity)
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            y=equity, mode='lines', name='Portfolio Value',
            line=dict(color='#00b09b', width=2),
            hovertemplate='<b>Value: ₹%{y:,.0f}</b><extra></extra>'
        ))
        
        # Add drawdown as shaded area
        fig.add_trace(go.Scatter(
            y=drawdown * 100, fill='tozeroy', name='Drawdown %',
            line=dict(color='rgba(252, 74, 26, 0.3)', width=0),
            fillcolor='rgba(252, 74, 26, 0.2)',
            hovertemplate='<b>Drawdown: %{y:.2f}%</b><extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{title} (Max Drawdown: {max_dd*100:.2f}%)",
            xaxis_title="Time Period",
            yaxis_title="Portfolio Value (₹)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.8)",
            height=400,
            hovermode='x unified',
            font=dict(color="white")
        )
        
        return fig


class RiskReturnAnalyzer:
    """Analyze risk-return profiles of stocks/portfolios."""
    
    @staticmethod
    def calculate_stock_metrics(price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate return and volatility for a stock.
        
        Args:
            price_data: DataFrame with 'Close' column
            
        Returns:
            Dictionary with annualized return and volatility
        """
        returns = price_data['Close'].pct_change().dropna()
        if len(returns) == 0:
            return {"return": 0, "volatility": 0, "sharpe": 0}
        
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / max(annual_volatility, 1e-6)
        
        return {
            "return": annual_return,
            "volatility": annual_volatility,
            "sharpe": sharpe
        }
    
    @staticmethod
    def plot_efficient_frontier(stocks_data: Dict[str, Dict], allocations: np.ndarray = None) -> go.Figure:
        """
        Plot risk-return scatter of stocks with efficient frontier.
        
        Args:
            stocks_data: Dict with stock metrics {stock: {return, volatility, sharpe}}
            allocations: Current portfolio allocation weights
            
        Returns:
            Plotly figure
        """
        names = list(stocks_data.keys())
        returns = [stocks_data[s].get("return", 0) for s in names]
        volatilities = [stocks_data[s].get("volatility", 0) for s in names]
        sharpes = [stocks_data[s].get("sharpe", 0) for s in names]
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=[v*100 for v in volatilities],
            y=[r*100 for r in returns],
            mode='markers+text',
            text=names,
            marker=dict(
                size=[max(10, min(50, s*100)) for s in sharpes],
                color=sharpes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Risk-Return Profile (Efficient Frontier)",
            xaxis_title="Volatility (Risk) %",
            yaxis_title="Annual Return %",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.8)",
            height=500,
            font=dict(color="white")
        )
        
        return fig


class PerformanceAnalyzer:
    """Analyze and visualize performance metrics."""
    
    @staticmethod
    def plot_returns_distribution(returns: np.ndarray, title: str = "Returns Distribution") -> go.Figure:
        """Plot histogram of returns with normal distribution overlay."""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns*100, nbinsx=50, name="Returns",
            marker_color='rgba(102, 126, 234, 0.6)',
            hovertemplate='<b>Return: %{x:.2f}%</b><br>Frequency: %{y}<extra></extra>'
        ))
        
        # Normal distribution
        mu = np.mean(returns)
        sigma = np.std(returns)
        x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        normal = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mu)/sigma)**2)
        normal = normal / normal.max() * returns.shape[0]
        
        fig.add_trace(go.Scatter(
            x=x_range*100, y=normal,
            mode='lines', name='Normal Distribution',
            line=dict(color='#00b09b', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Daily Return %",
            yaxis_title="Frequency",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,35,0.8)",
            height=400,
            font=dict(color="white")
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns(daily_returns: np.ndarray, dates: pd.DatetimeIndex = None) -> go.Figure:
        """Plot monthly returns heatmap."""
        if dates is None:
            dates = pd.date_range(end=datetime.now(), periods=len(daily_returns), freq='D')
        
        returns_series = pd.Series(daily_returns, index=dates)
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create heatmap data
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        returns_matrix = monthly_returns.values.reshape(-1, 12) if len(monthly_returns) >= 12 else monthly_returns.values.reshape(1, -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=returns_matrix * 100,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(returns_matrix * 100, 2),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Return %")
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            font=dict(color="white")
        )
        
        return fig


class PortfolioOptimizer:
    """Optimize portfolio for maximum risk-adjusted returns."""
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray, returns: np.ndarray, volatilities: np.ndarray, 
                                    correlation_matrix: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights: Portfolio weights (normalized to 1.0)
            returns: Array of asset returns
            volatilities: Array of asset volatilities
            correlation_matrix: Correlation matrix of assets
            
        Returns:
            Dictionary with portfolio metrics
        """
        portfolio_return = np.sum(weights * returns)
        
        if correlation_matrix is None:
            correlation_matrix = np.eye(len(weights))
        
        # Portfolio volatility using correlation
        portfolio_variance = np.dot(weights**2, volatilities**2) + \
                            2 * np.sum([weights[i] * weights[j] * correlation_matrix[i, j] * volatilities[i] * volatilities[j] 
                                       for i in range(len(weights)-1) for j in range(i+1, len(weights))])
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        sharpe_ratio = portfolio_return / max(portfolio_volatility, 1e-6)
        
        return {
            "return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe": sharpe_ratio,
            "return_per_risk": portfolio_return / max(portfolio_volatility, 1e-6)
        }
    
    @staticmethod
    def optimize_for_profit(returns: np.ndarray, volatilities: np.ndarray, risk_tolerance: str = "medium") -> np.ndarray:
        """
        Optimize portfolio weights for maximum profit with controlled risk.
        
        Args:
            returns: Expected returns for each asset
            volatilities: Standard deviations for each asset
            risk_tolerance: 'low', 'medium', 'high'
            
        Returns:
            Optimized weights
        """
        n_assets = len(returns)
        
        # Risk-return ratio for each asset
        risk_return_ratio = returns / np.maximum(volatilities, 1e-6)
        
        if risk_tolerance == "low":
            # Conservative: favor low volatility
            weights = volatilities / np.sum(volatilities)  # Inverse volatility weighting
            weights = 1 - weights  # Invert to favor low volatility
            weights = weights / np.sum(weights)
        elif risk_tolerance == "high":
            # Aggressive: favor high returns
            weights = np.maximum(returns, 0)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_assets) / n_assets
        else:  # medium
            # Balanced: risk-return weighted
            positive_ratios = np.maximum(risk_return_ratio, 0.001)
            weights = positive_ratios / np.sum(positive_ratios)
            
            # Apply dynamic weighting: increase allocation to top performers
            sorted_idx = np.argsort(risk_return_ratio)[::-1]
            top_performance = sorted_idx[:max(2, n_assets//3)]
            
            weights[top_performance] *= 1.5
            weights = weights / np.sum(weights)
        
        return np.maximum(weights, 0) / np.sum(np.maximum(weights, 0))
    
    @staticmethod
    def plot_optimization_results(weights: np.ndarray, returns: np.ndarray, 
                                  names: List[str], metrics: Dict) -> go.Figure:
        """Plot optimized portfolio allocation."""
        fig = go.Figure()
        
        # Pie chart
        fig.add_trace(go.Pie(
            labels=names,
            values=weights * 100,
            hole=0.4,
            textinfo="label+percent",
            textposition="inside",
            marker=dict(line=dict(color='rgba(0,0,0,0.2)', width=2)),
            hovertemplate='<b>%{label}</b><br>Allocation: %{value:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Optimized Portfolio (Sharpe: {metrics.get('sharpe', 0):.3f}, Return: {metrics.get('return', 0)*100:.2f}%)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
            font=dict(color="white")
        )
        
        return fig


class CorrelationAnalyzer:
    """Analyze and visualize correlation structures."""
    
    @staticmethod
    def plot_correlation_heatmap(price_data: Dict[str, pd.DataFrame], title: str = "Correlation Matrix") -> go.Figure:
        """
        Plot correlation heatmap for multiple stocks.
        
        Args:
            price_data: Dict with {symbol: price_dataframe}
            title: Plot title
            
        Returns:
            Plotly heatmap figure
        """
        # Calculate returns for each stock
        returns_data = {}
        for symbol, df in price_data.items():
            if 'Close' in df.columns and len(df) > 1:
                returns_data[symbol] = df['Close'].pct_change().dropna().values
        
        if not returns_data:
            return go.Figure()
        
        # Create correlation matrix
        symbols = list(returns_data.keys())
        n = len(symbols)
        corr_matrix = np.corrcoef([returns_data[s] for s in symbols])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
            font=dict(color="white")
        )
        
        return fig
