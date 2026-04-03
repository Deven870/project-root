"""
Validation Framework for 70% Accuracy System
Tests predictions against real market data to validate accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class AccuracyValidator:
    """Validates prediction accuracy against real market data"""

    def __init__(self, prediction_model):
        """
        Initialize validator
        
        Args:
            prediction_model: Function that takes (features, prices, ticker) 
                            and returns (trend, confidence)
        """
        self.prediction_model = prediction_model
        self.predictions = []
        self.results = []

    def validate_timeframe(self, ticker, timeframe=5, test_days=50):
        """
        Validate predictions for a specific timeframe
        
        Args:
            ticker: Stock ticker
            timeframe: Horizon in days (5 for swing trading)
            test_days: Number of days to test
            
        Returns:
            dict: {
                'accuracy': float (0-1),
                'precision': float,
                'recall': float,
                'f1_score': float,
                'predictions': list of results
            }
        """
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=test_days + 30)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df is None or len(df) < test_days:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            predictions = []
            actuals = []

            # Generate predictions for each day
            for i in range(len(df) - timeframe - 5):
                current_price = float(df['Close'].iloc[i])
                future_price = float(df['Close'].iloc[i + timeframe])

                # Actual: did price go up?
                actual_trend = 1 if future_price > current_price else 0

                # Try to get prediction
                try:
                    # Build features from current data
                    lookback = df.iloc[max(0, i-100):i+1]
                    
                    # Simple features for demonstration
                    features = self._build_simple_features(lookback)
                    
                    # Get prediction
                    from modules.prediction_70_integration import predict_swing
                    pred_trend, confidence = predict_swing(features, lookback['Close'].values, ticker)

                    predictions.append({
                        'date': df.index[i],
                        'predicted': pred_trend,
                        'confidence': confidence,
                        'actual': actual_trend,
                        'correct': 1 if pred_trend == actual_trend else 0
                    })

                    actuals.append(actual_trend)

                except Exception as e:
                    logger.debug(f"Prediction failed for {ticker} at index {i}: {e}")
                    continue

            if not predictions:
                logger.error(f"No valid predictions for {ticker}")
                return None

            # Calculate metrics
            correct = sum(p['correct'] for p in predictions)
            accuracy = correct / len(predictions) if predictions else 0

            # Precision: TP / (TP + FP)
            predicted_1 = [p for p in predictions if p['predicted'] == 1]
            if predicted_1:
                tp = sum(1 for p in predicted_1 if p['correct'])
                precision = tp / len(predicted_1)
            else:
                precision = 0

            # Recall: TP / (TP + FN)
            actual_1 = sum(1 for p in predictions if p['actual'] == 1)
            if actual_1 > 0:
                tp = sum(1 for p in predictions if p['predicted'] == 1 and p['actual'] == 1)
                recall = tp / actual_1
            else:
                recall = 0

            # F1-Score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            result = {
                'ticker': ticker,
                'timeframe': timeframe,
                'test_days': len(predictions),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'correct': correct,
                'total': len(predictions),
                'predictions': predictions
            }

            logger.info(f"{ticker}: {accuracy:.1%} accuracy on {timeframe}-day horizon ({len(predictions)} samples)")

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Validation failed for {ticker}: {e}")
            return None

    def _build_simple_features(self, df):
        """Build simple features from price data"""
        close = df['Close'].values
        
        # Simple indicators
        ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        
        roc = ((close[-1] - close[-10]) / close[-10] * 100) if len(close) >= 10 else 0
        
        return {
            'close': close[-1],
            'ma5': ma5,
            'ma20': ma20,
            'roc': roc,
            'days_tested': len(close)
        }

    def validate_multi_ticker(self, tickers, timeframe=5):
        """
        Validate across multiple stocks
        
        Args:
            tickers: List of stock tickers
            timeframe: Prediction horizon in days
            
        Returns:
            dict: Summary statistics
        """
        results = []
        
        for ticker in tickers:
            result = self.validate_timeframe(ticker, timeframe=timeframe)
            if result:
                results.append(result)

        if not results:
            logger.warning("No validation results generated")
            return None

        # Summary
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])

        summary = {
            'tickers': len(results),
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'timeframe': timeframe,
            'details': results
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION SUMMARY ({timeframe}-day horizon)")
        logger.info(f"{'='*60}")
        logger.info(f"Stocks tested: {len(results)}")
        logger.info(f"Avg Accuracy: {avg_accuracy:.1%}")
        logger.info(f"Avg Precision: {avg_precision:.1%}")
        logger.info(f"Avg Recall: {avg_recall:.1%}")
        logger.info(f"Avg F1-Score: {avg_f1:.2f}")

        return summary

    def get_target_achievement(self, target_accuracy=0.70, target_f1=0.65):
        """Check if validation meets accuracy targets"""
        if not self.results:
            return {'achieved': False, 'reason': 'No results'}

        avg_accuracy = np.mean([r['accuracy'] for r in self.results])
        avg_f1 = np.mean([r['f1_score'] for r in self.results])

        achieved = avg_accuracy >= target_accuracy and avg_f1 >= target_f1

        return {
            'achieved': achieved,
            'accuracy': avg_accuracy,
            'target_accuracy': target_accuracy,
            'f1_score': avg_f1,
            'target_f1': target_f1,
            'status': '✓ TARGET MET' if achieved else '✗ TARGET NOT MET'
        }


class BacktestEngine:
    """Backtests trading strategy on historical data"""

    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.trades = []
        self.daily_pnl = []

    def backtest_ticker(self, ticker, start_date, end_date, prediction_func, 
                       timeframe=5, position_size=0.1):
        """
        Backtest strategy on single ticker
        
        Args:
            ticker: Stock ticker
            start_date: Backtest start date
            end_date: Backtest end date
            prediction_func: Function that returns (trend, confidence)
            timeframe: Holding period in days
            position_size: Position size as % of capital
            
        Returns:
            dict: Backtest results
        """
        try:
            # Fetch data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df is None or len(df) < timeframe + 5:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            capital = self.initial_capital
            position = None
            trades = []

            # Simulate each day
            for i in range(len(df) - timeframe):
                date = df.index[i]
                current_price = float(df['Close'].iloc[i])

                # Check if we need to close existing position
                if position:
                    hold_days = (date - position['entry_date']).days
                    
                    if hold_days >= timeframe:
                        # Close position
                        exit_price = current_price
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                        pnl_pct = (pnl / (position['quantity'] * position['entry_price'])) * 100

                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct
                        })

                        capital += pnl
                        position = None

                # Generate prediction
                try:
                    lookback = df.iloc[max(0, i-50):i+1]
                    features = self._build_features(lookback)
                    
                    pred_trend, confidence = prediction_func(features, lookback['Close'].values, ticker)

                    # Enter trade if no position and high confidence
                    if not position and confidence > 0.65:
                        quantity = int((capital * position_size) / current_price)
                        
                        if quantity > 0 and pred_trend == 1:  # Bullish
                            position = {
                                'entry_date': date,
                                'entry_price': current_price,
                                'quantity': quantity,
                                'trend': pred_trend
                            }

                except Exception as e:
                    logger.debug(f"Prediction error: {e}")
                    continue

            # Close any remaining position
            if position:
                exit_price = float(df['Close'].iloc[-1])
                pnl = (exit_price - position['entry_price']) * position['quantity']
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': df.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position['quantity'] * position['entry_price'])) * 100
                })
                capital += pnl

            # Calculate metrics
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                winning_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(trades) * 100

                result = {
                    'ticker': ticker,
                    'trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'final_capital': capital,
                    'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
                    'trades_detail': trades
                }

                logger.info(f"{ticker}: {len(trades)} trades, "
                           f"{win_rate:.0f}% win rate, ${total_pnl:+.0f} P&L")
                return result

            else:
                logger.warning(f"No trades executed for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")
            return None

    def _build_features(self, df):
        """Build features from price data"""
        close = df['Close'].values
        
        ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        
        return {
            'close': close[-1],
            'ma5': ma5,
            'ma20': ma20,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example: Validate accuracy
    print("\n" + "=" * 60)
    print("ACCURACY VALIDATION")
    print("=" * 60)

    validator = AccuracyValidator(prediction_model=None)

    # Validate on NSE stocks
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    summary = validator.validate_multi_ticker(tickers, timeframe=5)

    if summary:
        print(f"\nValidation achieved {summary['avg_accuracy']:.1%} accuracy")

    # Check targets
    target = validator.get_target_achievement(target_accuracy=0.70)
    print(f"\n{target['status']}")
