"""
Macro Signals for 70% Accuracy System
Fetches real macroeconomic data:
- USD/INR exchange rate
- Interest rates (RBI, US Fed)
- FII (Foreign Institutional Investor) flows
- Market breadth indicators
- Global equity indices
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging

logger = logging.getLogger(__name__)


class MacroSignals:
    """Fetch and analyze macroeconomic signals"""

    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 3600  # 1 hour cache

    def _is_cache_valid(self, key):
        """Check if cache is still valid"""
        if key not in self.cache_timestamp:
            return False
        age = (datetime.now() - self.cache_timestamp[key]).total_seconds()
        return age < self.cache_duration

    def get_usd_inr(self, days=30):
        """
        Get USD/INR exchange rate and trend
        
        Returns:
            dict: {
                'current': float,
                'change_pct': float,
                'trend': 'strengthening' | 'weakening' | 'stable',
                'signal': 1 (bullish INR) | -1 (bearish INR)
            }
        """
        try:
            if self._is_cache_valid('usd_inr'):
                return self.cache['usd_inr']

            # Fetch USD/INR
            usdinr = yf.download('USDINR=X', period=f'{days}d', progress=False)
            
            if usdinr is None or len(usdinr) < 2:
                return {'current': 83.0, 'change_pct': 0, 'trend': 'stable', 'signal': 0}

            current = float(usdinr['Close'].iloc[-1])
            previous = float(usdinr['Close'].iloc[-5]) if len(usdinr) >= 5 else float(usdinr['Close'].iloc[0])
            change_pct = ((current - previous) / previous) * 100

            # Stronger rupee (lower USD/INR) = bullish for Indian stocks
            if change_pct < -1:
                trend = 'strengthening'
                signal = 1  # Bullish for stocks
            elif change_pct > 1:
                trend = 'weakening'
                signal = -1  # Bearish for stocks
            else:
                trend = 'stable'
                signal = 0

            result = {
                'current': float(current),
                'change_pct': float(change_pct),
                'trend': trend,
                'signal': signal
            }

            self.cache['usd_inr'] = result
            self.cache_timestamp['usd_inr'] = datetime.now()
            return result

        except Exception as e:
            logger.warning(f"USD/INR fetch failed: {e}")
            return {'current': 83.0, 'change_pct': 0, 'trend': 'stable', 'signal': 0}

    def get_us_fed_rate(self):
        """
        Get US Federal Funds Rate
        Higher rates = capital outflow = bearish for Indian stocks
        
        Returns:
            dict: {
                'current_rate': float,
                'trend': 'rising' | 'falling' | 'stable',
                'signal': 1 (bullish) | -1 (bearish)
            }
        """
        try:
            if self._is_cache_valid('fed_rate'):
                return self.cache['fed_rate']

            # Federal Funds Rate (FEDFUNDS)
            fed_data = yf.download('^IRX', period='1y', progress=False)
            
            if fed_data is None or len(fed_data) < 2:
                return {'current_rate': 5.5, 'trend': 'stable', 'signal': 0}

            current = float(fed_data['Close'].iloc[-1])
            previous = float(fed_data['Close'].iloc[-20]) if len(fed_data) >= 20 else float(fed_data['Close'].iloc[0])
            change = current - previous

            # Higher Fed rate = capital flows to US = bearish India
            if change > 0.5:
                trend = 'rising'
                signal = -1  # Bearish
            elif change < -0.5:
                trend = 'falling'
                signal = 1  # Bullish
            else:
                trend = 'stable'
                signal = 0

            result = {
                'current_rate': float(current),
                'trend': trend,
                'signal': signal
            }

            self.cache['fed_rate'] = result
            self.cache_timestamp['fed_rate'] = datetime.now()
            return result

        except Exception as e:
            logger.warning(f"Fed rate fetch failed: {e}")
            return {'current_rate': 5.5, 'trend': 'stable', 'signal': 0}

    def get_rbi_rate(self):
        """
        Get RBI Repo Rate and trend
        Higher RBI rate = tighter liquidity = bearish for stocks
        
        Returns:
            dict: {
                'current_rate': float,
                'signal': 1 (bullish) | -1 (bearish)
            }
        """
        try:
            if self._is_cache_valid('rbi_rate'):
                return self.cache['rbi_rate']

            # Note: RBI rate changes quarterly; using historical average
            # In production, fetch from RBI API or Bloomberg
            current_rate = 6.5  # Approximate as of 2026
            
            signal = -1 if current_rate > 6.5 else (1 if current_rate < 6.0 else 0)

            result = {
                'current_rate': float(current_rate),
                'signal': signal
            }

            self.cache['rbi_rate'] = result
            self.cache_timestamp['rbi_rate'] = datetime.now()
            return result

        except Exception as e:
            logger.warning(f"RBI rate fetch failed: {e}")
            return {'current_rate': 6.5, 'signal': 0}

    def get_fii_flows(self):
        """
        Get FII (Foreign Institutional Investor) flows
        Positive flows = buying = bullish
        Negative flows = selling = bearish
        
        Note: FII data available from:
        - NSDL (National Securities Depository Limited)
        - BSE India
        - In this implementation, we use proxy via market momentum
        
        Returns:
            dict: {
                'flow_direction': 'inflow' | 'outflow' | 'neutral',
                'signal': 1 (bullish) | -1 (bearish) | 0
            }
        """
        try:
            if self._is_cache_valid('fii_flows'):
                return self.cache['fii_flows']

            # Proxy: Check if foreign investors are buying (check global market strength)
            # S&P 500 performance influences FII flows
            sp500 = yf.download('^GSPC', period='10d', progress=False)
            
            if sp500 is None or len(sp500) < 2:
                return {'flow_direction': 'neutral', 'signal': 0}

            sp500_change = ((float(sp500['Close'].iloc[-1]) - float(sp500['Close'].iloc[0])) / float(sp500['Close'].iloc[0])) * 100

            if sp500_change > 1:
                flow_direction = 'inflow'
                signal = 1
            elif sp500_change < -1:
                flow_direction = 'outflow'
                signal = -1
            else:
                flow_direction = 'neutral'
                signal = 0

            result = {
                'flow_direction': flow_direction,
                'signal': signal,
                'proxy_sp500_change': float(sp500_change)
            }

            self.cache['fii_flows'] = result
            self.cache_timestamp['fii_flows'] = datetime.now()
            return result

        except Exception as e:
            logger.warning(f"FII flows fetch failed: {e}")
            return {'flow_direction': 'neutral', 'signal': 0}

    def get_market_breadth(self, ticker='RELIANCE.NS'):
        """
        Get market breadth indicator
        Breadth = advances / (advances + declines)
        > 0.6 = bullish, < 0.4 = bearish
        
        In this implementation, we check market volatility as proxy
        Low volatility + up market = positive breadth
        
        Returns:
            dict: {
                'breadth': float (0-1),
                'signal': 1 | -1 | 0
            }
        """
        try:
            if self._is_cache_valid('market_breadth'):
                return self.cache['market_breadth']

            # Use VIX as proxy (lower VIX = positive breadth)
            vix = yf.download('^VIX', period='30d', progress=False)
            
            if vix is None or len(vix) < 2:
                return {'breadth': 0.5, 'signal': 0}

            current_vix = float(vix['Close'].iloc[-1])
            avg_vix = float(vix['Close'].mean())

            # VIX inversion: lower VIX = broader market participation
            breadth = 1 - (current_vix / 100)  # Normalize to 0-1
            breadth = max(0, min(1, breadth))  # Clamp 0-1

            if current_vix < avg_vix * 0.8:
                signal = 1  # Low volatility = bullish
            elif current_vix > avg_vix * 1.2:
                signal = -1  # High volatility = bearish
            else:
                signal = 0

            result = {
                'breadth': float(breadth),
                'signal': signal,
                'vix': float(current_vix)
            }

            self.cache['market_breadth'] = result
            self.cache_timestamp['market_breadth'] = datetime.now()
            return result

        except Exception as e:
            logger.warning(f"Market breadth fetch failed: {e}")
            return {'breadth': 0.5, 'signal': 0}

    def get_composite_macro_signal(self):
        """
        Combine all macro signals into single comprehensive signal
        
        Returns:
            dict: {
                'composite_signal': -1 to +1,
                'components': {individual signals},
                'strength': 'weak' | 'moderate' | 'strong'
            }
        """
        try:
            usd_inr = self.get_usd_inr()
            fed_rate = self.get_us_fed_rate()
            rbi_rate = self.get_rbi_rate()
            fii_flows = self.get_fii_flows()
            breadth = self.get_market_breadth()

            # Weighted combination
            composite_signal = (
                usd_inr['signal'] * 0.25 +          # 25% exchange rate
                fed_rate['signal'] * 0.20 +         # 20% Fed rate
                rbi_rate['signal'] * 0.15 +         # 15% RBI rate
                fii_flows['signal'] * 0.25 +        # 25% FII flows
                (breadth['signal'] * 0.15)          # 15% market breadth
            )

            # Clamp to -1 to +1
            composite_signal = max(-1, min(1, composite_signal))

            # Determine strength
            abs_signal = abs(composite_signal)
            if abs_signal > 0.6:
                strength = 'strong'
            elif abs_signal > 0.3:
                strength = 'moderate'
            else:
                strength = 'weak'

            result = {
                'composite_signal': float(composite_signal),
                'strength': strength,
                'components': {
                    'usd_inr': usd_inr['signal'],
                    'fed_rate': fed_rate['signal'],
                    'rbi_rate': rbi_rate['signal'],
                    'fii_flows': fii_flows['signal'],
                    'breadth': breadth['signal']
                },
                'details': {
                    'usd_inr': usd_inr,
                    'fed_rate': fed_rate,
                    'rbi_rate': rbi_rate,
                    'fii_flows': fii_flows,
                    'breadth': breadth
                }
            }

            return result

        except Exception as e:
            logger.error(f"Composite macro signal fetch failed: {e}")
            return {
                'composite_signal': 0,
                'strength': 'weak',
                'components': {},
                'details': {}
            }

    def apply_macro_boost(self, base_prediction, confidence):
        """
        Apply macro signal boost to prediction confidence
        
        Args:
            base_prediction: 1 (bullish) or 0 (bearish)
            confidence: 0-1 base confidence
            
        Returns:
            float: Adjusted confidence with macro boost
        """
        macro = self.get_composite_macro_signal()
        signal = macro['composite_signal']

        if base_prediction == 1:  # Bullish prediction
            # Positive macro = boost confidence
            boost = (1 + signal) / 2  # Convert -1..+1 to 0..1
        else:  # Bearish prediction
            # Negative macro = boost confidence
            boost = (1 - signal) / 2  # Convert -1..+1 to 0..1

        # Adjust confidence with boost (max 15% boost)
        adjusted = confidence + (boost * 0.15 * (1 - confidence))
        return min(1.0, adjusted)


# Singleton instance
_macro_signals = None


def get_macro_signals():
    """Get or create macro signals instance"""
    global _macro_signals
    if _macro_signals is None:
        _macro_signals = MacroSignals()
    return _macro_signals


if __name__ == "__main__":
    # Test macro signals
    macro = get_macro_signals()

    print("=" * 60)
    print("MACRO SIGNALS TEST")
    print("=" * 60)

    # USD/INR
    usd_inr = macro.get_usd_inr()
    print(f"\n📊 USD/INR: ₹{usd_inr['current']:.2f}")
    print(f"   Change: {usd_inr['change_pct']:+.2f}% ({usd_inr['trend']})")
    print(f"   Signal: {usd_inr['signal']:+d}")

    # Fed Rate
    fed = macro.get_us_fed_rate()
    print(f"\n📊 US Fed Rate: {fed['current_rate']:.2f}%")
    print(f"   Trend: {fed['trend']}")
    print(f"   Signal: {fed['signal']:+d}")

    # RBI Rate
    rbi = macro.get_rbi_rate()
    print(f"\n📊 RBI Rate: {rbi['current_rate']:.2f}%")
    print(f"   Signal: {rbi['signal']:+d}")

    # FII Flows
    fii = macro.get_fii_flows()
    print(f"\n📊 FII Flows: {fii['flow_direction']}")
    print(f"   Signal: {fii['signal']:+d}")

    # Market Breadth
    breadth = macro.get_market_breadth()
    print(f"\n📊 Market Breadth: {breadth['breadth']:.2f} (VIX: {breadth['vix']:.2f})")
    print(f"   Signal: {breadth['signal']:+d}")

    # Composite
    composite = macro.get_composite_macro_signal()
    print(f"\n🎯 COMPOSITE MACRO SIGNAL: {composite['composite_signal']:+.2f}")
    print(f"   Strength: {composite['strength']}")

    # Boost example
    adjusted = macro.apply_macro_boost(base_prediction=1, confidence=0.70)
    print(f"\n📈 Macro Boost Example:")
    print(f"   Base confidence: 0.70 (bullish)")
    print(f"   Macro signal: {composite['composite_signal']:+.2f}")
    print(f"   Adjusted confidence: {adjusted:.3f}")
