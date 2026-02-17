"""
Advanced Technical Signals Module

Implements popular TradingView-style indicators for Indian markets:
- MACD (Moving Average Convergence Divergence)
- Williams %R with Trend Exhaustion
- Williams VixFix (Volatility bottom detector)
- Hull Moving Average Suite
- Laguerre RSI/Filter
- Supertrend
- RSI with divergence detection
- Ichimoku Cloud

These indicators are combined to generate buy/sell signals with
profit targets based on holding period.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class FundamentalData:
    """Fundamental/news data for a stock."""
    has_recent_news: bool = False
    catalysts: List[Dict[str, Any]] = field(default_factory=list)
    news_sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    news_score: float = 0.0  # -100 to +100
    earnings_surprise: Optional[str] = None  # BEAT, MISS, INLINE
    pead_score: float = 0.0  # -100 to +100
    recent_headlines: List[str] = field(default_factory=list)
    sentiment_method: str = "Keyword"  # FinBERT, VADER, or Keyword

    # Institutional data
    fii_sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    dii_sentiment: str = "NEUTRAL"
    institutional_score: float = 0.0  # -50 to +50
    bulk_deals: List[Dict[str, Any]] = field(default_factory=list)
    insider_activity: str = "NEUTRAL"  # BUYING, SELLING, NEUTRAL

    # Policy/Macro impact
    has_policy_boost: bool = False
    policy_score: float = 0.0  # -50 to +50
    affected_sectors: List[str] = field(default_factory=list)
    relevant_policies: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics (fundamental screener criteria)
    quality_score: float = 0.0  # 0 to 100
    passes_quality: bool = True  # Passes minimum quality checks
    eps_growth_3y: Optional[float] = None  # 3-year EPS CAGR %
    revenue_growth_3y: Optional[float] = None  # 3-year revenue CAGR %
    peg_ratio: Optional[float] = None  # P/E divided by growth rate
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity %
    profit_margin: Optional[float] = None  # Net profit margin %
    market_cap_cr: Optional[float] = None  # Market cap in Crores
    quality_flags: List[str] = field(default_factory=list)  # Quality issues/highlights

    def to_dict(self) -> Dict:
        return {
            'has_recent_news': self.has_recent_news,
            'catalysts': self.catalysts,
            'news_sentiment': self.news_sentiment,
            'news_score': round(self.news_score, 1),
            'earnings_surprise': self.earnings_surprise,
            'pead_score': round(self.pead_score, 1),
            'recent_headlines': self.recent_headlines[:3],
            'sentiment_method': self.sentiment_method,
            # Institutional
            'fii_sentiment': self.fii_sentiment,
            'dii_sentiment': self.dii_sentiment,
            'institutional_score': round(self.institutional_score, 1),
            'bulk_deals': self.bulk_deals[:3],
            'insider_activity': self.insider_activity,
            # Policy
            'has_policy_boost': self.has_policy_boost,
            'policy_score': round(self.policy_score, 1),
            'affected_sectors': self.affected_sectors,
            'relevant_policies': self.relevant_policies[:3],
            # Quality
            'quality_score': round(self.quality_score, 1),
            'passes_quality': self.passes_quality,
            'peg_ratio': round(self.peg_ratio, 2) if self.peg_ratio else None,
            'roe': round(self.roe, 1) if self.roe else None,
            'debt_to_equity': round(self.debt_to_equity, 2) if self.debt_to_equity else None,
            'quality_flags': self.quality_flags,
        }


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    ticker: str
    timestamp: str
    direction: SignalDirection
    confidence: float  # 0-100

    # Price info
    current_price: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float

    # Expected returns
    expected_return_pct: float
    risk_reward_ratio: float

    # Holding period
    suggested_hold_days: int

    # Individual indicator signals
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Analysis
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    # Fundamental data (optional)
    fundamental: Optional[FundamentalData] = None

    # Combined score (technical + fundamental)
    combined_score: float = 0.0

    # Enhanced metadata (regime, momentum, volume, RS, MTA, etc.)
    regime: str = ""
    regime_adx: float = 0.0
    momentum_grade: str = ""
    momentum_score: float = 0.0
    volume_ratio: float = 0.0
    volume_confirmed: bool = False
    rs_mrs: float = 0.0
    rs_outperforming: bool = False
    weekly_trend: str = ""
    squeeze_active: bool = False
    breakout_detected: bool = False
    price_action: str = ""

    def to_dict(self) -> Dict:
        result = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'direction': self.direction.value,
            'confidence': round(self.confidence, 1),
            'combined_score': round(self.combined_score, 1),
            'current_price': round(self.current_price, 2),
            'entry_price': round(self.entry_price, 2),
            'stop_loss': round(self.stop_loss, 2),
            'targets': {
                'target_1': round(self.target_1, 2),
                'target_2': round(self.target_2, 2),
                'target_3': round(self.target_3, 2),
            },
            'expected_return_pct': round(self.expected_return_pct, 2),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'suggested_hold_days': self.suggested_hold_days,
            'indicators': self.indicators,
            'reasons': self.reasons,
            'risks': self.risks,
            'regime': self.regime,
            'regime_adx': self.regime_adx,
            'momentum_grade': self.momentum_grade,
            'momentum_score': self.momentum_score,
            'volume_ratio': self.volume_ratio,
            'volume_confirmed': self.volume_confirmed,
            'rs_mrs': self.rs_mrs,
            'rs_outperforming': self.rs_outperforming,
            'weekly_trend': self.weekly_trend,
            'squeeze_active': self.squeeze_active,
            'breakout_detected': self.breakout_detected,
            'price_action': self.price_action,
        }
        if self.fundamental:
            result['fundamental'] = self.fundamental.to_dict()
        return result


class AdvancedIndicators:
    """
    Advanced technical indicators implementation.

    Includes TradingView-style indicators optimized for Indian markets.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # =========================================================================
    # MACD - Moving Average Convergence Divergence
    # =========================================================================

    def macd(
        self,
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD indicator.

        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def macd_signal(self, close: np.ndarray) -> Dict[str, Any]:
        """Generate MACD trading signal."""
        macd_line, signal_line, histogram = self.macd(close)

        # Current values
        curr_macd = macd_line[-1]
        curr_signal = signal_line[-1]
        curr_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0

        # Signal detection
        bullish_cross = prev_hist <= 0 and curr_hist > 0
        bearish_cross = prev_hist >= 0 and curr_hist < 0

        signal = "NEUTRAL"
        strength = 0

        if bullish_cross:
            signal = "BUY"
            strength = min(abs(curr_hist) * 100, 100)
        elif bearish_cross:
            signal = "SELL"
            strength = min(abs(curr_hist) * 100, 100)
        elif curr_hist > 0 and curr_hist > prev_hist:
            signal = "BULLISH"
            strength = 50
        elif curr_hist < 0 and curr_hist < prev_hist:
            signal = "BEARISH"
            strength = 50

        return {
            'signal': signal,
            'strength': strength,
            'macd': round(curr_macd, 4),
            'signal_line': round(curr_signal, 4),
            'histogram': round(curr_hist, 4),
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross,
        }

    # =========================================================================
    # Williams %R with Trend Exhaustion
    # =========================================================================

    def williams_r(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Calculate Williams %R indicator.

        Range: -100 to 0
        - Above -20: Overbought
        - Below -80: Oversold
        """
        result = np.full(len(close), np.nan)

        for i in range(period - 1, len(close)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])

            if highest_high != lowest_low:
                result[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                result[i] = -50

        return result

    def williams_r_exhaustion(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
        exhaustion_periods: int = 3
    ) -> Dict[str, Any]:
        """
        Williams %R with Trend Exhaustion detection.

        Catches tops/bottoms by detecting when %R stays in
        extreme zones for extended periods then reverses.
        """
        wr = self.williams_r(high, low, close, period)

        # Current value
        curr_wr = wr[-1]

        # Check for exhaustion patterns
        recent_wr = wr[-exhaustion_periods:] if len(wr) >= exhaustion_periods else wr

        # Oversold exhaustion (potential bottom)
        oversold_exhaustion = all(w < -80 for w in recent_wr[:-1]) and curr_wr > -80

        # Overbought exhaustion (potential top)
        overbought_exhaustion = all(w > -20 for w in recent_wr[:-1]) and curr_wr < -20

        signal = "NEUTRAL"
        strength = 0

        if oversold_exhaustion:
            signal = "BUY"
            strength = 80
        elif overbought_exhaustion:
            signal = "SELL"
            strength = 80
        elif curr_wr < -80:
            signal = "OVERSOLD"
            strength = 60
        elif curr_wr > -20:
            signal = "OVERBOUGHT"
            strength = 60

        return {
            'signal': signal,
            'strength': strength,
            'williams_r': round(curr_wr, 2),
            'oversold_exhaustion': oversold_exhaustion,
            'overbought_exhaustion': overbought_exhaustion,
            'zone': 'oversold' if curr_wr < -80 else ('overbought' if curr_wr > -20 else 'neutral'),
        }

    # =========================================================================
    # Williams VixFix - Volatility Bottom Detector
    # =========================================================================

    def williams_vix_fix(
        self,
        close: np.ndarray,
        low: np.ndarray,
        period: int = 22,
        bband_length: int = 20,
        bband_mult: float = 2.0
    ) -> Dict[str, Any]:
        """
        CM_Williams_Vix_Fix indicator.

        Detects market bottoms using implied volatility logic.
        Great for stocks on higher timeframes.
        """
        # Calculate VixFix
        highest_close = np.full(len(close), np.nan)
        for i in range(period - 1, len(close)):
            highest_close[i] = np.max(close[i - period + 1:i + 1])

        vix_fix = ((highest_close - low) / highest_close) * 100

        # Bollinger Bands on VixFix
        vix_sma = self._sma(vix_fix, bband_length)
        vix_std = self._rolling_std(vix_fix, bband_length)
        upper_band = vix_sma + (bband_mult * vix_std)

        # Current values
        curr_vix = vix_fix[-1]
        curr_upper = upper_band[-1]

        # Signal: VixFix above upper band = potential bottom
        is_bottom_signal = curr_vix > curr_upper if not np.isnan(curr_upper) else False

        signal = "NEUTRAL"
        strength = 0

        if is_bottom_signal:
            signal = "BUY"
            strength = 75
        elif curr_vix > 20:  # High volatility
            signal = "CAUTION"
            strength = 50

        return {
            'signal': signal,
            'strength': strength,
            'vix_fix': round(curr_vix, 2) if not np.isnan(curr_vix) else 0,
            'upper_band': round(curr_upper, 2) if not np.isnan(curr_upper) else 0,
            'is_bottom_signal': is_bottom_signal,
        }

    # =========================================================================
    # Hull Moving Average Suite
    # =========================================================================

    def hull_ma(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Hull Moving Average - Trend identification on steroids.

        More responsive than traditional MAs while reducing lag.
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        wma_half = self._wma(close, half_period)
        wma_full = self._wma(close, period)

        raw_hull = 2 * wma_half - wma_full
        hull = self._wma(raw_hull, sqrt_period)

        return hull

    def hull_suite_signal(
        self,
        close: np.ndarray,
        period: int = 20
    ) -> Dict[str, Any]:
        """
        Hull Suite signal generation.

        Identifies trend direction and strength.
        """
        hull = self.hull_ma(close, period)

        curr_hull = hull[-1]
        prev_hull = hull[-2] if len(hull) > 1 else curr_hull
        curr_price = close[-1]

        # Trend direction
        hull_rising = curr_hull > prev_hull
        price_above_hull = curr_price > curr_hull

        signal = "NEUTRAL"
        strength = 0

        if hull_rising and price_above_hull:
            signal = "BUY"
            strength = 70
        elif not hull_rising and not price_above_hull:
            signal = "SELL"
            strength = 70
        elif hull_rising:
            signal = "BULLISH"
            strength = 50
        elif not hull_rising:
            signal = "BEARISH"
            strength = 50

        return {
            'signal': signal,
            'strength': strength,
            'hull_ma': round(curr_hull, 2),
            'hull_rising': hull_rising,
            'price_above_hull': price_above_hull,
            'trend': 'UP' if hull_rising else 'DOWN',
        }

    # =========================================================================
    # Laguerre RSI/Filter
    # =========================================================================

    def laguerre_rsi(
        self,
        close: np.ndarray,
        gamma: float = 0.7
    ) -> np.ndarray:
        """
        Laguerre RSI - Better moving average filter.

        Uses Laguerre polynomials for smoother signals.
        """
        l0 = np.zeros(len(close))
        l1 = np.zeros(len(close))
        l2 = np.zeros(len(close))
        l3 = np.zeros(len(close))
        lrsi = np.zeros(len(close))

        for i in range(1, len(close)):
            l0[i] = (1 - gamma) * close[i] + gamma * l0[i-1]
            l1[i] = -gamma * l0[i] + l0[i-1] + gamma * l1[i-1]
            l2[i] = -gamma * l1[i] + l1[i-1] + gamma * l2[i-1]
            l3[i] = -gamma * l2[i] + l2[i-1] + gamma * l3[i-1]

            cu = 0.0
            cd = 0.0

            if l0[i] >= l1[i]:
                cu += l0[i] - l1[i]
            else:
                cd += l1[i] - l0[i]

            if l1[i] >= l2[i]:
                cu += l1[i] - l2[i]
            else:
                cd += l2[i] - l1[i]

            if l2[i] >= l3[i]:
                cu += l2[i] - l3[i]
            else:
                cd += l3[i] - l2[i]

            if cu + cd != 0:
                lrsi[i] = cu / (cu + cd) * 100
            else:
                lrsi[i] = 50

        return lrsi

    def laguerre_signal(self, close: np.ndarray, gamma: float = 0.7) -> Dict[str, Any]:
        """Generate Laguerre RSI signal."""
        lrsi = self.laguerre_rsi(close, gamma)

        curr_lrsi = lrsi[-1]
        prev_lrsi = lrsi[-2] if len(lrsi) > 1 else curr_lrsi

        # Crossover detection
        cross_up = prev_lrsi < 20 and curr_lrsi >= 20
        cross_down = prev_lrsi > 80 and curr_lrsi <= 80

        signal = "NEUTRAL"
        strength = 0

        if cross_up:
            signal = "BUY"
            strength = 75
        elif cross_down:
            signal = "SELL"
            strength = 75
        elif curr_lrsi < 20:
            signal = "OVERSOLD"
            strength = 60
        elif curr_lrsi > 80:
            signal = "OVERBOUGHT"
            strength = 60

        return {
            'signal': signal,
            'strength': strength,
            'laguerre_rsi': round(curr_lrsi, 2),
            'cross_up': cross_up,
            'cross_down': cross_down,
        }

    # =========================================================================
    # Supertrend
    # =========================================================================

    def supertrend(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Supertrend indicator - Excellent for trailing stops.

        Returns: (supertrend_line, direction)
        direction: 1 = uptrend, -1 = downtrend
        """
        atr = self._atr(high, low, close, period)

        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = np.zeros(len(close))
        direction = np.zeros(len(close))

        supertrend[0] = upper_band[0]
        direction[0] = 1

        for i in range(1, len(close)):
            if close[i] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1]) if direction[i-1] == 1 else lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1]) if direction[i-1] == -1 else upper_band[i]
                direction[i] = -1

        return supertrend, direction

    def supertrend_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Dict[str, Any]:
        """Generate Supertrend signal."""
        st, direction = self.supertrend(high, low, close, period, multiplier)

        curr_st = st[-1]
        curr_dir = direction[-1]
        prev_dir = direction[-2] if len(direction) > 1 else curr_dir
        curr_price = close[-1]

        # Trend change detection
        trend_change_up = prev_dir == -1 and curr_dir == 1
        trend_change_down = prev_dir == 1 and curr_dir == -1

        signal = "NEUTRAL"
        strength = 0

        if trend_change_up:
            signal = "BUY"
            strength = 80
        elif trend_change_down:
            signal = "SELL"
            strength = 80
        elif curr_dir == 1:
            signal = "BULLISH"
            strength = 60
        else:
            signal = "BEARISH"
            strength = 60

        return {
            'signal': signal,
            'strength': strength,
            'supertrend': round(curr_st, 2),
            'direction': 'UP' if curr_dir == 1 else 'DOWN',
            'trend_change_up': trend_change_up,
            'trend_change_down': trend_change_down,
            'trailing_stop': round(curr_st, 2),
        }

    # =========================================================================
    # RSI with Divergence
    # =========================================================================

    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))

        # Initial SMA
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])

        # EMA for rest
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period

        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def rsi_divergence_signal(
        self,
        close: np.ndarray,
        period: int = 14,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        RSI with divergence detection.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        rsi_values = self.rsi(close, period)

        curr_rsi = rsi_values[-1]

        # Find recent lows/highs for divergence
        recent_close = close[-lookback:]
        recent_rsi = rsi_values[-lookback:]

        # Simple divergence detection
        price_lower_low = close[-1] < np.min(close[-lookback:-1])
        rsi_higher_low = curr_rsi > np.min(rsi_values[-lookback:-1])
        bullish_div = price_lower_low and rsi_higher_low and curr_rsi < 40

        price_higher_high = close[-1] > np.max(close[-lookback:-1])
        rsi_lower_high = curr_rsi < np.max(rsi_values[-lookback:-1])
        bearish_div = price_higher_high and rsi_lower_high and curr_rsi > 60

        signal = "NEUTRAL"
        strength = 0

        if bullish_div:
            signal = "BUY"
            strength = 70
        elif bearish_div:
            signal = "SELL"
            strength = 70
        elif curr_rsi < 30:
            signal = "OVERSOLD"
            strength = 60
        elif curr_rsi > 70:
            signal = "OVERBOUGHT"
            strength = 60

        return {
            'signal': signal,
            'strength': strength,
            'rsi': round(curr_rsi, 2),
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'zone': 'oversold' if curr_rsi < 30 else ('overbought' if curr_rsi > 70 else 'neutral'),
        }

    # =========================================================================
    # Ichimoku Cloud
    # =========================================================================

    def ichimoku(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52
    ) -> Dict[str, np.ndarray]:
        """
        Ichimoku Cloud indicator.

        Returns dict with:
        - tenkan_sen (Conversion Line)
        - kijun_sen (Base Line)
        - senkou_span_a (Leading Span A)
        - senkou_span_b (Leading Span B)
        - chikou_span (Lagging Span)
        """
        def donchian(h, l, period):
            result = np.full(len(h), np.nan)
            for i in range(period - 1, len(h)):
                highest = np.max(h[i - period + 1:i + 1])
                lowest = np.min(l[i - period + 1:i + 1])
                result[i] = (highest + lowest) / 2
            return result

        tenkan_sen = donchian(high, low, tenkan)
        kijun_sen = donchian(high, low, kijun)

        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = donchian(high, low, senkou_b)

        chikou_span = np.roll(close, -kijun)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
        }

    def ichimoku_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, Any]:
        """Generate Ichimoku trading signal."""
        ichi = self.ichimoku(high, low, close)

        curr_price = close[-1]
        curr_tenkan = ichi['tenkan_sen'][-1]
        curr_kijun = ichi['kijun_sen'][-1]
        curr_span_a = ichi['senkou_span_a'][-1]
        curr_span_b = ichi['senkou_span_b'][-1]

        prev_tenkan = ichi['tenkan_sen'][-2] if len(ichi['tenkan_sen']) > 1 else curr_tenkan
        prev_kijun = ichi['kijun_sen'][-2] if len(ichi['kijun_sen']) > 1 else curr_kijun

        # Cloud boundaries
        cloud_top = max(curr_span_a, curr_span_b)
        cloud_bottom = min(curr_span_a, curr_span_b)

        # Signal detection
        tk_cross_up = prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun
        tk_cross_down = prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun

        above_cloud = curr_price > cloud_top
        below_cloud = curr_price < cloud_bottom
        in_cloud = not above_cloud and not below_cloud

        signal = "NEUTRAL"
        strength = 0

        if tk_cross_up and above_cloud:
            signal = "STRONG_BUY"
            strength = 90
        elif tk_cross_up:
            signal = "BUY"
            strength = 70
        elif tk_cross_down and below_cloud:
            signal = "STRONG_SELL"
            strength = 90
        elif tk_cross_down:
            signal = "SELL"
            strength = 70
        elif above_cloud and curr_tenkan > curr_kijun:
            signal = "BULLISH"
            strength = 60
        elif below_cloud and curr_tenkan < curr_kijun:
            signal = "BEARISH"
            strength = 60

        return {
            'signal': signal,
            'strength': strength,
            'tenkan_sen': round(curr_tenkan, 2) if not np.isnan(curr_tenkan) else 0,
            'kijun_sen': round(curr_kijun, 2) if not np.isnan(curr_kijun) else 0,
            'cloud_top': round(cloud_top, 2) if not np.isnan(cloud_top) else 0,
            'cloud_bottom': round(cloud_bottom, 2) if not np.isnan(cloud_bottom) else 0,
            'tk_cross_up': tk_cross_up,
            'tk_cross_down': tk_cross_down,
            'position': 'above_cloud' if above_cloud else ('below_cloud' if below_cloud else 'in_cloud'),
        }

    # =========================================================================
    # Market Regime Detection
    # =========================================================================

    def adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ADX (Average Directional Index) with +DI and -DI.

        Returns: (adx, plus_di, minus_di)
        """
        n = len(close)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        atr = self._atr(high, low, close, period)
        smooth_plus_dm = self._ema(plus_dm, period)
        smooth_minus_dm = self._ema(minus_dm, period)

        safe_atr = np.maximum(atr, 1e-10)
        plus_di = np.where(atr > 0, (smooth_plus_dm / safe_atr) * 100, 0)
        minus_di = np.where(atr > 0, (smooth_minus_dm / safe_atr) * 100, 0)

        di_sum = plus_di + minus_di
        dx = np.where(
            di_sum > 0,
            np.abs(plus_di - minus_di) / np.maximum(di_sum, 1e-10) * 100,
            0
        )
        adx_values = self._ema(dx, period)

        return adx_values, plus_di, minus_di

    def detect_market_regime(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect market regime using ADX + Bollinger Band width percentile.

        Regimes:
        - TRENDING: ADX > 25 and BB width above median
        - RANGING: ADX < 20
        - LOW_VOLATILITY: ADX < 20 and BB width below 25th percentile (squeeze)
        - HIGH_VOLATILITY: BB width above 80th percentile
        - TRANSITIONAL: Everything else
        """
        adx_vals, plus_di, minus_di = self.adx(high, low, close)
        adx_current = adx_vals[-1]

        # Bollinger Band width percentile
        bb_mid = self._sma(close, 20)
        bb_std = self._rolling_std(close, 20)

        bb_widths = []
        for i in range(max(len(close) - 60, 20), len(close)):
            if not np.isnan(bb_mid[i]) and bb_mid[i] > 0 and not np.isnan(bb_std[i]):
                bb_widths.append((2 * bb_std[i]) / bb_mid[i] * 100)

        if not bb_widths:
            return {'regime': 'TRANSITIONAL', 'adx': adx_current, 'bb_width_percentile': 50}

        current_bb_width = bb_widths[-1] if bb_widths else 0
        bb_percentile = sum(1 for w in bb_widths if w < current_bb_width) / len(bb_widths) * 100

        if adx_current > 25 and bb_percentile > 50:
            regime = "TRENDING"
        elif adx_current < 20 and bb_percentile < 25:
            regime = "LOW_VOLATILITY"
        elif adx_current < 20:
            regime = "RANGING"
        elif bb_percentile > 80:
            regime = "HIGH_VOLATILITY"
        else:
            regime = "TRANSITIONAL"

        return {
            'regime': regime,
            'adx': round(adx_current, 1),
            'adx_rising': adx_vals[-1] > adx_vals[-5] if len(adx_vals) > 5 else False,
            'plus_di': round(plus_di[-1], 1),
            'minus_di': round(minus_di[-1], 1),
            'bb_width_percentile': round(bb_percentile, 1),
            'trend_direction': 'UP' if plus_di[-1] > minus_di[-1] else 'DOWN',
        }

    # =========================================================================
    # Volume Analysis
    # =========================================================================

    def on_balance_volume(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume — cumulative volume flow indicator."""
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        return obv

    def accumulation_distribution(
        self, high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """Accumulation/Distribution Line."""
        hl_range = high - low
        numerator = (close - low) - (high - close)
        mfm = np.divide(
            numerator, hl_range,
            out=np.zeros_like(numerator, dtype=float),
            where=hl_range != 0
        )
        mfv = mfm * volume
        return np.cumsum(mfv)

    def volume_analysis(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Comprehensive volume analysis for signal confirmation.

        Checks: volume ratio, OBV trend, A/D trend, OBV divergence.
        """
        vol_sma = np.mean(volume[-lookback:]) if len(volume) >= lookback else np.mean(volume)
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 0

        obv = self.on_balance_volume(close, volume)
        obv_rising = obv[-1] > obv[-5] if len(obv) > 5 else False

        ad = self.accumulation_distribution(high, low, close, volume)
        ad_rising = ad[-1] > ad[-5] if len(ad) > 5 else False

        # OBV divergence detection
        price_higher = close[-1] > close[-lookback] if len(close) > lookback else False
        obv_higher = obv[-1] > obv[-lookback] if len(obv) > lookback else False
        bearish_divergence = price_higher and not obv_higher
        bullish_divergence = (not price_higher) and obv_higher

        # Volume trend (3-bar)
        vol_expanding = (volume[-1] > volume[-2] > volume[-3]) if len(volume) > 3 else False

        score = 0
        if vol_ratio >= 2.0:
            score += 30
        elif vol_ratio >= 1.5:
            score += 20
        elif vol_ratio >= 1.0:
            score += 10

        if obv_rising:
            score += 20
        if ad_rising:
            score += 15
        if vol_expanding:
            score += 10
        if bearish_divergence:
            score -= 25
        if bullish_divergence:
            score += 15

        return {
            'volume_ratio': round(vol_ratio, 2),
            'volume_confirmed': vol_ratio >= 1.5,
            'obv_rising': obv_rising,
            'ad_rising': ad_rising,
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'vol_expanding': vol_expanding,
            'volume_score': max(min(score, 100), -50),
        }

    # =========================================================================
    # Relative Strength (Mansfield RS)
    # =========================================================================

    def mansfield_relative_strength(
        self,
        stock_close: np.ndarray,
        index_close: np.ndarray,
        period: int = 200
    ) -> Dict[str, Any]:
        """
        Mansfield Relative Strength — stock performance vs benchmark.

        MRS > 0: outperforming index.  MRS < 0: underperforming.
        MRS rising: strengthening.  MRS falling: weakening.
        """
        min_len = min(len(stock_close), len(index_close))
        if min_len < period + 5:
            return {'mrs': 0.0, 'mrs_trend': 'UNKNOWN', 'outperforming': False}

        stock = stock_close[-min_len:]
        index = index_close[-min_len:]

        rp = stock / index * 100
        rp_sma = self._sma(rp, period)

        mrs_current = ((rp[-1] / rp_sma[-1]) - 1) * 100 if rp_sma[-1] > 0 and not np.isnan(rp_sma[-1]) else 0
        mrs_prev = ((rp[-5] / rp_sma[-5]) - 1) * 100 if rp_sma[-5] > 0 and not np.isnan(rp_sma[-5]) else mrs_current

        return {
            'mrs': round(mrs_current, 2),
            'mrs_trend': 'RISING' if mrs_current > mrs_prev else 'FALLING',
            'outperforming': mrs_current > 0,
        }

    # =========================================================================
    # Momentum Quality Scoring
    # =========================================================================

    def momentum_quality(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray
    ) -> Dict[str, Any]:
        """
        Assess momentum quality: ADX strength, ROC acceleration,
        MACD histogram expansion, volume alignment, RSI divergence.

        Returns score 0-100 and grade A/B/C/D.
        """
        score = 0
        details = {}

        # ADX trend strength (0-30 points)
        adx_vals, plus_di, minus_di = self.adx(high, low, close)
        adx_current = adx_vals[-1]
        adx_rising = adx_vals[-1] > adx_vals[-5] if len(adx_vals) > 5 else False

        if adx_current > 40:
            score += 25
            details['adx'] = f"Very strong ({adx_current:.0f})"
        elif adx_current > 25:
            score += 20
            details['adx'] = f"Strong ({adx_current:.0f})"
        elif adx_current > 20:
            score += 10
            details['adx'] = f"Moderate ({adx_current:.0f})"
        else:
            details['adx'] = f"Weak ({adx_current:.0f})"

        if adx_rising and adx_current > 20:
            score += 5

        # ROC acceleration (0-25 points)
        if len(close) > 16:
            roc_10 = (close[-1] / close[-11] - 1) * 100
            roc_10_prev = (close[-6] / close[-16] - 1) * 100
            roc_accel = roc_10 - roc_10_prev

            if roc_10 > 0 and roc_accel > 0:
                score += 25
                details['roc'] = f"Accelerating bullish ({roc_10:.1f}%)"
            elif roc_10 > 0:
                score += 15
                details['roc'] = f"Decelerating bullish ({roc_10:.1f}%)"
            elif roc_10 < 0 and roc_accel < 0:
                details['roc'] = f"Accelerating bearish ({roc_10:.1f}%)"
            else:
                score += 10
                details['roc'] = f"Mixed ({roc_10:.1f}%)"

        # MACD histogram expansion (0-20 points)
        _, _, hist = self.macd(close)
        if len(hist) > 3:
            hist_expanding = abs(hist[-1]) > abs(hist[-2]) > abs(hist[-3])
            if hist_expanding and hist[-1] > 0:
                score += 20
                details['macd_hist'] = "Expanding bullish"
            elif hist_expanding and hist[-1] < 0:
                score += 5
                details['macd_hist'] = "Expanding bearish"
            else:
                score += 10
                details['macd_hist'] = "Contracting"

        # Volume alignment (0-15 points)
        vol_sma = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 0
        obv = self.on_balance_volume(close, volume)
        price_dir = 1 if close[-1] > close[-10] else -1 if len(close) > 10 else 0
        obv_dir = 1 if obv[-1] > obv[-10] else -1 if len(obv) > 10 else 0

        if price_dir == obv_dir and vol_ratio > 1.0:
            score += 15
            details['vol_align'] = "Confirming"
        elif price_dir == obv_dir:
            score += 10
            details['vol_align'] = "Weak confirming"
        else:
            details['vol_align'] = "DIVERGING"

        # RSI divergence check (0-10 points)
        rsi_vals = self.rsi(close)
        if len(rsi_vals) > 20:
            price_up = close[-1] > close[-20]
            rsi_up = rsi_vals[-1] > rsi_vals[-20]
            if price_up == rsi_up:
                score += 10
                details['divergence'] = "None"
            else:
                details['divergence'] = "RSI divergence (caution)"

        grade = 'A' if score >= 80 else ('B' if score >= 60 else ('C' if score >= 40 else 'D'))

        return {
            'momentum_score': score,
            'grade': grade,
            'details': details,
            'tradeable': score >= 40,
        }

    # =========================================================================
    # Structure-Based Support/Resistance
    # =========================================================================

    def find_swing_highs(self, high: np.ndarray, lookback: int = 5) -> List[float]:
        """Find swing high points (pivot highs)."""
        swings = []
        for i in range(lookback, len(high) - lookback):
            if high[i] == np.max(high[i - lookback:i + lookback + 1]):
                swings.append(float(high[i]))
        return swings

    def find_swing_lows(self, low: np.ndarray, lookback: int = 5) -> List[float]:
        """Find swing low points (pivot lows)."""
        swings = []
        for i in range(lookback, len(low) - lookback):
            if low[i] == np.min(low[i - lookback:i + lookback + 1]):
                swings.append(float(low[i]))
        return swings

    def calculate_structure_targets(
        self,
        high: np.ndarray, low: np.ndarray, close: np.ndarray,
        entry_price: float, direction: str = "BUY",
        hold_days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate targets using multiple methods and find confluence zones.

        Methods: ATR-based, swing S/R, Fibonacci extensions, pivot points.
        Returns best targets ranked by confluence count.
        """
        atr_val = self._atr(high, low, close)[-1]
        hold_mult = np.sqrt(hold_days / 7)

        swing_highs = self.find_swing_highs(high)
        swing_lows = self.find_swing_lows(low)

        all_targets = []

        # Method 1: ATR-based
        if direction == "BUY":
            all_targets.extend([
                entry_price + 1.5 * atr_val * hold_mult,
                entry_price + 2.0 * atr_val * hold_mult,
                entry_price + 3.0 * atr_val * hold_mult,
            ])
        else:
            all_targets.extend([
                entry_price - 1.5 * atr_val * hold_mult,
                entry_price - 2.0 * atr_val * hold_mult,
                entry_price - 3.0 * atr_val * hold_mult,
            ])

        # Method 2: Swing S/R levels
        if direction == "BUY":
            relevant = sorted([h for h in swing_highs if h > entry_price * 1.005])
            all_targets.extend(relevant[:4])
        else:
            relevant = sorted([l for l in swing_lows if l < entry_price * 0.995], reverse=True)
            all_targets.extend(relevant[:4])

        # Method 3: Fibonacci extensions from last swing
        if swing_lows and swing_highs:
            last_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
            last_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
            swing_range = last_high - last_low
            if swing_range > 0:
                fib_levels = [0.618, 1.0, 1.272, 1.618]
                if direction == "BUY":
                    all_targets.extend([last_high + swing_range * f for f in fib_levels])
                else:
                    all_targets.extend([last_low - swing_range * f for f in fib_levels])

        # Method 4: Pivot-based targets
        if len(high) > 2:
            pp = (high[-2] + low[-2] + close[-2]) / 3
            r = high[-2] - low[-2]
            if direction == "BUY":
                all_targets.extend([pp + r * 0.382, pp + r * 0.618, pp + r * 1.0])
            else:
                all_targets.extend([pp - r * 0.382, pp - r * 0.618, pp - r * 1.0])

        # Filter out invalid targets
        if direction == "BUY":
            all_targets = [t for t in all_targets if t > entry_price * 1.002]
        else:
            all_targets = [t for t in all_targets if t < entry_price * 0.998]

        if not all_targets:
            # Fallback to simple ATR targets
            if direction == "BUY":
                return {
                    'target_1': round(entry_price + 1.5 * atr_val * hold_mult, 2),
                    'target_2': round(entry_price + 2.0 * atr_val * hold_mult, 2),
                    'target_3': round(entry_price + 3.0 * atr_val * hold_mult, 2),
                    'confluence_count': 1,
                }
            else:
                return {
                    'target_1': round(entry_price - 1.5 * atr_val * hold_mult, 2),
                    'target_2': round(entry_price - 2.0 * atr_val * hold_mult, 2),
                    'target_3': round(entry_price - 3.0 * atr_val * hold_mult, 2),
                    'confluence_count': 1,
                }

        # Cluster targets within 1.5% of each other
        sorted_targets = sorted(all_targets) if direction == "BUY" else sorted(all_targets, reverse=True)
        clusters = []
        used = set()

        for i, t in enumerate(sorted_targets):
            if i in used:
                continue
            cluster = [t]
            used.add(i)
            for j in range(i + 1, len(sorted_targets)):
                if j in used:
                    continue
                if abs(sorted_targets[j] - t) / t * 100 <= 1.5:
                    cluster.append(sorted_targets[j])
                    used.add(j)
            clusters.append((sum(cluster) / len(cluster), len(cluster)))

        # Sort by confluence count (most methods agreeing first)
        clusters.sort(key=lambda x: x[1], reverse=True)

        # Pick top 3 confluence levels
        result_targets = [round(c[0], 2) for c in clusters[:3]]
        max_confluence = clusters[0][1] if clusters else 1

        # Pad if fewer than 3
        while len(result_targets) < 3:
            if direction == "BUY":
                last = result_targets[-1]
                result_targets.append(round(last + atr_val * hold_mult, 2))
            else:
                last = result_targets[-1]
                result_targets.append(round(last - atr_val * hold_mult, 2))

        return {
            'target_1': result_targets[0],
            'target_2': result_targets[1],
            'target_3': result_targets[2],
            'confluence_count': max_confluence,
        }

    def calculate_structure_stop(
        self,
        high: np.ndarray, low: np.ndarray, close: np.ndarray,
        entry_price: float, direction: str = "BUY"
    ) -> float:
        """
        Calculate stop-loss using multiple methods, pick the tightest
        that still gives breathing room.

        Methods: Chandelier Exit, swing structure, Supertrend.
        """
        atr_val = self._atr(high, low, close)[-1]
        stops = []

        # Method 1: Chandelier Exit (3x ATR from 22-bar extreme)
        if direction == "BUY":
            highest_22 = np.max(high[-22:]) if len(high) >= 22 else np.max(high)
            stops.append(highest_22 - 3.0 * atr_val)
        else:
            lowest_22 = np.min(low[-22:]) if len(low) >= 22 else np.min(low)
            stops.append(lowest_22 + 3.0 * atr_val)

        # Method 2: Swing structure stop
        swing_lows = self.find_swing_lows(low)
        swing_highs = self.find_swing_highs(high)

        if direction == "BUY" and swing_lows:
            recent_lows = [s for s in swing_lows if s < entry_price]
            if recent_lows:
                stops.append(max(recent_lows) * 0.995)  # 0.5% buffer
        elif direction == "SELL" and swing_highs:
            recent_highs = [s for s in swing_highs if s > entry_price]
            if recent_highs:
                stops.append(min(recent_highs) * 1.005)

        # Method 3: Supertrend as stop
        st_line, st_dir = self.supertrend(high, low, close)
        stops.append(st_line[-1])

        # Select the tightest valid stop with minimum 1 ATR breathing room
        if direction == "BUY":
            valid_stops = [s for s in stops if s < entry_price]
            if valid_stops:
                best_stop = max(valid_stops)
                min_stop = entry_price - 1.0 * atr_val
                return round(min(best_stop, min_stop), 2)
            return round(entry_price - 2.0 * atr_val, 2)
        else:
            valid_stops = [s for s in stops if s > entry_price]
            if valid_stops:
                best_stop = min(valid_stops)
                min_stop = entry_price + 1.0 * atr_val
                return round(max(best_stop, min_stop), 2)
            return round(entry_price + 2.0 * atr_val, 2)

    # =========================================================================
    # Signal Quality Gate
    # =========================================================================

    def signal_quality_gate(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray,
        direction: str, indicators: Dict[str, Dict]
    ) -> Tuple[bool, float, List[str]]:
        """
        Multi-gate signal quality filter.
        A signal must pass 3+ of 5 gates to be considered valid.

        Gates:
        1. Trend alignment (price vs 200-SMA)
        2. Volume confirmation (>= 1.2x average)
        3. ADX trend strength (> 18)
        4. Multi-indicator confirmation (3+ of 5 agree)
        5. No volatility squeeze (BB width not at extreme low)

        Returns: (passes, quality_score, reasons)
        """
        gates_passed = 0
        total_gates = 5
        reasons = []

        # Gate 1: Trend alignment
        if len(close) >= 200:
            sma_200 = self._sma(close, 200)
            if direction == "BUY" and close[-1] > sma_200[-1]:
                gates_passed += 1
                reasons.append("Above 200-SMA")
            elif direction == "SELL" and close[-1] < sma_200[-1]:
                gates_passed += 1
                reasons.append("Below 200-SMA")
            else:
                reasons.append("Counter-trend signal")
        else:
            # Not enough data for 200-SMA, use 50-SMA
            sma_50 = self._sma(close, 50)
            if not np.isnan(sma_50[-1]):
                if direction == "BUY" and close[-1] > sma_50[-1]:
                    gates_passed += 1
                elif direction == "SELL" and close[-1] < sma_50[-1]:
                    gates_passed += 1

        # Gate 2: Volume confirmation (relaxed to 1.2x for broader inclusion)
        vol_sma = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 0
        if vol_ratio >= 1.2:
            gates_passed += 1
            reasons.append(f"Volume {vol_ratio:.1f}x avg")
        else:
            reasons.append(f"Low volume ({vol_ratio:.1f}x)")

        # Gate 3: ADX trend strength
        adx_vals, _, _ = self.adx(high, low, close)
        if adx_vals[-1] > 18:
            gates_passed += 1
            reasons.append(f"ADX {adx_vals[-1]:.0f}")

        # Gate 4: Multi-indicator confirmation (3+ of available indicators agree)
        confirming = 0
        total_checked = 0
        for name, sig in indicators.items():
            signal = sig.get('signal', 'NEUTRAL')
            if direction == "BUY" and signal in ['BUY', 'STRONG_BUY', 'BULLISH', 'OVERSOLD']:
                confirming += 1
            elif direction == "SELL" and signal in ['SELL', 'STRONG_SELL', 'BEARISH', 'OVERBOUGHT']:
                confirming += 1
            total_checked += 1

        if total_checked > 0 and confirming / total_checked >= 0.375:
            gates_passed += 1
            reasons.append(f"{confirming}/{total_checked} indicators confirm")

        # Gate 5: Not in extreme squeeze
        bb_mid = self._sma(close, 20)
        bb_std = self._rolling_std(close, 20)
        if not np.isnan(bb_mid[-1]) and not np.isnan(bb_std[-1]) and bb_mid[-1] > 0:
            bb_width = (2 * bb_std[-1]) / bb_mid[-1]
            recent_widths = []
            for i in range(max(len(close) - 20, 0), len(close)):
                if not np.isnan(bb_mid[i]) and not np.isnan(bb_std[i]) and bb_mid[i] > 0:
                    recent_widths.append((2 * bb_std[i]) / bb_mid[i])
            if recent_widths:
                width_pctl = sum(1 for w in recent_widths if w < bb_width) / len(recent_widths) * 100
                if width_pctl > 15:
                    gates_passed += 1
                else:
                    reasons.append("Volatility squeeze")
            else:
                gates_passed += 1
        else:
            gates_passed += 1  # Can't compute, pass by default

        quality_score = (gates_passed / total_gates) * 100
        passes = gates_passed >= 3

        return passes, quality_score, reasons

    # =========================================================================
    # Liquidity Filter
    # =========================================================================

    def liquidity_check(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        min_turnover_cr: float = 0.5
    ) -> Dict[str, Any]:
        """
        Check minimum liquidity for reliable signal execution.
        Daily turnover must meet threshold (in Crores).
        """
        avg_vol_20 = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        avg_price = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        daily_turnover = avg_vol_20 * avg_price
        turnover_cr = daily_turnover / 1e7

        return {
            'daily_turnover_cr': round(turnover_cr, 2),
            'passes': turnover_cr >= min_turnover_cr,
            'avg_volume': int(avg_vol_20),
        }

    # =========================================================================
    # Multi-Timeframe Analysis (MTA)
    # =========================================================================

    def multi_timeframe_analysis(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray
    ) -> Dict[str, Any]:
        """
        Multi-timeframe analysis: resample daily data to weekly bars
        and check weekly EMA(13) trend direction for confirmation.

        Weekly uptrend = daily buy signals are higher conviction.
        Weekly downtrend = daily buy signals are suspect.
        """
        n = len(close)
        if n < 30:
            return {'weekly_trend': 'UNKNOWN', 'aligned': True, 'weekly_ema': 0}

        # Resample daily to weekly (groups of 5 trading days)
        week_count = n // 5
        if week_count < 13:
            return {'weekly_trend': 'UNKNOWN', 'aligned': True, 'weekly_ema': 0}

        weekly_close = np.array([close[min((i + 1) * 5 - 1, n - 1)] for i in range(week_count)])
        weekly_high = np.array([np.max(high[i * 5:min((i + 1) * 5, n)]) for i in range(week_count)])
        weekly_low = np.array([np.min(low[i * 5:min((i + 1) * 5, n)]) for i in range(week_count)])

        # Weekly EMA(13) trend
        weekly_ema13 = self._ema(weekly_close, 13)

        current_weekly_close = weekly_close[-1]
        current_weekly_ema = weekly_ema13[-1]
        prev_weekly_ema = weekly_ema13[-2] if len(weekly_ema13) > 1 else current_weekly_ema

        ema_rising = current_weekly_ema > prev_weekly_ema
        price_above_ema = current_weekly_close > current_weekly_ema

        if ema_rising and price_above_ema:
            weekly_trend = 'UP'
        elif not ema_rising and not price_above_ema:
            weekly_trend = 'DOWN'
        else:
            weekly_trend = 'MIXED'

        # Weekly Supertrend for additional confirmation
        weekly_atr = self._atr(weekly_high, weekly_low, weekly_close, period=min(10, week_count - 1))
        weekly_st, weekly_st_dir = self.supertrend(weekly_high, weekly_low, weekly_close, period=min(10, week_count - 1))
        weekly_st_trend = 'UP' if weekly_st_dir[-1] == 1 else 'DOWN'

        return {
            'weekly_trend': weekly_trend,
            'weekly_ema': round(current_weekly_ema, 2),
            'weekly_ema_rising': ema_rising,
            'price_above_weekly_ema': price_above_ema,
            'weekly_supertrend': weekly_st_trend,
            'aligned': weekly_trend == weekly_st_trend,
        }

    # =========================================================================
    # TTM Squeeze Detection
    # =========================================================================

    def ttm_squeeze(
        self,
        high: np.ndarray, low: np.ndarray, close: np.ndarray,
        bb_length: int = 20, bb_mult: float = 2.0,
        kc_length: int = 20, kc_mult: float = 1.5
    ) -> Dict[str, Any]:
        """
        TTM Squeeze: Bollinger Bands inside Keltner Channels = compression.
        BB expanding beyond KC = momentum breakout.

        When squeeze fires (BB exits KC), momentum direction predicts breakout.
        Uses linear regression momentum for direction.
        """
        # Bollinger Bands
        bb_mid = self._sma(close, bb_length)
        bb_std = self._rolling_std(close, bb_length)
        bb_upper = bb_mid + bb_mult * bb_std
        bb_lower = bb_mid - bb_mult * bb_std

        # Keltner Channels
        kc_mid = self._ema(close, kc_length)
        atr = self._atr(high, low, close, kc_length)
        kc_upper = kc_mid + kc_mult * atr
        kc_lower = kc_mid - kc_mult * atr

        # Squeeze detection: BB inside KC
        squeeze_on = []
        for i in range(max(bb_length, kc_length) - 1, len(close)):
            if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
                squeeze_on.append(bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i])
            else:
                squeeze_on.append(False)

        is_squeezing = squeeze_on[-1] if squeeze_on else False

        # Count consecutive squeeze bars
        squeeze_bars = 0
        for sq in reversed(squeeze_on):
            if sq:
                squeeze_bars += 1
            else:
                break

        # Squeeze just fired (was squeezing, now released)
        squeeze_fired = False
        if len(squeeze_on) >= 2:
            squeeze_fired = squeeze_on[-2] and not squeeze_on[-1]

        # Momentum direction using linear regression slope of close
        lookback = min(20, len(close))
        recent = close[-lookback:]
        x = np.arange(lookback)
        if len(recent) > 1:
            slope = np.polyfit(x, recent, 1)[0]
            momentum_direction = 'UP' if slope > 0 else 'DOWN'
        else:
            slope = 0
            momentum_direction = 'NEUTRAL'

        signal = 'NEUTRAL'
        strength = 0

        if squeeze_fired:
            if momentum_direction == 'UP':
                signal = 'BUY'
                strength = 85
            elif momentum_direction == 'DOWN':
                signal = 'SELL'
                strength = 85
        elif is_squeezing and squeeze_bars >= 6:
            signal = 'BUILDING'
            strength = 60

        return {
            'signal': signal,
            'strength': strength,
            'is_squeezing': is_squeezing,
            'squeeze_bars': squeeze_bars,
            'squeeze_fired': squeeze_fired,
            'momentum_direction': momentum_direction,
            'momentum_slope': round(slope, 4),
        }

    # =========================================================================
    # Breakout Detection (Darvas Box / Consolidation)
    # =========================================================================

    def detect_breakout(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray,
        consolidation_bars: int = 10, range_pct: float = 6.0
    ) -> Dict[str, Any]:
        """
        Detect breakout from consolidation (Darvas Box concept).

        A consolidation is defined as price staying within range_pct%
        for at least consolidation_bars bars. Breakout occurs when
        price closes above the range high with volume expansion.
        """
        n = len(close)
        if n < consolidation_bars + 5:
            return {'breakout': False, 'consolidating': False, 'direction': 'NONE'}

        # Look for consolidation range in recent bars (exclude last bar)
        lookback_start = max(n - 40, 0)
        recent_high = high[lookback_start:n - 1]
        recent_low = low[lookback_start:n - 1]

        # Find the tightest consolidation zone
        best_range = None
        best_range_pct_val = float('inf')

        for start in range(len(recent_high) - consolidation_bars):
            end = start + consolidation_bars
            zone_high = np.max(recent_high[start:end])
            zone_low = np.min(recent_low[start:end])
            zone_range = (zone_high - zone_low) / zone_low * 100 if zone_low > 0 else 999

            if zone_range <= range_pct and zone_range < best_range_pct_val:
                best_range = (zone_low, zone_high)
                best_range_pct_val = zone_range

        if best_range is None:
            return {'breakout': False, 'consolidating': False, 'direction': 'NONE'}

        box_low, box_high = best_range
        current_close = close[-1]
        current_vol = volume[-1]
        avg_vol = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
        vol_expansion = current_vol > avg_vol * 1.5

        # Breakout detection
        breakout_up = current_close > box_high * 1.002 and vol_expansion
        breakout_down = current_close < box_low * 0.998 and vol_expansion
        still_consolidating = box_low * 0.998 <= current_close <= box_high * 1.002

        # Measured move target
        box_range = box_high - box_low
        if breakout_up:
            measured_target = box_high + box_range
        elif breakout_down:
            measured_target = box_low - box_range
        else:
            measured_target = 0

        direction = 'UP' if breakout_up else ('DOWN' if breakout_down else 'NONE')

        return {
            'breakout': breakout_up or breakout_down,
            'direction': direction,
            'consolidating': still_consolidating,
            'box_high': round(box_high, 2),
            'box_low': round(box_low, 2),
            'box_range_pct': round(best_range_pct_val, 1),
            'volume_expansion': vol_expansion,
            'measured_target': round(measured_target, 2),
            'consolidation_bars': consolidation_bars,
        }

    # =========================================================================
    # Mean Reversion (for Ranging Markets)
    # =========================================================================

    def mean_reversion_signal(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, volume: np.ndarray
    ) -> Dict[str, Any]:
        """
        Mean reversion signal for ranging/low-volatility markets.

        Uses BB %B + RSI to find oversold bounces and overbought fades.
        Only valid when ADX < 25 (no strong trend).
        Targets are smaller (1-1.5 ATR vs 2.5-3 ATR for trends).
        """
        # Check if market is ranging
        adx_vals, _, _ = self.adx(high, low, close)
        adx_current = adx_vals[-1] if len(adx_vals) > 0 else 30

        if adx_current > 25:
            return {'signal': 'NOT_RANGING', 'strength': 0, 'adx': round(adx_current, 1)}

        # Bollinger Band %B
        bb_mid = self._sma(close, 20)
        bb_std = self._rolling_std(close, 20)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        if np.isnan(bb_upper[-1]) or np.isnan(bb_lower[-1]) or bb_upper[-1] == bb_lower[-1]:
            return {'signal': 'NEUTRAL', 'strength': 0, 'adx': round(adx_current, 1)}

        bb_pctb = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])

        # RSI
        rsi_vals = self.rsi(close)
        rsi_current = rsi_vals[-1]

        atr_val = self._atr(high, low, close)[-1]

        signal = 'NEUTRAL'
        strength = 0
        target_distance = 0
        stop_distance = 0

        # Mean reversion buy: near lower BB + RSI oversold
        if bb_pctb < 0.15 and rsi_current < 35:
            signal = 'MR_BUY'
            strength = 75
            target_distance = round(1.5 * atr_val, 2)  # Smaller target for MR
            stop_distance = round(1.0 * atr_val, 2)
        elif bb_pctb < 0.25 and rsi_current < 40:
            signal = 'MR_BUY'
            strength = 55
            target_distance = round(1.0 * atr_val, 2)
            stop_distance = round(0.8 * atr_val, 2)

        # Mean reversion sell: near upper BB + RSI overbought
        elif bb_pctb > 0.85 and rsi_current > 65:
            signal = 'MR_SELL'
            strength = 75
            target_distance = round(1.5 * atr_val, 2)
            stop_distance = round(1.0 * atr_val, 2)
        elif bb_pctb > 0.75 and rsi_current > 60:
            signal = 'MR_SELL'
            strength = 55
            target_distance = round(1.0 * atr_val, 2)
            stop_distance = round(0.8 * atr_val, 2)

        return {
            'signal': signal,
            'strength': strength,
            'bb_pctb': round(bb_pctb, 3),
            'rsi': round(rsi_current, 1),
            'adx': round(adx_current, 1),
            'target_distance': target_distance,
            'stop_distance': stop_distance,
            'bb_mid': round(bb_mid[-1], 2) if not np.isnan(bb_mid[-1]) else 0,
        }

    # =========================================================================
    # Price Action Patterns (HH/HL, LH/LL, Inside Bars)
    # =========================================================================

    def price_action_patterns(
        self,
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Detect price action patterns:
        - Higher Highs / Higher Lows (bullish structure)
        - Lower Highs / Lower Lows (bearish structure)
        - Inside bars (compression before breakout)
        - Swing failure pattern
        """
        n = len(close)
        if n < lookback + 5:
            return {'pattern': 'UNKNOWN', 'hh_count': 0, 'll_count': 0, 'inside_bars': 0}

        recent_high = high[-lookback:]
        recent_low = low[-lookback:]

        # Count HH/HL and LH/LL sequences
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0

        # Use swing points (5-bar pivots)
        swing_h = []
        swing_l = []
        for i in range(2, len(recent_high) - 2):
            if recent_high[i] >= max(recent_high[i - 2:i]) and recent_high[i] >= max(recent_high[i + 1:i + 3]):
                swing_h.append(recent_high[i])
            if recent_low[i] <= min(recent_low[i - 2:i]) and recent_low[i] <= min(recent_low[i + 1:i + 3]):
                swing_l.append(recent_low[i])

        # Count consecutive HH/LH
        for i in range(1, len(swing_h)):
            if swing_h[i] > swing_h[i - 1]:
                hh_count += 1
            elif swing_h[i] < swing_h[i - 1]:
                lh_count += 1

        # Count consecutive HL/LL
        for i in range(1, len(swing_l)):
            if swing_l[i] > swing_l[i - 1]:
                hl_count += 1
            elif swing_l[i] < swing_l[i - 1]:
                ll_count += 1

        # Inside bars (current bar's range inside previous bar's range)
        inside_bars = 0
        for i in range(n - 5, n):
            if i > 0 and high[i] <= high[i - 1] and low[i] >= low[i - 1]:
                inside_bars += 1

        # Determine dominant pattern
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        if bullish_score >= 3 and bullish_score > bearish_score * 1.5:
            pattern = 'BULLISH_STRUCTURE'
            strength = min(bullish_score * 15, 80)
        elif bearish_score >= 3 and bearish_score > bullish_score * 1.5:
            pattern = 'BEARISH_STRUCTURE'
            strength = min(bearish_score * 15, 80)
        elif inside_bars >= 2:
            pattern = 'COMPRESSION'
            strength = 60
        else:
            pattern = 'MIXED'
            strength = 30

        return {
            'pattern': pattern,
            'strength': strength,
            'hh_count': hh_count,
            'hl_count': hl_count,
            'lh_count': lh_count,
            'll_count': ll_count,
            'inside_bars': inside_bars,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
        }

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        ema = np.zeros(len(data))
        multiplier = 2 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def _wma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.sum(data[i - period + 1:i + 1] * weights) / np.sum(weights)
        return result

    def _rolling_std(self, data: np.ndarray, period: int) -> np.ndarray:
        """Rolling standard deviation."""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.std(data[i - period + 1:i + 1])
        return result

    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Average True Range."""
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]

        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        return self._ema(tr, period)


class FundamentalScreener:
    """
    Fundamental quality screener based on proven screening criteria:
    - Consistent EPS growth (3-year CAGR)
    - Healthy revenue growth
    - PEG ratio (valuation relative to growth)
    - Debt levels (D/E ratio)
    - Profitability (ROE, profit margins)
    - Market cap guardrails

    Note on PEG: A low PEG (<1) is a shortlisting tool, not proof of value.
    It may reflect cyclical upswing rather than sustainable growth.
    """

    # Quality thresholds
    MIN_EPS_GROWTH = 10.0  # Minimum 3-year EPS CAGR %
    MIN_REVENUE_GROWTH = 8.0  # Minimum 3-year revenue CAGR %
    MAX_PEG_RATIO = 2.0  # Maximum PEG for consideration
    GOOD_PEG_RATIO = 1.0  # PEG < 1 is attractive (with caveats)
    MAX_DEBT_EQUITY = 1.5  # Maximum D/E ratio
    LOW_DEBT_EQUITY = 0.5  # Low debt threshold
    MIN_ROE = 12.0  # Minimum ROE %
    GOOD_ROE = 18.0  # Good ROE threshold
    MIN_PROFIT_MARGIN = 5.0  # Minimum net profit margin %
    MIN_MARKET_CAP_CR = 500  # Minimum market cap in Crores

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Dict] = {}

    def get_quality_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch quality metrics for a stock from Yahoo Finance.
        Returns dict with EPS growth, revenue growth, PEG, D/E, ROE, margins, market cap.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        metrics = {
            'eps_growth_3y': None,
            'revenue_growth_3y': None,
            'peg_ratio': None,
            'debt_to_equity': None,
            'roe': None,
            'profit_margin': None,
            'market_cap_cr': None,
            'pe_ratio': None,
            'pb_ratio': None,
            'current_ratio': None,
        }

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            # Market cap (convert to Crores: 1 Cr = 10 million)
            if 'marketCap' in info and info['marketCap']:
                metrics['market_cap_cr'] = info['marketCap'] / 10_000_000

            # PEG ratio (directly from Yahoo)
            if 'pegRatio' in info and info['pegRatio']:
                metrics['peg_ratio'] = info['pegRatio']

            # Debt to Equity
            if 'debtToEquity' in info and info['debtToEquity']:
                metrics['debt_to_equity'] = info['debtToEquity'] / 100  # Yahoo returns as %

            # ROE (Return on Equity)
            if 'returnOnEquity' in info and info['returnOnEquity']:
                metrics['roe'] = info['returnOnEquity'] * 100  # Convert to %

            # Profit margin
            if 'profitMargins' in info and info['profitMargins']:
                metrics['profit_margin'] = info['profitMargins'] * 100  # Convert to %

            # P/E ratio
            if 'trailingPE' in info and info['trailingPE']:
                metrics['pe_ratio'] = info['trailingPE']

            # P/B ratio
            if 'priceToBook' in info and info['priceToBook']:
                metrics['pb_ratio'] = info['priceToBook']

            # Current ratio (liquidity)
            if 'currentRatio' in info and info['currentRatio']:
                metrics['current_ratio'] = info['currentRatio']

            # Revenue growth (YoY from Yahoo)
            if 'revenueGrowth' in info and info['revenueGrowth']:
                metrics['revenue_growth_3y'] = info['revenueGrowth'] * 100  # Use as proxy

            # EPS growth (calculate from earnings growth or forward PE difference)
            if 'earningsGrowth' in info and info['earningsGrowth']:
                metrics['eps_growth_3y'] = info['earningsGrowth'] * 100
            elif 'earningsQuarterlyGrowth' in info and info['earningsQuarterlyGrowth']:
                metrics['eps_growth_3y'] = info['earningsQuarterlyGrowth'] * 100

            self._cache[ticker] = metrics

        except Exception as e:
            self.logger.debug(f"Error fetching quality metrics for {ticker}: {e}")

        return metrics

    def calculate_quality_score(self, metrics: Dict[str, Any]) -> Tuple[float, List[str], bool]:
        """
        Calculate quality score (0-100) based on fundamental metrics.
        Returns (score, flags, passes_minimum).

        Scoring breakdown:
        - EPS Growth: 0-25 points
        - Revenue Growth: 0-15 points
        - PEG Ratio: 0-20 points
        - Debt/Equity: 0-15 points
        - ROE: 0-15 points
        - Profit Margin: 0-10 points
        """
        score = 0.0
        flags = []
        passes = True

        # 1. EPS Growth (25 points max)
        eps_growth = metrics.get('eps_growth_3y')
        if eps_growth is not None:
            if eps_growth >= 25:
                score += 25
                flags.append("Strong EPS growth (>25%)")
            elif eps_growth >= 15:
                score += 20
                flags.append("Good EPS growth (15-25%)")
            elif eps_growth >= self.MIN_EPS_GROWTH:
                score += 15
            elif eps_growth >= 5:
                score += 8
            elif eps_growth < 0:
                flags.append("⚠ Negative EPS growth")
                passes = False

        # 2. Revenue Growth (15 points max)
        rev_growth = metrics.get('revenue_growth_3y')
        if rev_growth is not None:
            if rev_growth >= 20:
                score += 15
                flags.append("Strong revenue growth (>20%)")
            elif rev_growth >= self.MIN_REVENUE_GROWTH:
                score += 12
            elif rev_growth >= 5:
                score += 8
            elif rev_growth < 0:
                flags.append("⚠ Negative revenue growth")

        # 3. PEG Ratio (20 points max)
        peg = metrics.get('peg_ratio')
        if peg is not None and peg > 0:
            if peg < 0.5:
                score += 20
                flags.append("Very low PEG (<0.5) - verify sustainability")
            elif peg < self.GOOD_PEG_RATIO:
                score += 18
                flags.append("Attractive PEG (<1)")
            elif peg < 1.5:
                score += 12
            elif peg < self.MAX_PEG_RATIO:
                score += 6
            else:
                flags.append("⚠ High PEG (>2) - expensive relative to growth")

        # 4. Debt/Equity (15 points max)
        de = metrics.get('debt_to_equity')
        if de is not None:
            if de < self.LOW_DEBT_EQUITY:
                score += 15
                flags.append("Low debt (D/E <0.5)")
            elif de < 1.0:
                score += 12
            elif de < self.MAX_DEBT_EQUITY:
                score += 6
            else:
                flags.append("⚠ High debt (D/E >1.5)")
                passes = False

        # 5. ROE (15 points max)
        roe = metrics.get('roe')
        if roe is not None:
            if roe >= self.GOOD_ROE:
                score += 15
                flags.append("Excellent ROE (>18%)")
            elif roe >= self.MIN_ROE:
                score += 12
            elif roe >= 8:
                score += 6
            elif roe < 5:
                flags.append("⚠ Low ROE (<5%)")

        # 6. Profit Margin (10 points max)
        margin = metrics.get('profit_margin')
        if margin is not None:
            if margin >= 20:
                score += 10
                flags.append("High profit margin (>20%)")
            elif margin >= 12:
                score += 8
            elif margin >= self.MIN_PROFIT_MARGIN:
                score += 5
            elif margin < 0:
                flags.append("⚠ Negative profit margin")
                passes = False

        # 7. Market cap check (no points, just filter)
        mcap = metrics.get('market_cap_cr')
        if mcap is not None and mcap < self.MIN_MARKET_CAP_CR:
            flags.append(f"⚠ Small cap (<₹{self.MIN_MARKET_CAP_CR}Cr)")

        return score, flags, passes

    def screen_stock(self, ticker: str) -> Tuple[float, Dict[str, Any], List[str], bool]:
        """
        Screen a stock for quality.
        Returns (quality_score, metrics, flags, passes_minimum).
        """
        metrics = self.get_quality_metrics(ticker)
        score, flags, passes = self.calculate_quality_score(metrics)
        return score, metrics, flags, passes


class FundamentalEnhancer:
    """
    Enhances technical signals with fundamental data from:
    - Live news from Yahoo Finance and Google News (default)
    - NSE/BSE circulars database (if available)
    - Press releases
    - News catalysts
    - PEAD (Post-Earnings Announcement Drift) signals

    Uses NLP-based sentiment analysis (FinBERT/VADER) when available,
    falls back to keyword matching otherwise.
    """

    def __init__(self, circulars_db: str = None, days_lookback: int = 30,
                 use_nlp: bool = True, use_live_news: bool = True):
        self.logger = logging.getLogger(__name__)
        self.days_lookback = days_lookback
        self.use_nlp = use_nlp
        self.use_live_news = use_live_news

        # Find circulars database
        self.circulars_db = self._find_database(circulars_db)
        self._catalyst_cache: Dict[str, FundamentalData] = {}

        # Initialize live news fetcher
        self.news_fetcher = None
        if use_live_news:
            try:
                # Use absolute import for reliability
                from finagent.analysis.live_news_fetcher import LiveNewsFetcher
                self.news_fetcher = LiveNewsFetcher()
                self.logger.info("Live news fetching enabled")
            except ImportError as e:
                self.logger.warning(f"Could not load live news fetcher: {e}")
            except Exception as e:
                self.logger.warning(f"Error initializing live news fetcher: {e}")

        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        if use_nlp:
            try:
                # Use absolute import for reliability
                from finagent.analysis.sentiment_analyzer import FinancialSentimentAnalyzer
                self.sentiment_analyzer = FinancialSentimentAnalyzer()
                methods = self.sentiment_analyzer.get_available_methods()
                self.logger.info(f"Sentiment analysis available: {methods}")
            except ImportError as e:
                self.logger.warning(f"Could not load sentiment analyzer: {e}")
            except Exception as e:
                self.logger.warning(f"Error initializing sentiment analyzer: {e}")

        # Initialize macro/policy analyzer
        self.macro_analyzer = None
        if use_live_news:
            try:
                from finagent.analysis.live_news_fetcher import MacroPolicyAnalyzer
                self.macro_analyzer = MacroPolicyAnalyzer()
                self.logger.info("Macro/Policy analyzer enabled")
            except ImportError as e:
                self.logger.debug(f"Could not load macro analyzer: {e}")
            except Exception as e:
                self.logger.debug(f"Error initializing macro analyzer: {e}")

        # Cache for macro news (shared across all stocks)
        self._macro_news_cache = None
        self._fii_dii_cache = None

        # Initialize fundamental screener for quality metrics
        self.quality_screener = FundamentalScreener()
        self.use_quality_screen = True  # Enable quality screening by default

    def _find_database(self, db_path: Optional[str] = None) -> Optional[str]:
        """Find the circulars database."""
        from pathlib import Path

        if db_path and Path(db_path).exists():
            return db_path

        # Try common locations
        possible_paths = [
            "data/cache/circulars.db",
            "~/.finagent/circulars.db",
            "./circulars.db",
        ]

        for path in possible_paths:
            expanded = Path(path).expanduser()
            if expanded.exists():
                return str(expanded)

        return None

    def get_fundamental_data(self, ticker: str) -> FundamentalData:
        """Get fundamental data for a ticker from database or live news."""
        # Check cache first
        if ticker in self._catalyst_cache:
            return self._catalyst_cache[ticker]

        fundamental = FundamentalData()
        headlines = []
        catalysts = []
        bullish_count = 0
        bearish_count = 0

        # ALWAYS try live news first if enabled (regardless of database)
        if self.use_live_news and self.news_fetcher:
            try:
                news_items = self.news_fetcher.fetch_news(ticker, self.days_lookback)
                self.logger.debug(f"Live news for {ticker}: {len(news_items) if news_items else 0} items")

                if news_items:
                    fundamental.has_recent_news = True

                    for item in news_items[:15]:  # Process up to 15 items
                        title = item.title or ''
                        description = item.description or ''
                        text = f"{title} {description}".lower()

                        headlines.append(title[:100])

                        # Detect catalyst type and sentiment
                        catalyst_info = self._analyze_catalyst(title, description, '')
                        if catalyst_info:
                            catalysts.append(catalyst_info)
                            if catalyst_info['sentiment'] == 'BULLISH':
                                bullish_count += 1
                            elif catalyst_info['sentiment'] == 'BEARISH':
                                bearish_count += 1

                        # Check for earnings
                        if any(kw in text for kw in ['result', 'quarter', 'earning', 'profit', 'revenue']):
                            fundamental.earnings_surprise = self._detect_earnings_surprise(text)
                            if fundamental.earnings_surprise == 'BEAT':
                                fundamental.pead_score = 30.0
                            elif fundamental.earnings_surprise == 'MISS':
                                fundamental.pead_score = -30.0

            except Exception as e:
                self.logger.warning(f"Error fetching live news for {ticker}: {e}")

        # Also check database for historical circulars (supplements live news)
        if self.circulars_db:
            try:
                import sqlite3
                from datetime import datetime, timedelta

                # Normalize ticker for database lookup
                base_ticker = ticker.replace('.NS', '').replace('.BO', '').upper()
                cutoff_date = (datetime.now() - timedelta(days=self.days_lookback)).isoformat()

                conn = sqlite3.connect(self.circulars_db)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Fetch recent circulars for this ticker
                sql = """
                    SELECT id, ticker, company_name, title, description,
                           doc_type, category, filing_date, parsed_text
                    FROM circulars
                    WHERE (ticker LIKE ? OR company_name LIKE ?)
                      AND filing_date >= ?
                    ORDER BY filing_date DESC
                    LIMIT 20
                """
                cursor.execute(sql, (f"%{base_ticker}%", f"%{base_ticker}%", cutoff_date))
                rows = cursor.fetchall()
                conn.close()

                if rows:
                    fundamental.has_recent_news = True

                    for row in rows:
                        title = row['title'] or ''
                        description = row['description'] or ''
                        category = row['category'] or ''
                        text = f"{title} {description}".lower()

                        headlines.append(title[:100])

                        # Detect catalyst type and sentiment
                        catalyst_info = self._analyze_catalyst(title, description, category)
                        if catalyst_info:
                            catalysts.append(catalyst_info)
                            if catalyst_info['sentiment'] == 'BULLISH':
                                bullish_count += 1
                            elif catalyst_info['sentiment'] == 'BEARISH':
                                bearish_count += 1

                        # Check for earnings
                        if any(kw in text for kw in ['result', 'quarter', 'earning', 'profit', 'revenue']):
                            fundamental.earnings_surprise = self._detect_earnings_surprise(text)
                            if fundamental.earnings_surprise == 'BEAT':
                                fundamental.pead_score = 30.0
                            elif fundamental.earnings_surprise == 'MISS':
                                fundamental.pead_score = -30.0

            except Exception as e:
                self.logger.debug(f"Error fetching from database for {ticker}: {e}")

        # Process collected data if we have news (from either source)
        if fundamental.has_recent_news:
            fundamental.catalysts = catalysts[:5]
            fundamental.recent_headlines = headlines[:5]

            # Calculate news sentiment using NLP if available
            if self.sentiment_analyzer and headlines:
                # Analyze all headlines and aggregate
                sentiment_scores = []
                for headline in headlines[:10]:  # Analyze up to 10 headlines
                    if headline:
                        result = self.sentiment_analyzer.analyze(headline)
                        sentiment_scores.append(result.score)

                if sentiment_scores:
                    avg_score = sum(sentiment_scores) / len(sentiment_scores)

                    # Map score to sentiment label and news_score
                    if avg_score >= 0.3:
                        fundamental.news_sentiment = 'BULLISH'
                        fundamental.news_score = min(avg_score * 70, 50)
                    elif avg_score <= -0.3:
                        fundamental.news_sentiment = 'BEARISH'
                        fundamental.news_score = max(avg_score * 70, -50)
                    elif avg_score >= 0.1:
                        fundamental.news_sentiment = 'BULLISH'
                        fundamental.news_score = avg_score * 50
                    elif avg_score <= -0.1:
                        fundamental.news_sentiment = 'BEARISH'
                        fundamental.news_score = avg_score * 50
                    else:
                        fundamental.news_sentiment = 'NEUTRAL'
                        fundamental.news_score = avg_score * 30

                    # Store sentiment method for display
                    fundamental.sentiment_method = self.sentiment_analyzer._active_method
            else:
                # Fallback to keyword counting
                if bullish_count > bearish_count:
                    fundamental.news_sentiment = 'BULLISH'
                    fundamental.news_score = min((bullish_count - bearish_count) * 15, 50)
                elif bearish_count > bullish_count:
                    fundamental.news_sentiment = 'BEARISH'
                    fundamental.news_score = max((bullish_count - bearish_count) * 15, -50)
                else:
                    fundamental.news_sentiment = 'NEUTRAL'
                    fundamental.news_score = 0
                fundamental.sentiment_method = 'Keyword'

        # Add macro/policy impact analysis
        if self.macro_analyzer:
            try:
                # Fetch macro news once and cache it
                if self._macro_news_cache is None:
                    self._macro_news_cache = self.macro_analyzer.fetch_macro_news(self.days_lookback)

                # Get policy impact for this stock
                policy_impact = self.macro_analyzer.get_policy_impact(ticker, self._macro_news_cache)
                fundamental.has_policy_boost = policy_impact.get('has_policy_boost', False)
                fundamental.policy_score = policy_impact.get('policy_score', 0.0)
                fundamental.affected_sectors = policy_impact.get('affected_sectors', [])
                fundamental.relevant_policies = policy_impact.get('relevant_policies', [])

                # Get FII/DII sentiment (cached)
                if self._fii_dii_cache is None:
                    self._fii_dii_cache = self.macro_analyzer.get_fii_dii_sentiment()

                fundamental.fii_sentiment = self._fii_dii_cache.get('fii_sentiment', 'NEUTRAL')
                fundamental.dii_sentiment = self._fii_dii_cache.get('dii_sentiment', 'NEUTRAL')

                # Calculate institutional score from FII/DII
                fii_score = self._fii_dii_cache.get('fii_score', 0)
                dii_score = self._fii_dii_cache.get('dii_score', 0)
                fundamental.institutional_score = (fii_score + dii_score) / 2

                self.logger.debug(f"{ticker}: Policy boost={fundamental.has_policy_boost}, "
                                f"FII={fundamental.fii_sentiment}, DII={fundamental.dii_sentiment}")

            except Exception as e:
                self.logger.debug(f"Error getting macro/policy data for {ticker}: {e}")

        # Add quality metrics if enabled
        if self.use_quality_screen and self.quality_screener:
            try:
                quality_score, metrics, flags, passes = self.quality_screener.screen_stock(ticker)
                fundamental.quality_score = quality_score
                fundamental.passes_quality = passes
                fundamental.quality_flags = flags
                fundamental.eps_growth_3y = metrics.get('eps_growth_3y')
                fundamental.revenue_growth_3y = metrics.get('revenue_growth_3y')
                fundamental.peg_ratio = metrics.get('peg_ratio')
                fundamental.debt_to_equity = metrics.get('debt_to_equity')
                fundamental.roe = metrics.get('roe')
                fundamental.profit_margin = metrics.get('profit_margin')
                fundamental.market_cap_cr = metrics.get('market_cap_cr')
                self.logger.debug(f"{ticker}: Quality score={quality_score:.0f}, passes={passes}")
            except Exception as e:
                self.logger.debug(f"Error getting quality metrics for {ticker}: {e}")

        # Cache result
        self._catalyst_cache[ticker] = fundamental
        return fundamental

    def _analyze_catalyst(self, title: str, description: str, category: str) -> Optional[Dict]:
        """Analyze a circular/news item for catalyst information."""
        text = f"{title} {description}".lower()

        # Bullish catalysts
        bullish_patterns = {
            'contract_win': ['contract', 'order', 'won', 'secured', 'awarded', 'bagged'],
            'expansion': ['expansion', 'capacity', 'new plant', 'new facility', 'capex'],
            'strong_results': ['profit up', 'revenue up', 'growth', 'beat', 'exceeded', 'record'],
            'dividend': ['dividend', 'bonus', 'buyback'],
            'partnership': ['partnership', 'alliance', 'tie-up', 'collaboration', 'joint venture'],
            'approval': ['approval', 'clearance', 'license', 'patent'],
            'upgrade': ['upgrade', 'rating', 'target raised'],
            'acquisition': ['acquisition', 'acquire', 'merger'],
        }

        # Bearish catalysts
        bearish_patterns = {
            'loss': ['loss', 'decline', 'fell', 'dropped', 'missed', 'below'],
            'downgrade': ['downgrade', 'cut', 'lowered', 'reduced'],
            'penalty': ['penalty', 'fine', 'litigation', 'lawsuit'],
            'fraud': ['fraud', 'scam', 'investigation', 'sebi notice'],
            'management': ['resignation', 'stepped down', 'quit', 'exit'],
        }

        # Check for bullish catalysts
        for catalyst_type, keywords in bullish_patterns.items():
            if any(kw in text for kw in keywords):
                return {
                    'type': catalyst_type,
                    'headline': title[:100],
                    'sentiment': 'BULLISH',
                    'impact': self._estimate_impact(catalyst_type, text),
                }

        # Check for bearish catalysts
        for catalyst_type, keywords in bearish_patterns.items():
            if any(kw in text for kw in keywords):
                return {
                    'type': catalyst_type,
                    'headline': title[:100],
                    'sentiment': 'BEARISH',
                    'impact': self._estimate_impact(catalyst_type, text),
                }

        return None

    def _estimate_impact(self, catalyst_type: str, text: str) -> str:
        """Estimate impact level of a catalyst."""
        high_impact = ['contract_win', 'acquisition', 'strong_results', 'fraud', 'loss']
        medium_impact = ['expansion', 'dividend', 'partnership', 'approval', 'penalty']

        # Check for value mentions (large values = higher impact)
        import re
        value_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:cr|crore|billion)', text)
        if value_match:
            try:
                value = float(value_match.group(1).replace(',', ''))
                if value > 500:  # > 500 crore
                    return 'HIGH'
                elif value > 100:
                    return 'MEDIUM'
            except ValueError:
                pass

        if catalyst_type in high_impact:
            return 'HIGH'
        elif catalyst_type in medium_impact:
            return 'MEDIUM'
        return 'LOW'

    def _detect_earnings_surprise(self, text: str) -> str:
        """Detect earnings surprise from text."""
        text = text.lower()

        beat_keywords = ['beat', 'exceeded', 'surpassed', 'above', 'strong', 'robust',
                         'profit up', 'revenue up', 'growth', 'record', 'highest']
        miss_keywords = ['missed', 'below', 'fell', 'declined', 'weak', 'disappointing',
                         'profit down', 'revenue down', 'loss']

        beat_count = sum(1 for kw in beat_keywords if kw in text)
        miss_count = sum(1 for kw in miss_keywords if kw in text)

        if beat_count > miss_count:
            return 'BEAT'
        elif miss_count > beat_count:
            return 'MISS'
        return 'INLINE'

    def preload_fundamentals(self, tickers: List[str]):
        """Preload fundamental data for multiple tickers."""
        for ticker in tickers:
            self.get_fundamental_data(ticker)


class AdvancedSignalGenerator:
    """
    Generates combined signals from multiple indicators.

    Scans Indian stocks and provides buy/sell recommendations
    with profit targets based on holding period.

    Optionally incorporates fundamental data from NSE circulars,
    press releases, and news for enhanced signals.
    """

    def __init__(self, use_fundamentals: bool = True, days_lookback: int = 30):
        self.indicators = AdvancedIndicators()
        self.logger = logging.getLogger(__name__)
        self.use_fundamentals = use_fundamentals
        self.fundamental_enhancer = FundamentalEnhancer(days_lookback=days_lookback) if use_fundamentals else None

    def analyze_stock(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        hold_days: int = 7,
        index_close: Optional[np.ndarray] = None
    ) -> Optional[TechnicalSignal]:
        """
        Analyze a stock using all indicators, quality gates, and
        structure-based targets.

        Improvements over baseline:
        - Market regime detection (ADX + BB width)
        - Volume confirmation requirement
        - Signal quality gate (multi-gate filter)
        - Relative strength ranking vs Nifty 50
        - Momentum quality scoring
        - Structure-based targets (swing S/R + Fibonacci + ATR confluence)
        - Chandelier/swing-based stop-loss
        - Dynamic R:R calculation
        - Liquidity filter

        Args:
            ticker: Stock symbol
            ohlcv: DataFrame with Open, High, Low, Close, Volume
            hold_days: Suggested holding period in days
            index_close: Optional benchmark index close prices for RS calculation

        Returns:
            TechnicalSignal if actionable, None otherwise
        """
        if len(ohlcv) < 60:
            return None

        # Normalize column names
        df = ohlcv.copy()
        df.columns = [c.lower() for c in df.columns]

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        current_price = close[-1]

        # ── STEP 0: Liquidity filter ──
        liquidity = self.indicators.liquidity_check(close, volume, min_turnover_cr=0.3)
        if not liquidity['passes']:
            self.logger.debug(f"{ticker}: Failed liquidity filter ({liquidity['daily_turnover_cr']:.1f} Cr)")
            return None

        # ── STEP 1: Market regime detection ──
        regime_info = self.indicators.detect_market_regime(high, low, close)
        regime = regime_info['regime']

        # ── STEP 2: Calculate all indicator signals ──
        signals = {}
        scores = []

        # Category-based scoring to avoid redundancy:
        # Trend: MACD, Hull, Supertrend, Ichimoku
        # Momentum: Williams %R, Laguerre, RSI
        # Volatility: VixFix
        trend_scores = []
        momentum_scores = []

        try:
            # MACD
            macd_sig = self.indicators.macd_signal(close)
            signals['macd'] = macd_sig
            s = macd_sig['strength'] if macd_sig['signal'] in ['BUY', 'BULLISH'] else \
                (-macd_sig['strength'] if macd_sig['signal'] in ['SELL', 'BEARISH'] else 0)
            if s != 0:
                scores.append(s)
                trend_scores.append(s)

            # Williams %R Exhaustion
            wr_sig = self.indicators.williams_r_exhaustion(high, low, close)
            signals['williams_r'] = wr_sig
            s = wr_sig['strength'] if wr_sig['signal'] == 'BUY' else \
                (-wr_sig['strength'] if wr_sig['signal'] == 'SELL' else 0)
            if s != 0:
                scores.append(s)
                momentum_scores.append(s)

            # VixFix
            vix_sig = self.indicators.williams_vix_fix(close, low)
            signals['vix_fix'] = vix_sig
            if vix_sig['signal'] == 'BUY':
                scores.append(vix_sig['strength'])

            # Hull Suite
            hull_sig = self.indicators.hull_suite_signal(close)
            signals['hull_ma'] = hull_sig
            s = hull_sig['strength'] if hull_sig['signal'] in ['BUY', 'BULLISH'] else \
                (-hull_sig['strength'] if hull_sig['signal'] in ['SELL', 'BEARISH'] else 0)
            if s != 0:
                scores.append(s)
                trend_scores.append(s)

            # Laguerre RSI
            lag_sig = self.indicators.laguerre_signal(close)
            signals['laguerre'] = lag_sig
            s = lag_sig['strength'] if lag_sig['signal'] in ['BUY', 'OVERSOLD'] else \
                (-lag_sig['strength'] if lag_sig['signal'] in ['SELL', 'OVERBOUGHT'] else 0)
            if s != 0:
                scores.append(s)
                momentum_scores.append(s)

            # Supertrend
            st_sig = self.indicators.supertrend_signal(high, low, close)
            signals['supertrend'] = st_sig
            s = st_sig['strength'] if st_sig['signal'] in ['BUY', 'BULLISH'] else \
                (-st_sig['strength'] if st_sig['signal'] in ['SELL', 'BEARISH'] else 0)
            if s != 0:
                scores.append(s)
                trend_scores.append(s)

            # RSI Divergence
            rsi_sig = self.indicators.rsi_divergence_signal(close)
            signals['rsi'] = rsi_sig
            s = rsi_sig['strength'] if rsi_sig['signal'] in ['BUY', 'OVERSOLD'] else \
                (-rsi_sig['strength'] if rsi_sig['signal'] in ['SELL', 'OVERBOUGHT'] else 0)
            if s != 0:
                scores.append(s)
                momentum_scores.append(s)

            # Ichimoku
            ichi_sig = self.indicators.ichimoku_signal(high, low, close)
            signals['ichimoku'] = ichi_sig
            s = ichi_sig['strength'] if ichi_sig['signal'] in ['STRONG_BUY', 'BUY', 'BULLISH'] else \
                (-ichi_sig['strength'] if ichi_sig['signal'] in ['STRONG_SELL', 'SELL', 'BEARISH'] else 0)
            if s != 0:
                scores.append(s)
                trend_scores.append(s)

        except Exception as e:
            self.logger.warning(f"Error calculating indicators for {ticker}: {e}")
            return None

        if not scores:
            return None

        # ── STEP 3: Category-weighted scoring ──
        # Use median within each category to prevent redundant indicators
        # from inflating the score, then weight by regime
        trend_vote = float(np.median(trend_scores)) if trend_scores else 0
        momentum_vote = float(np.median(momentum_scores)) if momentum_scores else 0
        raw_avg = float(np.mean(scores))

        # Regime-adaptive weighting
        if regime == "TRENDING":
            weighted_score = trend_vote * 0.55 + momentum_vote * 0.25 + raw_avg * 0.20
        elif regime == "RANGING":
            weighted_score = momentum_vote * 0.50 + trend_vote * 0.20 + raw_avg * 0.30
        else:
            weighted_score = raw_avg * 0.50 + trend_vote * 0.25 + momentum_vote * 0.25

        # Determine direction
        if weighted_score >= 40:
            direction = SignalDirection.STRONG_BUY
        elif weighted_score >= 18:
            direction = SignalDirection.BUY
        elif weighted_score <= -40:
            direction = SignalDirection.STRONG_SELL
        elif weighted_score <= -18:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        if direction == SignalDirection.NEUTRAL:
            return None

        dir_str = "BUY" if direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY] else "SELL"

        # ── STEP 4: Signal quality gate ──
        gate_passes, gate_score, gate_reasons = self.indicators.signal_quality_gate(
            high, low, close, volume, dir_str, signals
        )
        if not gate_passes:
            self.logger.debug(f"{ticker}: Failed quality gate ({gate_score:.0f}%): {gate_reasons}")
            return None

        # ── STEP 5: Volume analysis ──
        vol_analysis = self.indicators.volume_analysis(high, low, close, volume)

        # Penalize signals with OBV divergence
        if dir_str == "BUY" and vol_analysis['bearish_divergence']:
            weighted_score *= 0.7  # 30% penalty for volume divergence
        elif dir_str == "SELL" and vol_analysis['bullish_divergence']:
            weighted_score *= 0.7

        # ── STEP 6: Momentum quality ──
        mom_quality = self.indicators.momentum_quality(high, low, close, volume)

        # ── STEP 7: Relative Strength (if benchmark available) ──
        rs_info = {'mrs': 0, 'outperforming': True, 'mrs_trend': 'UNKNOWN'}
        if index_close is not None and len(index_close) > 0:
            rs_info = self.indicators.mansfield_relative_strength(close, index_close)
            # Penalize buy signals on underperforming stocks
            if dir_str == "BUY" and not rs_info['outperforming']:
                weighted_score *= 0.8  # 20% penalty
            elif dir_str == "SELL" and rs_info['outperforming']:
                weighted_score *= 0.8

        # ── STEP 7b: Multi-Timeframe Analysis ──
        mta_info = self.indicators.multi_timeframe_analysis(high, low, close, volume)

        # Penalize signals that fight the weekly trend
        if dir_str == "BUY" and mta_info['weekly_trend'] == 'DOWN':
            weighted_score *= 0.8
        elif dir_str == "SELL" and mta_info['weekly_trend'] == 'UP':
            weighted_score *= 0.8
        # Boost signals aligned with weekly trend
        elif dir_str == "BUY" and mta_info['weekly_trend'] == 'UP' and mta_info.get('aligned', False):
            weighted_score *= 1.1
        elif dir_str == "SELL" and mta_info['weekly_trend'] == 'DOWN' and mta_info.get('aligned', False):
            weighted_score *= 1.1

        # ── STEP 7c: TTM Squeeze ──
        squeeze_info = self.indicators.ttm_squeeze(high, low, close)
        signals['ttm_squeeze'] = squeeze_info

        # Squeeze firing adds conviction
        if squeeze_info['squeeze_fired']:
            if squeeze_info['momentum_direction'] == 'UP' and dir_str == "BUY":
                weighted_score *= 1.15
            elif squeeze_info['momentum_direction'] == 'DOWN' and dir_str == "SELL":
                weighted_score *= 1.15

        # ── STEP 7d: Breakout Detection ──
        breakout_info = self.indicators.detect_breakout(high, low, close, volume)
        signals['breakout'] = breakout_info

        # Breakout adds conviction and can override targets
        if breakout_info['breakout']:
            if breakout_info['direction'] == 'UP' and dir_str == "BUY":
                weighted_score *= 1.2
            elif breakout_info['direction'] == 'DOWN' and dir_str == "SELL":
                weighted_score *= 1.2

        # ── STEP 7e: Mean Reversion (for ranging markets) ──
        mr_info = self.indicators.mean_reversion_signal(high, low, close, volume)
        signals['mean_reversion'] = mr_info

        # If ranging market, mean reversion signals can supplement
        if regime in ['RANGING', 'LOW_VOLATILITY']:
            if mr_info['signal'] == 'MR_BUY' and dir_str == "BUY":
                weighted_score *= 1.1
            elif mr_info['signal'] == 'MR_SELL' and dir_str == "SELL":
                weighted_score *= 1.1

        # ── STEP 7f: Price Action Patterns ──
        pa_info = self.indicators.price_action_patterns(high, low, close)
        signals['price_action'] = pa_info

        # Structural confirmation
        if pa_info['pattern'] == 'BULLISH_STRUCTURE' and dir_str == "BUY":
            weighted_score *= 1.05
        elif pa_info['pattern'] == 'BEARISH_STRUCTURE' and dir_str == "SELL":
            weighted_score *= 1.05
        elif pa_info['pattern'] == 'BULLISH_STRUCTURE' and dir_str == "SELL":
            weighted_score *= 0.85  # Fighting the structure
        elif pa_info['pattern'] == 'BEARISH_STRUCTURE' and dir_str == "BUY":
            weighted_score *= 0.85

        # Re-check direction after all penalties/boosts
        if dir_str == "BUY" and weighted_score < 15:
            return None
        if dir_str == "SELL" and weighted_score > -15:
            return None

        # ── STEP 8: Structure-based targets ──
        entry = current_price
        targets = self.indicators.calculate_structure_targets(
            high, low, close, entry, dir_str, hold_days
        )
        target_1 = targets['target_1']
        target_2 = targets['target_2']
        target_3 = targets['target_3']

        # Override target with breakout measured move if applicable
        if breakout_info['breakout'] and breakout_info['measured_target'] > 0:
            measured = breakout_info['measured_target']
            if dir_str == "BUY" and measured > target_2:
                target_2 = measured
            elif dir_str == "SELL" and measured < target_2:
                target_2 = measured

        # For mean reversion in ranging markets, use tighter targets
        if regime in ['RANGING', 'LOW_VOLATILITY'] and mr_info['signal'] in ['MR_BUY', 'MR_SELL']:
            bb_mid_val = mr_info.get('bb_mid', 0)
            if bb_mid_val > 0:
                if dir_str == "BUY":
                    target_1 = min(target_1, bb_mid_val)  # Target the mean
                else:
                    target_1 = max(target_1, bb_mid_val)

        # ── STEP 9: Structure-based stop-loss ──
        stop_loss = self.indicators.calculate_structure_stop(
            high, low, close, entry, dir_str
        )

        # For breakouts, use box boundary as stop
        if breakout_info['breakout']:
            if dir_str == "BUY" and breakout_info['box_low'] > 0:
                box_stop = breakout_info['box_low'] * 0.995
                stop_loss = max(stop_loss, box_stop)  # Tighter stop
            elif dir_str == "SELL" and breakout_info['box_high'] > 0:
                box_stop = breakout_info['box_high'] * 1.005
                stop_loss = min(stop_loss, box_stop)

        # ── STEP 10: Dynamic R:R calculation ──
        if dir_str == "BUY":
            risk = abs(entry - stop_loss)
            reward = abs(target_2 - entry)
            expected_return = ((target_2 - entry) / entry) * 100
        else:
            risk = abs(stop_loss - entry)
            reward = abs(entry - target_2)
            expected_return = ((entry - target_2) / entry) * 100

        risk_reward = reward / risk if risk > 0 else 0

        # Require minimum R:R of 1.3 for quality signals
        if risk_reward < 1.3:
            self.logger.debug(f"{ticker}: R:R too low ({risk_reward:.2f})")
            return None

        # ── STEP 11: Build reasons and risks ──
        reasons = []
        risks_list = []

        # Regime info
        reasons.append(f"Regime: {regime} (ADX {regime_info['adx']:.0f})")

        # Multi-timeframe confirmation
        if mta_info['weekly_trend'] != 'UNKNOWN':
            if (dir_str == "BUY" and mta_info['weekly_trend'] == 'UP') or \
               (dir_str == "SELL" and mta_info['weekly_trend'] == 'DOWN'):
                reasons.append(f"Weekly trend: {mta_info['weekly_trend']} (confirmed)")
            elif mta_info['weekly_trend'] == 'MIXED':
                risks_list.append("Weekly trend: MIXED")
            else:
                risks_list.append(f"Counter weekly trend ({mta_info['weekly_trend']})")

        # Momentum grade
        if mom_quality['grade'] in ['A', 'B']:
            reasons.append(f"Momentum: {mom_quality['grade']}-grade ({mom_quality['momentum_score']})")
        elif mom_quality['grade'] == 'D':
            risks_list.append(f"Weak momentum (grade {mom_quality['grade']})")

        # Volume
        if vol_analysis['volume_confirmed']:
            reasons.append(f"Volume {vol_analysis['volume_ratio']:.1f}x avg")
        else:
            risks_list.append(f"Low volume ({vol_analysis['volume_ratio']:.1f}x)")

        # RS
        if rs_info.get('mrs', 0) > 0:
            reasons.append(f"RS+ (MRS {rs_info['mrs']:.1f})")
        elif rs_info.get('mrs', 0) < -2:
            risks_list.append(f"Underperforming (MRS {rs_info['mrs']:.1f})")

        # TTM Squeeze
        if squeeze_info['squeeze_fired']:
            reasons.append(f"Squeeze fired ({squeeze_info['momentum_direction']})")
        elif squeeze_info['is_squeezing'] and squeeze_info['squeeze_bars'] >= 6:
            reasons.append(f"Squeeze building ({squeeze_info['squeeze_bars']} bars)")

        # Breakout
        if breakout_info['breakout']:
            reasons.append(f"Breakout {breakout_info['direction']} (range {breakout_info['box_range_pct']:.1f}%)")
        elif breakout_info['consolidating']:
            reasons.append(f"Consolidating ({breakout_info['box_range_pct']:.1f}% range)")

        # Mean reversion
        if mr_info['signal'] in ['MR_BUY', 'MR_SELL']:
            reasons.append(f"Mean reversion ({mr_info['signal']}, BB%B={mr_info['bb_pctb']:.2f})")

        # Price action structure
        if pa_info['pattern'] == 'BULLISH_STRUCTURE' and dir_str == "BUY":
            reasons.append(f"Bullish structure (HH:{pa_info['hh_count']} HL:{pa_info['hl_count']})")
        elif pa_info['pattern'] == 'BEARISH_STRUCTURE' and dir_str == "SELL":
            reasons.append(f"Bearish structure (LH:{pa_info['lh_count']} LL:{pa_info['ll_count']})")
        elif pa_info['pattern'] == 'COMPRESSION':
            reasons.append(f"Price compression ({pa_info['inside_bars']} inside bars)")
        elif pa_info['pattern'] == 'BULLISH_STRUCTURE' and dir_str == "SELL":
            risks_list.append("Counter bullish price structure")
        elif pa_info['pattern'] == 'BEARISH_STRUCTURE' and dir_str == "BUY":
            risks_list.append("Counter bearish price structure")

        # Target confluence
        if targets.get('confluence_count', 1) >= 3:
            reasons.append(f"Target confluence ({targets['confluence_count']} methods)")

        # Technical indicator reasons
        if signals['macd']['signal'] in ['BUY', 'BULLISH']:
            reasons.append(f"MACD {signals['macd']['signal'].lower()}")
        elif signals['macd']['signal'] in ['SELL', 'BEARISH']:
            if dir_str == "SELL":
                reasons.append(f"MACD {signals['macd']['signal'].lower()}")
            else:
                risks_list.append(f"MACD {signals['macd']['signal'].lower()}")

        if signals['supertrend']['signal'] in ['BUY', 'BULLISH']:
            if dir_str == "BUY":
                reasons.append(f"Supertrend {signals['supertrend']['direction']}")
            else:
                risks_list.append(f"Supertrend {signals['supertrend']['direction']}")
        elif signals['supertrend']['signal'] in ['SELL', 'BEARISH']:
            if dir_str == "SELL":
                reasons.append(f"Supertrend {signals['supertrend']['direction']}")
            else:
                risks_list.append(f"Supertrend {signals['supertrend']['direction']}")

        if signals['ichimoku']['signal'] in ['STRONG_BUY', 'BUY', 'BULLISH']:
            if dir_str == "BUY":
                reasons.append(f"Ichimoku {signals['ichimoku']['position']}")
        elif signals['ichimoku']['signal'] in ['STRONG_SELL', 'SELL', 'BEARISH']:
            if dir_str == "SELL":
                reasons.append(f"Ichimoku {signals['ichimoku']['position']}")

        if signals['rsi']['bullish_divergence']:
            reasons.append("RSI bullish divergence")
        if signals['rsi']['bearish_divergence']:
            if dir_str == "SELL":
                reasons.append("RSI bearish divergence")
            else:
                risks_list.append("RSI bearish divergence")

        if signals['williams_r']['oversold_exhaustion']:
            reasons.append("Williams %R oversold exhaustion")
        if signals['williams_r']['overbought_exhaustion']:
            if dir_str == "SELL":
                reasons.append("Williams %R overbought exhaustion")
            else:
                risks_list.append("Williams %R overbought exhaustion")

        if signals['vix_fix']['is_bottom_signal']:
            reasons.append("VixFix bottom signal")

        if signals['hull_ma']['hull_rising'] and dir_str == "BUY":
            reasons.append("Hull MA rising")
        elif not signals['hull_ma']['hull_rising'] and dir_str == "SELL":
            reasons.append("Hull MA falling")
        elif not signals['hull_ma']['hull_rising'] and dir_str == "BUY":
            risks_list.append("Hull MA falling")

        # Get fundamental data if enabled
        fundamental = None
        fundamental_score = 0.0

        if self.use_fundamentals and self.fundamental_enhancer:
            fundamental = self.fundamental_enhancer.get_fundamental_data(ticker)

            if fundamental.has_recent_news:
                # Check for fundamental/technical alignment
                news_aligned = (
                    (dir_str == "BUY" and fundamental.news_sentiment == 'BULLISH') or
                    (dir_str == "SELL" and fundamental.news_sentiment == 'BEARISH') or
                    fundamental.news_sentiment == 'NEUTRAL'
                )

                # Add fundamental reasons/risks based on alignment
                if fundamental.news_sentiment == 'BULLISH':
                    if dir_str == "BUY":
                        reasons.append(f"News: BULLISH (aligned)")
                        for catalyst in fundamental.catalysts[:2]:
                            if catalyst['sentiment'] == 'BULLISH':
                                reasons.append(f"{catalyst['type'].replace('_', ' ').title()}: {catalyst['headline'][:50]}...")
                    else:
                        risks_list.append(f"Contrary news: BULLISH")
                elif fundamental.news_sentiment == 'BEARISH':
                    if dir_str == "SELL":
                        reasons.append(f"News: BEARISH (aligned)")
                        for catalyst in fundamental.catalysts[:2]:
                            if catalyst['sentiment'] == 'BEARISH':
                                reasons.append(f"{catalyst['type'].replace('_', ' ').title()}: {catalyst['headline'][:50]}...")
                    else:
                        risks_list.append(f"Contrary news: BEARISH")

                # Earnings
                if fundamental.earnings_surprise == 'BEAT':
                    reasons.append("Recent earnings beat (PEAD)")
                elif fundamental.earnings_surprise == 'MISS':
                    risks_list.append("Recent earnings miss")

                # Institutional flow (market-wide context)
                if fundamental.fii_sentiment == 'BULLISH' and dir_str == "BUY":
                    reasons.append("FII flow: BULLISH")
                elif fundamental.fii_sentiment == 'BEARISH' and dir_str == "BUY":
                    risks_list.append("FII flow: BEARISH")

                # Policy
                if fundamental.has_policy_boost and fundamental.relevant_policies:
                    policy_info = fundamental.relevant_policies[0]
                    reasons.append(f"Policy: {policy_info.get('keyword', 'govt policy')}")
                elif fundamental.policy_score < -10:
                    risks_list.append("Negative policy impact")

                # Fundamental score with alignment consideration
                alignment_mult = 1.0 if news_aligned else 0.5
                fundamental_score = (
                    fundamental.news_score * alignment_mult +
                    fundamental.pead_score +
                    fundamental.institutional_score * 0.2 +
                    fundamental.policy_score * 0.2 +
                    fundamental.quality_score * 0.3
                )

                # Quality reasons
                if fundamental.quality_score >= 70:
                    reasons.append(f"Quality: {fundamental.quality_score:.0f}/100")
                elif fundamental.quality_score < 40 and fundamental.quality_score > 0:
                    risks_list.append(f"Low quality ({fundamental.quality_score:.0f}/100)")
                if not fundamental.passes_quality:
                    risks_list.append("Fails quality checks")

        # ── STEP 12: Combined score ──
        # Technical confidence includes quality gate and momentum
        tech_confidence = min(abs(weighted_score), 100)
        gate_bonus = gate_score * 0.1  # Up to 10 points from gate quality
        mom_bonus = mom_quality['momentum_score'] * 0.1  # Up to 10 from momentum

        combined_score = (tech_confidence * 0.55) + (fundamental_score * 0.25) + gate_bonus + mom_bonus

        # Alignment boost (only when tech and fundamental agree)
        if fundamental and fundamental.has_recent_news:
            if (dir_str == "BUY" and fundamental.news_sentiment == 'BULLISH'):
                combined_score *= 1.12
            elif (dir_str == "SELL" and fundamental.news_sentiment == 'BEARISH'):
                combined_score *= 1.12
            elif (dir_str == "BUY" and fundamental.news_sentiment == 'BEARISH'):
                combined_score *= 0.85  # Penalize conflicting signals
            elif (dir_str == "SELL" and fundamental.news_sentiment == 'BULLISH'):
                combined_score *= 0.85

        # Volume confirmation bonus
        if vol_analysis['volume_confirmed']:
            combined_score *= 1.05

        # RS bonus
        if rs_info.get('outperforming', False) and rs_info.get('mrs_trend') == 'RISING':
            combined_score *= 1.05

        # MTA alignment bonus
        if (dir_str == "BUY" and mta_info['weekly_trend'] == 'UP') or \
           (dir_str == "SELL" and mta_info['weekly_trend'] == 'DOWN'):
            combined_score *= 1.05

        # Squeeze/breakout bonus
        if squeeze_info['squeeze_fired']:
            combined_score *= 1.08
        if breakout_info['breakout']:
            combined_score *= 1.10

        # Price action structure alignment bonus
        if (pa_info['pattern'] == 'BULLISH_STRUCTURE' and dir_str == "BUY") or \
           (pa_info['pattern'] == 'BEARISH_STRUCTURE' and dir_str == "SELL"):
            combined_score *= 1.05

        return TechnicalSignal(
            ticker=ticker,
            timestamp=str(datetime.now()),
            direction=direction,
            confidence=tech_confidence,
            current_price=current_price,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            expected_return_pct=round(expected_return, 2),
            risk_reward_ratio=round(risk_reward, 2),
            suggested_hold_days=hold_days,
            indicators=signals,
            reasons=reasons,
            risks=risks_list,
            fundamental=fundamental,
            combined_score=min(combined_score, 100),
            regime=regime,
            regime_adx=regime_info['adx'],
            momentum_grade=mom_quality['grade'],
            momentum_score=mom_quality['momentum_score'],
            volume_ratio=vol_analysis['volume_ratio'],
            volume_confirmed=vol_analysis['volume_confirmed'],
            rs_mrs=rs_info.get('mrs', 0),
            rs_outperforming=rs_info.get('outperforming', False),
            weekly_trend=mta_info.get('weekly_trend', ''),
            squeeze_active=squeeze_info.get('is_squeezing', False) or squeeze_info.get('squeeze_fired', False),
            breakout_detected=breakout_info.get('breakout', False),
            price_action=pa_info.get('pattern', ''),
        )

    def scan_stocks(
        self,
        tickers: List[str],
        hold_days: int = 7,
        min_confidence: float = 30.0,
        progress_callback=None
    ) -> List[TechnicalSignal]:
        """
        Scan multiple stocks and return sorted signals.

        Fetches Nifty 50 index data for Mansfield Relative Strength.
        Applies liquidity filter, quality gates, and minimum R:R.

        Args:
            tickers: List of stock tickers to scan
            hold_days: Suggested holding period
            min_confidence: Minimum confidence to include
            progress_callback: Optional callback(current, total, ticker)

        Returns:
            List of TechnicalSignal sorted by combined score (technical + fundamental)
        """
        import yfinance as yf

        signals = []
        total = len(tickers)

        # Fetch benchmark index for Relative Strength calculation
        index_close = None
        try:
            index_data = yf.Ticker("^NSEI").history(period="1y")
            if len(index_data) > 60:
                index_close = index_data['Close'].values
        except Exception as e:
            self.logger.debug(f"Could not fetch Nifty index for RS: {e}")

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="6mo")

                if len(df) < 60:
                    continue

                signal = self.analyze_stock(ticker, df, hold_days, index_close)

                if signal and signal.confidence >= min_confidence:
                    signals.append(signal)

            except Exception as e:
                self.logger.debug(f"Error scanning {ticker}: {e}")
                continue

        # Sort by combined score descending
        if self.use_fundamentals:
            signals.sort(key=lambda s: s.combined_score, reverse=True)
        else:
            signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals


# Popular Indian stock tickers for scanning
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "HDFCLIFE.NS",
    "TECHM.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS", "ONGC.NS", "ADANIENT.NS",
    "TATASTEEL.NS", "GRASIM.NS", "ADANIPORTS.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "SBILIFE.NS", "BRITANNIA.NS", "COALINDIA.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "INDUSINDBK.NS", "BPCL.NS", "APOLLOHOSP.NS", "TATACONSUM.NS",
    "UPL.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "BEL.NS",
]

NIFTY_NEXT_50 = [
    "ADANIGREEN.NS", "AMBUJACEM.NS", "BANKBARODA.NS", "BERGEPAINT.NS",
    "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS",
    "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "ICICIGI.NS", "INDIGO.NS", "IOC.NS", "JINDALSTEL.NS", "LICI.NS",
    "MARICO.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PIDILITIND.NS", "PNB.NS",
    "SBICARD.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "TORNTPHARM.NS",
    "TRENT.NS", "VEDL.NS", "ETERNAL.NS", "ZYDUSLIFE.NS",
]

# Nifty 200 additional stocks (beyond Nifty 100)
NIFTY_200_EXTRA = [
    "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ALKEM.NS",
    "ASHOKLEY.NS", "ASTRAL.NS", "AUROPHARMA.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
    "BHEL.NS", "BIOCON.NS", "BHARATFORG.NS", "CANFINHOME.NS",
    "CGPOWER.NS", "CHAMBLFERT.NS", "COFORGE.NS", "CONCOR.NS", "CROMPTON.NS",
    "CUB.NS", "CUMMINSIND.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS",
    "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FSL.NS",
    "GLAND.NS", "GLAXO.NS", "GMRAIRPORT.NS", "GNFC.NS", "GODREJPROP.NS",
    "GSPL.NS", "GUJGASLTD.NS", "HAL.NS", "HDFCAMC.NS", "HINDPETRO.NS",
    "HONAUT.NS", "IDFCFIRSTB.NS", "IGL.NS", "IIFL.NS", "INDUSTOWER.NS",
    "IRCTC.NS", "IRFC.NS", "IPCALAB.NS", "JKCEMENT.NS", "JUBLFOOD.NS",
    "KANSAINER.NS", "KEI.NS", "KPITTECH.NS", "LTF.NS", "LAURUSLABS.NS",
    "LICHSGFIN.NS", "LUPIN.NS", "MANAPPURAM.NS", "MFSL.NS",
    "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "NAM-INDIA.NS",
    "NATIONALUM.NS", "NAVINFLUOR.NS", "NHPC.NS", "NMDC.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "OIL.NS", "PAGEIND.NS", "PAYTM.NS", "PERSISTENT.NS",
    "PETRONET.NS", "PFC.NS", "PFIZER.NS", "PIIND.NS", "PNB.NS",
    "POLYCAB.NS", "POONAWALLA.NS", "PRESTIGE.NS", "PVRINOX.NS", "RAMCOCEM.NS",
    "RECLTD.NS", "SAIL.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SONACOMS.NS",
    "STARHEALTH.NS", "SUNTV.NS", "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS",
    "TATACOMM.NS", "TATAELXSI.NS", "TATAPOWER.NS", "THERMAX.NS", "TIINDIA.NS",
    "TORNTPOWER.NS", "TVSMOTOR.NS", "UBL.NS", "UNIONBANK.NS", "UNITDSPR.NS",
    "VBL.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS", "ZEEL.NS",
]

NIFTY_200 = NIFTY_50 + NIFTY_NEXT_50 + NIFTY_200_EXTRA

# F&O Stocks (stocks available for futures and options trading)
FNO_STOCKS = NIFTY_200 + [
    "AARTIIND.NS", "APOLLOTYRE.NS", "ATUL.NS", "AUBANK.NS", "BATAINDIA.NS",
    "COROMANDEL.NS", "DELTACORP.NS", "EMAMILTD.NS", "GLENMARK.NS",
    "GRANULES.NS", "GRAPHITE.NS", "GSFC.NS", "HINDCOPPER.NS",
    "INDHOTEL.NS", "INTELLECT.NS", "IRB.NS",
    "JSWENERGY.NS", "LTTS.NS", "METROPOLIS.NS", "NIACL.NS",
    "NOCIL.NS", "RBLBANK.NS", "RELAXO.NS", "SOLARINDS.NS",
    "SUMICHEM.NS", "UJJIVANSFB.NS",
]

# Broad market - all major NSE stocks (~500)
BROAD_MARKET_EXTRA = [
    "3MINDIA.NS", "AARTIDRUGS.NS", "AAVAS.NS", "AWL.NS", "AFFLE.NS",
    "AJANTPHARM.NS", "ALKYLAMINE.NS", "ANGELONE.NS", "APLAPOLLO.NS", "APTUS.NS",
    "ASAHIINDIA.NS", "ASTRAZEN.NS", "AVANTIFEED.NS", "BASF.NS", "BAYERCROP.NS",
    "BCG.NS", "BEML.NS", "BORORENEW.NS", "BRIGADE.NS", "BSE.NS",
    "CARERATING.NS", "CASTROLIND.NS", "CEATLTD.NS", "ABREL.NS", "CENTURYPLY.NS",
    "CESC.NS", "CHALET.NS", "CLEAN.NS", "COCHINSHIP.NS", "CYIENT.NS",
    "DCMSHRIRAM.NS", "DEVYANI.NS", "DMART.NS", "ECLERX.NS",
    "EDELWEISS.NS", "ELGIEQUIP.NS", "EMUDHRA.NS", "ENDURANCE.NS", "EPL.NS",
    "EQUITASBNK.NS", "FINCABLES.NS", "FINPIPE.NS", "FLUOROCHEM.NS", "FORTIS.NS",
    "GARFIBRES.NS", "GILLETTE.NS", "GLOBUSSPR.NS", "GODREJIND.NS", "GRINDWELL.NS",
    "GRINFRA.NS", "HAPPSTMNDS.NS", "HATSUN.NS", "HGS.NS", "HIKAL.NS",
    "HSCL.NS", "HUDCO.NS", "ICRA.NS", "INDIGOPNTS.NS",
    "INDOCO.NS", "INOXWIND.NS", "IONEXCHANG.NS",
    "ITI.NS", "JAMNAAUTO.NS", "JBCHEPHARM.NS", "JINDALSAW.NS",
    "JKPAPER.NS", "JKTYRE.NS", "JSWINFRA.NS", "JTEKTINDIA.NS",
    "JUSTDIAL.NS", "KAJARIACER.NS", "KPIL.NS", "KALYANKJIL.NS", "KEC.NS",
    "KIOCL.NS", "KNRCON.NS", "KRBL.NS", "KSB.NS", "LATENTVIEW.NS",
    "LINDEINDIA.NS", "LUXIND.NS", "MAHABANK.NS",
    "MAHLOG.NS", "MAHSEAMLES.NS", "MANINFRA.NS", "MAPMYINDIA.NS",
    "MARKSANS.NS", "MASTEK.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MCX.NS",
    "MEDANTA.NS", "MEDPLUS.NS", "METROBRAND.NS", "MIDHANI.NS", "MINDACORP.NS",
    "MMTC.NS", "MOIL.NS", "MRPL.NS", "NATCOPHARM.NS", "NAUKRI.NS",
    "NBCC.NS", "NCC.NS", "NEWGEN.NS", "NSLNISP.NS", "NUCLEUS.NS",
    "OLECTRA.NS", "ORIENTELEC.NS", "PCBL.NS", "PDSL.NS", "PGHH.NS",
    "PHOENIXLTD.NS", "POLICYBZR.NS", "PRINCEPIPE.NS", "PRSMJOHNSN.NS", "PSB.NS",
    "QUESS.NS", "RADICO.NS", "RAILTEL.NS", "RAIN.NS", "RAJESHEXPO.NS",
    "RALLIS.NS", "RKFORGE.NS", "RVNL.NS", "SAPPHIRE.NS", "SARDAEN.NS",
    "SJVN.NS", "SKFINDIA.NS", "SOBHA.NS", "SOLARA.NS", "SPARC.NS",
    "SPANDANA.NS", "STLTECH.NS", "SUDARSCHEM.NS", "SUNDRMFAST.NS", "SUNFLAG.NS",
    "SUNTECK.NS", "SWSOLAR.NS", "TANLA.NS", "TASTYBITE.NS", "TATAINVEST.NS",
    "TEAMLEASE.NS", "TECHNOE.NS", "THYROCARE.NS", "TIMKEN.NS",
    "TRITURBINE.NS", "UCOBANK.NS", "UFLEX.NS", "UTIAMC.NS",
    "VAIBHAVGBL.NS", "VARROC.NS", "VGUARD.NS", "VINATIORGA.NS", "VIPIND.NS",
    "VRLLOG.NS", "VSTIND.NS", "WELCORP.NS", "WESTLIFE.NS",
    "WOCKPHARMA.NS", "ZENITHSTL.NS", "ZENTEC.NS", "ZFCVINDIA.NS",
]

BROAD_MARKET = FNO_STOCKS + BROAD_MARKET_EXTRA

# Remove duplicates while preserving order
BROAD_MARKET = list(dict.fromkeys(BROAD_MARKET))

# Extended NSE stocks - Mid-cap and Small-cap (~500 additional stocks)
# Verified and cleaned list - removed delisted/invalid tickers
NSE_MIDCAP_SMALLCAP = [
    # Midcap stocks (verified)
    "AARTISURF.NS", "AAVAS.NS", "ACCELYA.NS", "ACE.NS",
    "ADANIENSOL.NS", "ADANIPOWER.NS", "ADVENZYMES.NS", "AETHER.NS", "AFFLE.NS",
    "AGARIND.NS", "AIAENG.NS", "AJMERA.NS", "ALEMBICLTD.NS", "ALKYLAMINE.NS",
    "ALLCARGO.NS", "AMBER.NS", "AMRUTANJAN.NS", "ANANTRAJ.NS",
    "ANGELONE.NS", "APCOTEXIND.NS", "APLAPOLLO.NS", "APLLTD.NS", "ARCHIDPLY.NS",
    "ARVINDFASN.NS", "ARVIND.NS", "ASHIANA.NS", "ASHOKA.NS", "ASIANHOTNR.NS",
    "ASTERDM.NS", "ASTRAMICRO.NS", "ATUL.NS", "AURIONPRO.NS", "AUTOAXLES.NS",
    "AVTNPL.NS", "AXISCADES.NS",
    # Banking and finance (verified)
    "BAJAJHIND.NS", "BALAMINES.NS", "BALMLAWRIE.NS", "BANARISUG.NS", "BANKINDIA.NS",
    "BASF.NS", "BATAINDIA.NS", "BBTC.NS", "BDL.NS", "BEL.NS",
    "BEPL.NS", "BFUTILITIE.NS", "BLS.NS", "BLUESTARCO.NS", "BODALCHEM.NS",
    "BOROLTD.NS", "BORORENEW.NS", "BOSCHLTD.NS", "BPCL.NS", "BUTTERFLY.NS",
    # Capital goods (verified)
    "CAMLINFINE.NS", "CANFINHOME.NS", "CANTABIL.NS", "CAPACITE.NS", "CAPLIPOINT.NS",
    "CARBORUNIV.NS", "CARERATING.NS", "CASTROLIND.NS", "CCL.NS", "CDSL.NS",
    "CEATLTD.NS", "CENTURYPLY.NS", "CERA.NS", "CESC.NS", "CGCL.NS",
    "CGPOWER.NS", "CHALET.NS", "CHAMBLFERT.NS", "CHEMCON.NS", "CHENNPETRO.NS",
    "CHEVIOT.NS", "CHOICEIN.NS", "CLEAN.NS", "COALINDIA.NS",
    # Consumer goods (verified)
    "COCHINSHIP.NS", "COLPAL.NS", "CONFIPET.NS", "CONTROLPR.NS", "COROMANDEL.NS",
    "COSMOFIRST.NS", "CRAFTSMAN.NS", "CRISIL.NS", "CSBBANK.NS", "CUB.NS",
    "CYIENT.NS", "CYIENTDLM.NS", "DABUR.NS", "DALMIASUG.NS", "DATAMATICS.NS",
    "DATAPATTNS.NS", "DBCORP.NS", "DBREALTY.NS", "DCM.NS", "DCMSHRIRAM.NS",
    "DCW.NS", "DECCANCE.NS",
    # Diversified (verified)
    "DEEPAKFERT.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DENORA.NS",
    "DIAMONDYD.NS", "DICIND.NS", "DIXON.NS", "DLF.NS", "DMART.NS",
    "DOLLAR.NS", "DONEAR.NS", "DPABHUSHAN.NS", "DRREDDY.NS", "DWARKESH.NS",
    # Engineering (verified)
    "DYNAMATECH.NS", "EASEMYTRIP.NS", "ECLERX.NS", "EDELWEISS.NS", "EICHERMOT.NS",
    "EIDPARRY.NS", "EIHOTEL.NS", "ELECON.NS", "ELGIEQUIP.NS", "EMAMILTD.NS",
    "EMKAY.NS", "EMMBI.NS", "ENDURANCE.NS", "ENGINERSIN.NS", "ENIL.NS",
    "EPL.NS", "EQUITASBNK.NS", "ESABINDIA.NS", "ESCORTS.NS",
    # Financial services (verified)
    "ETHOSLTD.NS", "EVERESTIND.NS", "EXIDEIND.NS", "FACT.NS", "FDC.NS",
    "FEDERALBNK.NS", "FILATEX.NS", "FINCABLES.NS", "FINPIPE.NS", "FIRSTCRY.NS",
    "FMGOETZE.NS", "FORCEMOT.NS", "FOSECOIND.NS", "FRETAIL.NS", "FSL.NS",
    "GABRIEL.NS",
    # Healthcare (verified)
    "GAEL.NS", "GAIL.NS", "GALAXYSURF.NS", "GALLANTT.NS", "GANDHAR.NS",
    "GANDHITUBE.NS", "GANECOS.NS", "GANESHBE.NS", "GARFIBRES.NS", "GATEWAY.NS",
    "GEECEE.NS", "GENUSPOWER.NS", "GESHIP.NS", "GHCL.NS", "GICHSGFIN.NS",
    "GILLETTE.NS", "GIPCL.NS", "GLAND.NS",
    # Industrial (verified)
    "GLAXO.NS", "GLENMARK.NS", "GLOBUSSPR.NS", "GMMPFAUDLR.NS", "GMRAIRPORT.NS",
    "GNFC.NS", "GOACARBON.NS", "GODFRYPHLP.NS", "GODREJAGRO.NS", "GODREJCP.NS",
    "GODREJIND.NS", "GODREJPROP.NS", "GOKUL.NS", "GOLDIAM.NS", "GOLDENTOBC.NS",
    "GOODLUCK.NS", "GPIL.NS", "GPPL.NS", "GRANULES.NS", "GRAPHITE.NS",
    "GRASIM.NS", "GRAVITA.NS",
    # IT and tech (verified)
    "GREENLAM.NS", "GREENPLY.NS", "GRINDWELL.NS", "GRPLTD.NS", "GRSE.NS",
    "GSFC.NS", "GSPL.NS", "GTL.NS", "GTLINFRA.NS", "GUFICBIO.NS",
    "GUJALKALI.NS", "GUJGASLTD.NS", "GULFOILLUB.NS", "HAL.NS", "HAPPSTMNDS.NS",
    "HARIOMPIPE.NS", "HATHWAY.NS",
    # Metals and mining (verified)
    "HATSUN.NS", "HAVELLS.NS", "HCC.NS", "HCLTECH.NS", "HDFCAMC.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEG.NS", "HEIDELBERG.NS", "HERCULES.NS",
    "HERITGFOOD.NS", "HEROMOTOCO.NS", "HESTERBIO.NS", "HFCL.NS", "HGINFRA.NS",
    "HIKAL.NS", "HIMATSEIDE.NS",
    # Pharma (verified)
    "HINDALCO.NS", "HINDCOPPER.NS", "HINDOILEXP.NS", "HINDPETRO.NS", "HINDUNILVR.NS",
    "HINDZINC.NS", "HITECH.NS", "HMVL.NS", "HOMEFIRST.NS", "HONAUT.NS",
    "HSCL.NS", "HTMEDIA.NS", "HUDCO.NS", "IBREALEST.NS", "ICICIBANK.NS",
    # Real estate (verified)
    "ICICIGI.NS", "ICICIPRULI.NS", "ICRA.NS", "IDBI.NS", "IDEA.NS",
    "IDFCFIRSTB.NS", "IEX.NS", "IFBAGRO.NS", "IFBIND.NS", "IFCI.NS",
    "IFGLEXPOR.NS", "IGARASHI.NS", "IIFL.NS", "IIFLCAPS.NS", "IMAGICAA.NS",
    "IMFA.NS", "INDIACEM.NS", "INDIAMART.NS",
    # Telecom (verified)
    "INDIANB.NS", "INDIANHUME.NS", "INDIGO.NS", "INDIGOPNTS.NS", "INDNIPPON.NS",
    "INDOCO.NS", "INDORAMA.NS", "INDOSTAR.NS", "INDOWIND.NS", "INDUSTOWER.NS",
    "INFIBEAM.NS", "INFY.NS", "INOXGREEN.NS", "INOXWIND.NS", "INSECTICID.NS",
    "INTELLECT.NS", "IOB.NS", "IOC.NS", "IOLCP.NS",
    # Textiles (verified)
    "IONEXCHANG.NS", "IPCALAB.NS", "IRB.NS", "IRCON.NS", "IRCTC.NS",
    "IRFC.NS", "ISGEC.NS", "ITDC.NS", "ITI.NS",
    "JISLJALEQS.NS", "JKCEMENT.NS", "JKIL.NS", "JKLAKSHMI.NS", "JKPAPER.NS",
    "JKTYRE.NS", "JMFINANCIL.NS", "JPASSOCIAT.NS", "JSL.NS",
    # Utilities (verified)
    "JSWENERGY.NS", "JSWHL.NS", "JSWINFRA.NS", "JSWSTEEL.NS", "JTEKTINDIA.NS",
    "JTLIND.NS", "JUBLFOOD.NS", "JUBLPHARMA.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS",
    "KAJARIACER.NS", "KPIL.NS", "KALYANKJIL.NS", "KAMDHENU.NS",
    "KANSAINER.NS", "KAYNES.NS", "KDDL.NS", "KEC.NS", "KEI.NS",
    "KESORAMIND.NS", "KFINTECH.NS", "KILITCH.NS", "KIOCL.NS", "KIRIINDUS.NS",
    "KIRLOSENG.NS", "KIRLOSIND.NS", "KITEX.NS", "KNRCON.NS", "KOLTEPATIL.NS",
    "KOPRAN.NS", "KOTAKBANK.NS", "KPIGREEN.NS", "KPIL.NS", "KPITTECH.NS",
    "KRBL.NS", "KSB.NS", "KSCL.NS", "KTKBANK.NS",
    "LALPATHLAB.NS", "LAOPALA.NS", "LATENTVIEW.NS", "LAURUSLABS.NS", "LEMONTREE.NS",
    "LICHSGFIN.NS", "LINDEINDIA.NS", "LT.NS", "LTF.NS", "LTFOODS.NS",
    "LTIM.NS", "LTTS.NS", "LUMAXIND.NS", "LUMAXTECH.NS", "LUPIN.NS",
    "LUXIND.NS", "LXCHEM.NS",
]

# BSE-specific stocks (using named tickers with .BO suffix)
# Note: BSE numeric scrip codes (500xxx.BO) don't work on Yahoo Finance
# Use named tickers only. Many large-cap BSE tickers are duplicates of NSE
# tickers already in other lists, so this focuses on BSE-primary stocks.
BSE_ADDITIONAL = [
    # Large BSE-listed stocks (named tickers that work on Yahoo Finance)
    "AARTIIND.BO", "ABB.BO", "ACC.BO", "ADANIENT.BO", "ADANIPORTS.BO",
    "AMBUJACEM.BO", "APOLLOHOSP.BO", "ASIANPAINT.BO", "AXISBANK.BO", "BAJAJ-AUTO.BO",
    "BAJAJFINSV.BO", "BAJFINANCE.BO", "BHARTIARTL.BO", "BPCL.BO", "BRITANNIA.BO",
    "CIPLA.BO", "COALINDIA.BO", "DIVISLAB.BO", "DRREDDY.BO", "EICHERMOT.BO",
    "GAIL.BO", "GRASIM.BO", "HCLTECH.BO", "HDFCBANK.BO", "HDFCLIFE.BO",
    "HEROMOTOCO.BO", "HINDALCO.BO", "HINDUNILVR.BO", "ICICIBANK.BO", "INDUSINDBK.BO",
    "INFY.BO", "IOC.BO", "ITC.BO", "JSWSTEEL.BO", "KOTAKBANK.BO",
    "LT.BO", "MARUTI.BO", "NESTLEIND.BO", "NTPC.BO", "ONGC.BO",
    "POWERGRID.BO", "RELIANCE.BO", "SBIN.BO", "SBILIFE.BO", "SHREECEM.BO",
    "SUNPHARMA.BO", "TATACONSUM.BO", "TATASTEEL.BO", "TCS.BO",  # Removed TATAMOTORS.BO (use .NS)
    "TECHM.BO", "TITAN.BO", "ULTRACEMCO.BO", "UPL.BO", "WIPRO.BO",
]

# More NSE small-cap and micro-cap stocks
# Cleaned: removed delisted/invalid tickers verified against Yahoo Finance
NSE_SMALLCAP_EXTRA = [
    # M stocks (verified active)
    "MAHABANK.NS", "MAHLOG.NS", "MAHSEAMLES.NS", "MAITHANALL.NS",
    "MANAKSIA.NS", "MANALIPETC.NS", "MANAPPURAM.NS", "MANGALAM.NS",
    "MANINDS.NS", "MANINFRA.NS", "MANKIND.NS",
    "MANORG.NS", "MARATHON.NS", "MARKSANS.NS",
    "MASFIN.NS", "MASTEK.NS", "MATRIMONY.NS", "MAXHEALTH.NS",
    "MAYURUNIQ.NS", "MAZDOCK.NS",
    "MCX.NS", "MEDANTA.NS",
    "MEDPLUS.NS", "METROBRAND.NS",
    "METROPOLIS.NS", "MFSL.NS", "MGL.NS",
    "MHRIL.NS", "MIDHANI.NS", "MINDACORP.NS",
    "MINDTECK.NS", "MIRCELECTR.NS",
    "MMTC.NS", "MODIRUBBER.NS",
    "MOIL.NS", "MOL.NS",
    "MOREPENLAB.NS", "MOTHERSON.NS", "MOTILALOFS.NS",
    "MPHASIS.NS", "MPSLTD.NS", "MRF.NS", "MRPL.NS",
    "MSPL.NS", "MSTCLTD.NS", "MTNL.NS",
    "MUKANDLTD.NS", "MUKTAARTS.NS", "MUNJALAU.NS", "MUNJALSHOW.NS",
    "MUTHOOTCAP.NS", "MUTHOOTFIN.NS", "NACLIND.NS",
    "NAM-INDIA.NS", "NATCOPHARM.NS",
    "NATHBIOGEN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NAVINFLUOR.NS",
    "NBCC.NS", "NCC.NS",
    "NCLIND.NS", "NDTV.NS", "NECLIFE.NS",
    "NELCAST.NS", "NELCO.NS", "NEOGEN.NS", "NESCO.NS",
    "NESTLEIND.NS", "NETWORK18.NS", "NEULANDLAB.NS", "NEWGEN.NS",
    "NFL.NS", "NHPC.NS", "NIACL.NS",
    "NIITLTD.NS", "NILKAMAL.NS",
    "NLCINDIA.NS", "NMDC.NS", "NOCIL.NS",
    "NRBBEARING.NS", "NSLNISP.NS",
    "NTPC.NS", "NUCLEUS.NS", "NURECA.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "OIL.NS", "OLECTRA.NS",
    "ONGC.NS", "ONMOBILE.NS",
    "ORIENTBELL.NS",
    "ORIENTCEM.NS", "ORIENTELEC.NS", "ORIENTLTD.NS",
    "PAGEIND.NS",
    "PALREDTEC.NS", "PANACEABIO.NS", "PATANJALI.NS",
    "PARACABLES.NS", "PARAGMILK.NS",
    "PATELENG.NS", "PAUSHAKLTD.NS", "PAYTM.NS",
    "PCBL.NS", "PCJEWELLER.NS", "PDSL.NS",
    "PENIND.NS", "PERSISTENT.NS", "PETRONET.NS",
    "PFC.NS", "PFIZER.NS", "PGHH.NS",
    "PHOENIXLTD.NS", "PIDILITIND.NS",
    "PIIND.NS", "PILANIINVS.NS", "PNBGILTS.NS",
    "PNBHOUSING.NS", "PNB.NS", "PNCINFRA.NS", "POKARNA.NS",
    "POLICYBZR.NS", "POLYCAB.NS", "POLYMED.NS",
    "POLYPLEX.NS", "POONAWALLA.NS", "POWERGRID.NS", "POWERMECH.NS",
    "PRAJIND.NS", "PRAKASH.NS",
    "PRECOT.NS", "PRECWIRE.NS",
    "PRESTIGE.NS", "PRICOLLTD.NS", "PRINCEPIPE.NS",
    "PRSMJOHNSN.NS", "PSB.NS",
    "PSPPROJECT.NS", "PTC.NS", "PTL.NS",
    "PURVA.NS", "PVRINOX.NS", "QUESS.NS", "QUICKHEAL.NS",
    "RADICO.NS", "RAILTEL.NS", "RAIN.NS",
    "RAJESHEXPO.NS", "RAJRATAN.NS",
    "RALLIS.NS", "RAMASTEEL.NS",
    "RAMCOCEM.NS", "RAMCOIND.NS", "RAMCOSYS.NS",
    "RATNAMANI.NS", "RAYMOND.NS",
    "RBLBANK.NS", "RCF.NS", "RECLTD.NS",
    "REDINGTON.NS", "REFEX.NS", "RELAXO.NS", "RELIANCE.NS", "RELIGARE.NS",
    "RENUKA.NS", "REPCOHOME.NS",
    "RICOAUTO.NS",
    "RITES.NS", "RKFORGE.NS",
    "ROLEXRINGS.NS", "ROSSELLIND.NS", "ROUTE.NS", "RPGLIFE.NS",
    "RPOWER.NS", "RSWM.NS", "RSYSTEMS.NS",
    "RUCHIRA.NS",
    "RUPA.NS", "RVNL.NS", "SAFARI.NS",
    "SAIL.NS", "SAKSOFT.NS",
    "SALZERELEC.NS", "SANOFI.NS", "SAPPHIRE.NS",
    "SARDAEN.NS", "SAREGAMA.NS", "SARLAPOLY.NS", "SASKEN.NS", "SATIN.NS",
    "SBFC.NS", "SBIN.NS", "SBICARD.NS", "SBILIFE.NS", "SCHAEFFLER.NS",
    "SCHNEIDER.NS", "SEAMECLTD.NS",
    "SENCO.NS",
    "SEPC.NS", "SEQUENT.NS", "SESHAPAPER.NS",
    "SFL.NS", "SHAKTIPUMP.NS", "SHALBY.NS", "SHALPAINTS.NS",
    "SHANKARA.NS", "SHANTIGEAR.NS", "SHARDACROP.NS", "SHARDAMOTR.NS",
    "SHAREINDIA.NS", "SHEMAROO.NS", "SHILPAMED.NS",
]

# Combine all into comprehensive list
ALL_NSE_BSE = list(dict.fromkeys(
    BROAD_MARKET +
    NSE_MIDCAP_SMALLCAP +
    NSE_SMALLCAP_EXTRA +
    BSE_ADDITIONAL
))

ALL_INDIAN_STOCKS = NIFTY_50 + NIFTY_NEXT_50
