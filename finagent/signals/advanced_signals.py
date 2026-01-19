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

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'direction': self.direction.value,
            'confidence': round(self.confidence, 1),
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
        }


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


class AdvancedSignalGenerator:
    """
    Generates combined signals from multiple indicators.

    Scans Indian stocks and provides buy/sell recommendations
    with profit targets based on holding period.
    """

    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.logger = logging.getLogger(__name__)

    def analyze_stock(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        hold_days: int = 7
    ) -> Optional[TechnicalSignal]:
        """
        Analyze a stock using all indicators and generate a combined signal.

        Args:
            ticker: Stock symbol
            ohlcv: DataFrame with Open, High, Low, Close, Volume
            hold_days: Suggested holding period in days

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

        current_price = close[-1]

        # Calculate all indicator signals
        signals = {}
        scores = []

        try:
            # MACD
            macd_sig = self.indicators.macd_signal(close)
            signals['macd'] = macd_sig
            if macd_sig['signal'] in ['BUY', 'BULLISH']:
                scores.append(macd_sig['strength'])
            elif macd_sig['signal'] in ['SELL', 'BEARISH']:
                scores.append(-macd_sig['strength'])

            # Williams %R Exhaustion
            wr_sig = self.indicators.williams_r_exhaustion(high, low, close)
            signals['williams_r'] = wr_sig
            if wr_sig['signal'] == 'BUY':
                scores.append(wr_sig['strength'])
            elif wr_sig['signal'] == 'SELL':
                scores.append(-wr_sig['strength'])

            # VixFix
            vix_sig = self.indicators.williams_vix_fix(close, low)
            signals['vix_fix'] = vix_sig
            if vix_sig['signal'] == 'BUY':
                scores.append(vix_sig['strength'])

            # Hull Suite
            hull_sig = self.indicators.hull_suite_signal(close)
            signals['hull_ma'] = hull_sig
            if hull_sig['signal'] in ['BUY', 'BULLISH']:
                scores.append(hull_sig['strength'])
            elif hull_sig['signal'] in ['SELL', 'BEARISH']:
                scores.append(-hull_sig['strength'])

            # Laguerre RSI
            lag_sig = self.indicators.laguerre_signal(close)
            signals['laguerre'] = lag_sig
            if lag_sig['signal'] in ['BUY', 'OVERSOLD']:
                scores.append(lag_sig['strength'])
            elif lag_sig['signal'] in ['SELL', 'OVERBOUGHT']:
                scores.append(-lag_sig['strength'])

            # Supertrend
            st_sig = self.indicators.supertrend_signal(high, low, close)
            signals['supertrend'] = st_sig
            if st_sig['signal'] in ['BUY', 'BULLISH']:
                scores.append(st_sig['strength'])
            elif st_sig['signal'] in ['SELL', 'BEARISH']:
                scores.append(-st_sig['strength'])

            # RSI Divergence
            rsi_sig = self.indicators.rsi_divergence_signal(close)
            signals['rsi'] = rsi_sig
            if rsi_sig['signal'] in ['BUY', 'OVERSOLD']:
                scores.append(rsi_sig['strength'])
            elif rsi_sig['signal'] in ['SELL', 'OVERBOUGHT']:
                scores.append(-rsi_sig['strength'])

            # Ichimoku
            ichi_sig = self.indicators.ichimoku_signal(high, low, close)
            signals['ichimoku'] = ichi_sig
            if ichi_sig['signal'] in ['STRONG_BUY', 'BUY', 'BULLISH']:
                scores.append(ichi_sig['strength'])
            elif ichi_sig['signal'] in ['STRONG_SELL', 'SELL', 'BEARISH']:
                scores.append(-ichi_sig['strength'])

        except Exception as e:
            self.logger.warning(f"Error calculating indicators for {ticker}: {e}")
            return None

        # Calculate combined score
        if not scores:
            return None

        avg_score = np.mean(scores)

        # Determine direction
        if avg_score >= 40:
            direction = SignalDirection.STRONG_BUY
        elif avg_score >= 20:
            direction = SignalDirection.BUY
        elif avg_score <= -40:
            direction = SignalDirection.STRONG_SELL
        elif avg_score <= -20:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        # Skip neutral signals
        if direction == SignalDirection.NEUTRAL:
            return None

        # Calculate targets based on ATR
        atr = self.indicators._atr(high, low, close)[-1]
        atr_pct = (atr / current_price) * 100

        # Adjust targets based on holding period
        hold_multiplier = np.sqrt(hold_days / 7)  # Scale with sqrt of time

        if direction in [SignalDirection.STRONG_BUY, SignalDirection.BUY]:
            entry = current_price
            stop_loss = current_price - (2 * atr)
            target_1 = current_price + (2 * atr * hold_multiplier)
            target_2 = current_price + (3 * atr * hold_multiplier)
            target_3 = current_price + (4 * atr * hold_multiplier)
            expected_return = ((target_2 - entry) / entry) * 100
        else:
            entry = current_price
            stop_loss = current_price + (2 * atr)
            target_1 = current_price - (2 * atr * hold_multiplier)
            target_2 = current_price - (3 * atr * hold_multiplier)
            target_3 = current_price - (4 * atr * hold_multiplier)
            expected_return = ((entry - target_2) / entry) * 100

        # Risk/Reward
        risk = abs(entry - stop_loss)
        reward = abs(target_2 - entry)
        risk_reward = reward / risk if risk > 0 else 0

        # Build reasons and risks
        reasons = []
        risks = []

        if signals['macd']['signal'] in ['BUY', 'BULLISH']:
            reasons.append(f"MACD {signals['macd']['signal'].lower()}")
        elif signals['macd']['signal'] in ['SELL', 'BEARISH']:
            risks.append(f"MACD {signals['macd']['signal'].lower()}")

        if signals['supertrend']['signal'] in ['BUY', 'BULLISH']:
            reasons.append(f"Supertrend {signals['supertrend']['direction']}")
        elif signals['supertrend']['signal'] in ['SELL', 'BEARISH']:
            risks.append(f"Supertrend {signals['supertrend']['direction']}")

        if signals['ichimoku']['signal'] in ['STRONG_BUY', 'BUY', 'BULLISH']:
            reasons.append(f"Ichimoku {signals['ichimoku']['position']}")
        elif signals['ichimoku']['signal'] in ['STRONG_SELL', 'SELL', 'BEARISH']:
            risks.append(f"Ichimoku {signals['ichimoku']['position']}")

        if signals['rsi']['bullish_divergence']:
            reasons.append("RSI bullish divergence")
        if signals['rsi']['bearish_divergence']:
            risks.append("RSI bearish divergence")

        if signals['williams_r']['oversold_exhaustion']:
            reasons.append("Williams %R oversold exhaustion (bottom signal)")
        if signals['williams_r']['overbought_exhaustion']:
            risks.append("Williams %R overbought exhaustion (top signal)")

        if signals['vix_fix']['is_bottom_signal']:
            reasons.append("VixFix bottom signal")

        if signals['hull_ma']['hull_rising']:
            reasons.append("Hull MA rising trend")
        else:
            risks.append("Hull MA falling trend")

        return TechnicalSignal(
            ticker=ticker,
            timestamp=str(datetime.now()),
            direction=direction,
            confidence=min(abs(avg_score), 100),
            current_price=current_price,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            expected_return_pct=expected_return,
            risk_reward_ratio=risk_reward,
            suggested_hold_days=hold_days,
            indicators=signals,
            reasons=reasons,
            risks=risks,
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

        Args:
            tickers: List of stock tickers to scan
            hold_days: Suggested holding period
            min_confidence: Minimum confidence to include
            progress_callback: Optional callback(current, total, ticker)

        Returns:
            List of TechnicalSignal sorted by confidence
        """
        import yfinance as yf

        signals = []
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                # Fetch data
                stock = yf.Ticker(ticker)
                df = stock.history(period="6mo")

                if len(df) < 60:
                    continue

                signal = self.analyze_stock(ticker, df, hold_days)

                if signal and signal.confidence >= min_confidence:
                    signals.append(signal)

            except Exception as e:
                self.logger.debug(f"Error scanning {ticker}: {e}")
                continue

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals


# Popular Indian stock tickers for scanning
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "M&M.NS",
    "TECHM.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS", "ONGC.NS", "HDFCLIFE.NS",
    "TATASTEEL.NS", "GRASIM.NS", "ADANIPORTS.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "SBILIFE.NS", "BRITANNIA.NS", "COALINDIA.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "INDUSINDBK.NS", "BPCL.NS", "APOLLOHOSP.NS", "TATACONSUM.NS",
    "UPL.NS", "ADANIENT.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS",
]

NIFTY_NEXT_50 = [
    "ADANIGREEN.NS", "AMBUJACEM.NS", "BANKBARODA.NS", "BERGEPAINT.NS",
    "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS",
    "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "ICICIGI.NS", "INDIGO.NS", "IOC.NS", "JINDALSTEL.NS", "LICI.NS",
    "MARICO.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PIDILITIND.NS", "PNB.NS",
    "SBICARD.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "TORNTPHARM.NS",
    "TRENT.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS",
]

ALL_INDIAN_STOCKS = NIFTY_50 + NIFTY_NEXT_50
