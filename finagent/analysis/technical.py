"""
Technical Analysis Module
Computes indicators using TA-Lib (deterministic, no LLM math).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import talib, fall back to manual calculations if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using fallback calculations.")


class TechnicalAnalyzer:
    """
    Computes technical indicators.
    ALL calculations are programmatic - never estimated.

    Uses TA-Lib when available, falls back to pure numpy/pandas otherwise.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.talib_available = TALIB_AVAILABLE

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full technical analysis.

        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            {
                "rsi_14": float,
                "macd": {"value": float, "signal": float, "histogram": float},
                "bollinger": {"upper": float, "middle": float, "lower": float},
                "sma_50": float,
                "sma_200": float,
                "ema_12": float,
                "ema_26": float,
                "golden_cross": bool,  # 50 > 200
                "death_cross": bool,   # 50 < 200
                "atr": float,
                "volatility_30d": float,
                "signals": {
                    "rsi_signal": "OVERSOLD | NEUTRAL | OVERBOUGHT",
                    "macd_signal": "BULLISH | BEARISH | NEUTRAL",
                    "trend": "UPTREND | DOWNTREND | SIDEWAYS"
                }
            }
        """
        if df.empty:
            return {"error": "Empty dataframe provided"}

        # Ensure we have enough data
        if len(df) < 200:
            self.logger.warning(f"Only {len(df)} data points. Some indicators may be unavailable.")

        result = {
            "data_points": len(df),
            "start_date": df.index[0].isoformat() if len(df) > 0 else None,
            "end_date": df.index[-1].isoformat() if len(df) > 0 else None,
        }

        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)

        # RSI
        result["rsi_14"] = self.calculate_rsi(close, period=14)

        # MACD
        result["macd"] = self.calculate_macd(close)

        # Bollinger Bands
        result["bollinger"] = self.calculate_bollinger_bands(close)

        # Moving Averages
        result["sma_20"] = self.calculate_sma(close, 20)
        result["sma_50"] = self.calculate_sma(close, 50)
        result["sma_200"] = self.calculate_sma(close, 200)
        result["ema_12"] = self.calculate_ema(close, 12)
        result["ema_26"] = self.calculate_ema(close, 26)

        # Golden/Death Cross
        if result["sma_50"] is not None and result["sma_200"] is not None:
            result["golden_cross"] = result["sma_50"] > result["sma_200"]
            result["death_cross"] = result["sma_50"] < result["sma_200"]
        else:
            result["golden_cross"] = None
            result["death_cross"] = None

        # ATR
        result["atr"] = self.calculate_atr(high, low, close)

        # Volatility
        result["volatility_30d"] = self.calculate_volatility(close, 30)

        # Stochastic
        result["stochastic"] = self.calculate_stochastic(high, low, close)

        # ADX (trend strength)
        result["adx"] = self.calculate_adx(high, low, close)

        # Volume analysis
        if 'Volume' in df.columns:
            volume = df['Volume'].astype(float)
            result["volume_sma_20"] = self.calculate_sma(volume, 20)
            result["volume_trend"] = self._analyze_volume_trend(volume)

        # Generate signals
        result["signals"] = self._generate_signals(result)

        # Current price info
        result["current_price"] = float(close.iloc[-1]) if len(close) > 0 else None
        result["price_change_1d"] = self._calculate_price_change(close, 1)
        result["price_change_7d"] = self._calculate_price_change(close, 7)
        result["price_change_30d"] = self._calculate_price_change(close, 30)

        return result

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI using TA-Lib or fallback."""
        if len(close) < period + 1:
            return None

        try:
            if self.talib_available:
                rsi = talib.RSI(close.values, timeperiod=period)
                return float(rsi[-1]) if not np.isnan(rsi[-1]) else None
            else:
                return self._calculate_rsi_manual(close, period)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    def _calculate_rsi_manual(self, close: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI without TA-Lib."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None

    def calculate_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, Optional[float]]:
        """Calculate MACD (12, 26, 9)."""
        if len(close) < slow + signal:
            return {"value": None, "signal": None, "histogram": None}

        try:
            if self.talib_available:
                macd, macd_signal, macd_hist = talib.MACD(
                    close.values,
                    fastperiod=fast,
                    slowperiod=slow,
                    signalperiod=signal
                )
                return {
                    "value": float(macd[-1]) if not np.isnan(macd[-1]) else None,
                    "signal": float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None,
                    "histogram": float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None
                }
            else:
                return self._calculate_macd_manual(close, fast, slow, signal)
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {"value": None, "signal": None, "histogram": None}

    def _calculate_macd_manual(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, Optional[float]]:
        """Calculate MACD without TA-Lib."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "value": float(macd_line.iloc[-1]) if not np.isnan(macd_line.iloc[-1]) else None,
            "signal": float(signal_line.iloc[-1]) if not np.isnan(signal_line.iloc[-1]) else None,
            "histogram": float(histogram.iloc[-1]) if not np.isnan(histogram.iloc[-1]) else None
        }

    def calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands."""
        if len(close) < period:
            return {"upper": None, "middle": None, "lower": None}

        try:
            if self.talib_available:
                upper, middle, lower = talib.BBANDS(
                    close.values,
                    timeperiod=period,
                    nbdevup=std_dev,
                    nbdevdn=std_dev
                )
                return {
                    "upper": float(upper[-1]) if not np.isnan(upper[-1]) else None,
                    "middle": float(middle[-1]) if not np.isnan(middle[-1]) else None,
                    "lower": float(lower[-1]) if not np.isnan(lower[-1]) else None
                }
            else:
                sma = close.rolling(window=period).mean()
                std = close.rolling(window=period).std()
                upper = sma + (std * std_dev)
                lower = sma - (std * std_dev)
                return {
                    "upper": float(upper.iloc[-1]) if not np.isnan(upper.iloc[-1]) else None,
                    "middle": float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else None,
                    "lower": float(lower.iloc[-1]) if not np.isnan(lower.iloc[-1]) else None
                }
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return {"upper": None, "middle": None, "lower": None}

    def calculate_sma(self, series: pd.Series, period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(series) < period:
            return None

        try:
            if self.talib_available:
                sma = talib.SMA(series.values, timeperiod=period)
                return float(sma[-1]) if not np.isnan(sma[-1]) else None
            else:
                sma = series.rolling(window=period).mean()
                return float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else None
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return None

    def calculate_ema(self, series: pd.Series, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(series) < period:
            return None

        try:
            if self.talib_available:
                ema = talib.EMA(series.values, timeperiod=period)
                return float(ema[-1]) if not np.isnan(ema[-1]) else None
            else:
                ema = series.ewm(span=period, adjust=False).mean()
                return float(ema.iloc[-1]) if not np.isnan(ema.iloc[-1]) else None
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return None

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Optional[float]:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return None

        try:
            if self.talib_available:
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                return float(atr[-1]) if not np.isnan(atr[-1]) else None
            else:
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=period).mean()
                return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else None
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return None

    def calculate_volatility(self, close: pd.Series, period: int = 30) -> Optional[float]:
        """Calculate annualized volatility."""
        if len(close) < period:
            return None

        try:
            returns = close.pct_change().dropna()
            volatility = returns.tail(period).std() * np.sqrt(252)  # Annualized
            return float(volatility) if not np.isnan(volatility) else None
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return None

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, Optional[float]]:
        """Calculate Stochastic Oscillator."""
        if len(close) < k_period + d_period:
            return {"k": None, "d": None}

        try:
            if self.talib_available:
                slowk, slowd = talib.STOCH(
                    high.values, low.values, close.values,
                    fastk_period=k_period,
                    slowk_period=d_period,
                    slowd_period=d_period
                )
                return {
                    "k": float(slowk[-1]) if not np.isnan(slowk[-1]) else None,
                    "d": float(slowd[-1]) if not np.isnan(slowd[-1]) else None
                }
            else:
                lowest_low = low.rolling(window=k_period).min()
                highest_high = high.rolling(window=k_period).max()
                k = 100 * (close - lowest_low) / (highest_high - lowest_low)
                d = k.rolling(window=d_period).mean()
                return {
                    "k": float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else None,
                    "d": float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else None
                }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return {"k": None, "d": None}

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Optional[float]:
        """Calculate Average Directional Index."""
        if len(close) < period * 2:
            return None

        try:
            if self.talib_available:
                adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
                return float(adx[-1]) if not np.isnan(adx[-1]) else None
            else:
                # Simplified ADX calculation
                return None  # ADX is complex to calculate manually
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return None

    def calculate_beta(
        self,
        stock_returns: pd.Series,
        benchmark_returns: pd.Series,
        period: int = 252
    ) -> Optional[float]:
        """
        Calculate beta vs benchmark (e.g., NIFTY50).

        Args:
            stock_returns: Daily returns of the stock
            benchmark_returns: Daily returns of the benchmark
            period: Number of trading days

        Returns:
            Beta value
        """
        try:
            # Align the series
            aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
            if len(aligned) < 30:
                return None

            stock = aligned.iloc[:, 0].tail(period)
            benchmark = aligned.iloc[:, 1].tail(period)

            covariance = np.cov(stock, benchmark)[0][1]
            variance = np.var(benchmark)

            if variance == 0:
                return None

            beta = covariance / variance
            return float(beta)

        except Exception as e:
            self.logger.error(f"Error calculating beta: {e}")
            return None

    def _analyze_volume_trend(self, volume: pd.Series) -> str:
        """Analyze volume trend."""
        if len(volume) < 20:
            return "UNKNOWN"

        recent_avg = volume.tail(5).mean()
        period_avg = volume.tail(20).mean()

        if recent_avg > period_avg * 1.5:
            return "HIGH"
        elif recent_avg < period_avg * 0.5:
            return "LOW"
        else:
            return "NORMAL"

    def _calculate_price_change(self, close: pd.Series, days: int) -> Optional[float]:
        """Calculate percentage price change over N days."""
        if len(close) <= days:
            return None

        try:
            current = close.iloc[-1]
            previous = close.iloc[-(days + 1)]
            return float((current - previous) / previous * 100)
        except Exception:
            return None

    def _generate_signals(self, analysis: Dict) -> Dict[str, str]:
        """Generate trading signals from indicators."""
        signals = {}

        # RSI Signal
        rsi = analysis.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                signals["rsi_signal"] = "OVERSOLD"
            elif rsi > 70:
                signals["rsi_signal"] = "OVERBOUGHT"
            else:
                signals["rsi_signal"] = "NEUTRAL"
        else:
            signals["rsi_signal"] = "UNKNOWN"

        # MACD Signal
        macd = analysis.get("macd", {})
        if macd.get("histogram") is not None:
            if macd["histogram"] > 0 and macd.get("value", 0) > macd.get("signal", 0):
                signals["macd_signal"] = "BULLISH"
            elif macd["histogram"] < 0 and macd.get("value", 0) < macd.get("signal", 0):
                signals["macd_signal"] = "BEARISH"
            else:
                signals["macd_signal"] = "NEUTRAL"
        else:
            signals["macd_signal"] = "UNKNOWN"

        # Trend Signal (based on moving averages)
        sma_50 = analysis.get("sma_50")
        sma_200 = analysis.get("sma_200")
        current_price = analysis.get("current_price")

        if all(x is not None for x in [sma_50, sma_200, current_price]):
            if current_price > sma_50 > sma_200:
                signals["trend"] = "UPTREND"
            elif current_price < sma_50 < sma_200:
                signals["trend"] = "DOWNTREND"
            else:
                signals["trend"] = "SIDEWAYS"
        else:
            signals["trend"] = "UNKNOWN"

        # Bollinger Band Signal
        bb = analysis.get("bollinger", {})
        if all(bb.get(k) is not None for k in ["upper", "lower"]) and current_price:
            if current_price > bb["upper"]:
                signals["bollinger_signal"] = "OVERBOUGHT"
            elif current_price < bb["lower"]:
                signals["bollinger_signal"] = "OVERSOLD"
            else:
                signals["bollinger_signal"] = "NEUTRAL"
        else:
            signals["bollinger_signal"] = "UNKNOWN"

        # Stochastic Signal
        stoch = analysis.get("stochastic", {})
        if stoch.get("k") is not None:
            if stoch["k"] < 20:
                signals["stochastic_signal"] = "OVERSOLD"
            elif stoch["k"] > 80:
                signals["stochastic_signal"] = "OVERBOUGHT"
            else:
                signals["stochastic_signal"] = "NEUTRAL"
        else:
            signals["stochastic_signal"] = "UNKNOWN"

        return signals
