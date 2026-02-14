"""
Signal Combiner Module

Merges multiple signal sources into unified actionable recommendations.

Signal Sources:
- Technical Analysis (TA-Lib indicators)
- Fundamental Analysis (financial ratios)
- Institutional Flow (FII/DII activity)
- Bulk/Block Deals
- Insider Trading (SAST)
- News Sentiment & Catalysts

Weights can be configured based on time horizon:
- Intraday: Heavy on technical, institutional flow
- Short-term (1d-1w): Technical + news catalysts + deals
- Medium-term (1w-1m): Fundamentals + institutional + news
- Long-term (1m+): Fundamentals + insider activity
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    INSTITUTIONAL = "institutional"  # FII/DII flow
    BULK_BLOCK = "bulk_block"        # Large deals
    INSIDER = "insider"              # SAST filings
    NEWS = "news"                    # News sentiment
    CATALYST = "catalyst"            # Specific news events
    PEAD = "pead"                    # Post-Earnings Announcement Drift


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class TimeHorizon(Enum):
    """Investment time horizons."""
    INTRADAY = "intraday"
    SHORT = "1d-3d"
    WEEKLY = "1w"
    MONTHLY = "1m"
    QUARTERLY = "3m"
    LONG = "1y+"


@dataclass
class Signal:
    """Represents a single trading signal."""
    ticker: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1
    source: str        # e.g., "RSI oversold", "FII buying", "Contract win"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    expiry_horizon: Optional[TimeHorizon] = None

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'signal_type': self.signal_type.value,
            'strength': self.strength.name,
            'strength_value': self.strength.value,
            'confidence': self.confidence,
            'source': self.source,
            'details': self.details,
            'timestamp': self.timestamp,
            'expiry_horizon': self.expiry_horizon.value if self.expiry_horizon else None
        }


@dataclass
class CombinedRecommendation:
    """Combined recommendation from multiple signals."""
    ticker: str
    company_name: str
    action: str           # BUY, SELL, HOLD
    score: float          # -100 to +100
    confidence: float     # 0-1
    time_horizon: TimeHorizon
    signals: List[Signal]
    key_factors: List[str]
    risks: List[str]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    current_price: Optional[float] = None
    expected_return: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'action': self.action,
            'score': self.score,
            'confidence': self.confidence,
            'time_horizon': self.time_horizon.value,
            'signals': [s.to_dict() for s in self.signals],
            'key_factors': self.key_factors,
            'risks': self.risks,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'current_price': self.current_price,
            'expected_return': self.expected_return,
            'timestamp': self.timestamp
        }


class SignalCombiner:
    """
    Combines multiple signal sources into unified recommendations.

    Supports configurable weights per time horizon and signal type.
    Uses ensemble approach to reduce false signals.
    """

    # Default weights by time horizon
    DEFAULT_WEIGHTS = {
        TimeHorizon.INTRADAY: {
            SignalType.TECHNICAL: 0.40,
            SignalType.INSTITUTIONAL: 0.25,
            SignalType.NEWS: 0.20,
            SignalType.BULK_BLOCK: 0.10,
            SignalType.FUNDAMENTAL: 0.03,
            SignalType.INSIDER: 0.02,
            SignalType.CATALYST: 0.00,
        },
        TimeHorizon.SHORT: {
            SignalType.TECHNICAL: 0.30,
            SignalType.CATALYST: 0.25,
            SignalType.NEWS: 0.15,
            SignalType.BULK_BLOCK: 0.15,
            SignalType.INSTITUTIONAL: 0.10,
            SignalType.FUNDAMENTAL: 0.03,
            SignalType.INSIDER: 0.02,
        },
        TimeHorizon.WEEKLY: {
            SignalType.TECHNICAL: 0.25,
            SignalType.CATALYST: 0.20,
            SignalType.NEWS: 0.15,
            SignalType.INSTITUTIONAL: 0.15,
            SignalType.BULK_BLOCK: 0.15,
            SignalType.FUNDAMENTAL: 0.05,
            SignalType.INSIDER: 0.05,
        },
        TimeHorizon.MONTHLY: {
            SignalType.FUNDAMENTAL: 0.25,
            SignalType.TECHNICAL: 0.20,
            SignalType.INSTITUTIONAL: 0.20,
            SignalType.CATALYST: 0.15,
            SignalType.INSIDER: 0.10,
            SignalType.NEWS: 0.05,
            SignalType.BULK_BLOCK: 0.05,
        },
        TimeHorizon.QUARTERLY: {
            SignalType.FUNDAMENTAL: 0.35,
            SignalType.INSIDER: 0.20,
            SignalType.INSTITUTIONAL: 0.20,
            SignalType.CATALYST: 0.10,
            SignalType.TECHNICAL: 0.10,
            SignalType.NEWS: 0.03,
            SignalType.BULK_BLOCK: 0.02,
        },
        TimeHorizon.LONG: {
            SignalType.FUNDAMENTAL: 0.45,
            SignalType.INSIDER: 0.25,
            SignalType.INSTITUTIONAL: 0.15,
            SignalType.CATALYST: 0.10,
            SignalType.TECHNICAL: 0.03,
            SignalType.NEWS: 0.01,
            SignalType.BULK_BLOCK: 0.01,
        },
    }

    # Minimum confidence thresholds
    MIN_CONFIDENCE = {
        'BUY': 0.60,
        'STRONG_BUY': 0.75,
        'SELL': 0.55,
        'STRONG_SELL': 0.70,
    }

    def __init__(
        self,
        weights: Optional[Dict[TimeHorizon, Dict[SignalType, float]]] = None
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.logger = logging.getLogger(__name__)

    def combine_signals(
        self,
        ticker: str,
        signals: List[Signal],
        time_horizon: TimeHorizon,
        current_price: Optional[float] = None,
        company_name: str = ""
    ) -> CombinedRecommendation:
        """
        Combine multiple signals into a unified recommendation.

        Args:
            ticker: Stock ticker
            signals: List of Signal objects
            time_horizon: Investment time horizon
            current_price: Current stock price
            company_name: Company name

        Returns:
            CombinedRecommendation with action, score, and analysis
        """
        if not signals:
            return CombinedRecommendation(
                ticker=ticker,
                company_name=company_name,
                action='HOLD',
                score=0.0,
                confidence=0.0,
                time_horizon=time_horizon,
                signals=[],
                key_factors=['No signals available'],
                risks=['Insufficient data for analysis'],
                current_price=current_price
            )

        # Get weights for this time horizon
        horizon_weights = self.weights.get(time_horizon, self.DEFAULT_WEIGHTS[TimeHorizon.MONTHLY])

        # Calculate weighted score
        total_weighted_score = 0.0
        total_weight = 0.0
        confidence_scores = []

        for signal in signals:
            weight = horizon_weights.get(signal.signal_type, 0.1)
            signal_score = signal.strength.value * signal.confidence * 50  # Scale to -100 to +100

            total_weighted_score += signal_score * weight
            total_weight += weight
            confidence_scores.append(signal.confidence)

        # Normalize score
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.0

        # Calculate overall confidence
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            # Adjust confidence based on signal agreement
            agreement_factor = self._calculate_agreement(signals)
            overall_confidence = avg_confidence * agreement_factor
        else:
            overall_confidence = 0.0

        # Determine action
        action = self._determine_action(final_score, overall_confidence)

        # Extract key factors and risks
        key_factors = self._extract_key_factors(signals, action)
        risks = self._extract_risks(signals, action)

        # Calculate target and stop loss if enough data
        target_price, stop_loss, expected_return = self._calculate_targets(
            signals, current_price, action, time_horizon
        )

        return CombinedRecommendation(
            ticker=ticker,
            company_name=company_name,
            action=action,
            score=round(final_score, 2),
            confidence=round(overall_confidence, 2),
            time_horizon=time_horizon,
            signals=signals,
            key_factors=key_factors,
            risks=risks,
            target_price=target_price,
            stop_loss=stop_loss,
            current_price=current_price,
            expected_return=expected_return
        )

    def _calculate_agreement(self, signals: List[Signal]) -> float:
        """
        Calculate signal agreement factor.

        Returns higher value when signals agree, lower when they conflict.
        """
        if len(signals) < 2:
            return 1.0

        bullish_count = sum(1 for s in signals if s.strength.value > 0)
        bearish_count = sum(1 for s in signals if s.strength.value < 0)
        neutral_count = sum(1 for s in signals if s.strength.value == 0)

        total = len(signals)

        # Perfect agreement
        if bullish_count == total or bearish_count == total:
            return 1.0

        # Mostly agreeing
        max_direction = max(bullish_count, bearish_count)
        agreement_ratio = max_direction / total

        # Penalize heavy conflict
        if bullish_count > 0 and bearish_count > 0:
            conflict_penalty = min(bullish_count, bearish_count) / total
            agreement_ratio *= (1 - conflict_penalty * 0.5)

        return max(0.5, agreement_ratio)

    def _determine_action(self, score: float, confidence: float) -> str:
        """Determine action based on score and confidence."""
        if score >= 50 and confidence >= self.MIN_CONFIDENCE['STRONG_BUY']:
            return 'STRONG_BUY'
        elif score >= 25 and confidence >= self.MIN_CONFIDENCE['BUY']:
            return 'BUY'
        elif score <= -50 and confidence >= self.MIN_CONFIDENCE['STRONG_SELL']:
            return 'STRONG_SELL'
        elif score <= -25 and confidence >= self.MIN_CONFIDENCE['SELL']:
            return 'SELL'
        else:
            return 'HOLD'

    def _extract_key_factors(self, signals: List[Signal], action: str) -> List[str]:
        """Extract key factors supporting the recommendation."""
        factors = []

        # Sort by strength and confidence
        sorted_signals = sorted(
            signals,
            key=lambda s: (abs(s.strength.value), s.confidence),
            reverse=True
        )

        # Get top factors aligned with action
        is_bullish_action = action in ['BUY', 'STRONG_BUY']
        is_bearish_action = action in ['SELL', 'STRONG_SELL']

        for signal in sorted_signals[:5]:
            is_bullish_signal = signal.strength.value > 0
            is_bearish_signal = signal.strength.value < 0

            # Include aligned signals
            if (is_bullish_action and is_bullish_signal) or \
               (is_bearish_action and is_bearish_signal):
                factor = f"{signal.signal_type.value.title()}: {signal.source}"
                if signal.confidence >= 0.8:
                    factor += " (high confidence)"
                factors.append(factor)

        if not factors:
            factors.append("Mixed signals - no clear direction")

        return factors[:5]

    def _extract_risks(self, signals: List[Signal], action: str) -> List[str]:
        """Extract risk factors contrary to the recommendation."""
        risks = []

        is_bullish_action = action in ['BUY', 'STRONG_BUY']
        is_bearish_action = action in ['SELL', 'STRONG_SELL']

        for signal in signals:
            is_bullish_signal = signal.strength.value > 0
            is_bearish_signal = signal.strength.value < 0

            # Include contrary signals as risks
            if (is_bullish_action and is_bearish_signal) or \
               (is_bearish_action and is_bullish_signal):
                risk = f"Contrary {signal.signal_type.value}: {signal.source}"
                risks.append(risk)

        # Add general risks based on signal types present
        signal_types = set(s.signal_type for s in signals)

        if SignalType.INSTITUTIONAL not in signal_types:
            risks.append("No institutional flow data")
        if SignalType.FUNDAMENTAL not in signal_types:
            risks.append("No fundamental analysis")
        if len(signals) < 3:
            risks.append("Limited signal coverage")

        return risks[:5]

    def _calculate_targets(
        self,
        signals: List[Signal],
        current_price: Optional[float],
        action: str,
        time_horizon: TimeHorizon
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate target price, stop loss, and expected return.

        Uses ATR-based scaling per time horizon with confidence adjustment.
        If technical signal details include structure-based targets from
        the advanced signal generator, those are preferred.
        """
        if not current_price or current_price <= 0:
            return None, None, None

        # Check if any technical signal carries structure-based targets
        for sig in signals:
            if sig.signal_type == SignalType.TECHNICAL and sig.details:
                if 'target_price' in sig.details and 'stop_loss' in sig.details:
                    return (
                        sig.details['target_price'],
                        sig.details['stop_loss'],
                        sig.details.get('expected_return', 0),
                    )

        # ATR-scaled expected move per time horizon (target_mult, stop_mult of daily range)
        # These represent realistic multiples of typical daily ATR for each horizon
        horizon_atr_mults = {
            TimeHorizon.INTRADAY: (1.0, 0.5),    # 1 ATR target, 0.5 ATR stop
            TimeHorizon.SHORT: (2.0, 1.0),        # 2 ATR target, 1 ATR stop
            TimeHorizon.WEEKLY: (3.0, 1.5),       # 3 ATR target, 1.5 ATR stop
            TimeHorizon.MONTHLY: (4.5, 2.0),      # 4.5 ATR target, 2 ATR stop
            TimeHorizon.QUARTERLY: (7.0, 3.0),    # 7 ATR target, 3 ATR stop
            TimeHorizon.LONG: (10.0, 4.0),        # 10 ATR target, 4 ATR stop
        }

        target_mult, stop_mult = horizon_atr_mults.get(time_horizon, (3.0, 1.5))

        # Estimate ATR from recent price action (approx 1.5% daily for Indian equities)
        # This is a fallback when OHLCV data isn't available here
        estimated_atr = current_price * 0.015

        # Adjust target based on signal confidence and agreement
        avg_confidence = sum(s.confidence for s in signals) / len(signals) if signals else 0.5
        agreement = self._calculate_agreement(signals)
        adj_factor = avg_confidence * (0.7 + 0.3 * agreement)  # 70-100% of base target

        target_distance = estimated_atr * target_mult * adj_factor
        stop_distance = estimated_atr * stop_mult

        if action in ['BUY', 'STRONG_BUY']:
            target_price = current_price + target_distance
            stop_loss = current_price - stop_distance
            expected_return = (target_distance / current_price) * 100
        elif action in ['SELL', 'STRONG_SELL']:
            target_price = current_price - target_distance
            stop_loss = current_price + stop_distance
            expected_return = (target_distance / current_price) * 100
        else:
            return None, None, None

        return round(target_price, 2), round(stop_loss, 2), round(expected_return, 2)

    # =========================================================================
    # Signal Generation Helpers
    # =========================================================================

    def generate_technical_signal(
        self,
        ticker: str,
        indicators: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate technical signal from indicator values.

        Expected indicators:
        - rsi, macd_signal, bb_position, sma_trend, volume_trend
        """
        bullish_factors = 0
        bearish_factors = 0
        sources = []

        # RSI analysis
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < 30:
                bullish_factors += 2
                sources.append(f"RSI oversold ({rsi:.1f})")
            elif rsi < 40:
                bullish_factors += 1
                sources.append(f"RSI low ({rsi:.1f})")
            elif rsi > 70:
                bearish_factors += 2
                sources.append(f"RSI overbought ({rsi:.1f})")
            elif rsi > 60:
                bearish_factors += 1
                sources.append(f"RSI high ({rsi:.1f})")

        # MACD analysis
        macd_hist = indicators.get('macd_histogram')
        if macd_hist is not None:
            if macd_hist > 0:
                bullish_factors += 1
                sources.append("MACD bullish")
            else:
                bearish_factors += 1
                sources.append("MACD bearish")

        # Bollinger Bands position
        bb_position = indicators.get('bb_position')  # 0=lower, 0.5=middle, 1=upper
        if bb_position is not None:
            if bb_position < 0.2:
                bullish_factors += 1
                sources.append("Near BB lower band")
            elif bb_position > 0.8:
                bearish_factors += 1
                sources.append("Near BB upper band")

        # SMA trend
        sma_trend = indicators.get('sma_trend')  # above/below
        if sma_trend == 'above':
            bullish_factors += 1
            sources.append("Above SMA")
        elif sma_trend == 'below':
            bearish_factors += 1
            sources.append("Below SMA")

        # Volume trend
        volume_trend = indicators.get('volume_trend')
        if volume_trend == 'increasing':
            # Amplify current direction
            if bullish_factors > bearish_factors:
                bullish_factors += 1
                sources.append("Rising volume")
            elif bearish_factors > bullish_factors:
                bearish_factors += 1
                sources.append("Rising volume (distribution)")

        # Determine strength
        net_score = bullish_factors - bearish_factors
        max_factors = max(bullish_factors + bearish_factors, 1)
        confidence = min(max_factors / 5, 1.0)

        if net_score >= 3:
            strength = SignalStrength.STRONG_BUY
        elif net_score >= 1:
            strength = SignalStrength.BUY
        elif net_score <= -3:
            strength = SignalStrength.STRONG_SELL
        elif net_score <= -1:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.NEUTRAL

        source_str = "; ".join(sources) if sources else "No clear signals"

        return Signal(
            ticker=ticker,
            signal_type=SignalType.TECHNICAL,
            strength=strength,
            confidence=confidence,
            source=source_str,
            details=indicators,
            expiry_horizon=TimeHorizon.SHORT
        )

    def generate_fundamental_signal(
        self,
        ticker: str,
        ratios: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate fundamental signal from financial ratios.

        Expected ratios:
        - pe_ratio, pb_ratio, roe, debt_to_equity, revenue_growth, profit_growth
        """
        bullish_factors = 0
        bearish_factors = 0
        sources = []

        # P/E analysis (lower is better, but not too low)
        pe = ratios.get('pe_ratio')
        if pe is not None:
            if 5 < pe < 15:
                bullish_factors += 2
                sources.append(f"Attractive P/E ({pe:.1f})")
            elif pe < 5:
                bearish_factors += 1
                sources.append(f"Very low P/E ({pe:.1f}) - investigate")
            elif pe > 50:
                bearish_factors += 2
                sources.append(f"High P/E ({pe:.1f})")
            elif pe > 30:
                bearish_factors += 1
                sources.append(f"Elevated P/E ({pe:.1f})")

        # ROE analysis
        roe = ratios.get('roe')
        if roe is not None:
            if roe > 20:
                bullish_factors += 2
                sources.append(f"Strong ROE ({roe:.1f}%)")
            elif roe > 15:
                bullish_factors += 1
                sources.append(f"Good ROE ({roe:.1f}%)")
            elif roe < 5:
                bearish_factors += 2
                sources.append(f"Weak ROE ({roe:.1f}%)")
            elif roe < 10:
                bearish_factors += 1
                sources.append(f"Below average ROE ({roe:.1f}%)")

        # Debt analysis
        de = ratios.get('debt_to_equity')
        if de is not None:
            if de < 0.5:
                bullish_factors += 1
                sources.append("Low debt")
            elif de > 2:
                bearish_factors += 2
                sources.append(f"High debt (D/E: {de:.1f})")
            elif de > 1:
                bearish_factors += 1
                sources.append(f"Moderate debt (D/E: {de:.1f})")

        # Growth analysis
        rev_growth = ratios.get('revenue_growth')
        if rev_growth is not None:
            if rev_growth > 20:
                bullish_factors += 2
                sources.append(f"Strong revenue growth ({rev_growth:.1f}%)")
            elif rev_growth > 10:
                bullish_factors += 1
                sources.append(f"Healthy revenue growth ({rev_growth:.1f}%)")
            elif rev_growth < 0:
                bearish_factors += 2
                sources.append(f"Revenue decline ({rev_growth:.1f}%)")

        profit_growth = ratios.get('profit_growth')
        if profit_growth is not None:
            if profit_growth > 25:
                bullish_factors += 2
                sources.append(f"Strong profit growth ({profit_growth:.1f}%)")
            elif profit_growth > 10:
                bullish_factors += 1
                sources.append(f"Healthy profit growth ({profit_growth:.1f}%)")
            elif profit_growth < 0:
                bearish_factors += 2
                sources.append(f"Profit decline ({profit_growth:.1f}%)")

        # Determine strength
        net_score = bullish_factors - bearish_factors
        max_factors = max(bullish_factors + bearish_factors, 1)
        confidence = min(max_factors / 6, 1.0)

        if net_score >= 4:
            strength = SignalStrength.STRONG_BUY
        elif net_score >= 2:
            strength = SignalStrength.BUY
        elif net_score <= -4:
            strength = SignalStrength.STRONG_SELL
        elif net_score <= -2:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.NEUTRAL

        source_str = "; ".join(sources) if sources else "Mixed fundamentals"

        return Signal(
            ticker=ticker,
            signal_type=SignalType.FUNDAMENTAL,
            strength=strength,
            confidence=confidence,
            source=source_str,
            details=ratios,
            expiry_horizon=TimeHorizon.QUARTERLY
        )

    def generate_institutional_signal(
        self,
        ticker: str,
        fii_dii_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate signal from FII/DII flow data.

        Expected data:
        - fii_net_cr: FII net buying (positive) or selling (negative) in crores
        - dii_net_cr: DII net buying/selling
        - fii_sentiment, dii_sentiment: 'bullish', 'bearish', 'neutral'
        """
        sources = []
        bullish_factors = 0
        bearish_factors = 0

        fii_net = fii_dii_data.get('fii', {}).get('net_total_cr', 0)
        dii_net = fii_dii_data.get('dii', {}).get('net_total_cr', 0)
        fii_sentiment = fii_dii_data.get('fii', {}).get('sentiment', 'neutral')
        dii_sentiment = fii_dii_data.get('dii', {}).get('sentiment', 'neutral')

        # FII analysis (more weight as they are trend-setters)
        if fii_sentiment == 'bullish':
            bullish_factors += 2
            sources.append(f"FII buying (₹{fii_net:.0f}Cr)")
        elif fii_sentiment == 'bearish':
            bearish_factors += 2
            sources.append(f"FII selling (₹{abs(fii_net):.0f}Cr)")

        # DII analysis
        if dii_sentiment == 'bullish':
            bullish_factors += 1
            sources.append(f"DII buying (₹{dii_net:.0f}Cr)")
        elif dii_sentiment == 'bearish':
            bearish_factors += 1
            sources.append(f"DII selling (₹{abs(dii_net):.0f}Cr)")

        # Combined analysis
        combined = fii_dii_data.get('combined_sentiment', 'neutral')
        if combined == 'strongly_bullish':
            bullish_factors += 2
            sources.append("Strong institutional buying")
        elif combined == 'strongly_bearish':
            bearish_factors += 2
            sources.append("Strong institutional selling")

        # Determine strength
        net_score = bullish_factors - bearish_factors
        confidence = min(abs(net_score) / 4, 1.0) * 0.8 + 0.2

        if net_score >= 3:
            strength = SignalStrength.STRONG_BUY
        elif net_score >= 1:
            strength = SignalStrength.BUY
        elif net_score <= -3:
            strength = SignalStrength.STRONG_SELL
        elif net_score <= -1:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.NEUTRAL

        source_str = "; ".join(sources) if sources else "Neutral institutional flow"

        return Signal(
            ticker=ticker,
            signal_type=SignalType.INSTITUTIONAL,
            strength=strength,
            confidence=confidence,
            source=source_str,
            details=fii_dii_data,
            expiry_horizon=TimeHorizon.WEEKLY
        )

    def generate_bulk_block_signal(
        self,
        ticker: str,
        deal_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate signal from bulk/block deal activity.

        Expected data:
        - total_deals, buy/sell counts and values
        - net_value_cr, sentiment, top_clients
        """
        sources = []

        sentiment = deal_data.get('sentiment', 'no_data')
        net_value = deal_data.get('net_value_cr', 0)
        buy_count = deal_data.get('buy', {}).get('count', 0)
        sell_count = deal_data.get('sell', {}).get('count', 0)
        top_clients = deal_data.get('top_clients', [])

        if sentiment == 'no_data' or deal_data.get('total_deals', 0) == 0:
            return None

        if sentiment == 'bullish':
            strength = SignalStrength.BUY
            sources.append(f"Net buying ₹{net_value:.1f}Cr in large deals")
        elif sentiment == 'bearish':
            strength = SignalStrength.SELL
            sources.append(f"Net selling ₹{abs(net_value):.1f}Cr in large deals")
        else:
            strength = SignalStrength.NEUTRAL
            sources.append("Mixed large deal activity")

        # Check for notable clients
        if top_clients:
            client_names = [c[0] for c in top_clients[:2]]
            sources.append(f"Key players: {', '.join(client_names)}")

        confidence = min(deal_data.get('total_deals', 1) / 5, 1.0) * 0.7

        return Signal(
            ticker=ticker,
            signal_type=SignalType.BULK_BLOCK,
            strength=strength,
            confidence=confidence,
            source="; ".join(sources),
            details=deal_data,
            expiry_horizon=TimeHorizon.WEEKLY
        )

    def generate_insider_signal(
        self,
        ticker: str,
        insider_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate signal from insider trading (SAST) data.

        Expected data:
        - buys/sells with counts and values
        - promoter_buys/sells
        - sentiment
        """
        sources = []

        sentiment = insider_data.get('sentiment', 'no_data')
        if sentiment == 'no_data' or insider_data.get('total_filings', 0) == 0:
            return None

        promoter_buys = insider_data.get('buys', {}).get('promoter_buys', 0)
        promoter_sells = insider_data.get('sells', {}).get('promoter_sells', 0)

        # Promoter activity is strongest signal
        if promoter_buys > 0 and promoter_sells == 0:
            strength = SignalStrength.STRONG_BUY
            sources.append(f"Promoter buying ({promoter_buys} transactions)")
            confidence = 0.9
        elif promoter_sells > 0 and promoter_buys == 0:
            strength = SignalStrength.STRONG_SELL
            sources.append(f"Promoter selling ({promoter_sells} transactions)")
            confidence = 0.85
        elif sentiment == 'strongly_bullish':
            strength = SignalStrength.STRONG_BUY
            sources.append("Strong insider buying")
            confidence = 0.8
        elif sentiment == 'bullish':
            strength = SignalStrength.BUY
            sources.append("Insider buying")
            confidence = 0.7
        elif sentiment == 'strongly_bearish':
            strength = SignalStrength.STRONG_SELL
            sources.append("Strong insider selling")
            confidence = 0.8
        elif sentiment == 'bearish':
            strength = SignalStrength.SELL
            sources.append("Insider selling")
            confidence = 0.7
        else:
            strength = SignalStrength.NEUTRAL
            sources.append("Mixed insider activity")
            confidence = 0.5

        return Signal(
            ticker=ticker,
            signal_type=SignalType.INSIDER,
            strength=strength,
            confidence=confidence,
            source="; ".join(sources),
            details=insider_data,
            expiry_horizon=TimeHorizon.MONTHLY
        )

    def generate_news_signal(
        self,
        ticker: str,
        news_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate signal from news sentiment analysis.

        Expected data:
        - sentiment_score: -1 to +1
        - article_count
        - positive/negative/neutral counts
        """
        sentiment_score = news_data.get('sentiment_score', 0)
        article_count = news_data.get('article_count', 0)

        if article_count == 0:
            return None

        sources = []

        if sentiment_score > 0.5:
            strength = SignalStrength.STRONG_BUY
            sources.append(f"Very positive news ({article_count} articles)")
        elif sentiment_score > 0.2:
            strength = SignalStrength.BUY
            sources.append(f"Positive news coverage ({article_count} articles)")
        elif sentiment_score < -0.5:
            strength = SignalStrength.STRONG_SELL
            sources.append(f"Very negative news ({article_count} articles)")
        elif sentiment_score < -0.2:
            strength = SignalStrength.SELL
            sources.append(f"Negative news coverage ({article_count} articles)")
        else:
            strength = SignalStrength.NEUTRAL
            sources.append(f"Neutral news ({article_count} articles)")

        confidence = min(article_count / 10, 1.0) * 0.6 + 0.2

        return Signal(
            ticker=ticker,
            signal_type=SignalType.NEWS,
            strength=strength,
            confidence=confidence,
            source="; ".join(sources),
            details=news_data,
            expiry_horizon=TimeHorizon.SHORT
        )

    def generate_catalyst_signal(
        self,
        ticker: str,
        catalyst_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Generate signal from specific news catalyst.

        Expected data:
        - catalyst_type: contract_win, product_launch, etc.
        - headline
        - impact: bullish/bearish
        - magnitude: high/medium/low
        """
        catalyst_type = catalyst_data.get('catalyst_type', '')
        impact = catalyst_data.get('impact', 'neutral')
        magnitude = catalyst_data.get('magnitude', 'medium')
        headline = catalyst_data.get('headline', '')

        sources = []

        # Strong bullish catalysts
        strong_bullish = ['contract_win', 'strong_results', 'expansion', 'acquisition_positive']
        # Moderate bullish
        moderate_bullish = ['product_launch', 'upgrade', 'partnership', 'award']
        # Strong bearish
        strong_bearish = ['fraud', 'regulatory_action', 'debt_default', 'management_exodus']
        # Moderate bearish
        moderate_bearish = ['downgrade', 'lawsuit', 'weak_results', 'guidance_cut']

        if catalyst_type in strong_bullish or (impact == 'bullish' and magnitude == 'high'):
            strength = SignalStrength.STRONG_BUY
            confidence = 0.85
        elif catalyst_type in moderate_bullish or impact == 'bullish':
            strength = SignalStrength.BUY
            confidence = 0.7
        elif catalyst_type in strong_bearish or (impact == 'bearish' and magnitude == 'high'):
            strength = SignalStrength.STRONG_SELL
            confidence = 0.85
        elif catalyst_type in moderate_bearish or impact == 'bearish':
            strength = SignalStrength.SELL
            confidence = 0.7
        else:
            strength = SignalStrength.NEUTRAL
            confidence = 0.5

        sources.append(f"{catalyst_type.replace('_', ' ').title()}: {headline[:50]}")

        return Signal(
            ticker=ticker,
            signal_type=SignalType.CATALYST,
            strength=strength,
            confidence=confidence,
            source="; ".join(sources),
            details=catalyst_data,
            expiry_horizon=TimeHorizon.WEEKLY
        )

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def rank_recommendations(
        self,
        recommendations: List[CombinedRecommendation],
        action_filter: Optional[str] = None
    ) -> List[CombinedRecommendation]:
        """
        Rank recommendations by score and confidence.

        Args:
            recommendations: List of recommendations
            action_filter: Filter by action (BUY, SELL, etc.)

        Returns:
            Sorted list of recommendations
        """
        filtered = recommendations
        if action_filter:
            filtered = [r for r in recommendations if action_filter in r.action]

        # Sort by score * confidence (absolute value for sells)
        return sorted(
            filtered,
            key=lambda r: abs(r.score) * r.confidence,
            reverse=True
        )

    def get_top_recommendations(
        self,
        recommendations: List[CombinedRecommendation],
        n: int = 10,
        action: str = 'BUY'
    ) -> List[CombinedRecommendation]:
        """Get top N recommendations for given action."""
        filtered = [r for r in recommendations if action in r.action]
        ranked = self.rank_recommendations(filtered)
        return ranked[:n]
