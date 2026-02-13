"""
FinAgent Signals Module

This module contains signal generation and combination:
- SignalCombiner: Merges multiple signals into actionable recommendations
- PEADStrategy: Post-Earnings Announcement Drift signals
- AdvancedSignalGenerator: Multi-indicator technical analysis
- Signal types: Technical, Fundamental, Institutional, News-based, PEAD
"""

from .signal_combiner import (
    SignalCombiner,
    Signal,
    SignalType,
    SignalStrength,
    CombinedRecommendation,
)

from .pead_strategy import (
    PEADStrategy,
    PEADSignal,
    EarningsEvent,
    EarningsSurprise,
)

from .advanced_signals import (
    AdvancedSignalGenerator,
    AdvancedIndicators,
    TechnicalSignal,
    SignalDirection,
    FundamentalEnhancer,
    FundamentalData,
    NIFTY_50,
    NIFTY_NEXT_50,
    NIFTY_200,
    FNO_STOCKS,
    BROAD_MARKET,
    ALL_INDIAN_STOCKS,
)

__all__ = [
    "SignalCombiner",
    "Signal",
    "SignalType",
    "SignalStrength",
    "CombinedRecommendation",
    "PEADStrategy",
    "PEADSignal",
    "EarningsEvent",
    "EarningsSurprise",
    "AdvancedSignalGenerator",
    "AdvancedIndicators",
    "TechnicalSignal",
    "SignalDirection",
    "FundamentalEnhancer",
    "FundamentalData",
    "NIFTY_50",
    "NIFTY_NEXT_50",
    "NIFTY_200",
    "FNO_STOCKS",
    "BROAD_MARKET",
    "ALL_INDIAN_STOCKS",
]
