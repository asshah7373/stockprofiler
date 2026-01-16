"""
FinAgent Signals Module

This module contains signal generation and combination:
- SignalCombiner: Merges multiple signals into actionable recommendations
- PEADStrategy: Post-Earnings Announcement Drift signals
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
]
