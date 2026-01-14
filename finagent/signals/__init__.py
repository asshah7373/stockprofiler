"""
FinAgent Signals Module

This module contains signal generation and combination:
- SignalCombiner: Merges multiple signals into actionable recommendations
- Signal types: Technical, Fundamental, Institutional, News-based
"""

from .signal_combiner import (
    SignalCombiner,
    Signal,
    SignalType,
    SignalStrength,
    CombinedRecommendation,
)

__all__ = [
    "SignalCombiner",
    "Signal",
    "SignalType",
    "SignalStrength",
    "CombinedRecommendation",
]
