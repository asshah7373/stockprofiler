"""
FinAgent Strategies Module

This module contains trading strategies and backtesting:
- VectorBTFramework: Fast vectorized backtesting using VectorBT
- Pre-built strategies optimized for Indian markets
- Forecast signals for forward-looking analysis
"""

from .vectorbt_framework import (
    VectorBTFramework,
    StrategyResult,
    StrategyConfig,
    ForecastSignal,
    SignalAction,
)

__all__ = [
    "VectorBTFramework",
    "StrategyResult",
    "StrategyConfig",
    "ForecastSignal",
    "SignalAction",
]
