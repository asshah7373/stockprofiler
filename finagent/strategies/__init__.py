"""
FinAgent Strategies Module

This module contains trading strategies and backtesting:
- VectorBTFramework: Fast vectorized backtesting using VectorBT
- Pre-built strategies optimized for Indian markets
"""

from .vectorbt_framework import (
    VectorBTFramework,
    StrategyResult,
    StrategyConfig,
)

__all__ = [
    "VectorBTFramework",
    "StrategyResult",
    "StrategyConfig",
]
