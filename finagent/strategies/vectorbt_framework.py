"""
VectorBT Strategy Framework

Fast vectorized backtesting for trading strategies using VectorBT.
Optimized for Indian markets with support for:
- Technical strategies (RSI, MACD, Bollinger Bands, etc.)
- Signal-based strategies (from SignalCombiner)
- Custom entry/exit rules
- Parameter optimization

VectorBT provides:
- 100x faster than traditional backtesting
- Vectorized operations on NumPy arrays
- Built-in portfolio analytics
- Parameter grid search optimization
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    stop_loss_pct: float = 0.05       # 5% stop loss
    take_profit_pct: float = 0.10     # 10% take profit
    position_size: float = 1.0         # Fraction of capital per trade
    max_positions: int = 5             # Max concurrent positions
    holding_period_days: Optional[int] = None  # Max holding period


@dataclass
class StrategyResult:
    """Results from strategy backtesting."""
    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float           # Percentage
    cagr: float                   # Compound Annual Growth Rate
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float           # Maximum drawdown percentage
    win_rate: float               # Percentage of winning trades
    total_trades: int
    profitable_trades: int
    avg_trade_return: float
    avg_holding_days: float
    profit_factor: float          # Gross profit / Gross loss
    trades: List[Dict] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    benchmark_return: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'ticker': self.ticker,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_value': round(self.final_value, 2),
            'total_return': round(self.total_return, 2),
            'cagr': round(self.cagr, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'win_rate': round(self.win_rate, 2),
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'avg_trade_return': round(self.avg_trade_return, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
            'profit_factor': round(self.profit_factor, 2) if self.profit_factor else None,
            'benchmark_return': round(self.benchmark_return, 2) if self.benchmark_return else None,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Strategy: {self.strategy_name} on {self.ticker}
Period: {self.start_date} to {self.end_date}
Initial Capital: ₹{self.initial_capital:,.0f}
Final Value: ₹{self.final_value:,.0f}
Total Return: {self.total_return:.2f}%
CAGR: {self.cagr:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2f}%
Win Rate: {self.win_rate:.1f}%
Total Trades: {self.total_trades}
Profit Factor: {self.profit_factor:.2f}
"""


class VectorBTFramework:
    """
    Vectorized backtesting framework using VectorBT.

    Features:
    - Fast vectorized backtesting
    - Built-in technical indicators
    - Portfolio analytics
    - Parameter optimization
    - Multiple strategy support

    Usage:
        framework = VectorBTFramework()
        result = framework.backtest_rsi_strategy(price_data, rsi_oversold=30)
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.001,  # 0.1% (typical for Indian brokers)
        slippage_pct: float = 0.001     # 0.1% slippage
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.logger = logging.getLogger(__name__)

        # Check if vectorbt is available
        self._vbt = None
        self._check_vectorbt()

    def _check_vectorbt(self):
        """Check if VectorBT is installed."""
        try:
            import vectorbt as vbt
            self._vbt = vbt
            self.logger.debug("VectorBT loaded successfully")
        except ImportError:
            self.logger.warning(
                "VectorBT not installed. Install with: pip install vectorbt"
            )
            self._vbt = None

    def _ensure_vectorbt(self):
        """Ensure VectorBT is available."""
        if self._vbt is None:
            raise ImportError(
                "VectorBT is required for backtesting. "
                "Install with: pip install vectorbt"
            )
        return self._vbt

    # =========================================================================
    # Technical Indicator Strategies
    # =========================================================================

    def backtest_rsi_strategy(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> StrategyResult:
        """
        Backtest RSI mean reversion strategy.

        Entry: RSI crosses below oversold threshold
        Exit: RSI crosses above overbought threshold OR stop loss/take profit

        Args:
            price_data: DataFrame with 'close' column
            rsi_period: RSI calculation period
            rsi_oversold: Entry threshold
            rsi_overbought: Exit threshold
        """
        vbt = self._ensure_vectorbt()

        # Ensure we have close prices
        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        # Calculate RSI
        rsi = vbt.RSI.run(close, window=rsi_period).rsi.to_numpy()

        # Generate signals
        entries = self._crossover_below(rsi, rsi_oversold)
        exits = self._crossover_above(rsi, rsi_overbought)

        # Run backtest
        return self._run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            ticker=ticker,
            strategy_name=f"RSI({rsi_period}, {rsi_oversold}/{rsi_overbought})",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

    def backtest_macd_strategy(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> StrategyResult:
        """
        Backtest MACD crossover strategy.

        Entry: MACD line crosses above signal line
        Exit: MACD line crosses below signal line
        """
        vbt = self._ensure_vectorbt()

        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        # Calculate MACD
        macd = vbt.MACD.run(
            close,
            fast_window=fast_period,
            slow_window=slow_period,
            signal_window=signal_period
        )

        macd_line = macd.macd.to_numpy()
        signal_line = macd.signal.to_numpy()

        # Generate signals
        entries = self._crossover_above_series(macd_line, signal_line)
        exits = self._crossover_below_series(macd_line, signal_line)

        return self._run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            ticker=ticker,
            strategy_name=f"MACD({fast_period},{slow_period},{signal_period})",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

    def backtest_bollinger_strategy(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        bb_period: int = 20,
        bb_std: float = 2.0,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> StrategyResult:
        """
        Backtest Bollinger Bands mean reversion strategy.

        Entry: Price crosses below lower band
        Exit: Price crosses above upper band OR middle band
        """
        vbt = self._ensure_vectorbt()

        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        # Calculate Bollinger Bands
        bb = vbt.BBANDS.run(close, window=bb_period, alpha=bb_std)

        lower = bb.lower.to_numpy()
        upper = bb.upper.to_numpy()
        middle = bb.middle.to_numpy()
        close_arr = close.to_numpy()

        # Generate signals
        entries = self._crossover_below_series(close_arr, lower)
        exits = self._crossover_above_series(close_arr, middle)

        return self._run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            ticker=ticker,
            strategy_name=f"BollingerBands({bb_period}, {bb_std})",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

    def backtest_sma_crossover(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        fast_period: int = 20,
        slow_period: int = 50,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15
    ) -> StrategyResult:
        """
        Backtest SMA crossover trend-following strategy.

        Entry: Fast SMA crosses above slow SMA
        Exit: Fast SMA crosses below slow SMA
        """
        vbt = self._ensure_vectorbt()

        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        # Calculate SMAs
        fast_sma = vbt.MA.run(close, window=fast_period).ma.to_numpy()
        slow_sma = vbt.MA.run(close, window=slow_period).ma.to_numpy()

        # Generate signals
        entries = self._crossover_above_series(fast_sma, slow_sma)
        exits = self._crossover_below_series(fast_sma, slow_sma)

        return self._run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            ticker=ticker,
            strategy_name=f"SMA_Crossover({fast_period}/{slow_period})",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

    def backtest_combined_strategy(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        rsi_period: int = 14,
        rsi_oversold: int = 35,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_period: int = 50,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.12
    ) -> StrategyResult:
        """
        Combined strategy using RSI, MACD, and SMA filter.

        Entry conditions (all must be true):
        - RSI < oversold threshold
        - MACD histogram positive (bullish momentum)
        - Price above SMA (uptrend)

        Exit conditions (any):
        - RSI > 60
        - MACD histogram turns negative
        - Stop loss / Take profit hit
        """
        vbt = self._ensure_vectorbt()

        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        # Calculate indicators
        rsi = vbt.RSI.run(close, window=rsi_period).rsi.to_numpy()

        macd = vbt.MACD.run(close, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
        macd_hist = macd.hist.to_numpy()

        sma = vbt.MA.run(close, window=sma_period).ma.to_numpy()
        close_arr = close.to_numpy()

        # Entry: RSI oversold AND MACD bullish AND above SMA
        rsi_entry = rsi < rsi_oversold
        macd_entry = macd_hist > 0
        sma_filter = close_arr > sma

        entries = rsi_entry & macd_entry & sma_filter
        # Shift to next bar for execution
        entries = np.roll(entries, 1)
        entries[0] = False

        # Exit: RSI > 60 OR MACD turns bearish
        rsi_exit = rsi > 60
        macd_exit = macd_hist < 0
        exits = rsi_exit | macd_exit
        exits = np.roll(exits, 1)
        exits[0] = False

        return self._run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            ticker=ticker,
            strategy_name=f"Combined(RSI+MACD+SMA)",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

    # =========================================================================
    # Signal-Based Strategy
    # =========================================================================

    def backtest_signals(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        ticker: str = "UNKNOWN",
        entry_threshold: float = 25,   # Score threshold for entry
        exit_threshold: float = -10,   # Score threshold for exit
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> StrategyResult:
        """
        Backtest strategy based on combined signal scores.

        Args:
            price_data: DataFrame with 'close' column
            signals: DataFrame with 'score' column aligned with price_data
            entry_threshold: Score above which to enter
            exit_threshold: Score below which to exit
        """
        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        # Align signals with price data
        if len(signals) != len(close):
            self.logger.warning("Signal and price data length mismatch, aligning...")
            signals = signals.reindex(close.index, method='ffill')

        score = signals['score'].to_numpy() if 'score' in signals.columns else np.zeros(len(close))

        # Generate entry/exit signals
        entries = score > entry_threshold
        exits = score < exit_threshold

        # Shift to next bar
        entries = np.roll(entries, 1)
        exits = np.roll(exits, 1)
        entries[0] = False
        exits[0] = False

        return self._run_backtest(
            close=close,
            entries=entries,
            exits=exits,
            ticker=ticker,
            strategy_name=f"SignalBased(entry>{entry_threshold})",
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

    # =========================================================================
    # Optimization
    # =========================================================================

    def optimize_rsi_strategy(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        rsi_periods: List[int] = [7, 14, 21],
        oversold_levels: List[int] = [20, 25, 30, 35],
        overbought_levels: List[int] = [65, 70, 75, 80]
    ) -> Dict[str, Any]:
        """
        Optimize RSI strategy parameters using grid search.

        Returns best parameters and all results.
        """
        vbt = self._ensure_vectorbt()

        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

        results = []
        best_result = None
        best_sharpe = float('-inf')

        for rsi_period in rsi_periods:
            for oversold in oversold_levels:
                for overbought in overbought_levels:
                    if oversold >= overbought:
                        continue

                    try:
                        result = self.backtest_rsi_strategy(
                            price_data=price_data,
                            ticker=ticker,
                            rsi_period=rsi_period,
                            rsi_oversold=oversold,
                            rsi_overbought=overbought
                        )

                        results.append({
                            'params': {
                                'rsi_period': rsi_period,
                                'oversold': oversold,
                                'overbought': overbought
                            },
                            'sharpe': result.sharpe_ratio,
                            'return': result.total_return,
                            'win_rate': result.win_rate,
                            'trades': result.total_trades
                        })

                        if result.sharpe_ratio > best_sharpe and result.total_trades >= 5:
                            best_sharpe = result.sharpe_ratio
                            best_result = result

                    except Exception as e:
                        self.logger.warning(f"Error testing params: {e}")
                        continue

        return {
            'best_params': results[results.index(
                max(results, key=lambda x: x['sharpe'] if x['trades'] >= 5 else float('-inf'))
            )]['params'] if results else None,
            'best_result': best_result,
            'all_results': sorted(results, key=lambda x: x['sharpe'], reverse=True)
        }

    def optimize_sma_crossover(
        self,
        price_data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        fast_periods: List[int] = [10, 20, 30],
        slow_periods: List[int] = [50, 100, 200]
    ) -> Dict[str, Any]:
        """Optimize SMA crossover strategy parameters."""
        results = []
        best_result = None
        best_sharpe = float('-inf')

        for fast in fast_periods:
            for slow in slow_periods:
                if fast >= slow:
                    continue

                try:
                    result = self.backtest_sma_crossover(
                        price_data=price_data,
                        ticker=ticker,
                        fast_period=fast,
                        slow_period=slow
                    )

                    results.append({
                        'params': {'fast_period': fast, 'slow_period': slow},
                        'sharpe': result.sharpe_ratio,
                        'return': result.total_return,
                        'win_rate': result.win_rate,
                        'trades': result.total_trades
                    })

                    if result.sharpe_ratio > best_sharpe and result.total_trades >= 3:
                        best_sharpe = result.sharpe_ratio
                        best_result = result

                except Exception as e:
                    self.logger.warning(f"Error testing params: {e}")
                    continue

        return {
            'best_params': max(results, key=lambda x: x['sharpe'] if x['trades'] >= 3 else float('-inf'))['params'] if results else None,
            'best_result': best_result,
            'all_results': sorted(results, key=lambda x: x['sharpe'], reverse=True)
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _crossover_above(self, arr: np.ndarray, threshold: float) -> np.ndarray:
        """Detect when array crosses above threshold."""
        above = arr > threshold
        crossed = above & ~np.roll(above, 1)
        crossed[0] = False
        return crossed

    def _crossover_below(self, arr: np.ndarray, threshold: float) -> np.ndarray:
        """Detect when array crosses below threshold."""
        below = arr < threshold
        crossed = below & ~np.roll(below, 1)
        crossed[0] = False
        return crossed

    def _crossover_above_series(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Detect when arr1 crosses above arr2."""
        above = arr1 > arr2
        crossed = above & ~np.roll(above, 1)
        crossed[0] = False
        return crossed

    def _crossover_below_series(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Detect when arr1 crosses below arr2."""
        below = arr1 < arr2
        crossed = below & ~np.roll(below, 1)
        crossed[0] = False
        return crossed

    def _run_backtest(
        self,
        close: pd.Series,
        entries: np.ndarray,
        exits: np.ndarray,
        ticker: str,
        strategy_name: str,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> StrategyResult:
        """
        Run backtest with given entry/exit signals.

        Uses VectorBT's Portfolio.from_signals for fast vectorized backtesting.
        """
        vbt = self._ensure_vectorbt()

        # Create portfolio
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission_pct,
            slippage=self.slippage_pct,
            sl_stop=stop_loss_pct,
            tp_stop=take_profit_pct,
            freq='1D'
        )

        # Extract metrics
        stats = pf.stats()

        # Get trade records
        trades_df = pf.trades.records_readable
        trades_list = []
        if len(trades_df) > 0:
            for _, trade in trades_df.iterrows():
                trades_list.append({
                    'entry_date': str(trade.get('Entry Timestamp', '')),
                    'exit_date': str(trade.get('Exit Timestamp', '')),
                    'entry_price': float(trade.get('Avg Entry Price', 0)),
                    'exit_price': float(trade.get('Avg Exit Price', 0)),
                    'return_pct': float(trade.get('Return', 0)) * 100,
                    'pnl': float(trade.get('PnL', 0))
                })

        # Calculate additional metrics
        total_trades = int(stats.get('Total Trades', 0))
        profitable_trades = int(pf.trades.winning.count())

        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        # Get profit factor
        gross_profit = pf.trades.winning.pnl.sum() if pf.trades.winning.count() > 0 else 0
        gross_loss = abs(pf.trades.losing.pnl.sum()) if pf.trades.losing.count() > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average holding time
        avg_holding = float(pf.trades.duration.mean().days) if total_trades > 0 else 0

        return StrategyResult(
            strategy_name=strategy_name,
            ticker=ticker,
            start_date=str(close.index[0].date()) if hasattr(close.index[0], 'date') else str(close.index[0]),
            end_date=str(close.index[-1].date()) if hasattr(close.index[-1], 'date') else str(close.index[-1]),
            initial_capital=self.initial_capital,
            final_value=float(stats.get('End Value', self.initial_capital)),
            total_return=float(stats.get('Total Return [%]', 0)),
            cagr=float(stats.get('Annualized Return [%]', 0)),
            sharpe_ratio=float(stats.get('Sharpe Ratio', 0)),
            sortino_ratio=float(stats.get('Sortino Ratio', 0)),
            max_drawdown=float(stats.get('Max Drawdown [%]', 0)),
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            avg_trade_return=float(stats.get('Avg Winning Trade [%]', 0)),
            avg_holding_days=avg_holding,
            profit_factor=profit_factor,
            trades=trades_list,
            equity_curve=pf.value()
        )

    # =========================================================================
    # Fallback Implementation (without VectorBT)
    # =========================================================================

    def backtest_simple(
        self,
        price_data: pd.DataFrame,
        entries: np.ndarray,
        exits: np.ndarray,
        ticker: str = "UNKNOWN",
        strategy_name: str = "Custom",
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> StrategyResult:
        """
        Simple backtesting fallback without VectorBT.

        Uses basic loop-based approach. Much slower but works without dependencies.
        """
        close = price_data['close'] if 'close' in price_data.columns else price_data['Close']
        close_arr = close.to_numpy()

        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity = [capital]

        for i in range(1, len(close_arr)):
            price = close_arr[i]

            if position == 0 and entries[i]:
                # Enter position
                position = (capital * 0.95) / price  # 95% of capital (leave room for commission)
                entry_price = price
                capital -= position * price * (1 + self.commission_pct)

            elif position > 0:
                # Check stop loss / take profit
                pnl_pct = (price - entry_price) / entry_price

                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct or exits[i]:
                    # Exit position
                    exit_price = price
                    capital += position * exit_price * (1 - self.commission_pct)

                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return_pct': pnl_pct * 100,
                        'pnl': position * (exit_price - entry_price)
                    })

                    position = 0
                    entry_price = 0

            equity.append(capital + position * price)

        # Final metrics
        final_value = equity[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Calculate Sharpe (simplified)
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Max drawdown
        equity_series = pd.Series(equity)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        # Trade stats
        total_trades = len(trades)
        profitable = sum(1 for t in trades if t['return_pct'] > 0)
        win_rate = (profitable / total_trades * 100) if total_trades > 0 else 0

        return StrategyResult(
            strategy_name=strategy_name,
            ticker=ticker,
            start_date=str(close.index[0]),
            end_date=str(close.index[-1]),
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            cagr=0,  # Not calculated in simple version
            sharpe_ratio=sharpe,
            sortino_ratio=0,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable,
            avg_trade_return=sum(t['return_pct'] for t in trades) / total_trades if total_trades > 0 else 0,
            avg_holding_days=0,
            profit_factor=0,
            trades=trades,
            equity_curve=pd.Series(equity, index=close.index)
        )

    # =========================================================================
    # Reporting
    # =========================================================================

    def compare_strategies(
        self,
        results: List[StrategyResult]
    ) -> pd.DataFrame:
        """
        Compare multiple strategy results.

        Returns DataFrame with key metrics for comparison.
        """
        comparison = []

        for result in results:
            comparison.append({
                'Strategy': result.strategy_name,
                'Ticker': result.ticker,
                'Return %': result.total_return,
                'CAGR %': result.cagr,
                'Sharpe': result.sharpe_ratio,
                'Max DD %': result.max_drawdown,
                'Win Rate %': result.win_rate,
                'Trades': result.total_trades,
                'Profit Factor': result.profit_factor
            })

        df = pd.DataFrame(comparison)
        return df.sort_values('Sharpe', ascending=False)

    def generate_report(
        self,
        result: StrategyResult,
        include_trades: bool = True
    ) -> str:
        """Generate detailed strategy report."""
        report = f"""
================================================================================
                        STRATEGY BACKTEST REPORT
================================================================================

Strategy: {result.strategy_name}
Ticker: {result.ticker}
Period: {result.start_date} to {result.end_date}

PERFORMANCE SUMMARY
-------------------
Initial Capital:     ₹{result.initial_capital:,.0f}
Final Value:         ₹{result.final_value:,.0f}
Total Return:        {result.total_return:.2f}%
CAGR:                {result.cagr:.2f}%

RISK METRICS
------------
Sharpe Ratio:        {result.sharpe_ratio:.2f}
Sortino Ratio:       {result.sortino_ratio:.2f}
Max Drawdown:        {result.max_drawdown:.2f}%

TRADE STATISTICS
----------------
Total Trades:        {result.total_trades}
Profitable Trades:   {result.profitable_trades}
Win Rate:            {result.win_rate:.1f}%
Profit Factor:       {result.profit_factor:.2f}
Avg Trade Return:    {result.avg_trade_return:.2f}%
Avg Holding Days:    {result.avg_holding_days:.1f}
"""

        if include_trades and result.trades:
            report += """
RECENT TRADES
-------------
"""
            for i, trade in enumerate(result.trades[-10:], 1):
                report += f"{i}. Entry: {trade.get('entry_price', 0):.2f} -> Exit: {trade.get('exit_price', 0):.2f} | Return: {trade.get('return_pct', 0):.2f}%\n"

        report += """
================================================================================
"""
        return report
