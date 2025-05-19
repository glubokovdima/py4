# core/backtesting/__init__.py

"""
Backtesting Submodule.

This submodule provides tools for evaluating model performance
on historical data. It includes metrics calculation, plotting of
accuracy over time, and confusion matrix generation.
"""

from .backtester import main_backtest_logic, run_backtest_on_data

__all__ = [
    "main_backtest_logic",
    "run_backtest_on_data" # If you want to expose the core backtest run separately
]
