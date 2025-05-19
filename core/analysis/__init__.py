# core/analysis/__init__.py

"""
Analysis Submodule.

This submodule contains tools and scripts for data analysis tasks,
such as symbol selection based on liquidity and volatility.
"""

from .symbol_selector import main_symbol_selection_logic

__all__ = [
    "main_symbol_selection_logic"
]
