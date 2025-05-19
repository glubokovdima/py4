# core/data_ingestion/__init__.py

"""
Data Ingestion Submodule.

This submodule is responsible for fetching and storing raw market data,
primarily historical klines from exchanges like Binance.
"""

from .historical_data_loader import (
    main_historical_load_logic,
    update_single_symbol_tf_historical, # If you want to expose this specific function
    update_timeframe_parallel_historical # If you want to expose this specific function
)

__all__ = [
    "main_historical_load_logic",
    "update_single_symbol_tf_historical",
    "update_timeframe_parallel_historical"
]
