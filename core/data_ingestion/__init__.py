# core/data_ingestion/__init__.py

"""
Data Ingestion Submodule.

This submodule is responsible for fetching and storing raw market data,
including historical klines and incremental updates from exchanges like Binance.
"""

from .historical_data_loader import (
    main_historical_load_logic,
    update_single_symbol_tf_historical,
    update_timeframe_parallel_historical
)

from .incremental_data_loader import ( # Added this import block
    main_incremental_load_logic
)

__all__ = [
    "main_historical_load_logic",
    "update_single_symbol_tf_historical",
    "update_timeframe_parallel_historical",
    "main_incremental_load_logic" # Added to __all__
]
