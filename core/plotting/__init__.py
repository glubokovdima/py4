# core/plotting/__init__.py

"""
Plotting Submodule.

This submodule contains functions for generating various plots
used in reports, backtesting analysis, and potentially live monitoring.
Examples include confusion matrices, accuracy over time, etc.
"""

from .report_plots import (
    plot_confusion_matrix_custom,  # Renamed for clarity if it's a custom version
    plot_daily_accuracy_custom   # Renamed for clarity
)

__all__ = [
    "plot_confusion_matrix_custom",
    "plot_daily_accuracy_custom"
]
