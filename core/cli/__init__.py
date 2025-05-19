# core/cli/__init__.py

"""
Command-Line Interface Submodule.

This submodule is responsible for parsing command-line arguments
and orchestrating the execution of different parts of the
crypto prediction application via the main CLI entry point.
"""

from .main import main_cli_entry_point

__all__ = [
    "main_cli_entry_point"
]
