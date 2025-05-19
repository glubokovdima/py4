# core/pipelines/__init__.py

"""
Pipelines Submodule.

This submodule contains scripts that define and execute multi-step
workflows (pipelines) by combining functionalities from other modules.
For example, a pipeline might include data update, feature preprocessing,
model training, and prediction generation.
"""

from .main_pipeline import main_pipeline_logic

__all__ = [
    "main_pipeline_logic"
]
