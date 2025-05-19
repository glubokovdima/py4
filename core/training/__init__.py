# core/training/__init__.py

"""
Model Training Submodule.

This submodule is responsible for training machine learning models
(e.g., CatBoost classifiers and regressors) using the preprocessed features.
It includes logic for feature selection, hyperparameter tuning (optional),
model fitting, evaluation, and saving trained models.
"""

from .trainer import (
    main_train_logic,
    # Potentially expose other key functions if they are meant for reuse:
    # select_top_features_for_model, # Example if feature selection is a distinct reusable step
    # train_single_catboost_model,    # Example if you want to train one model type specifically
)

__all__ = [
    "main_train_logic",
    # "select_top_features_for_model",
    # "train_single_catboost_model",
]
