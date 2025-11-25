"""Model training utilities for ACXF."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Optional, Dict, Any
import logging
import pickle
import os

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

logger = logging.getLogger(__name__)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train a classification model.
    
    Args:
        X: Feature array
        y: Target array
        model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
        test_size: Proportion of test set
        random_state: Random seed
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            random_state=random_state
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            random_state=random_state,
            solver='lbfgs'
        )
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (128, 64)),
            activation=kwargs.get('activation', 'relu'),
            learning_rate_init=kwargs.get('learning_rate_init', 0.001),
            max_iter=kwargs.get('max_iter', 500),
            random_state=random_state
        )
    elif model_type == 'xgboost':
        if xgb is None:
            raise ImportError(
                "xgboost is not installed. Please run `pip install xgboost` "
                "or install from requirements.txt."
            )
        model = xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 300),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.05),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            eval_metric=kwargs.get('eval_metric', 'logloss'),
            random_state=random_state,
            tree_method=kwargs.get('tree_method', 'hist')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model_type': model_type
    }
    
    logger.info(f"Model trained - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return model, X_train, X_test, y_train, y_test, metrics


def save_model(model: Any, filepath: str) -> None:
    """Save model to file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load model from file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {filepath}")
    return model

