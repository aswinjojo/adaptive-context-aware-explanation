"""Preprocessing utilities for ACXF."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: Optional[List[str]] = None,
    handle_missing: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess features and target for machine learning.
    
    Args:
        X: Feature dataframe
        y: Target series
        categorical_features: List of categorical feature names
        handle_missing: Strategy for missing values ('mean', 'median', 'drop', 'mode')
        
    Returns:
        Tuple of (X_processed, y_processed, preprocessing_info)
    """
    X = X.copy()
    y = y.copy()
    
    # Handle missing values
    if handle_missing == 'drop':
        mask = ~(X.isnull().any(axis=1))
        X = X[mask]
        y = y[mask]
    elif handle_missing == 'mean':
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        X = X.fillna(X.mode().iloc[0] if len(X.mode()) > 0 else X.iloc[0])
    elif handle_missing == 'median':
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X = X.fillna(X.mode().iloc[0] if len(X.mode()) > 0 else X.iloc[0])
    elif handle_missing == 'mode':
        X = X.fillna(X.mode().iloc[0] if len(X.mode()) > 0 else X.iloc[0])
    
    # Identify categorical features if not provided
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode target if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        target_encoder = le
    else:
        target_encoder = None
        y = y.values
    
    # Preprocess features
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    if len(categorical_features) > 0 and len(numeric_features) > 0:
        # Mixed features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        X_processed = preprocessor.fit_transform(X)
        feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    elif len(categorical_features) > 0:
        # Only categorical
        preprocessor = OneHotEncoder(drop='first', sparse_output=False)
        X_processed = preprocessor.fit_transform(X[categorical_features])
        feature_names = list(preprocessor.get_feature_names_out(categorical_features))
    else:
        # Only numeric
        preprocessor = StandardScaler()
        X_processed = preprocessor.fit_transform(X)
        feature_names = numeric_features
    
    preprocessing_info = {
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'target_encoder': target_encoder
    }
    
    logger.info(f"Preprocessed data: {X_processed.shape}, features: {len(feature_names)}")
    
    return X_processed, y, preprocessing_info


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numeric features.
    
    Args:
        X: Feature dataframe
        
    Returns:
        Tuple of (categorical_features, numeric_features)
    """
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    return categorical, numeric


