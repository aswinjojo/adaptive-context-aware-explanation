"""Data loading utilities for ACXF datasets."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_german_credit(filepath: str) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Load and preprocess German Credit dataset.
    
    Args:
        filepath: Path to german_credit.csv
        
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded German Credit dataset: {df.shape}")
        
        # Assume target is last column or named 'target'/'class'
        if 'target' in df.columns:
            target = df['target']
            features = df.drop('target', axis=1)
        elif 'class' in df.columns:
            target = df['class']
            features = df.drop('class', axis=1)
        else:
            # Assume last column is target
            target = df.iloc[:, -1]
            features = df.iloc[:, :-1]
        
        feature_names = features.columns.tolist()
        return features, target, feature_names
    except Exception as e:
        logger.error(f"Error loading German Credit: {e}")
        raise


def load_diabetes(filepath: str) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Load and preprocess Diabetes dataset.
    
    Args:
        filepath: Path to diabetes.csv
        
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded Diabetes dataset: {df.shape}")
        
        if 'Outcome' in df.columns:
            target = df['Outcome']
            features = df.drop('Outcome', axis=1)
        elif 'target' in df.columns:
            target = df['target']
            features = df.drop('target', axis=1)
        else:
            target = df.iloc[:, -1]
            features = df.iloc[:, :-1]
        
        feature_names = features.columns.tolist()
        return features, target, feature_names
    except Exception as e:
        logger.error(f"Error loading Diabetes: {e}")
        raise


def load_telco_churn(filepath: str) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Load and preprocess Telco Churn dataset.
    
    Args:
        filepath: Path to telco_churn.csv
        
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded Telco Churn dataset: {df.shape}")
        
        if 'Churn' in df.columns:
            target = df['Churn']
            features = df.drop('Churn', axis=1)
        elif 'churn' in df.columns:
            target = df['churn']
            features = df.drop('churn', axis=1)
        elif 'target' in df.columns:
            target = df['target']
            features = df.drop('target', axis=1)
        else:
            target = df.iloc[:, -1]
            features = df.iloc[:, :-1]
        
        # Convert target to binary if needed
        if target.dtype == 'object':
            target = (target == 'Yes') | (target == 'yes') | (target == '1')
            target = target.astype(int)
        
        feature_names = features.columns.tolist()
        return features, target, feature_names
    except Exception as e:
        logger.error(f"Error loading Telco Churn: {e}")
        raise


def load_dataset(dataset_name: str, filepath: str) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Generic dataset loader.
    
    Args:
        dataset_name: Name of dataset ('german_credit', 'diabetes', 'telco_churn')
        filepath: Path to CSV file
        
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    loaders = {
        'german_credit': load_german_credit,
        'diabetes': load_diabetes,
        'telco_churn': load_telco_churn
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    
    return loaders[dataset_name](filepath)


