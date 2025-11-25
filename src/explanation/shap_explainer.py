"""SHAP explainer wrapper for ACXF."""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wrapper for SHAP explanations with support for global and local explanations.
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Initialize explainer based on model type
        if hasattr(model, 'predict_proba'):
            try:
                # Try TreeExplainer for tree-based models
                self.explainer = shap.TreeExplainer(model)
                self.explainer_type = 'tree'
            except:
                # Fall back to KernelExplainer
                self.explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
                self.explainer_type = 'kernel'
        else:
            self.explainer = shap.KernelExplainer(model.predict, X_train[:100])
            self.explainer_type = 'kernel'
        
        logger.info(f"Initialized SHAP explainer (type: {self.explainer_type})")
    
    def explain_global(self, X_sample: Optional[np.ndarray] = None, n_samples: int = 100) -> Dict[str, Any]:
        """
        Generate global explanation (feature importance).
        
        Args:
            X_sample: Sample data (optional, uses training data if None)
            n_samples: Number of samples for kernel explainer
            
        Returns:
            Dictionary with global explanation data
        """
        if X_sample is None:
            X_sample = self.X_train[:n_samples]
        
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use positive class
        else:
            shap_values = self.explainer.shap_values(X_sample, nsamples=n_samples)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Ensure shap_values is a numpy array
        shap_values = np.asarray(shap_values)
        
        # Calculate mean absolute SHAP values for feature importance
        if shap_values.ndim > 1:
            # If 2D, calculate mean along the first axis (samples)
            feature_importance = np.abs(shap_values).mean(axis=0)
        else:
            # If 1D, use it directly (single sample case)
            feature_importance = np.abs(shap_values)
        
        # Ensure feature_importance is 1D
        feature_importance = np.asarray(feature_importance).flatten()
        
        # Ensure lengths match
        n_features = len(self.feature_names)
        if len(feature_importance) != n_features:
            if len(feature_importance) > n_features:
                feature_importance = feature_importance[:n_features]
            elif len(feature_importance) < n_features:
                logger.warning(f"Feature importance has {len(feature_importance)} values but expected {n_features}. Padding with zeros.")
                feature_importance = np.pad(feature_importance, (0, n_features - len(feature_importance)), mode='constant')
        
        # Create ranking
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'feature_importance': importance_df.to_dict('records'),
            'shap_values': shap_values,
            'type': 'global'
        }
    
    def explain_local(
        self,
        instance: np.ndarray,
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance.
        
        Args:
            instance: Single instance to explain (1D array)
            class_idx: Class index for multi-class (optional)
            
        Returns:
            Dictionary with local explanation data
        """
        # Ensure instance is 2D
        instance = np.asarray(instance)
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        # Ensure instance has correct number of features
        if instance.shape[1] != len(self.feature_names):
            raise ValueError(f"Instance has {instance.shape[1]} features but {len(self.feature_names)} feature names provided")
        
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if class_idx is None else shap_values[class_idx]
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if class_idx is None else base_value[class_idx]
        else:
            shap_values = self.explainer.shap_values(instance, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if class_idx is None else shap_values[class_idx]
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if class_idx is None else base_value[class_idx]
        
        # Ensure shap_values is the right shape
        shap_values = np.asarray(shap_values)
        if shap_values.ndim > 1:
            # If 2D, take the first row (for single instance explanation)
            if shap_values.shape[0] == 1:
                shap_values_flat = shap_values[0]
            else:
                shap_values_flat = shap_values.flatten()
        else:
            shap_values_flat = shap_values
        
        # Ensure it's 1D
        shap_values_flat = np.asarray(shap_values_flat).flatten()
        
        # Ensure lengths match
        n_features = len(self.feature_names)
        if len(shap_values_flat) != n_features:
            # If SHAP returned more values (e.g., bias term), take first n_features
            if len(shap_values_flat) > n_features:
                shap_values_flat = shap_values_flat[:n_features]
            # If SHAP returned fewer values, pad with zeros
            elif len(shap_values_flat) < n_features:
                logger.warning(f"SHAP returned {len(shap_values_flat)} values but expected {n_features}. Padding with zeros.")
                shap_values_flat = np.pad(shap_values_flat, (0, n_features - len(shap_values_flat)), mode='constant')
        
        # Ensure we have exactly n_features values now
        assert len(shap_values_flat) == n_features, f"Length mismatch: shap_values_flat has {len(shap_values_flat)} values, expected {n_features}"
        assert len(self.feature_names) == n_features, f"Length mismatch: feature_names has {len(self.feature_names)} values, expected {n_features}"
        
        # Create feature contribution ranking
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values_flat,
            'abs_shap_value': np.abs(shap_values_flat)
        }).sort_values('abs_shap_value', ascending=False)
        
        prediction = self.model.predict_proba(instance)[0] if hasattr(self.model, 'predict_proba') else self.model.predict(instance)[0]
        
        return {
            'shap_values': shap_values_flat,
            'base_value': float(base_value),
            'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else [float(prediction)],
            'contributions': contributions.to_dict('records'),
            'instance': instance.flatten().tolist(),
            'type': 'local'
        }
    
    def get_force_plot_data(
        self,
        instance: np.ndarray,
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get data for SHAP force plot visualization.
        
        Args:
            instance: Single instance to explain
            class_idx: Class index (optional)
            
        Returns:
            Dictionary with force plot data
        """
        local_explanation = self.explain_local(instance, class_idx)
        
        return {
            'shap_values': local_explanation['shap_values'].tolist(),
            'base_value': local_explanation['base_value'],
            'feature_names': self.feature_names,
            'instance_values': local_explanation['instance']
        }

