"""LIME explainer wrapper for ACXF."""

import numpy as np
import pandas as pd
from lime import lime_tabular
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Wrapper for LIME explanations for tabular data.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'classification',
        discretize_continuous: bool = False
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            X_train: Training data
            feature_names: Names of features
            class_names: Names of classes
            mode: 'classification' or 'regression'
            discretize_continuous: Whether to discretize continuous features (default: False)
        """
        self.model = model
        # Ensure X_train is a proper numpy array
        self.X_train = np.asarray(X_train)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        self.class_names = class_names or ['Class_0', 'Class_1']
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        
        # Always disable discretization to avoid known LIME bugs with broadcasting
        # The discretizer causes "could not broadcast" errors with certain data formats
        discretize_continuous = False
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            discretize_continuous=discretize_continuous
        )
        
        logger.info(f"Initialized LIME explainer (mode: {mode})")
    
    def explain_local(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        top_labels: int = 1
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance.
        
        Args:
            instance: Single instance to explain (1D array)
            num_features: Number of top features to show
            top_labels: Number of top labels to explain
            
        Returns:
            Dictionary with local explanation data
        """
        # Ensure instance is 1D numpy array for LIME (which expects a flat array)
        instance = np.asarray(instance, dtype=self.X_train.dtype)
        if instance.ndim > 1:
            # If 2D, take first row and flatten
            instance = instance[0].flatten()
        else:
            # If already 1D, ensure it's a contiguous array
            instance = np.ascontiguousarray(instance.flatten())
        
        # Ensure instance matches training data format
        if instance.shape[0] != self.X_train.shape[1]:
            raise ValueError(f"Instance has {instance.shape[0]} features but training data has {self.X_train.shape[1]} features")
        
        # Ensure instance is the same dtype as training data
        instance = instance.astype(self.X_train.dtype)
        
        # Proactively check if discretization is enabled and disable it to avoid errors
        # LIME's discretizer has known issues with certain data formats
        if self.discretize_continuous:
            logger.info("Discretization is enabled but may cause errors. Disabling it proactively.")
            self.explainer = lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=self.mode,
                discretize_continuous=False
            )
            self.discretize_continuous = False
        
        # Try to explain, with fallback if discretizer fails
        try:
            explanation = self.explainer.explain_instance(
                instance,
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=num_features,
                top_labels=top_labels
            )
        except (ValueError, TypeError, IndexError) as e:
            # If discretizer fails (broadcasting error), try without discretization
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["could not broadcast", "broadcast", "into shape", "shape"]):
                logger.warning(f"LIME discretizer failed ({e}), retrying with discretize_continuous=False")
                # Recreate explainer without discretization and update instance
                self.explainer = lime_tabular.LimeTabularExplainer(
                    self.X_train,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode=self.mode,
                    discretize_continuous=False
                )
                self.discretize_continuous = False
                # Retry with new explainer
                explanation = self.explainer.explain_instance(
                    instance,
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    num_features=num_features,
                    top_labels=top_labels
                )
            else:
                raise
        except Exception as e:
            # Catch any other exception and check if it's a shape/broadcast issue
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["could not broadcast", "broadcast", "into shape", "shape"]):
                logger.warning(f"LIME error detected ({e}), retrying with discretize_continuous=False")
                self.explainer = lime_tabular.LimeTabularExplainer(
                    self.X_train,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode=self.mode,
                    discretize_continuous=False
                )
                self.discretize_continuous = False
                explanation = self.explainer.explain_instance(
                    instance,
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    num_features=num_features,
                    top_labels=top_labels
                )
            else:
                raise
        
        # Get prediction first to determine which label to use
        instance_2d = instance.reshape(1, -1) if instance.ndim == 1 else instance
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(instance_2d)[0]
            predicted_class = int(np.argmax(prediction))
        else:
            prediction = self.model.predict(instance_2d)[0]
            predicted_class = int(prediction) if not isinstance(prediction, np.ndarray) else int(prediction[0])
        
        # Extract explanation data - use the predicted class label
        # LIME's as_list() needs a label, and we should use the predicted class
        try:
            exp_list = explanation.as_list(label=predicted_class)
        except (KeyError, IndexError):
            # If the predicted class doesn't exist in explanation, try available labels
            available_labels = list(explanation.local_exp.keys()) if hasattr(explanation, 'local_exp') else [0]
            if available_labels:
                exp_list = explanation.as_list(label=available_labels[0])
            else:
                # Fallback: try without specifying label
                exp_list = explanation.as_list()
        
        # Format contributions
        contributions = []
        for feature, value in exp_list:
            contributions.append({
                'feature': feature,
                'contribution': value
            })
        
        # Sort by absolute contribution
        contributions = sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True)
        
        # Format prediction for return
        if isinstance(prediction, np.ndarray):
            prediction_list = prediction.tolist()
        elif isinstance(prediction, (list, tuple)):
            prediction_list = [float(p) for p in prediction]
        else:
            prediction_list = [float(prediction)]
        
        return {
            'contributions': contributions,
            'prediction': prediction_list,
            'instance': instance.tolist() if isinstance(instance, np.ndarray) else instance,
            'explanation_object': explanation,  # Keep for visualization
            'type': 'local'
        }
    
    def generate_counterfactual(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        num_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation.
        
        Args:
            instance: Instance to explain
            target_class: Target class for counterfactual (optional)
            num_features: Number of features to modify
            
        Returns:
            Dictionary with counterfactual data
        """
        # Ensure instance is 1D for LIME (which expects a flat array)
        if instance.ndim > 1:
            instance = instance.flatten() if instance.shape[0] == 1 else instance[0]
        else:
            instance = instance.flatten()  # Ensure it's a proper 1D array
        
        # Get current prediction (model expects 2D array)
        instance_2d = instance.reshape(1, -1)
        current_pred = self.model.predict_proba(instance_2d)[0] if hasattr(self.model, 'predict_proba') else [self.model.predict(instance_2d)[0]]
        current_class = np.argmax(current_pred)
        
        # Get explanation
        explanation = self.explain_local(instance, num_features=num_features)
        
        # Create counterfactual by modifying top contributing features
        counterfactual = instance.copy()
        modifications = []
        
        for contrib in explanation['contributions'][:num_features]:
            feature_name = contrib['feature']
            # Extract feature index from name (handles both "Feature_X" and direct names)
            try:
                if '(' in feature_name:
                    # LIME format: "Feature_Name (value)"
                    feat_name = feature_name.split('(')[0].strip()
                    if feat_name in self.feature_names:
                        feat_idx = self.feature_names.index(feat_name)
                    else:
                        # Try to extract index
                        feat_idx = int(feature_name.split('_')[-1].split('(')[0])
                else:
                    if feature_name in self.feature_names:
                        feat_idx = self.feature_names.index(feature_name)
                    else:
                        feat_idx = int(feature_name.split('_')[-1])
                
                # Modify feature (simple approach: flip direction)
                original_value = counterfactual[feat_idx]
                # Change by moving towards opposite direction
                if contrib['contribution'] > 0:
                    # Feature pushes towards positive class, reduce it
                    counterfactual[feat_idx] = original_value * 0.5
                else:
                    # Feature pushes towards negative class, increase it
                    counterfactual[feat_idx] = original_value * 1.5
                
                modifications.append({
                    'feature': feature_name,
                    'original_value': float(original_value),
                    'new_value': float(counterfactual[feat_idx]),
                    'contribution': contrib['contribution']
                })
            except (ValueError, IndexError):
                continue
        
        # Get counterfactual prediction
        cf_pred = self.model.predict_proba(counterfactual.reshape(1, -1))[0] if hasattr(self.model, 'predict_proba') else [self.model.predict(counterfactual.reshape(1, -1))[0]]
        cf_class = np.argmax(cf_pred)
        
        return {
            'original_instance': instance.tolist(),
            'counterfactual_instance': counterfactual.tolist(),
            'original_prediction': current_pred.tolist() if isinstance(current_pred, np.ndarray) else current_pred,
            'counterfactual_prediction': cf_pred.tolist() if isinstance(cf_pred, np.ndarray) else cf_pred,
            'original_class': int(current_class),
            'counterfactual_class': int(cf_class),
            'modifications': modifications,
            'class_changed': current_class != cf_class
        }

