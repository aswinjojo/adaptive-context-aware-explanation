"""Context-aware explanation engine for ACXF."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

from ..profiling.user_profiler import UserProfiler, UserCategory
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer

logger = logging.getLogger(__name__)


class DecisionCriticality(Enum):
    """Decision criticality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TimePressure(Enum):
    """Time pressure levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContextAwareEngine:
    """
    Context-aware explanation engine that adapts explanations based on user profile and context.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: List[str],
        user_profiler: UserProfiler
    ):
        """
        Initialize context-aware engine.
        
        Args:
            model: Trained model
            X_train: Training data
            feature_names: Feature names
            user_profiler: User profiler instance
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.user_profiler = user_profiler
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, X_train, feature_names)
        self.lime_explainer = LIMEExplainer(model, X_train, feature_names)
        
        logger.info("Initialized context-aware explanation engine")
    
    def generate_explanation(
        self,
        instance: np.ndarray,
        decision_criticality: DecisionCriticality = DecisionCriticality.MEDIUM,
        time_pressure: TimePressure = TimePressure.NONE,
        regulatory_required: bool = False,
        desired_detail: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate context-aware explanation.
        
        Args:
            instance: Instance to explain
            decision_criticality: Criticality of decision
            time_pressure: Time pressure level
            regulatory_required: Whether regulatory compliance is required
            desired_detail: Desired detail level ('low', 'medium', 'high')
            
        Returns:
            Dictionary with explanation data
        """
        user_category = self.user_profiler.get_category()
        preferences = self.user_profiler.get_explanation_preferences()
        
        # Determine explanation method
        explanation_method = self._select_method(
            user_category,
            decision_criticality,
            time_pressure,
            regulatory_required,
            desired_detail or preferences['detail_level']
        )
        
        # Generate base explanation
        if explanation_method == 'shap_global':
            explanation = self.shap_explainer.explain_global()
            explanation['method'] = 'SHAP (Global)'
        elif explanation_method == 'shap_local':
            explanation = self.shap_explainer.explain_local(instance)
            explanation['method'] = 'SHAP (Local)'
        elif explanation_method == 'lime':
            num_features = preferences.get('max_features', 10) or len(self.feature_names)
            explanation = self.lime_explainer.explain_local(instance, num_features=num_features)
            explanation['method'] = 'LIME'
        else:
            # Default to LIME
            explanation = self.lime_explainer.explain_local(instance, num_features=10)
            explanation['method'] = 'LIME'
        
        # Add context information
        explanation['context'] = {
            'user_category': user_category.value,
            'decision_criticality': decision_criticality.value,
            'time_pressure': time_pressure.value,
            'regulatory_required': regulatory_required,
            'detail_level': desired_detail or preferences['detail_level']
        }
        
        # Format explanation based on user preferences
        formatted_explanation = self._format_explanation(explanation, preferences, user_category)
        
        return formatted_explanation
    
    def _select_method(
        self,
        user_category: UserCategory,
        decision_criticality: DecisionCriticality,
        time_pressure: TimePressure,
        regulatory_required: bool,
        detail_level: str
    ) -> str:
        """
        Select appropriate explanation method based on context.
        
        Returns:
            Method name ('shap_global', 'shap_local', 'lime')
        """
        # Regulatory requirements -> SHAP (more interpretable)
        if regulatory_required:
            if user_category == UserCategory.EXPERT:
                return 'shap_local'
            else:
                return 'lime'
        
        # Expert users -> SHAP
        if user_category == UserCategory.EXPERT:
            if decision_criticality in [DecisionCriticality.HIGH, DecisionCriticality.CRITICAL]:
                return 'shap_local'
            else:
                return 'shap_global'
        
        # High time pressure -> simpler explanations
        if time_pressure in [TimePressure.HIGH, TimePressure.MEDIUM]:
            return 'lime'
        
        # Default: LIME for intermediate/novice
        if user_category == UserCategory.INTERMEDIATE:
            return 'lime'
        
        # Novice: LIME with fewer features
        return 'lime'
    
    def _format_explanation(
        self,
        explanation: Dict[str, Any],
        preferences: Dict[str, Any],
        user_category: UserCategory
    ) -> Dict[str, Any]:
        """
        Format explanation based on user preferences.
        
        Args:
            explanation: Raw explanation data
            preferences: User preferences
            user_category: User category
            
        Returns:
            Formatted explanation
        """
        formatted = explanation.copy()
        
        # Limit features for novice/intermediate
        max_features = preferences.get('max_features')
        if max_features and 'contributions' in explanation:
            contributions = explanation['contributions']
            if len(contributions) > max_features:
                formatted['contributions'] = contributions[:max_features]
        
        # Add summary for novice/intermediate
        if preferences.get('include_summary', True) and user_category != UserCategory.EXPERT:
            formatted['summary'] = self._generate_summary(explanation, user_category)
        
        return formatted
    
    def _generate_summary(self, explanation: Dict[str, Any], user_category: UserCategory) -> str:
        """
        Generate text summary of explanation.
        
        Args:
            explanation: Explanation data
            user_category: User category
            
        Returns:
            Summary text
        """
        if 'contributions' not in explanation:
            return "Explanation generated successfully."
        
        contributions = explanation['contributions']
        if not contributions:
            return "No significant features identified."
        
        # Get top features
        top_features = contributions[:3] if len(contributions) >= 3 else contributions
        
        if user_category == UserCategory.NOVICE:
            summary = f"The model's decision is primarily influenced by: "
            feature_names = [c.get('feature', 'Unknown') for c in top_features]
            summary += ", ".join(feature_names[:3])
            summary += "."
        else:
            summary = f"Top contributing features: "
            for i, contrib in enumerate(top_features):
                feat = contrib.get('feature', 'Unknown')
                val = contrib.get('contribution', contrib.get('shap_value', 0))
                summary += f"{feat} ({val:.3f})"
                if i < len(top_features) - 1:
                    summary += ", "
            summary += "."
        
        return summary
    
    def generate_counterfactual(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation.
        
        Args:
            instance: Instance to explain
            target_class: Target class (optional)
            
        Returns:
            Counterfactual explanation
        """
        return self.lime_explainer.generate_counterfactual(instance, target_class)


