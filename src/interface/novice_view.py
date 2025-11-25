"""Novice-level explanation interface for ACXF."""

import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class NoviceView:
    """
    Simple, text-based explanation view for novice users.
    """
    
    def __init__(self):
        """Initialize novice view."""
        pass
    
    def render(self, explanation: Dict[str, Any]) -> str:
        """
        Render explanation as simple text summary.
        
        Args:
            explanation: Explanation data
            
        Returns:
            Formatted text explanation
        """
        # Get summary if available
        summary = explanation.get('summary', '')
        
        if summary:
            return summary
        
        # Generate summary from contributions
        contributions = explanation.get('contributions', [])
        if not contributions:
            return "The model made a prediction, but no detailed explanation is available."
        
        # Get top 3 features
        top_features = contributions[:3]
        
        text = "The model's decision is primarily based on:\n\n"
        for i, contrib in enumerate(top_features, 1):
            feature = contrib.get('feature', 'Unknown feature')
            value = contrib.get('contribution', contrib.get('shap_value', 0))
            
            # Simplify feature name
            if '(' in feature:
                feature = feature.split('(')[0].strip()
            
            direction = "increases" if value > 0 else "decreases"
            text += f"{i}. {feature} - This {direction} the likelihood of the prediction.\n"
        
        # Add prediction
        prediction = explanation.get('prediction', [])
        if prediction:
            if isinstance(prediction, list) and len(prediction) > 0:
                pred_val = prediction[0] if len(prediction) == 1 else max(prediction)
                text += f"\nPrediction confidence: {pred_val:.1%}"
        
        return text
    
    def visualize(self, explanation: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Create simple visualization for novice users.
        
        Args:
            explanation: Explanation data
            save_path: Optional path to save figure
        """
        contributions = explanation.get('contributions', [])
        if not contributions:
            logger.warning("No contributions to visualize")
            return
        
        # Get top 5 features
        top_features = contributions[:5]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        features = [c.get('feature', 'Unknown') for c in top_features]
        # Simplify feature names
        features = [f.split('(')[0].strip() if '(' in f else f for f in features]
        values = [abs(c.get('contribution', c.get('shap_value', 0))) for c in top_features]
        
        colors = ['green' if c.get('contribution', c.get('shap_value', 0)) > 0 else 'red' 
                 for c in top_features]
        
        bars = ax.barh(features, values, color=colors)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top Factors Influencing the Decision', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


