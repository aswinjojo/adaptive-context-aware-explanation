"""Intermediate-level explanation interface for ACXF."""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IntermediateView:
    """
    Visual feature ranking view for intermediate users.
    """
    
    def __init__(self):
        """Initialize intermediate view."""
        sns.set_style("whitegrid")
    
    def render(self, explanation: Dict[str, Any]) -> str:
        """
        Render explanation as detailed text with feature rankings.
        
        Args:
            explanation: Explanation data
            
        Returns:
            Formatted text explanation
        """
        contributions = explanation.get('contributions', [])
        if not contributions:
            return "No explanation data available."
        
        text = "Feature Contribution Analysis:\n\n"
        text += f"Method: {explanation.get('method', 'Unknown')}\n\n"
        
        # Show top 10 features
        top_features = contributions[:10]
        
        for i, contrib in enumerate(top_features, 1):
            feature = contrib.get('feature', 'Unknown')
            value = contrib.get('contribution', contrib.get('shap_value', 0))
            
            direction = "↑" if value > 0 else "↓"
            text += f"{i:2d}. {feature:30s} {direction} {abs(value):.4f}\n"
        
        # Add prediction info
        prediction = explanation.get('prediction', [])
        if prediction:
            if isinstance(prediction, list):
                if len(prediction) == 2:
                    text += f"\nPredicted probabilities:\n"
                    text += f"  Class 0: {prediction[0]:.3f}\n"
                    text += f"  Class 1: {prediction[1]:.3f}\n"
                else:
                    text += f"\nPrediction: {prediction[0]:.3f}\n"
        
        return text
    
    def visualize(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[str] = None,
        max_features: int = 10
    ) -> None:
        """
        Create feature ranking visualization.
        
        Args:
            explanation: Explanation data
            save_path: Optional path to save figure
            max_features: Maximum number of features to show
        """
        contributions = explanation.get('contributions', [])
        if not contributions:
            logger.warning("No contributions to visualize")
            return
        
        # Get top features
        top_features = contributions[:max_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = [c.get('feature', 'Unknown') for c in top_features]
        values = [c.get('contribution', c.get('shap_value', 0)) for c in top_features]
        
        # Color by sign
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        bars = ax.barh(range(len(features)), values, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Contribution Value', fontsize=12)
        ax.set_title(f'Feature Contributions ({explanation.get("method", "Unknown")})', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


