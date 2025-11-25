"""Expert-level explanation interface for ACXF."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExpertView:
    """
    Comprehensive explanation view for expert users with SHAP plots and counterfactuals.
    """
    
    def __init__(self):
        """Initialize expert view."""
        sns.set_style("whitegrid")
    
    def render(self, explanation: Dict[str, Any]) -> str:
        """
        Render detailed explanation text.
        
        Args:
            explanation: Explanation data
            
        Returns:
            Formatted detailed text
        """
        text = f"Detailed Explanation Report\n"
        text += f"{'='*50}\n\n"
        text += f"Method: {explanation.get('method', 'Unknown')}\n"
        text += f"Type: {explanation.get('type', 'Unknown')}\n\n"
        
        # Feature contributions
        contributions = explanation.get('contributions', [])
        if contributions:
            text += "Feature Contributions:\n"
            text += "-" * 50 + "\n"
            for i, contrib in enumerate(contributions, 1):
                feature = contrib.get('feature', 'Unknown')
                value = contrib.get('contribution', contrib.get('shap_value', 0))
                text += f"{i:3d}. {feature:35s} {value:10.6f}\n"
        
        # Prediction details
        prediction = explanation.get('prediction', [])
        base_value = explanation.get('base_value')
        
        if base_value is not None:
            text += f"\nBase Value (Expected): {base_value:.6f}\n"
        
        if prediction:
            if isinstance(prediction, list):
                if len(prediction) == 2:
                    text += f"\nPredicted Probabilities:\n"
                    text += f"  Class 0: {prediction[0]:.6f}\n"
                    text += f"  Class 1: {prediction[1]:.6f}\n"
                    text += f"  Predicted Class: {np.argmax(prediction)}\n"
                else:
                    text += f"\nPrediction: {prediction[0]:.6f}\n"
        
        # Instance values
        instance = explanation.get('instance', [])
        if instance:
            text += f"\nInstance Values:\n"
            feature_names = explanation.get('feature_names', [f"Feature_{i}" for i in range(len(instance))])
            for feat, val in zip(feature_names[:10], instance[:10]):  # Show first 10
                text += f"  {feat}: {val:.4f}\n"
        
        return text
    
    def visualize_shap_bar(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP bar plot.
        
        Args:
            explanation: Explanation data
            save_path: Optional path to save figure
        """
        contributions = explanation.get('contributions', [])
        if not contributions:
            logger.warning("No contributions to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = [c.get('feature', 'Unknown') for c in contributions]
        values = [c.get('contribution', c.get('shap_value', 0)) for c in contributions]
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(values))[::-1]
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        
        bars = ax.barh(range(len(features)), values, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('SHAP Value', fontsize=12)
        ax.set_title('SHAP Feature Contributions', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved SHAP bar plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_force_plot(
        self,
        explanation: Dict[str, Any],
        shap_explainer: Any,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP force plot (if SHAP explainer is available).
        
        Args:
            explanation: Explanation data
            shap_explainer: SHAP explainer instance
            save_path: Optional path to save figure
        """
        try:
            shap_values = explanation.get('shap_values')
            base_value = explanation.get('base_value')
            instance = explanation.get('instance')
            feature_names = explanation.get('feature_names')
            
            if shap_values is None or base_value is None or instance is None:
                logger.warning("Missing data for force plot")
                return
            
            # Convert to numpy arrays
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            if isinstance(instance, list):
                instance = np.array(instance)
            
            # Create force plot
            shap.force_plot(
                base_value,
                shap_values,
                instance,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved force plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Error creating force plot: {e}")
    
    def visualize_counterfactual(
        self,
        counterfactual: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize counterfactual explanation.
        
        Args:
            counterfactual: Counterfactual data
            save_path: Optional path to save figure
        """
        modifications = counterfactual.get('modifications', [])
        if not modifications:
            logger.warning("No modifications to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Modifications
        features = [m.get('feature', 'Unknown') for m in modifications]
        original_vals = [m.get('original_value', 0) for m in modifications]
        new_vals = [m.get('new_value', 0) for m in modifications]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax1.bar(x - width/2, original_vals, width, label='Original', color='#3498db')
        ax1.bar(x + width/2, new_vals, width, label='Counterfactual', color='#e74c3c')
        ax1.set_xlabel('Features', fontsize=11)
        ax1.set_ylabel('Values', fontsize=11)
        ax1.set_title('Feature Modifications', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Prediction comparison
        orig_pred = counterfactual.get('original_prediction', [])
        cf_pred = counterfactual.get('counterfactual_prediction', [])
        
        if orig_pred and cf_pred:
            classes = ['Class 0', 'Class 1'] if len(orig_pred) == 2 else ['Prediction']
            x_pos = np.arange(len(classes))
            
            if len(orig_pred) == 2:
                orig_vals = orig_pred
                cf_vals = cf_pred
            else:
                orig_vals = orig_pred
                cf_vals = cf_pred
            
            ax2.bar(x_pos - width/2, orig_vals, width, label='Original', color='#3498db')
            ax2.bar(x_pos + width/2, cf_vals, width, label='Counterfactual', color='#e74c3c')
            ax2.set_xlabel('Classes', fontsize=11)
            ax2.set_ylabel('Probability', fontsize=11)
            ax2.set_title('Prediction Comparison', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(classes)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved counterfactual visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


