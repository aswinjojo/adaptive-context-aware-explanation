"""Evaluation metrics for ACXF."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for explanation quality.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        pass
    
    def compute_fidelity(
        self,
        model: Any,
        X: np.ndarray,
        explanations: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Compute explanation fidelity (how well explanation matches model behavior).
        
        Args:
            model: Trained model
            X: Instances
            explanations: List of explanations
            top_k: Number of top features to consider
            
        Returns:
            Dictionary with fidelity metrics
        """
        if len(explanations) == 0:
            return {'fidelity_score': 0.0, 'num_explanations': 0}
        
        # Get model predictions
        predictions = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
        
        # Reconstruct predictions from explanations
        reconstructed_scores = []
        valid_indices = []  # Track which explanations had valid contributions
        
        for i, explanation in enumerate(explanations):
            contributions = explanation.get('contributions', [])
            if not contributions:
                continue
            
            # Get top-k features
            top_features = contributions[:top_k]
            
            # Sum contributions as proxy for prediction
            total_contribution = sum(c.get('contribution', c.get('shap_value', 0)) 
                                    for c in top_features)
            base_value = explanation.get('base_value', 0.0)
            
            # Reconstructed probability (simplified)
            reconstructed_prob = 1.0 / (1.0 + np.exp(-(base_value + total_contribution)))
            reconstructed_scores.append(reconstructed_prob)
            valid_indices.append(i)
        
        if len(reconstructed_scores) == 0:
            return {'fidelity_score': 0.0, 'num_explanations': 0}
        
        # Compare with actual predictions
        # Ensure predictions is properly formatted
        predictions = np.asarray(predictions)
        if predictions.ndim > 1:
            # For binary classification, use positive class probability
            if predictions.shape[1] > 1:
                actual_probs = predictions[:, 1]
            else:
                actual_probs = predictions[:, 0]
        else:
            actual_probs = predictions
        
        # Ensure actual_probs is 1D
        actual_probs = np.asarray(actual_probs).flatten()
        
        # Only use predictions for valid explanations
        if len(valid_indices) > 0 and len(valid_indices) <= len(actual_probs):
            actual_probs = actual_probs[valid_indices]
        else:
            # Fallback: take first len(reconstructed_scores) predictions
            actual_probs = actual_probs[:len(reconstructed_scores)]
        
        # Ensure lengths match
        min_len = min(len(reconstructed_scores), len(actual_probs))
        if min_len == 0:
            return {'fidelity_score': 0.0, 'num_explanations': 0}
        
        reconstructed_scores = np.array(reconstructed_scores[:min_len])
        actual_probs = np.array(actual_probs[:min_len])
        
        # Compute correlation and MSE
        if len(reconstructed_scores) < 2:
            # Need at least 2 points for correlation
            correlation = 0.0
            mse = np.mean((reconstructed_scores - actual_probs) ** 2)
        else:
            correlation = np.corrcoef(reconstructed_scores, actual_probs)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            mse = np.mean((reconstructed_scores - actual_probs) ** 2)
        
        # Fidelity score (higher is better)
        fidelity_score = max(0.0, correlation * (1.0 - mse))
        
        return {
            'fidelity_score': float(fidelity_score),
            'correlation': float(correlation),
            'mse': float(mse),
            'num_explanations': len(reconstructed_scores)
        }
    
    def compute_consistency(
        self,
        explanations: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> Dict[str, float]:
        """
        Compute explanation consistency across similar instances.
        
        Args:
            explanations: List of explanations
            similarity_threshold: Threshold for considering explanations similar
            
        Returns:
            Dictionary with consistency metrics
        """
        if len(explanations) < 2:
            return {'consistency_score': 0.0, 'num_pairs': 0}
        
        similarities = []
        
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                exp1 = explanations[i]
                exp2 = explanations[j]
                
                sim = self._explanation_similarity(exp1, exp2)
                similarities.append(sim)
        
        if len(similarities) == 0:
            return {'consistency_score': 0.0, 'num_pairs': 0}
        
        avg_similarity = np.mean(similarities)
        consistency_score = avg_similarity
        
        return {
            'consistency_score': float(consistency_score),
            'avg_similarity': float(avg_similarity),
            'std_similarity': float(np.std(similarities)),
            'num_pairs': len(similarities)
        }
    
    def compute_comprehensibility(
        self,
        explanations: List[Dict[str, Any]],
        user_category: str = 'intermediate'
    ) -> Dict[str, float]:
        """
        Compute explanation comprehensibility.
        
        Args:
            explanations: List of explanations
            user_category: User category ('novice', 'intermediate', 'expert')
            
        Returns:
            Dictionary with comprehensibility metrics
        """
        if len(explanations) == 0:
            return {'comprehensibility_score': 0.0, 'num_explanations': 0}
        
        scores = []
        
        for explanation in explanations:
            # Factors affecting comprehensibility
            num_features = len(explanation.get('contributions', []))
            has_summary = 'summary' in explanation
            method = explanation.get('method', '')
            
            # Score based on user category
            if user_category == 'novice':
                # Prefer fewer features, summaries, simpler methods
                score = 1.0 if num_features <= 5 else max(0.0, 1.0 - (num_features - 5) * 0.1)
                score += 0.2 if has_summary else 0.0
                score += 0.1 if 'LIME' in method else 0.0
            elif user_category == 'intermediate':
                # Moderate number of features
                score = 1.0 if 5 <= num_features <= 15 else max(0.0, 1.0 - abs(num_features - 10) * 0.05)
                score += 0.1 if has_summary else 0.0
            else:  # expert
                # More features, detailed methods
                score = 1.0 if num_features >= 10 else num_features / 10.0
                score += 0.2 if 'SHAP' in method else 0.0
            
            scores.append(min(1.0, score))
        
        return {
            'comprehensibility_score': float(np.mean(scores)),
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'num_explanations': len(explanations)
        }
    
    def compute_cognitive_load(
        self,
        explanations: List[Dict[str, Any]],
        time_spent: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute cognitive load (NASA-TLX proxy).
        
        Args:
            explanations: List of explanations
            time_spent: Optional list of time spent on each explanation (seconds)
            
        Returns:
            Dictionary with cognitive load metrics
        """
        if len(explanations) == 0:
            return {'cognitive_load': 0.0, 'num_explanations': 0}
        
        load_scores = []
        
        for i, explanation in enumerate(explanations):
            # Factors: number of features, complexity of method
            num_features = len(explanation.get('contributions', []))
            method = explanation.get('method', '')
            
            # Base load from features (more features = higher load)
            feature_load = min(1.0, num_features / 20.0)
            
            # Method complexity
            method_load = 0.3 if 'SHAP' in method else 0.2 if 'LIME' in method else 0.1
            
            # Time factor (if available)
            time_load = 0.0
            if time_spent and i < len(time_spent):
                # Normalize time (assume 60s is high load)
                time_load = min(1.0, time_spent[i] / 60.0)
            
            # Combined load (weighted)
            total_load = (feature_load * 0.4 + method_load * 0.3 + time_load * 0.3)
            load_scores.append(total_load)
        
        return {
            'cognitive_load': float(np.mean(load_scores)),
            'avg_load': float(np.mean(load_scores)),
            'std_load': float(np.std(load_scores)),
            'num_explanations': len(explanations)
        }
    
    def compute_decision_quality(
        self,
        user_decisions: List[int],
        correct_decisions: List[int],
        confidence_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute decision quality metrics.
        
        Args:
            user_decisions: User's decisions
            correct_decisions: Correct decisions
            confidence_scores: Optional confidence scores
            
        Returns:
            Dictionary with decision quality metrics
        """
        if len(user_decisions) != len(correct_decisions):
            raise ValueError("User decisions and correct decisions must have same length")
        
        accuracy = np.mean(np.array(user_decisions) == np.array(correct_decisions))
        
        # Calibration (if confidence available)
        calibration = None
        if confidence_scores and len(confidence_scores) == len(user_decisions):
            # Brier score
            brier_scores = []
            for i, (user_dec, correct_dec, conf) in enumerate(zip(user_decisions, correct_decisions, confidence_scores)):
                prob = conf if user_dec == 1 else (1 - conf)
                actual = 1.0 if user_dec == correct_dec else 0.0
                brier = (prob - actual) ** 2
                brier_scores.append(brier)
            calibration = 1.0 - np.mean(brier_scores)  # Higher is better
        
        return {
            'decision_accuracy': float(accuracy),
            'calibration': float(calibration) if calibration is not None else None,
            'num_decisions': len(user_decisions)
        }
    
    def compute_trust_calibration(
        self,
        trust_scores: List[float],
        model_performance: float,
        explanation_quality: float
    ) -> Dict[str, float]:
        """
        Compute trust calibration (alignment between trust and actual quality).
        
        Args:
            trust_scores: User trust scores (0-1)
            model_performance: Actual model performance (0-1)
            explanation_quality: Explanation quality score (0-1)
            
        Returns:
            Dictionary with trust calibration metrics
        """
        avg_trust = np.mean(trust_scores)
        
        # Expected trust based on quality
        expected_trust = (model_performance + explanation_quality) / 2.0
        
        # Calibration error
        calibration_error = abs(avg_trust - expected_trust)
        
        # Calibration score (higher is better, lower error)
        calibration_score = 1.0 - calibration_error
        
        return {
            'trust_calibration': float(calibration_score),
            'avg_trust': float(avg_trust),
            'expected_trust': float(expected_trust),
            'calibration_error': float(calibration_error),
            'num_ratings': len(trust_scores)
        }
    
    def _explanation_similarity(
        self,
        exp1: Dict[str, Any],
        exp2: Dict[str, Any]
    ) -> float:
        """
        Compute similarity between two explanations.
        
        Args:
            exp1: First explanation
            exp2: Second explanation
            
        Returns:
            Similarity score (0-1)
        """
        contrib1 = {c.get('feature'): c.get('contribution', c.get('shap_value', 0))
                   for c in exp1.get('contributions', [])}
        contrib2 = {c.get('feature'): c.get('contribution', c.get('shap_value', 0))
                   for c in exp2.get('contributions', [])}
        
        if len(contrib1) == 0 or len(contrib2) == 0:
            return 0.0
        
        # Common features
        common_features = set(contrib1.keys()).intersection(set(contrib2.keys()))
        
        if len(common_features) == 0:
            return 0.0
        
        # Compare contributions
        similarities = []
        for feat in common_features:
            val1 = contrib1[feat]
            val2 = contrib2[feat]
            if abs(val1) + abs(val2) > 0:
                sim = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-6)
                similarities.append(max(0.0, sim))
        
        return np.mean(similarities) if similarities else 0.0

