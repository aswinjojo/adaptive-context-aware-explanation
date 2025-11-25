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
            
            # Extract contribution values (handle both 'contribution' and 'shap_value' keys)
            contribution_values = []
            for c in top_features:
                contrib_val = c.get('contribution', c.get('shap_value', 0))
                if contrib_val != 0:  # Only include non-zero contributions
                    contribution_values.append(contrib_val)
            
            if not contribution_values:
                continue
            
            # Sum contributions
            total_contribution = sum(contribution_values)
            
            # Get base value (SHAP has it, LIME doesn't)
            base_value = explanation.get('base_value', None)
            method = explanation.get('method', '').lower()
            
            # Reconstruct prediction based on explanation type
            if 'shap' in method and base_value is not None:
                # SHAP: contributions are additive to base value
                # For binary classification, use sigmoid transformation
                logit = base_value + total_contribution
                reconstructed_prob = 1.0 / (1.0 + np.exp(-logit))
            else:
                # LIME: contributions are already in probability/logit space
                # Use the prediction from explanation if available
                pred_from_exp = explanation.get('prediction', None)
                if pred_from_exp is not None:
                    if isinstance(pred_from_exp, list):
                        # For binary classification, use positive class probability
                        if len(pred_from_exp) > 1:
                            reconstructed_prob = float(pred_from_exp[1])
                        else:
                            reconstructed_prob = float(pred_from_exp[0])
                    else:
                        reconstructed_prob = float(pred_from_exp)
                else:
                    # Fallback: use contributions as logit and apply sigmoid
                    # Estimate base from mean of contributions (rough approximation)
                    estimated_base = -np.mean(contribution_values) if contribution_values else 0.0
                    logit = estimated_base + total_contribution
                    reconstructed_prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
            
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
        
        # Compute correlation and normalized MSE
        if len(reconstructed_scores) < 2:
            # Need at least 2 points for correlation
            correlation = 0.0
            mse = np.mean((reconstructed_scores - actual_probs) ** 2)
        else:
            correlation = np.corrcoef(reconstructed_scores, actual_probs)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            mse = np.mean((reconstructed_scores - actual_probs) ** 2)
        
        # Normalize MSE to [0, 1] range (assuming probabilities are in [0, 1])
        # Maximum possible MSE for probabilities is 1.0 (when predictions are completely wrong)
        normalized_mse = min(1.0, mse)
        
        # Fidelity score: combination of correlation and accuracy
        # Use both correlation (rank ordering) and 1 - normalized_mse (accuracy)
        # Weight them equally
        if correlation > 0:
            fidelity_score = 0.5 * correlation + 0.5 * (1.0 - normalized_mse)
        else:
            # If correlation is negative or zero, rely mainly on MSE
            fidelity_score = max(0.0, 1.0 - normalized_mse)
        
        # Ensure score is in [0, 1]
        fidelity_score = np.clip(fidelity_score, 0.0, 1.0)
        
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
    
    def _normalize_feature_name(self, feature_name: str) -> str:
        """
        Normalize feature name for comparison (remove parentheses, extra spaces, etc.).
        
        Args:
            feature_name: Feature name string
            
        Returns:
            Normalized feature name
        """
        if not isinstance(feature_name, str):
            return str(feature_name)
        
        # Remove content in parentheses (e.g., "Feature_0 (0.5)" -> "Feature_0")
        if '(' in feature_name:
            feature_name = feature_name.split('(')[0].strip()
        
        # Remove extra whitespace
        feature_name = feature_name.strip()
        
        return feature_name
    
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
        # Extract contributions with normalized feature names
        contrib1 = {}
        for c in exp1.get('contributions', []):
            feat = c.get('feature', '')
            if feat:
                feat_normalized = self._normalize_feature_name(feat)
                contrib_val = c.get('contribution', c.get('shap_value', 0))
                # Use maximum absolute value if feature appears multiple times
                if feat_normalized in contrib1:
                    if abs(contrib_val) > abs(contrib1[feat_normalized]):
                        contrib1[feat_normalized] = contrib_val
                else:
                    contrib1[feat_normalized] = contrib_val
        
        contrib2 = {}
        for c in exp2.get('contributions', []):
            feat = c.get('feature', '')
            if feat:
                feat_normalized = self._normalize_feature_name(feat)
                contrib_val = c.get('contribution', c.get('shap_value', 0))
                # Use maximum absolute value if feature appears multiple times
                if feat_normalized in contrib2:
                    if abs(contrib_val) > abs(contrib2[feat_normalized]):
                        contrib2[feat_normalized] = contrib_val
                else:
                    contrib2[feat_normalized] = contrib_val
        
        if len(contrib1) == 0 or len(contrib2) == 0:
            return 0.0
        
        # Get all unique features from both explanations
        all_features = set(contrib1.keys()).union(set(contrib2.keys()))
        
        if len(all_features) == 0:
            return 0.0
        
        # Create vectors for cosine similarity
        vec1 = np.array([contrib1.get(f, 0.0) for f in all_features])
        vec2 = np.array([contrib2.get(f, 0.0) for f in all_features])
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Also compute rank correlation for top features
        # Get top features by absolute value
        top_k = min(10, len(contrib1), len(contrib2))
        top_features1 = sorted(contrib1.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        top_features2 = sorted(contrib2.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        # Create feature rankings
        rank1 = {feat: idx for idx, (feat, _) in enumerate(top_features1)}
        rank2 = {feat: idx for idx, (feat, _) in enumerate(top_features2)}
        
        # Compute rank similarity for common top features
        common_top = set(rank1.keys()).intersection(set(rank2.keys()))
        if len(common_top) > 0:
            rank_diffs = [abs(rank1[f] - rank2[f]) for f in common_top]
            max_rank_diff = max(len(rank1), len(rank2)) - 1
            rank_sim = 1.0 - np.mean(rank_diffs) / max_rank_diff if max_rank_diff > 0 else 1.0
            rank_sim = max(0.0, rank_sim)
        else:
            rank_sim = 0.0
        
        # Combine cosine similarity and rank similarity
        # Weight cosine similarity more (0.7) as it captures magnitude
        combined_sim = 0.7 * cosine_sim + 0.3 * rank_sim
        
        # Ensure result is in [0, 1]
        return float(np.clip(combined_sim, 0.0, 1.0))

