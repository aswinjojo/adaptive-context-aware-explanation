"""Temporal consistency tracking for ACXF."""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple, Any
import logging

from .case_cache import CaseCache

logger = logging.getLogger(__name__)


class ConsistencyTracker:
    """
    Tracks temporal consistency of explanations using clustering and case caching.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        cache_size: int = 100,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize consistency tracker.
        
        Args:
            n_clusters: Number of clusters for feature-space clustering
            cache_size: Size of case cache
            similarity_threshold: Similarity threshold for case matching
        """
        self.n_clusters = n_clusters
        self.case_cache = CaseCache(cache_size, similarity_threshold)
        self.clusterer = None
        self.cluster_centers = None
        self.explanation_history = []
        self.feature_clusters = None
        
        logger.info(f"Initialized consistency tracker (n_clusters={n_clusters})")
    
    def fit_clusters(self, X: np.ndarray) -> None:
        """
        Fit clustering model on feature space.
        
        Args:
            X: Feature data for clustering
        """
        if len(X) < self.n_clusters:
            self.n_clusters = max(2, len(X) // 2)
        
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.clusterer.fit(X)
        self.cluster_centers = self.clusterer.cluster_centers_
        
        logger.info(f"Fitted {self.n_clusters} clusters on feature space")
    
    def get_cluster(self, instance: np.ndarray) -> int:
        """
        Get cluster assignment for an instance.
        
        Args:
            instance: Instance to cluster
            
        Returns:
            Cluster index
        """
        if self.clusterer is None:
            return 0
        
        return int(self.clusterer.predict(instance.reshape(1, -1))[0])
    
    def add_explanation(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> str:
        """
        Add explanation to history and cache.
        
        Args:
            instance: Instance data
            explanation: Explanation data
            case_id: Optional case ID
            
        Returns:
            Case ID
        """
        case_id = self.case_cache.add_case(instance, explanation, case_id)
        self.explanation_history.append({
            'case_id': case_id,
            'instance': instance,
            'explanation': explanation,
            'cluster': self.get_cluster(instance) if self.clusterer is not None else None
        })
        
        return case_id
    
    def compute_consistency_score(
        self,
        instance: np.ndarray,
        new_explanation: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Compute consistency score for a new explanation against cached explanations.
        
        Args:
            instance: New instance
            new_explanation: New explanation
            top_k: Number of similar cases to consider
            
        Returns:
            Dictionary with consistency metrics
        """
        # Find similar cases
        similar_cases = self.case_cache.find_similar(instance, metric='cosine')
        
        if len(similar_cases) == 0:
            return {
                'consistency_score': 0.0,
                'num_similar_cases': 0,
                'explanation_similarity': 0.0
            }
        
        # Get top-k similar cases
        top_similar = similar_cases[:top_k]
        
        # Compare explanations
        explanation_similarities = []
        
        for case_id, instance_sim, _ in top_similar:
            cached_explanation = self.case_cache.get_explanation(case_id)
            if cached_explanation is None:
                continue
            
            # Compare feature contributions
            exp_sim = self._compare_explanations(new_explanation, cached_explanation)
            explanation_similarities.append(exp_sim * instance_sim)  # Weight by instance similarity
        
        if len(explanation_similarities) == 0:
            return {
                'consistency_score': 0.0,
                'num_similar_cases': len(top_similar),
                'explanation_similarity': 0.0
            }
        
        avg_explanation_sim = np.mean(explanation_similarities)
        consistency_score = avg_explanation_sim
        
        return {
            'consistency_score': float(consistency_score),
            'num_similar_cases': len(top_similar),
            'explanation_similarity': float(avg_explanation_sim)
        }
    
    def _compare_explanations(
        self,
        exp1: Dict[str, Any],
        exp2: Dict[str, Any]
    ) -> float:
        """
        Compare two explanations and return similarity score.
        
        Args:
            exp1: First explanation
            exp2: Second explanation
            
        Returns:
            Similarity score (0-1)
        """
        # Extract feature contributions
        contrib1 = self._extract_contributions(exp1)
        contrib2 = self._extract_contributions(exp2)
        
        if len(contrib1) == 0 or len(contrib2) == 0:
            return 0.0
        
        # Get common features
        features1 = set(contrib1.keys())
        features2 = set(contrib2.keys())
        common_features = features1.intersection(features2)
        
        if len(common_features) == 0:
            return 0.0
        
        # Compare contributions for common features
        similarities = []
        for feat in common_features:
            val1 = contrib1[feat]
            val2 = contrib2[feat]
            # Normalize and compare
            if abs(val1) + abs(val2) > 0:
                sim = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-6)
                similarities.append(max(0.0, sim))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _extract_contributions(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract feature contributions from explanation.
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Dictionary mapping feature names to contribution values
        """
        contributions = {}
        
        if 'contributions' in explanation:
            for contrib in explanation['contributions']:
                feat = contrib.get('feature', '')
                val = contrib.get('contribution', contrib.get('shap_value', 0.0))
                contributions[feat] = float(val)
        elif 'shap_values' in explanation:
            # SHAP format
            shap_vals = explanation['shap_values']
            if isinstance(shap_vals, np.ndarray):
                shap_vals = shap_vals.flatten()
            feature_names = explanation.get('feature_names', [f"Feature_{i}" for i in range(len(shap_vals))])
            for i, val in enumerate(shap_vals):
                contributions[feature_names[i]] = float(val)
        
        return contributions
    
    def smooth_explanation(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any],
        smoothing_factor: float = 0.3
    ) -> Dict[str, Any]:
        """
        Smooth explanation using similar cached explanations.
        
        Args:
            instance: Instance to explain
            explanation: Original explanation
            smoothing_factor: Weight for smoothing (0-1)
            
        Returns:
            Smoothed explanation
        """
        similar_cases = self.case_cache.find_similar(instance, metric='cosine')
        
        if len(similar_cases) == 0:
            return explanation
        
        # Get top similar case
        top_case_id, top_sim, _ = similar_cases[0]
        cached_explanation = self.case_cache.get_explanation(top_case_id)
        
        if cached_explanation is None:
            return explanation
        
        # Smooth contributions
        smoothed = explanation.copy()
        
        if 'contributions' in explanation and 'contributions' in cached_explanation:
            contrib1 = {c.get('feature'): c.get('contribution', c.get('shap_value', 0)) 
                       for c in explanation['contributions']}
            contrib2 = {c.get('feature'): c.get('contribution', c.get('shap_value', 0)) 
                       for c in cached_explanation['contributions']}
            
            # Smooth
            smoothed_contribs = []
            all_features = set(contrib1.keys()).union(set(contrib2.keys()))
            
            for feat in all_features:
                val1 = contrib1.get(feat, 0.0)
                val2 = contrib2.get(feat, 0.0)
                smoothed_val = (1 - smoothing_factor) * val1 + smoothing_factor * val2 * top_sim
                smoothed_contribs.append({
                    'feature': feat,
                    'contribution': smoothed_val
                })
            
            smoothed_contribs.sort(key=lambda x: abs(x['contribution']), reverse=True)
            smoothed['contributions'] = smoothed_contribs
        
        return smoothed


