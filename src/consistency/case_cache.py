"""Case caching for temporal consistency in ACXF."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CaseCache:
    """
    Cache for storing representative cases and their explanations for consistency.
    """
    
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.8):
        """
        Initialize case cache.
        
        Args:
            max_size: Maximum number of cases to cache
            similarity_threshold: Similarity threshold for retrieval
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cases = deque(maxlen=max_size)
        self.explanations = {}
        self.case_ids = []
        
        logger.info(f"Initialized case cache (max_size={max_size}, threshold={similarity_threshold})")
    
    def add_case(
        self,
        instance: np.ndarray,
        explanation: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> str:
        """
        Add a case to the cache.
        
        Args:
            instance: Instance data
            explanation: Explanation for the instance
            case_id: Optional case ID
            
        Returns:
            Case ID
        """
        if case_id is None:
            case_id = f"case_{len(self.cases)}"
        
        self.cases.append(instance)
        self.explanations[case_id] = explanation
        self.case_ids.append(case_id)
        
        logger.debug(f"Added case {case_id} to cache")
        return case_id
    
    def find_similar(
        self,
        instance: np.ndarray,
        metric: str = 'cosine'
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Find similar cases in cache.
        
        Args:
            instance: Instance to find similar cases for
            metric: Similarity metric ('cosine', 'euclidean')
            
        Returns:
            List of (case_id, similarity_score, cached_instance) tuples
        """
        if len(self.cases) == 0:
            return []
        
        similarities = []
        
        for i, cached_instance in enumerate(self.cases):
            case_id = self.case_ids[i]
            
            if metric == 'cosine':
                similarity = self._cosine_similarity(instance, cached_instance)
            elif metric == 'euclidean':
                similarity = 1.0 / (1.0 + np.linalg.norm(instance - cached_instance))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if similarity >= self.similarity_threshold:
                similarities.append((case_id, similarity, cached_instance))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def get_explanation(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get explanation for a cached case."""
        return self.explanations.get(case_id)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
        
        return np.dot(a, b) / (a_norm * b_norm)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cases.clear()
        self.explanations.clear()
        self.case_ids.clear()
        logger.info("Case cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cases)


