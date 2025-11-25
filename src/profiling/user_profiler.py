"""User profiling module for ACXF."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UserCategory(Enum):
    """User expertise categories."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class UserProfiler:
    """
    User profiling system that adapts based on questionnaire, interactions, and performance.
    """
    
    def __init__(self, initial_category: Optional[UserCategory] = None):
        """
        Initialize user profiler.
        
        Args:
            initial_category: Initial user category (optional)
        """
        self.category = initial_category or UserCategory.NOVICE
        self.questionnaire_score = 0.0
        self.interaction_log = []
        self.performance_scores = []
        self.adaptation_history = []
        
    def update_from_questionnaire(
        self,
        ml_experience: int,  # 1-5
        stats_knowledge: int,  # 1-5
        domain_expertise: int,  # 1-5
        explanation_preference: int  # 1-5 (1=simple, 5=detailed)
    ) -> None:
        """
        Update profile from questionnaire responses.
        
        Args:
            ml_experience: ML experience level (1-5)
            stats_knowledge: Statistics knowledge (1-5)
            domain_expertise: Domain expertise (1-5)
            explanation_preference: Explanation detail preference (1-5)
        """
        # Weighted scoring
        self.questionnaire_score = (
            ml_experience * 0.3 +
            stats_knowledge * 0.3 +
            domain_expertise * 0.2 +
            explanation_preference * 0.2
        )
        
        # Map to category
        if self.questionnaire_score >= 4.0:
            self.category = UserCategory.EXPERT
        elif self.questionnaire_score >= 2.5:
            self.category = UserCategory.INTERMEDIATE
        else:
            self.category = UserCategory.NOVICE
        
        logger.info(f"Questionnaire score: {self.questionnaire_score:.2f}, Category: {self.category.value}")
    
    def log_interaction(
        self,
        explanation_type: str,
        time_spent: float,
        satisfaction: Optional[int] = None,
        requested_detail: Optional[str] = None
    ) -> None:
        """
        Log user interaction for adaptation.
        
        Args:
            explanation_type: Type of explanation viewed
            time_spent: Time spent viewing (seconds)
            satisfaction: Satisfaction rating (1-5, optional)
            requested_detail: Requested detail level (optional)
        """
        interaction = {
            'explanation_type': explanation_type,
            'time_spent': time_spent,
            'satisfaction': satisfaction,
            'requested_detail': requested_detail
        }
        self.interaction_log.append(interaction)
        
        # Adapt based on interactions
        if len(self.interaction_log) > 0:
            self._adapt_from_interactions()
    
    def update_performance(
        self,
        task_accuracy: float,
        decision_confidence: float,
        time_to_decision: float
    ) -> None:
        """
        Update profile based on user performance.
        
        Args:
            task_accuracy: Accuracy on decision tasks (0-1)
            decision_confidence: User's confidence (0-1)
            time_to_decision: Time to make decision (seconds)
        """
        performance_score = (
            task_accuracy * 0.4 +
            decision_confidence * 0.3 +
            (1.0 - min(time_to_decision / 60.0, 1.0)) * 0.3  # Normalize time
        )
        self.performance_scores.append(performance_score)
        
        # Refine category based on performance
        if len(self.performance_scores) >= 3:
            avg_performance = np.mean(self.performance_scores[-3:])
            if avg_performance > 0.8 and self.category == UserCategory.INTERMEDIATE:
                self.category = UserCategory.EXPERT
            elif avg_performance < 0.5 and self.category == UserCategory.INTERMEDIATE:
                self.category = UserCategory.NOVICE
        
        logger.info(f"Performance score: {performance_score:.2f}, Category: {self.category.value}")
    
    def _adapt_from_interactions(self) -> None:
        """Adapt user category based on interaction patterns."""
        if len(self.interaction_log) < 3:
            return
        
        recent_interactions = self.interaction_log[-5:]
        
        # Check if user requests more detail
        detail_requests = sum(1 for i in recent_interactions 
                             if i.get('requested_detail') in ['more', 'detailed', 'expert'])
        
        # Check satisfaction trends
        satisfactions = [i.get('satisfaction') for i in recent_interactions if i.get('satisfaction')]
        if satisfactions:
            avg_satisfaction = np.mean(satisfactions)
            if avg_satisfaction < 3.0 and self.category == UserCategory.NOVICE:
                # User might need intermediate level
                pass  # Don't downgrade from novice
            elif avg_satisfaction < 3.0 and self.category == UserCategory.INTERMEDIATE:
                # Might need simpler explanations
                self.category = UserCategory.NOVICE
            elif detail_requests >= 2 and self.category == UserCategory.INTERMEDIATE:
                # User wants more detail
                self.category = UserCategory.EXPERT
    
    def get_category(self) -> UserCategory:
        """Get current user category."""
        return self.category
    
    def get_explanation_preferences(self) -> Dict[str, any]:
        """
        Get explanation preferences based on user category.
        
        Returns:
            Dictionary of preferences
        """
        if self.category == UserCategory.NOVICE:
            return {
                'detail_level': 'low',
                'use_visualizations': True,
                'use_simple_language': True,
                'max_features': 5,
                'include_summary': True
            }
        elif self.category == UserCategory.INTERMEDIATE:
            return {
                'detail_level': 'medium',
                'use_visualizations': True,
                'use_simple_language': False,
                'max_features': 10,
                'include_summary': True
            }
        else:  # EXPERT
            return {
                'detail_level': 'high',
                'use_visualizations': True,
                'use_simple_language': False,
                'max_features': None,  # Show all
                'include_summary': False
            }
    
    def get_profile_summary(self) -> Dict[str, any]:
        """Get summary of user profile."""
        return {
            'category': self.category.value,
            'questionnaire_score': self.questionnaire_score,
            'num_interactions': len(self.interaction_log),
            'avg_performance': np.mean(self.performance_scores) if self.performance_scores else None,
            'preferences': self.get_explanation_preferences()
        }


