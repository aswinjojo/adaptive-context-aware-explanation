"""User study simulation for ACXF evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path

from ..profiling.user_profiler import UserProfiler, UserCategory
from ..explanation.context_engine import ContextAwareEngine, DecisionCriticality, TimePressure
from ..consistency.consistency_tracker import ConsistencyTracker
from ..interface.novice_view import NoviceView
from ..interface.intermediate_view import IntermediateView
from ..interface.expert_view import ExpertView
from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class UserPersona:
    """Simulated user persona."""
    
    def __init__(
        self,
        name: str,
        category: UserCategory,
        ml_experience: int,
        stats_knowledge: int,
        domain_expertise: int,
        explanation_preference: int
    ):
        """
        Initialize user persona.
        
        Args:
            name: Persona name
            category: User category
            ml_experience: ML experience (1-5)
            stats_knowledge: Stats knowledge (1-5)
            domain_expertise: Domain expertise (1-5)
            explanation_preference: Explanation preference (1-5)
        """
        self.name = name
        self.category = category
        self.ml_experience = ml_experience
        self.stats_knowledge = stats_knowledge
        self.domain_expertise = domain_expertise
        self.explanation_preference = explanation_preference
        
        self.profiler = UserProfiler(category)
        self.profiler.update_from_questionnaire(
            ml_experience, stats_knowledge, domain_expertise, explanation_preference
        )
    
    def simulate_decision(
        self,
        explanation: Dict[str, Any],
        correct_decision: int,
        base_accuracy: float = 0.7
    ) -> Tuple[int, float, float]:
        """
        Simulate user decision based on explanation.
        
        Args:
            explanation: Explanation data
            correct_decision: Correct decision
            base_accuracy: Base accuracy without explanation
            
        Returns:
            Tuple of (decision, confidence, time_spent)
        """
        # Base accuracy varies by category
        if self.category == UserCategory.EXPERT:
            accuracy = base_accuracy + 0.2
        elif self.category == UserCategory.INTERMEDIATE:
            accuracy = base_accuracy + 0.1
        else:
            accuracy = base_accuracy
        
        # Decision quality depends on explanation quality
        num_features = len(explanation.get('contributions', []))
        has_summary = 'summary' in explanation
        
        # Better explanations improve decision quality
        if has_summary and num_features <= 10:
            accuracy += 0.1
        
        # Make decision
        decision = correct_decision if np.random.random() < accuracy else (1 - correct_decision)
        
        # Confidence (higher for experts, better explanations)
        confidence = 0.5 + (accuracy - 0.5) * 0.5
        confidence += np.random.normal(0, 0.1)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Time spent (varies by category and explanation complexity)
        base_time = 10.0 if self.category == UserCategory.NOVICE else 5.0
        complexity_factor = 1.0 + (num_features / 20.0)
        time_spent = base_time * complexity_factor + np.random.normal(0, 2.0)
        time_spent = max(3.0, time_spent)
        
        return int(decision), float(confidence), float(time_spent)
    
    def rate_trust(self, explanation: Dict[str, Any], model_performance: float) -> float:
        """
        Simulate trust rating.
        
        Args:
            explanation: Explanation data
            model_performance: Model performance
            
        Returns:
            Trust score (0-1)
        """
        # Base trust from model performance
        trust = model_performance
        
        # Adjust based on explanation quality
        num_features = len(explanation.get('contributions', []))
        has_summary = 'summary' in explanation
        method = explanation.get('method', '')
        
        # Better explanations increase trust
        if has_summary:
            trust += 0.1
        if 'SHAP' in method and self.category == UserCategory.EXPERT:
            trust += 0.1
        elif 'LIME' in method and self.category != UserCategory.EXPERT:
            trust += 0.05
        
        # Add noise
        trust += np.random.normal(0, 0.05)
        trust = np.clip(trust, 0.0, 1.0)
        
        return float(trust)


class UserStudySimulator:
    """
    Simulator for user study evaluation.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        consistency_tracker: ConsistencyTracker
    ):
        """
        Initialize user study simulator.
        
        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data
            y_test: Test labels
            feature_names: Feature names
            consistency_tracker: Consistency tracker
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.consistency_tracker = consistency_tracker
        
        self.metrics = EvaluationMetrics()
        
        # Create personas
        self.personas = [
            UserPersona("Novice_User", UserCategory.NOVICE, 1, 2, 2, 2),
            UserPersona("Intermediate_User", UserCategory.INTERMEDIATE, 3, 3, 3, 3),
            UserPersona("Expert_User", UserCategory.EXPERT, 5, 5, 4, 5)
        ]
        
        logger.info("Initialized user study simulator")
    
    def run_study(
        self,
        n_tasks: int = 50,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run simulated user study.
        
        Args:
            n_tasks: Number of tasks per persona
            random_seed: Random seed
            
        Returns:
            Dictionary with study results
        """
        np.random.seed(random_seed)
        
        results = {
            'personas': {},
            'overall_metrics': {},
            'task_details': []
        }
        
        # Select test instances
        n_available = len(self.X_test)
        n_tasks = min(n_tasks, n_available)
        task_indices = np.random.choice(n_available, n_tasks, replace=False)
        
        for persona in self.personas:
            logger.info(f"Running study for {persona.name}")
            
            persona_results = {
                'decisions': [],
                'correct_decisions': [],
                'confidence_scores': [],
                'time_spent': [],
                'trust_scores': [],
                'explanations': [],
                'metrics': {}
            }
            
            # Create context engine for this persona
            context_engine = ContextAwareEngine(
                self.model, self.X_train, self.feature_names, persona.profiler
            )
            
            for task_idx in task_indices:
                instance = self.X_test[task_idx]
                correct_decision = int(self.y_test[task_idx])
                
                # Generate explanation
                explanation = context_engine.generate_explanation(
                    instance,
                    decision_criticality=DecisionCriticality.MEDIUM,
                    time_pressure=TimePressure.NONE
                )
                
                # Add to consistency tracker
                consistency_score = self.consistency_tracker.compute_consistency_score(
                    instance, explanation
                )
                self.consistency_tracker.add_explanation(instance, explanation)
                
                # Simulate user interaction
                decision, confidence, time_spent = persona.simulate_decision(
                    explanation, correct_decision
                )
                trust = persona.rate_trust(explanation, 0.85)  # Assume 85% model performance
                
                # Log interaction
                persona.profiler.log_interaction(
                    explanation.get('method', 'unknown'),
                    time_spent,
                    satisfaction=4 if decision == correct_decision else 3
                )
                
                persona_results['decisions'].append(decision)
                persona_results['correct_decisions'].append(correct_decision)
                persona_results['confidence_scores'].append(confidence)
                persona_results['time_spent'].append(time_spent)
                persona_results['trust_scores'].append(trust)
                persona_results['explanations'].append(explanation)
                
                results['task_details'].append({
                    'persona': persona.name,
                    'task_idx': int(task_idx),
                    'decision': decision,
                    'correct': correct_decision,
                    'confidence': confidence,
                    'time_spent': time_spent,
                    'trust': trust,
                    'consistency_score': consistency_score.get('consistency_score', 0.0)
                })
            
            # Compute metrics for this persona
            persona_results['metrics'] = {
                'decision_quality': self.metrics.compute_decision_quality(
                    persona_results['decisions'],
                    persona_results['correct_decisions'],
                    persona_results['confidence_scores']
                ),
                'trust_calibration': self.metrics.compute_trust_calibration(
                    persona_results['trust_scores'],
                    0.85,  # Model performance
                    0.8    # Explanation quality (assumed)
                ),
                'comprehensibility': self.metrics.compute_comprehensibility(
                    persona_results['explanations'],
                    persona.category.value
                ),
                'cognitive_load': self.metrics.compute_cognitive_load(
                    persona_results['explanations'],
                    persona_results['time_spent']
                ),
                'fidelity': self.metrics.compute_fidelity(
                    self.model,
                    self.X_test[task_indices],
                    persona_results['explanations']
                ),
                'consistency': self.metrics.compute_consistency(
                    persona_results['explanations']
                )
            }
            
            results['personas'][persona.name] = persona_results
        
        # Compute overall metrics
        all_decisions = []
        all_correct = []
        all_trust = []
        all_explanations = []
        
        for persona_name, persona_data in results['personas'].items():
            all_decisions.extend(persona_data['decisions'])
            all_correct.extend(persona_data['correct_decisions'])
            all_trust.extend(persona_data['trust_scores'])
            all_explanations.extend(persona_data['explanations'])
        
        results['overall_metrics'] = {
            'decision_quality': self.metrics.compute_decision_quality(all_decisions, all_correct),
            'trust_calibration': self.metrics.compute_trust_calibration(all_trust, 0.85, 0.8),
            'comprehensibility': self.metrics.compute_comprehensibility(all_explanations),
            'cognitive_load': self.metrics.compute_cognitive_load(all_explanations),
            'fidelity': self.metrics.compute_fidelity(
                self.model, self.X_test[task_indices], all_explanations
            ),
            'consistency': self.metrics.compute_consistency(all_explanations)
        }
        
        logger.info("User study completed")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save study results to files.
        
        Args:
            results: Study results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_path / 'results.json'
        # Convert numpy types to native Python types for JSON
        def convert_to_serializable(obj):
            # Handle NumPy scalar types
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # Skip heavy / non-serializable objects from explanation payloads
            if obj.__class__.__name__ == 'Explanation':
                # LIME Explanation objects are large and not JSON-serializable.
                # We drop them from persisted results (they can be regenerated).
                return None
            
            if isinstance(obj, dict):
                serializable_dict = {}
                for k, v in obj.items():
                    # Drop embedded explanation objects
                    if k == 'explanation_object':
                        continue
                    serializable_dict[k] = convert_to_serializable(v)
                return serializable_dict
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {json_path}")
        
        # Save CSV summary
        csv_path = output_path / 'task_summary.csv'
        df = pd.DataFrame(results['task_details'])
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved task summary to {csv_path}")

