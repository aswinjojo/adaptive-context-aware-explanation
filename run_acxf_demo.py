"""
Main demo script for ACXF system.

Run this script to demonstrate the Adaptive Context-Aware Explanation Generation system.
"""

import sys
import os
import logging
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.loaders import load_dataset
from src.utils.preprocessing import preprocess_data
from src.models.train_models import train_model
from src.profiling.user_profiler import UserProfiler, UserCategory
from src.explanation.context_engine import ContextAwareEngine, DecisionCriticality, TimePressure
from src.consistency.consistency_tracker import ConsistencyTracker
from src.interface.novice_view import NoviceView
from src.interface.intermediate_view import IntermediateView
from src.interface.expert_view import ExpertView

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the ACXF demo with selectable baseline models."
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=[
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "mlp",
            "xgboost",
        ],
        help="Baseline model to train before generating explanations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data reserved for testing (default: 0.2).",
    )
    return parser.parse_args()


def main():
    """Run ACXF demo."""
    args = parse_args()
    print("=" * 70)
    print("ACXF - Adaptive Context-Aware Explanation Generation")
    print("=" * 70)
    print()
    
    # Check for dataset
    data_dir = Path(__file__).parent / 'data'
    datasets = {
        'german_credit': data_dir / 'german_credit.csv',
        'diabetes': data_dir / 'diabetes.csv',
        'telco_churn': data_dir / 'telco_churn.csv'
    }
    
    # Also check for alternative telco churn filename
    telco_alt = data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if telco_alt.exists():
        datasets['telco_churn'] = telco_alt
    
    # Find available dataset
    available_dataset = None
    dataset_name = None
    
    for name, path in datasets.items():
        if path.exists():
            available_dataset = path
            dataset_name = name
            break
    
    if available_dataset is None:
        # Try to find any CSV file in data directory
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            available_dataset = csv_files[0]
            # Try to guess dataset type from filename
            filename_lower = available_dataset.name.lower()
            if 'telco' in filename_lower or 'churn' in filename_lower:
                dataset_name = 'telco_churn'
            elif 'german' in filename_lower or 'credit' in filename_lower:
                dataset_name = 'german_credit'
            elif 'diabetes' in filename_lower or 'diab' in filename_lower:
                dataset_name = 'diabetes'
            else:
                dataset_name = 'telco_churn'  # Default fallback
            print(f"Found dataset: {available_dataset.name}")
            print(f"Using as: {dataset_name}")
        else:
            print("ERROR: No dataset found in data/ directory.")
            print("Please add one of: german_credit.csv, diabetes.csv, telco_churn.csv")
            print("\nFor demo purposes, creating a synthetic dataset...")
        
        # Create synthetic dataset
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=5,
            n_redundant=2, n_classes=2, random_state=42
        )
        import pandas as pd
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        print("Using synthetic dataset for demonstration.")
    else:
        print(f"Loading dataset: {dataset_name}")
        X_df, y_series, feature_names = load_dataset(dataset_name, str(available_dataset))
        print(f"Dataset loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    
    # Preprocess
    print("\nPreprocessing data...")
    X_processed, y_processed, preprocess_info = preprocess_data(X_df, y_series)
    feature_names_processed = preprocess_info['feature_names']
    print(f"Preprocessed: {X_processed.shape}")
    
    # Train model
    print(f"\nTraining model ({args.model})...")
    model, X_train, X_test, y_train, y_test, metrics = train_model(
        X_processed,
        y_processed,
        model_type=args.model,
        test_size=args.test_size
    )
    print(f"Model trained - Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    # Initialize consistency tracker
    print("\nInitializing consistency tracker...")
    consistency_tracker = ConsistencyTracker(n_clusters=10)
    consistency_tracker.fit_clusters(X_train)
    
    # Create user personas
    print("\nCreating user personas...")
    personas = {
        'novice': UserProfiler(UserCategory.NOVICE),
        'intermediate': UserProfiler(UserCategory.INTERMEDIATE),
        'expert': UserProfiler(UserCategory.EXPERT)
    }
    
    # Update personas with questionnaire
    personas['novice'].update_from_questionnaire(1, 2, 2, 2)
    personas['intermediate'].update_from_questionnaire(3, 3, 3, 3)
    personas['expert'].update_from_questionnaire(5, 5, 4, 5)
    
    # Create views
    views = {
        'novice': NoviceView(),
        'intermediate': IntermediateView(),
        'expert': ExpertView()
    }
    
    # Demo: Generate explanations for different user types
    print("\n" + "=" * 70)
    print("Generating Adaptive Explanations")
    print("=" * 70)
    
    # Select a test instance
    test_idx = 0
    test_instance = X_test[test_idx]
    true_label = y_test[test_idx]
    prediction = model.predict_proba(test_instance.reshape(1, -1))[0]
    predicted_class = np.argmax(prediction)
    
    print(f"\nTest Instance #{test_idx}")
    print(f"True Label: {true_label}, Predicted: {predicted_class} (confidence: {prediction[predicted_class]:.3f})")
    print()
    
    for user_type, profiler in personas.items():
        print(f"\n{'=' * 70}")
        print(f"User Type: {user_type.upper()}")
        print(f"{'=' * 70}")
        
        # Create context engine
        context_engine = ContextAwareEngine(
            model, X_train, feature_names_processed, profiler
        )
        
        # Generate explanation
        explanation = context_engine.generate_explanation(
            test_instance,
            decision_criticality=DecisionCriticality.MEDIUM,
            time_pressure=TimePressure.NONE
        )
        
        print(f"\nExplanation Method: {explanation.get('method', 'Unknown')}")
        print(f"Detail Level: {explanation.get('context', {}).get('detail_level', 'Unknown')}")
        
        # Render explanation
        view = views[user_type]
        text_explanation = view.render(explanation)
        print(f"\n{text_explanation}")
        
        # Compute consistency
        consistency = consistency_tracker.compute_consistency_score(test_instance, explanation)
        print(f"\nConsistency Score: {consistency.get('consistency_score', 0.0):.3f}")
        
        # Add to tracker
        consistency_tracker.add_explanation(test_instance, explanation)
    
    # Generate counterfactual for expert
    print(f"\n{'=' * 70}")
    print("Counterfactual Explanation (Expert View)")
    print(f"{'=' * 70}")
    
    expert_engine = ContextAwareEngine(
        model, X_train, feature_names_processed, personas['expert']
    )
    counterfactual = expert_engine.generate_counterfactual(test_instance)
    
    print(f"\nOriginal Prediction: {counterfactual.get('original_prediction', [])}")
    print(f"Counterfactual Prediction: {counterfactual.get('counterfactual_prediction', [])}")
    print(f"Class Changed: {counterfactual.get('class_changed', False)}")
    
    if counterfactual.get('modifications'):
        print("\nModifications:")
        for mod in counterfactual['modifications'][:5]:
            print(f"  {mod.get('feature', 'Unknown')}: "
                  f"{mod.get('original_value', 0):.3f} -> {mod.get('new_value', 0):.3f}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run notebooks/demo_acxf.ipynb for interactive exploration")
    print("2. Run notebooks/evaluation_results.ipynb for full evaluation")
    print("3. Check experiments/ directory for results")


if __name__ == '__main__':
    main()

