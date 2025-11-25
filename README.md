# ACXF: Adaptive Context-Aware Explanation Generation for Tabular Data Classification

## 📌 Project Overview

ACXF is a comprehensive system for generating adaptive, context-aware explanations for tabular data classification models. The system dynamically adapts explanation formats based on user expertise, decision criticality, time pressure, and regulatory requirements.

### Key Features

- **User Profiling Module**: Adapts to user expertise (novice, intermediate, expert)
- **Context-Aware Explanation Engine**: Integrates SHAP (global) and LIME (local) explanations
- **Temporal Consistency Mechanism**: Ensures consistent explanations across similar instances
- **Multi-Level Explanation Interface**: Different views for different user types
- **Comprehensive Evaluation Framework**: Measures fidelity, consistency, comprehensibility, cognitive load, decision quality, and trust calibration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ACXF System Architecture                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│ User Profiler│─────▶│ Context Engine    │─────▶│ Explanation  │
│              │      │                  │      │ Interface    │
└──────────────┘      └──────────────────┘      └──────────────┘
       │                       │                          │
       │              ┌────────┴────────┐                │
       │              │                 │                │
       │         ┌────▼────┐      ┌────▼────┐           │
       │         │  SHAP   │      │  LIME   │           │
       │         │Explainer│      │Explainer│           │
       │         └─────────┘      └─────────┘           │
       │                                                  │
       │         ┌──────────────────────────┐            │
       └────────▶│ Consistency Tracker     │◀───────────┘
                 │ - Clustering            │
                 │ - Case Caching          │
                 │ - Smoothing             │
                 └──────────────────────────┘
```

## 📁 Project Structure

```
acxf/
├── data/                      # Dataset directory
│   ├── german_credit.csv
│   ├── diabetes.csv
│   └── telco_churn.csv
├── src/
│   ├── profiling/            # User profiling module
│   │   └── user_profiler.py
│   ├── explanation/           # Explanation engines
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── context_engine.py
│   ├── consistency/          # Temporal consistency
│   │   ├── consistency_tracker.py
│   │   └── case_cache.py
│   ├── interface/             # Multi-level interfaces
│   │   ├── novice_view.py
│   │   ├── intermediate_view.py
│   │   └── expert_view.py
│   ├── evaluation/            # Evaluation framework
│   │   ├── metrics.py
│   │   └── run_user_study.py
│   ├── models/                # Model training
│   │   └── train_models.py
│   └── utils/                 # Utilities
│       ├── preprocessing.py
│       └── loaders.py
├── notebooks/                 # Jupyter notebooks
│   ├── demo_acxf.ipynb
│   └── evaluation_results.ipynb
├── experiments/              # Results and plots
│   ├── results.json
│   └── plots/
├── README.md
├── requirements.txt
└── run_acxf_demo.py          # Main demo script
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd acxf
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Add datasets to `data/` directory**
   - Place `german_credit.csv`, `diabetes.csv`, or `telco_churn.csv` in the `data/` folder
   - If no datasets are provided, the system will use synthetic data for demonstration

## 💻 Usage

### Quick Start

Run the main demo script (defaults to Random Forest):

```bash
python run_acxf_demo.py
```

Select a baseline explicitly with `--model` (`random_forest`, `gradient_boosting`, `logistic_regression`, `mlp`, or `xgboost`):

```bash
python run_acxf_demo.py --model xgboost
```

This will:
1. Load or generate a dataset
2. Train the requested classification model
3. Initialize ACXF components
4. Generate adaptive explanations for different user types
5. Display results

### Using Jupyter Notebooks

1. **Demo Notebook** (`notebooks/demo_acxf.ipynb`):
   - Interactive exploration of ACXF features
   - Generate explanations for different user personas
   - Visualize explanations and counterfactuals

2. **Evaluation Notebook** (`notebooks/evaluation_results.ipynb`):
   - Run comprehensive user study simulation
   - Compute evaluation metrics
   - Generate plots and visualizations

### Programmatic Usage

```python
from src.utils.loaders import load_dataset
from src.utils.preprocessing import preprocess_data
from src.models.train_models import train_model
from src.profiling.user_profiler import UserProfiler, UserCategory
from src.explanation.context_engine import ContextAwareEngine, DecisionCriticality, TimePressure
from src.interface.novice_view import NoviceView

# Load and preprocess data
X_df, y_series, feature_names = load_dataset('diabetes', 'data/diabetes.csv')
X_processed, y_processed, preprocess_info = preprocess_data(X_df, y_series)

# Train model
model, X_train, X_test, y_train, y_test, metrics = train_model(
    X_processed, y_processed, model_type='random_forest'
)

# Create user profiler
profiler = UserProfiler(UserCategory.NOVICE)
profiler.update_from_questionnaire(1, 2, 2, 2)

# Create context engine
engine = ContextAwareEngine(model, X_train, preprocess_info['feature_names'], profiler)

# Generate explanation
explanation = engine.generate_explanation(
    X_test[0],
    decision_criticality=DecisionCriticality.MEDIUM,
    time_pressure=TimePressure.NONE
)

# Render explanation
view = NoviceView()
print(view.render(explanation))
```

## 🔧 Components

### 1. User Profiling Module

The `UserProfiler` class adapts to user expertise through:
- **Questionnaire-based scoring**: Initial assessment via ML experience, stats knowledge, domain expertise
- **Interaction-log-based adaptation**: Learns from user interactions
- **Performance-based refinement**: Updates based on user decision quality

**User Categories:**
- **Novice**: Simple text summaries, limited features, visual aids
- **Intermediate**: Feature rankings, moderate detail, visualizations
- **Expert**: Full SHAP plots, detailed attributions, counterfactuals

### 2. Context-Aware Explanation Engine

The `ContextAwareEngine` selects appropriate explanation methods:

- **SHAP (SHapley Additive exPlanations)**:
  - Global explanations for expert users
  - Local explanations for high-criticality decisions
  - Force plots and bar plots

- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - General-purpose local explanations
  - Suitable for intermediate and novice users
  - Counterfactual generation

**Context Factors:**
- User expertise level
- Decision criticality (low, medium, high, critical)
- Time pressure (none, low, medium, high)
- Regulatory requirements

### 3. Temporal Consistency Mechanism

The `ConsistencyTracker` ensures explanations remain consistent:

- **Feature-space clustering**: Groups similar instances using KMeans
- **Case caching**: Stores representative explanations
- **Similarity-based smoothing**: Smooths explanations using cached cases
- **Consistency scoring**: Measures explanation stability

### 4. Multi-Level Explanation Interface

Three specialized views:

- **NoviceView**: Simple text summaries with basic visualizations
- **IntermediateView**: Feature ranking charts with moderate detail
- **ExpertView**: Comprehensive SHAP plots, force plots, counterfactuals

### 5. Evaluation Framework

Comprehensive metrics:

- **Fidelity**: How well explanations match model behavior
- **Consistency**: Stability across similar instances
- **Comprehensibility**: Ease of understanding (user-category dependent)
- **Cognitive Load**: Mental effort required (NASA-TLX proxy)
- **Decision Quality**: Accuracy of user decisions with explanations
- **Trust Calibration**: Alignment between trust and actual quality

## 📊 Example Outputs

### Novice Explanation
```
The model's decision is primarily based on:

1. Feature_0 - This increases the likelihood of the prediction.
2. Feature_2 - This decreases the likelihood of the prediction.
3. Feature_5 - This increases the likelihood of the prediction.

Prediction confidence: 75.3%
```

### Intermediate Explanation
```
Feature Contribution Analysis:

Method: LIME

 1. Feature_0            ↑ 0.3421
 2. Feature_2            ↓ -0.2156
 3. Feature_5            ↑ 0.1892
 ...
```

### Expert Explanation
```
Detailed Explanation Report
==================================================

Method: SHAP (Local)
Type: local

Feature Contributions:
  1. Feature_0                                   0.342156
  2. Feature_2                                  -0.215643
  3. Feature_5                                   0.189234
  ...

Base Value (Expected): 0.523456

Predicted Probabilities:
  Class 0: 0.234567
  Class 1: 0.765433
  Predicted Class: 1
```

## 🧪 Experiments

### Running Evaluation

1. **Via Notebook:**
   ```bash
   jupyter notebook notebooks/evaluation_results.ipynb
   ```

2. **Via Python:**
   ```python
   from src.evaluation.run_user_study import UserStudySimulator
   
   simulator = UserStudySimulator(model, X_train, X_test, y_test, feature_names, tracker)
   results = simulator.run_study(n_tasks=50)
   simulator.save_results(results, 'experiments/')
   ```

### Results

Results are saved in `experiments/`:
- `results.json`: Complete evaluation results
- `task_summary.csv`: Task-level summary
- `plots/`: Generated visualizations

## 📈 Evaluation Metrics

The system evaluates explanations across six dimensions:

1. **Fidelity Score**: Correlation between explanation-based predictions and actual model predictions
2. **Consistency Score**: Average similarity of explanations for similar instances
3. **Comprehensibility Score**: User-category-appropriate explanation complexity
4. **Cognitive Load**: Estimated mental effort (based on features, method, time)
5. **Decision Accuracy**: User decision correctness with explanations
6. **Trust Calibration**: Alignment between user trust and system quality

## 🔬 Technical Details

### Dependencies

- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computations
- `scikit-learn>=1.3.0`: Machine learning models
- `shap>=0.42.0`: SHAP explanations
- `lime>=0.2.0.1`: LIME explanations
- `matplotlib>=3.7.0`: Visualization
- `seaborn>=0.12.0`: Statistical visualization

### Model Support

Currently supports:
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Multi-Layer Perceptron (MLPClassifier)
- XGBoost Classifier (`pip install xgboost`)

Extensible to any scikit-learn-compatible classifier.

### Poster Workflow

Need presentation-ready visuals like the sample poster? Follow `POSTER_GUIDE.md` for
step-by-step instructions covering evaluation reruns, plot generation, metrics export,
and recommended layout of the resulting assets.

## 📝 Citation

If you use ACXF in your research, please cite:

```bibtex
@software{acxf2024,
  title={ACXF: Adaptive Context-Aware Explanation Generation for Tabular Data Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/acxf}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SHAP library for Shapley value explanations
- LIME library for local explanations
- scikit-learn for machine learning models

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a research implementation. For production use, additional testing and validation are recommended.

