# ACXF Quick Start Guide

## Installation

```bash
cd acxf
pip install -r requirements.txt
```

## Run Demo

```bash
python run_acxf_demo.py
```

## Run Jupyter Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/demo_acxf.ipynb` for interactive demo
3. Open `notebooks/evaluation_results.ipynb` for full evaluation

## Expected Output

The demo will:
1. Load or generate a dataset
2. Train a Random Forest model
3. Generate explanations for three user types (novice, intermediate, expert)
4. Display adaptive explanations
5. Show counterfactual examples
6. Demonstrate temporal consistency

## Key Components

- **User Profiling**: Adapts to user expertise level
- **SHAP & LIME**: Dual explanation methods
- **Context-Aware**: Adapts to decision criticality and time pressure
- **Consistency Tracking**: Ensures stable explanations
- **Multi-Level Views**: Different interfaces for different users

## Next Steps

1. Add your own datasets to `data/` directory
2. Customize user personas in the notebooks
3. Run full evaluation with 50 tasks per persona
4. Explore the generated plots in `experiments/plots/`


