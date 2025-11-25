# How to Run ACXF with Your Dataset

## Your Dataset
You have: `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` folder

## Step-by-Step Instructions

### 1. Install Dependencies (First Time Only)
```bash
cd acxf
pip install -r requirements.txt
```

### 2. Run the Demo (Choose One Method)

#### Method A: Quick Command-Line Demo
```bash
python run_acxf_demo.py --model random_forest
```

This will:
- ✅ Detect your Telco Churn dataset automatically
- ✅ Load and preprocess the data
- ✅ Train the model you specify (`--model xgboost`, `--model mlp`, etc.)
- ✅ Generate explanations for Novice, Intermediate, and Expert users
- ✅ Show counterfactual examples
- ✅ Display all results

#### Method B: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/demo_acxf.ipynb
```
Then click "Run All" to execute all cells.

#### Method C: Full Evaluation Study
```bash
jupyter notebook notebooks/evaluation_results.ipynb
```
This runs a comprehensive evaluation with 50 tasks per user persona.

## Expected Output

When you run `python run_acxf_demo.py`, you should see:

```
======================================================================
ACXF - Adaptive Context-Aware Explanation Generation
======================================================================

Loading dataset: telco_churn
Dataset loaded: [rows] samples, [cols] features

Preprocessing data...
Preprocessed: [shape]

Training model...
Model trained - Test Accuracy: [accuracy]

Initializing consistency tracker...

======================================================================
Generating Adaptive Explanations
======================================================================

Test Instance #0
True Label: [label], Predicted: [prediction] (confidence: [prob])

======================================================================
User Type: NOVICE
======================================================================

Explanation Method: LIME
Detail Level: low

[Simple text explanation]

======================================================================
User Type: INTERMEDIATE
======================================================================

[Feature ranking explanation]

======================================================================
User Type: EXPERT
======================================================================

[Detailed SHAP explanation]

======================================================================
Counterfactual Explanation (Expert View)
======================================================================

[Counterfactual details]

Demo completed successfully!
```

## Troubleshooting

### If you get import errors:
```bash
# Make sure you're in the acxf directory
cd acxf

# Verify Python can find the modules
python -c "import sys; sys.path.insert(0, 'src'); from src.utils.loaders import load_dataset; print('✓ OK')"
```

### If the dataset isn't detected:
The script now automatically looks for `WA_Fn-UseC_-Telco-Customer-Churn.csv`. If it still doesn't work, check:
- File is in `acxf/data/` folder
- File name is exactly `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- File has a 'Churn' column (case-sensitive)

## What Happens Next?

After running, you'll see:
1. **Model Performance**: Accuracy metrics
2. **Three Explanation Types**: One for each user expertise level
3. **Visualizations**: Saved in `experiments/plots/` (if using notebooks)
4. **Consistency Scores**: How stable explanations are across similar instances

## Building the Poster

Need the final presentation board? Follow `POSTER_GUIDE.md` for the reproducible
workflow that re-generates metrics, exports the persona visuals, and maps each
asset to the recommended poster sections.

## Quick Test

To quickly verify everything works:
```bash
cd acxf
python run_acxf_demo.py
```

The script will automatically use your Telco Churn dataset!

