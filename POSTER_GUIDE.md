# Poster Creation Workflow

Use this guide to regenerate all quantitative results and visual assets needed to
assemble a conference-style poster (similar to the sample provided).

---

## 1. Environment Setup

```bash
cd acxf
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

If you intend to benchmark the XGBoost baseline, the dependency is already listed
in `requirements.txt`. Install GPU extras as needed.

---

## 2. Reproduce Core Demo (Optional Sanity Check)

```bash
python run_acxf_demo.py --model random_forest
```

Key outputs:
- Console summaries for novice/intermediate/expert personas
- Counterfactual example for expert persona

---

## 3. Full Evaluation + Plot Generation

All poster figures originate from the evaluation notebook. You can re-run it
either interactively or headlessly.

### Interactive
```bash
jupyter notebook notebooks/evaluation_results.ipynb
# Kernel > Restart & Run All
```

### Headless (repeatable for automation)
```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 \
  --output evaluation_results-ran.ipynb \
  notebooks/evaluation_results.ipynb
```

Artifacts are written to `experiments/`:

| File | Usage on Poster |
| --- | --- |
| `experiments/results.json` | Metrics table for Model Performance section |
| `experiments/task_summary.csv` | Highlights for Summary/Future Work |
| `experiments/plots/metrics_comparison.png` | Model bar chart (Performance panel) |
| `experiments/plots/novice_explanation.png` | Screenshot for persona panel |
| `experiments/plots/intermediate_explanation.png` | Persona panel |
| `experiments/plots/expert_explanation.png` | Persona panel |
| `experiments/plots/counterfactual.png` | Counterfactual callout |
| `experiments/plots/decision_accuracy.png` | Summary/Future work chart |

---

## 4. Suggested Poster Layout

Match the provided sample by arranging panels as follows:

1. **Background & Motivation** – derive bullet points from Section 1 of the paper +
   dataset descriptions inside `README.md`.
2. **Overview / Workflow** – reuse architecture diagram (`README.md`) or paste
   the system graphic from the notebook (export to PNG if needed).
3. **Distribution of Datasets** – create quick histograms inside
   `notebooks/demo_acxf.ipynb` (cells already generate these plots) and export
   via `File > Download as PNG`.
4. **Model Performance Assessment** – use `metrics_comparison.png` plus a table
   summarizing `results.json` (import into Excel/PowerPoint for formatting).
5. **Persona-specific Explanations** – place the three explanation PNGs side by side.
6. **Counterfactual + Consistency** – include `counterfactual.png` and cite the
   `ConsistencyTracker` outputs reported in `results.json`.
7. **Summary & Future Work** – quote `decision_accuracy.png` and bullet the key
   metrics (`fidelity`, `consistency`, `trust_calibration`) from `results.json`.

Logos (IBM Research, MIT, university) can be swapped with your institution’s logos.

---

## 5. Exporting the Poster

1. Assemble the layout using PowerPoint, Keynote, Figma, or Canva (36x48 in. canvas
   matches the sample).
2. Insert each PNG at 300 DPI.
3. Copy textual snippets from README/notebooks into the corresponding panels.
4. Export as PDF and high-resolution PNG for printing/deployment.

---

## 6. Quick Validation Checklist

- [ ] `results.json` updated timestamp matches latest run.
- [ ] All PNG assets re-generated (check modified time).
- [ ] Poster includes citations [1]-[9] consistent with manuscript.
- [ ] Mention adaptive profiling, context engine, and temporal consistency.
- [ ] Future-work bullet reflects freshest experimental insights.

Following the steps above ensures the codebase outputs and poster narrative stay
perfectly aligned with the ACXF proposal. Happy presenting!


