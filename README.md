# Job Salary & Vacation Prediction — Classification & Regression on Tabular Data

An end-to-end machine learning project on a job/salary dataset, covering exploratory data analysis, preprocessing, and the training/evaluation of classification and regression models with scikit-learn.

**Tasks:**
- **Classification (multiclass):** predict `vacation` type — `No Vacation`, `Small`, `Medium`, `Large`
- **Regression:** predict `salary`

## Dataset

The dataset contains job-related records with the following attributes:

| Attribute | Type | Description |
|---|---|---|
| `job_title` | categorical | Job role |
| `experience_years` | numeric | Years of experience |
| `education_level` | categorical | Highest education level completed |
| `skills_count` | numeric | Number of skills the employee has |
| `industry` | categorical | Industry of employment |
| `company_size` | categorical | Size of the company |
| `remote_work` | categorical | Whether the job is remote |
| `certifications` | numeric | Number of certifications |
| `location` | categorical | Work location (or Remote) |
| `vacation` | categorical (target) | Vacation allowance tier |
| `salary` | numeric (target) | Salary |

> Data files are not included in this repository. See [`data/README.md`](data/README.md) for details on obtaining them.

## Repository Structure

```
.
├── notebook/
│   └── tema1_ml.ipynb        # Full pipeline: EDA, preprocessing, modeling, evaluation
├── report/
│   └── raport_tema1.pdf      # Written report with analysis and interpretation
├── data/
│   └── README.md             # Notes on the dataset source and structure (no raw data)
├── requirements.txt
└── README.md
```

## Methodology

The project follows a staged, ablation-style approach rather than a single exhaustive grid search — each modeling decision is introduced and justified individually so its effect on performance can be tracked.

### 1. Exploratory Data Analysis
- Missing value inspection
- Attribute type analysis (numeric continuous vs. categorical) with summary statistics, boxplots, and histograms
- Class balance analysis for the classification target
- Correlation analysis: numeric–numeric (Pearson), categorical–categorical (Chi-squared), and attribute–target correlations

### 2. Preprocessing
- Missing value imputation (with justification per column)
- Outlier detection and treatment using the IQR method
- Redundant/highly-correlated attribute analysis and removal
- Feature scaling with `StandardScaler`

### 3. Classification
- **Baseline:** `DecisionTreeClassifier`
- **Ablation:** systematic variation of hyperparameters (`max_depth`, `min_samples_leaf`, class weighting) tracked in a cumulative comparison table
- **Final evaluation:** classification report (precision/recall/F1 per class), confusion matrix, and feature importance on both validation and test sets

### 4. Regression
- **Baseline:** `LinearRegression`
- **Regularization:** `Ridge` (L2) and `Lasso` (L1), with hyperparameter (alpha) tuning
- **Final evaluation:** comparison table (MAE, RMSE, R²) across models, predicted-vs-actual plot, and train/validation error curves

## Results Summary

| Task | Best model | Key metric |
|---|---|---|
| Classification | Decision Tree (tuned, `class_weight='balanced'`) | **73%** accuracy on test |
| Regression | Linear Regression | **R² ≈ 0.95** on validation and test |

Detailed per-class metrics, ablation tables, and discussion of the results are available in the [notebook](notebook/tema1_ml.ipynb) and the [full report](report/raport_tema1.pdf).

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (preprocessing, `DecisionTreeClassifier`, `RandomForestClassifier`, `LinearRegression`, `Ridge`, `Lasso`, metrics)
- SciPy (Chi-squared test)
- matplotlib, seaborn

## Getting Started

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebook/tema1_ml.ipynb
```

## License

This project is shared for educational and portfolio purposes.
