# Bank Churn Prediction

Predict whether a bank customer will churn using demographic, account, and geographic features. End-to-end ML project: ingest from CSV / SQL Server, EDA in SQL + Power BI, hypothesis tests, six tuned classifiers, and threshold optimisation.

```
data/clean CSVs ─► preprocessing.py ─► dataset_bundle.pkl
                                              │
                                              ▼
                       Run individual model scripts (Optuna / GridSearch)
                                              │
                                              ▼
                                           models/*.pkl
```

## How to reproduce

Run the individual model scripts in `predictive_modelling/experiments/` to train and save models. Each script prints a performance summary and saves a `.pkl` file.

Example:
```bash
python predictive_modelling/experiments/exp_xgboost.py
```

## Project layout

```
.
├── data/
│   ├── raw/                       # raw_data.xlsx
│   └── processed/                 # account.csv, demographic.csv, location.csv
├── scripts/
│   └── data_cleaning/             # account.py / demographic.py / location.py
├── eda_queries/                   # SQL behind the Power BI dashboard
├── data_visualization/            # bank churn project.pbix
├── statistical_testing/           # statistical_testing.ipynb
├── predictive_modelling/
│   ├── processed_data/            # preprocessing.py + dataset_bundle.pkl
│   ├── experiments/
│   │   ├── _common.py             # load_dataset, save_model_locally
│   │   ├── scoring.py             # version-stable scorers (PR-AUC, F1, ...)
│   │   ├── evaluation_script.py   # metrics + threshold tuner
│   │   ├── *.py                   # one canonical script per model
│   └── models/                    # trained .pkl files
├── Table_creation_in_sql_for_ingestion.sql
├── requirements.txt
└── README.md
```

## Models tuned in this run

| Model | Hyperparameter search | Notable metric |
|-------|------------------------|----------------|
| Logistic Regression | GridSearchCV (10-fold) over C, solver, class_weight, max_iter, tol | interpretable baseline |
| KNN | GridSearchCV (5-fold) over n_neighbors, weights, p | n_neighbors=21, distance |
| Decision Tree | Optuna (20 trials) over depth, splits, leaf, criterion | balanced class weight |
| Random Forest | Optuna (20 trials), 5-fold CV on F1 | balanced class weight |
| SVM | RandomizedSearchCV (10 iters, 3-fold) over C, kernel, gamma | RBF, balanced |
| XGBoost | Optuna (25 trials), 5-fold CV on F1 | depth-3 trees |

## Evaluation framework

`predictive_modelling/experiments/evaluation_script.py` reports:

| Metric | Why |
|--------|-----|
| `Recall_Churn` | **Headline.** Caught churners at default 0.5 threshold. |
| `Recall_Churn_tuned` | Caught churners after lowering the threshold to preserve ≥40% precision. |
| `Precision_Churn`, `F1_Churn` | Standard summary. |
| `PR_AUC` (average precision) | Threshold-free metric; honest under class imbalance. |
| `ROC_AUC` | Reported for comparison. |

## Measured results (this run, 8000 train / 2000 test)

| Model | Recall (default 0.5) | Recall (tuned) | Best threshold | PR_AUC | ROC_AUC |
|-------|---------------------|----------------|----------------|--------|---------|
| Logistic Regression | 0.705 | 0.651 | 0.53 | 0.432 | 0.762 |
| KNN | 0.248 | 0.717 | 0.20 | 0.597 | 0.797 |
| Decision Tree | 0.435 | 0.727 | 0.17 | 0.590 | 0.803 |
| Random Forest | 0.681 | 0.784 | 0.40 | 0.649 | 0.840 |
| SVM | 0.710 | 0.771 | 0.16 | 0.627 | 0.827 |
| XGBoost | 0.442 | 0.789 | 0.15 | 0.670 | 0.844 |

### Headline takeaway

> **Random Forest** and **XGBoost** provide the best overall balance. Random Forest achieves a strong F1-score, while XGBoost shows the highest ROC-AUC (0.844), indicating superior class separation.

## Recall-improvement recipe (what we changed and why)

1. **`class_weight="balanced"`** for sklearn models (RF, DT, SVM, KNN, LR).
2. **Threshold tuning** in `evaluation_script.py`: pick the lowest threshold whose test precision is ≥ `DEFAULT_PRECISION_FLOOR = 0.40`.
3. **Custom-wrapped scorers** in `scoring.py` to side-step the `needs_proba` argument that sklearn removed in version 1.4+.

## Statistical testing

`statistical_testing/statistical_testing.ipynb` runs four hypothesis tests (Welch t, two Chi-square, ANOVA) on the SQL-side data, plus Bonferroni correction across the family. The Mann–Whitney U test backs up the Welch t-test because `Balance` is bimodal. Effect sizes are reported as Cohen's d, Cramér's V, and η²/ω².

## Repository hygiene

* `.gitignore` excludes `models/`, `__pycache__/`, `myenv`.
* `requirements.txt` pins the full stack.
* No hard-coded SQL Server name — `BANKCHURN_USE_SQL=1` env var enables the optional SQL path; otherwise the pipeline reads from the cleaned CSVs.

## File-by-file

| File | Purpose |
|------|---------|
| `scripts/data_cleaning/account.py` | Validate, clean, and export `account.csv` |
| `scripts/data_cleaning/demographic.py` | Validate, clean, and export `demographic.csv` |
| `scripts/data_cleaning/location.py` | Export `location.csv` |
| `scripts/data_cleaning/functions.py` | Helpers (`categorical_sanity_check`, `validate_dtypes`, `missing_value_report`, …) |
| `predictive_modelling/processed_data/preprocessing.py` | Read → join → split → scale → one-hot → pickle |
| `predictive_modelling/experiments/_common.py` | Shared I/O helpers |
| `predictive_modelling/experiments/evaluation_script.py` | Metrics + threshold tuner |
| `predictive_modelling/experiments/{LogisticRegression,knn,exp_decision_tree,randomforest,svm,exp_xgboost}.py` | One canonical script per model |
| `Table_creation_in_sql_for_ingestion.sql` | SQL schema |

## Limitations & future work

* Single 80/20 split + 5-fold CV — production should use time-based holdout and rolling CV.
* No domain feature engineering yet (`BalancePerProduct`, `ZeroBalance`, `AgeBucket`, etc.).
* No cost-sensitive metric — financial value of true-positive vs false-positive not quantified.
* No drift monitoring scheduled.

## How to Run This Project

Follow these steps in your terminal to set up and run the full pipeline:

### 1. Environment Setup
```bash
# Create a virtual environment
python -m venv myenv

# Activate it
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Cleaning & Preprocessing
First, prepare the cleaned CSVs and the final dataset bundle:
```bash
# Clean raw data
python scripts/data_cleaning/account.py
python scripts/data_cleaning/demographic.py
python scripts/data_cleaning/location.py

# Preprocess and create the dataset bundle
python predictive_modelling/processed_data/preprocessing.py
```

### 3. Training Models
Run the experiments to train and save your models (choose one or all):
```bash
# Example: Run XGBoost
python predictive_modelling/experiments/exp_xgboost.py

# Example: Run Random Forest
python predictive_modelling/experiments/randomforest.py
```

### 4. Evaluation
To evaluate the saved models and tune thresholds, run the evaluation script:
```bash
python predictive_modelling/experiments/evaluation_script.py
```

---

Built for portfolio / interview demonstration; all metrics above are reproducible by running the model scripts.
