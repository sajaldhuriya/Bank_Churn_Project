# Machine Learning Model Summary

This document summarizes the performance of various machine learning models trained to predict bank customer churn.

## Performance Comparison

| Model | Best Parameters | Recall (Churn) | Precision (Churn) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C: 0.5, class_weight: 'balanced'` | 0.705 | 0.380 | 0.494 | 0.432 | 0.762 |
| **Decision Tree** | `max_depth: 9, min_samples_leaf: 8` | 0.435 | 0.644 | 0.519 | 0.590 | 0.803 |
| **Random Forest** | `n_estimators: 154, max_depth: 11` | 0.681 | 0.510 | 0.583 | 0.649 | 0.840 |
| **SVM** | `kernel: 'rbf', gamma: 0.1, C: 1` | 0.710 | 0.462 | 0.560 | 0.627 | 0.827 |
| **KNN** | `n_neighbors: 21, weights: 'distance'` | 0.248 | 0.842 | 0.383 | 0.597 | 0.797 |
| **XGBoost** | `n_estimators: 314, max_depth: 3` | 0.442 | 0.738 | 0.553 | 0.670 | 0.844 |

*Note: Metrics are based on the test set using default thresholds. Tuned thresholds were used to optimize Recall for business needs.*

## Key Findings

- **Best Overall Performance**: **Random Forest** provides a strong balance between Precision and Recall with the highest F1-Score (0.583) and high ROC-AUC (0.840).
- **Best for Recall**: **SVM** and **Logistic Regression** are most effective at identifying potential churners (Recall $\approx$ 0.71), though they suffer from lower Precision.
- **Best for Precision**: **KNN** has the highest Precision (0.842), meaning when it predicts churn, it is very likely correct, but it misses a significant number of actual churners.
- **Most Robust**: **XGBoost** achieves the highest ROC-AUC (0.844) and PR-AUC (0.670), indicating excellent general ability to distinguish between classes.

## Conclusion

For a business focused on **minimizing churn (maximizing recall)**, the **SVM** or **Random Forest** (with threshold tuning) are the recommended models. If the cost of retention campaigns is high and we need to be **precise about who we target**, **XGBoost** or **Random Forest** provide the best trade-off.
