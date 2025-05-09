# Models

This folder stores trained machine learning models used for predicting activity and selectivity of S1R/S2R compounds.

## Subfolders
- `classification/` – Binary classifiers (e.g., XGBoost, ExtraTrees)
- `regression/` – pActivity prediction models
- `multiclass/` – Models for multiclass activity/selectivity prediction

> Note: Large models may be stored externally or compressed using joblib's `compress=3`.
