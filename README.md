# SRs_ML-based_workflows

A collection of cheminformatics and machine learning workflows for modeling Sigma-1 and Sigma-2 receptor (S1R, S2R) selectivity and activity using Mordred descriptors, molecular fingerprints, and multiple classifiers/regressors.

---

## Overview
This project provides data preprocessing, model training (nested CV + final models), and external validation scripts for classification, regression, and multiclass approaches applied to S1R/S2R ligands.

---

## Folder Structure
```
SRs_ML-based_workflows/
├── data/                      # All datasets (raw, features, external, etc.)
├── scripts/                  # Python scripts organized by task
│   ├── data_preprocessing/   # Cleaning, featurization
│   ├── nested_cv/            # Nested CV workflows
│   ├── final_models/         # Final model training (5-fold CV)
│   └── external_validation/  # Scripts for external evaluation
├── requirements.txt          # Python dependencies
├── .gitignore                # Files/folders to exclude from Git
└── README.md                 # You are here
```

---

## Installation
Create a conda or virtualenv with Python ≥ 3.8 and install dependencies:

```bash
pip install -r requirements.txt
```

You may also need to install RDKit via conda:
```bash
conda install -c rdkit rdkit
```

---

## How to Run

### 1. Preprocessing
```bash
python scripts/data_preprocessing/SRs_data-cleaning.py
```

### 2. Model Training
- **Nested CV**: Run models in `scripts/nested_cv/[classification|regression|multiclass]/`
- **Final CV**: Scripts in `scripts/final_models/`

### 3. External Validation
```bash
python scripts/external_validation/classification.py --input data/external/... --output results/
```

---

## Models
Saved `.joblib` pipelines for:
- XGBoost, ExtraTrees, SVM, kNN classifiers
- ExtraTrees and XGBoost regressors
- ECFP-based selectivity models

> 🧱 If models exceed GitHub’s size limit, download them via a shared link (see `download_instructions.md`).

---

##  License
Include an open license (e.g., MIT) if you want others to reuse this work legally.

---

##  Acknowledgements
This work leverages RDKit, Mordred, scikit-learn, XGBoost, and Optuna.

---

##  Contact
Feel free to open an [issue](https://github.com/lisa-lombardo/SRs_ML-based_workflows/issues) for questions or suggestions.

