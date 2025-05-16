#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from imblearn.under_sampling import RandomUnderSampler
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train final SVM model with cross-validation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input dataset (CSV)')
    parser.add_argument('--output_dir', type=str, default='results/classical_cv', help='Directory to save outputs')
    args = parser.parse_args()

    metrics_dir = os.path.join(args.output_dir, 'metrics')
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(args.input_file)
    X = df.drop(columns=['selectivity', 'pActivity', 'activity', 'labels'])
    y = df['labels']

    feature_names = X.columns
    joblib.dump(feature_names, os.path.join(args.output_dir, "selected_features.joblib"))

    stratkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    def objective(trial):
        clf = SVC(
            kernel='rbf',
            C=trial.suggest_float('C', 2, 61, log=True),
            gamma=trial.suggest_float('gamma', 2.0e-4, 8.9e-4, log=True),
            class_weight='balanced',
            probability=True
        )
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
        auc_results = cross_val_score(pipeline, X, y, cv=stratkfold, scoring='roc_auc')
        return np.mean(auc_results)

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    clf_best = SVC(**best_params, probability=True)
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf_best)])

    results_df = pd.DataFrame()
    rus = RandomUnderSampler(random_state=1)

    auc_scores, f1_scores, precision_scores, recall_scores, mcc_scores = [], [], [], [], []
    fold = 0

    for train_idx, test_idx in stratkfold.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

        pipeline.fit(X_train_resampled, y_train_resampled)
        joblib.dump(pipeline, os.path.join(models_dir, f"svm_pipeline_fold_{fold}.joblib"))

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plot_confusion_matrix(cm, fold, [0, 1], metrics_dir)

        results_df = pd.concat([results_df, pd.DataFrame({
            'Fold': [fold], 'AUC': [auc], 'F1': [f1], 'Precision': [precision],
            'Recall': [recall], 'MCC': [mcc], 'Confusion Matrix': [cm.tolist()]
        })], ignore_index=True)

        auc_scores.append(auc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mcc_scores.append(mcc)

    results_df.loc['mean'] = ['mean'] + [
        np.mean(auc_scores), np.mean(f1_scores), np.mean(precision_scores),
        np.mean(recall_scores), np.mean(mcc_scores), 'NA'
    ]

    results_df.to_csv(os.path.join(metrics_dir, "classical_cv_results_summary_with_cm.csv"), index=False)
    print(f"Results saved to {metrics_dir}/classical_cv_results_summary_with_cm.csv")

    pipeline.fit(X, y)
    joblib.dump(pipeline, os.path.join(models_dir, "svm_final_pipeline_s2r.joblib"))
    print(f"Final model saved to {models_dir}/svm_final_pipeline_s2r.joblib")

if __name__ == '__main__':
    main()
