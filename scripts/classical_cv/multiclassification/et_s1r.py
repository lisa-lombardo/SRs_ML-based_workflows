#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import ExtraTreesClassifier
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
import matplotlib
matplotlib.use('Agg')

def plot_confusion_matrix(cm, fold, labels, output_dir):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.title(f'Confusion Matrix - Fold {fold}', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'confusion_matrix_fold{fold}.png')
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train ExtraTreesClassifier model for multiclass with cross-validation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input dataset (CSV)')
    parser.add_argument('--output_dir', type=str, default='results/classical_cv/multiclass', help='Directory to save outputs')
    args = parser.parse_args()

    metrics_dir = os.path.join(args.output_dir, 'metrics')
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(args.input_file)
    X = df.drop(columns=['selectivity', 'pActivity', 'activity', 'class_label'])
    y = df['class_label']

    feature_names = X.columns
    joblib.dump(feature_names, os.path.join(args.output_dir, "selected_features.joblib"))

    stratkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    def objective(trial):
        clf = ExtraTreesClassifier(
            n_estimators=trial.suggest_int('n_estimators', 417, 823),
            max_depth=trial.suggest_int('max_depth', 10, 29),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 5),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 3),
            bootstrap=False,
            class_weight='balanced_subsample',
            random_state=1,
            n_jobs=3
        )
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
        auc_results = cross_val_score(pipeline, X, y, cv=stratkfold, scoring='roc_auc_ovr_weighted', n_jobs=3)
        return np.mean(auc_results)

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_params['bootstrap'] = False
    clf_best = ExtraTreesClassifier(**best_params, random_state=1, n_jobs=3)
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
        joblib.dump(pipeline, os.path.join(models_dir, f"et_pipeline_fold_{fold}.joblib"))

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        auc = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        plot_confusion_matrix(cm, fold, [0, 1, 2], metrics_dir)

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
    joblib.dump(pipeline, os.path.join(models_dir, "et_final_pipeline.joblib"))
    print(f"Final model saved to {models_dir}/et_final_pipeline.joblib")

if __name__ == '__main__':
    main()
